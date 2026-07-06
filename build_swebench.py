#!/usr/bin/env python3
"""Build missing SWE-bench instance images and publish them to GHCR.

``publish.py`` mirrors images that already exist in a public source registry.
Some SWE-bench train instances do not have public ``swebench/sweb.eval.*``
images, so they must be built from the SWE-bench harness first. This script
uses the official harness to build those images locally, tags them directly as
``ghcr.io/<org>/sweb.eval.<arch>.<instance_id>:<tag>``, pushes them, and reuses
the visibility checks from ``publish.py``.

The script is intentionally conservative: unsupported rows are recorded in the
manifest and skipped, existing GHCR targets can be skipped, and every successful
push is resumable through the JSONL manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from publish import (
    DEFAULT_ORG,
    REGISTRY,
    Manifest,
    docker_exists,
    docker_login,
    do_verify,
    normalize_repo_link,
    resolve_creds,
)


@dataclass(frozen=True)
class BuildTarget:
    """One SWE-bench row plus the image name the harness will build."""

    row: dict[str, Any]
    spec: Any
    dataset: str
    split: str

    @property
    def instance_id(self) -> str:
        return self.row["instance_id"]

    @property
    def target_ref(self) -> str:
        return self.spec.instance_image_key

    @property
    def target_name(self) -> str:
        # ``ghcr.io/<org>/<package>:<tag>`` -> ``<package>``.
        return self.target_ref.split("/", 2)[2].rsplit(":", 1)[0]


class BuildManifest(Manifest):
    """Manifest variant that also tracks unsupported rows."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.unsupported: set[str] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "unsupported" and rec.get("instance_id"):
                    self.unsupported.add(rec["instance_id"])

    def is_unsupported(self, instance_id: str) -> bool:
        return instance_id in self.unsupported


def _run(cmd: list[str], *, input_text: str | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def _parse_instance_ids(args) -> set[str] | None:
    values: list[str] = []
    values.extend(args.instance_id or [])
    if args.instance_ids:
        values.extend(args.instance_ids.replace(",", "\n").splitlines())
    if args.instance_ids_file:
        values.extend(Path(args.instance_ids_file).read_text().replace(",", "\n").splitlines())
    ids = {v.strip() for v in values if v.strip()}
    return ids or None


def _load_rows(args) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs = {}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    ds = load_dataset(args.dataset_name, split=args.split, **kwargs)
    rows = [dict(row) for row in ds]

    wanted = _parse_instance_ids(args)
    if wanted is not None:
        found = {row["instance_id"] for row in rows}
        missing = sorted(wanted - found)
        if missing:
            print(f"! {len(missing)} requested instance id(s) not in {args.dataset_name}/{args.split}:")
            for iid in missing[:20]:
                print(f"  {iid}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        rows = [row for row in rows if row["instance_id"] in wanted]

    if args.limit is not None:
        rows = rows[: args.limit]

    if args.shard:
        idx, total = (int(x) for x in args.shard.split("/", 1))
        if not (1 <= idx <= total):
            raise SystemExit(f"--shard {args.shard}: index must be in 1..{total}")
        rows = rows[idx - 1 :: total]
    return rows


def _make_targets(args, rows: list[dict[str, Any]], manifest: BuildManifest) -> tuple[list[BuildTarget], int]:
    from swebench.harness.test_spec.test_spec import make_test_spec

    targets: list[BuildTarget] = []
    unsupported = 0
    namespace = f"{REGISTRY}/{args.org}"
    for row in rows:
        iid = row["instance_id"]
        if manifest.is_unsupported(iid):
            unsupported += 1
            continue
        try:
            spec = make_test_spec(
                row,
                namespace=namespace,
                instance_image_tag=args.tag,
                env_image_tag=args.env_image_tag,
                arch=args.arch,
            )
        except Exception as exc:  # noqa: BLE001 - record and keep the shard moving.
            unsupported += 1
            rec = {
                "status": "unsupported",
                "dataset": args.dataset_name,
                "split": args.split,
                "instance_id": iid,
                "repo": row.get("repo"),
                "version": row.get("version"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if not args.dry_run:
                manifest.record(rec)
            if unsupported <= 20:
                print(
                    f"[unsupported] {iid} repo={row.get('repo')} version={row.get('version')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            continue
        targets.append(BuildTarget(row=row, spec=spec, dataset=args.dataset_name, split=args.split))
    return targets, unsupported


def _target_record(target: BuildTarget, status: str, **extra) -> dict:
    return {
        "status": status,
        "source": "swebench-harness-build",
        "target": target.target_ref,
        "target_name": target.target_name,
        "dataset": target.dataset,
        "split": target.split,
        "instance_id": target.instance_id,
        "repo": target.row.get("repo"),
        "version": target.row.get("version"),
        **extra,
    }


def _label_image(target_ref: str, repo_link: str) -> tuple[bool, str]:
    dockerfile = (
        f"FROM {target_ref}\n"
        f"LABEL org.opencontainers.image.source={json.dumps(repo_link)}\n"
    )
    return _run(["docker", "build", "-t", target_ref, "-"], input_text=dockerfile)


def _push_image(target_ref: str, retries: int) -> tuple[bool, str]:
    log: list[str] = []
    for attempt in range(1, retries + 2):
        rc, out = _run(["docker", "push", target_ref])
        log.append(out)
        if rc == 0:
            return True, "\n".join(log)
        if attempt > retries:
            return False, "\n".join(log)
        time.sleep(2 * attempt)
    return False, "\n".join(log)


def _build(args, targets: list[BuildTarget]) -> None:
    import docker

    from swebench.harness.docker_build import build_instance_images

    client = docker.from_env()
    build_instance_images(
        client=client,
        dataset=[target.spec for target in targets],
        force_rebuild=args.force_rebuild,
        max_workers=args.jobs,
    )


def _image_present_locally(target_ref: str) -> bool:
    import docker

    client = docker.from_env()
    try:
        client.images.get(target_ref)
    except docker.errors.ImageNotFound:
        return False
    return True


def _filter_targets(args, targets: list[BuildTarget], manifest: BuildManifest) -> list[BuildTarget]:
    kept: list[BuildTarget] = []
    for target in targets:
        if manifest.is_done(target.target_ref, args.repo_link):
            print(f"[skip] {target.target_ref} (manifest)", flush=True)
            continue
        if args.skip_existing and docker_exists(target.target_ref):
            manifest.record(_target_record(target, "ok", reason="already-present", repo_link=args.repo_link))
            print(f"[skip] {target.target_ref} (registry)", flush=True)
            continue
        kept.append(target)
    return kept


def _push_targets(args, targets: list[BuildTarget], manifest: BuildManifest) -> int:
    errors = 0
    for i, target in enumerate(targets, start=1):
        if not _image_present_locally(target.target_ref):
            errors += 1
            manifest.record(_target_record(target, "error", reason="not-built"))
            print(f"[{i}/{len(targets)}] x {target.target_ref} (not built)", flush=True)
            continue

        if args.repo_link:
            rc, out = _label_image(target.target_ref, args.repo_link)
            if rc != 0:
                errors += 1
                manifest.record(_target_record(target, "error", reason="label-failed", log=out[-2000:]))
                print(f"[{i}/{len(targets)}] x {target.target_ref} (label failed)", flush=True)
                continue

        ok, out = _push_image(target.target_ref, args.retries)
        if ok:
            manifest.record(_target_record(target, "ok", repo_link=args.repo_link))
            print(f"[{i}/{len(targets)}] + {target.target_ref}", flush=True)
        else:
            errors += 1
            manifest.record(_target_record(target, "error", reason="push-failed", log=out[-2000:]))
            print(f"[{i}/{len(targets)}] x {target.target_ref} (push failed)", flush=True)
    return errors


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build missing SWE-bench instance images and publish them to GHCR."
    )
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-bench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--org", default=DEFAULT_ORG, help=f"GHCR org/user below {REGISTRY}")
    parser.add_argument("--arch", default="x86_64")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--env-image-tag", default="latest")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instance-id", action="append", help="build only this instance id; repeatable")
    parser.add_argument("--instance-ids", default="", help="comma/newline-separated instance id allow-list")
    parser.add_argument("--instance-ids-file", default=None)
    parser.add_argument("--shard", default=None, metavar="i/n")
    parser.add_argument("--jobs", type=int, default=2, help="parallel SWE-bench image builds")
    parser.add_argument("--retries", type=int, default=3, help="per-image docker push retries")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="skip targets already present on GHCR")
    parser.add_argument("--no-push", action="store_true", help="build locally but do not push")
    parser.add_argument("--dry-run", action="store_true", help="enumerate/preflight only")
    parser.add_argument(
        "--repo-link",
        default="swe-images/swe-images",
        help="stamp org.opencontainers.image.source before pushing; empty disables",
    )
    parser.add_argument("--ghcr-user", default=None)
    parser.add_argument("--ghcr-token", default=None)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parent / ".local" / "build-manifest.jsonl"),
    )
    parser.add_argument("--set-public", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    args.repo_link = normalize_repo_link(args.repo_link)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = BuildManifest(manifest_path)

    print(f"loading {args.dataset_name} split={args.split}", flush=True)
    rows = _load_rows(args)
    print(f"{len(rows)} row(s) selected", flush=True)

    targets, unsupported = _make_targets(args, rows, manifest)
    print(f"{len(targets)} supported target(s); unsupported/skipped={unsupported}", flush=True)

    targets = _filter_targets(args, targets, manifest)
    print(f"{len(targets)} target(s) left after manifest/registry skips", flush=True)

    if args.dry_run:
        for target in targets:
            print(f"would build {target.instance_id} -> {target.target_ref}")
        return 0
    if not targets:
        return 0

    creds = resolve_creds(args)
    if not args.no_push:
        docker_login(creds)
        if not creds:
            print(
                "! no GHCR credentials resolved; docker push may fail unless the daemon is already logged in",
                file=sys.stderr,
            )

    _build(args, targets)

    errors = 0
    if not args.no_push:
        errors = _push_targets(args, targets, manifest)

    if not args.no_push and not args.no_verify:
        token = args.ghcr_token or os.environ.get("GHCR_TOKEN") or os.environ.get("GITHUB_TOKEN")
        do_verify(args, targets, token)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
