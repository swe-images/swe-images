#!/usr/bin/env python3
"""Mirror SWE-bench / R2E-Gym per-instance container images to ghcr.io.

Pulls the public source images for the supported datasets and re-pushes them
under a ghcr.io organisation (default: ``swe-images``), then verifies that each
resulting package is publicly visible.

Copy engines
------------
* ``skopeo`` copies registry -> registry with no Docker daemon and no local
  disk — the right tool for thousands of multi-GB images, but it cannot set
  image labels.
* ``crane`` (``go-containerregistry``) also copies registry -> registry and
  *can* rewrite the image config to add labels, so it is preferred whenever
  ``--repo-link`` is requested.
* ``docker`` (fallback) does pull / tag / push (or ``build FROM`` when labeling)
  and needs a running daemon plus enough local disk for the largest image.

The engine is auto-detected (crane→skopeo→docker without linking; crane→docker
with ``--repo-link`` since skopeo can't label); force one with ``--engine``.

Linking images to a GitHub repo
-------------------------------
Pass ``--repo-link swe-images/swe-images`` (or a full URL). This stamps the
``org.opencontainers.image.source`` label onto each image config; GHCR reads
that label and automatically links the package to the repo (showing the repo's
README/source and letting the package inherit the repo's access). Images already
pushed without the label are re-pushed when this flag is set.

Authentication
--------------
The ghcr.io destination needs a GitHub token with ``write:packages`` (and
``read:packages`` for the visibility check). Provide it via ``--ghcr-token`` or
the ``GHCR_TOKEN`` / ``GITHUB_TOKEN`` env var, and the owning user via
``--ghcr-user`` / ``GHCR_USER``. The public source images need no credentials.

Public visibility
------------------
A freshly pushed ghcr package is **always private**, and GitHub has **no
working API to flip an org-owned package to public** (the undocumented
``PATCH /orgs/{org}/packages/container/{name}`` 404s for organizations; it only
works for personal-account packages). The org "Package creation" setting only
controls which visibilities are *allowed*, and linking to a repo inherits the
repo's access *permissions* but not its visibility.

So making these public is a two-step manual/semi-manual process:

  1. Prerequisite: enable public packages for the org
     (Org -> Settings -> Packages -> "Package creation" -> check Public).
  2. Flip each package to Public via its settings UI (Danger Zone), or in bulk
     via an authenticated browser session (cookie + CSRF, not a PAT).

This script *verifies* visibility through
``GET /orgs/{org}/packages/container/{name}`` and prints a loud warning listing
everything still private (public -> private is not reversible).

Resumability
------------
Every successful copy is appended to a JSONL manifest (``--manifest``). Re-runs
skip targets already recorded as done. ``--skip-existing`` additionally probes
the destination registry so a manifest-less re-run still avoids re-pushing.

Examples
--------
    # dry run: list what *would* be mirrored for SWE-bench Verified
    python publish.py --dataset swe-bench-verified --dry-run

    # mirror first 5 (smoke test), then verify visibility
    python publish.py --dataset swe-bench-verified --limit 5

    # everything, 6 concurrent copies
    python publish.py --dataset all --jobs 6

    # just re-check visibility of what's already pushed
    python publish.py --dataset all --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import sources as S

DEFAULT_ORG = "swe-images"
REGISTRY = "ghcr.io"


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
class Manifest:
    """Append-only JSONL record of completed copies, keyed by target ref."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.done: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "ok" and rec.get("target"):
                    self.done[rec["target"]] = rec

    def is_done(self, target: str, repo_link: Optional[str]) -> bool:
        rec = self.done.get(target)
        # a target labeled differently (or not yet) must be re-pushed
        return rec is not None and rec.get("repo_link") == repo_link

    def record(self, rec: dict) -> None:
        with self._lock:
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if rec.get("status") == "ok":
                self.done[rec["target"]] = rec


# --------------------------------------------------------------------------- #
# copy engines
# --------------------------------------------------------------------------- #
def detect_engine(*, need_label: bool) -> str:
    # crane can label; skopeo can't. When linking, prefer crane and skip skopeo.
    order = ["crane", "docker"] if need_label else ["crane", "skopeo", "docker"]
    for eng in order:
        if shutil.which(eng):
            return eng
    raise SystemExit(
        "no usable container engine found. Install one of: "
        + ", ".join(order)
        + " (or pass --engine explicitly)."
        + ("\nNote: --repo-link needs crane or docker; skopeo cannot set labels."
           if need_label else "")
    )


def _run(cmd: list[str], *, env: Optional[dict] = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout


def _run_stdin(cmd: list[str], data: str) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return proc.returncode, proc.stdout


def skopeo_exists(target_ref: str, creds: Optional[str]) -> bool:
    cmd = ["skopeo", "inspect", "--raw"]
    if creds:
        cmd += ["--creds", creds]
    cmd.append(f"docker://{target_ref}")
    rc, _ = _run(cmd)
    return rc == 0


def skopeo_copy(source: str, target_ref: str, creds: Optional[str], retries: int) -> tuple[bool, str]:
    cmd = [
        "skopeo",
        "copy",
        "--retry-times",
        str(retries),
        # source images are amd64-only; --all is a harmless no-op but keeps
        # multi-arch sources intact if any appear.
        "--all",
    ]
    if creds:
        cmd += ["--dest-creds", creds]
    cmd += [f"docker://{source}", f"docker://{target_ref}"]
    rc, out = _run(cmd)
    return rc == 0, out


def crane_login(creds: Optional[str]) -> None:
    if not creds:
        return
    user, _, token = creds.partition(":")
    proc = subprocess.run(
        ["crane", "auth", "login", REGISTRY, "-u", user, "--password-stdin"],
        input=token,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise SystemExit(f"crane auth login {REGISTRY} failed:\n{proc.stdout}")


def crane_exists(target_ref: str) -> bool:
    rc, _ = _run(["crane", "manifest", target_ref])
    return rc == 0


def crane_copy(
    source: str, target_ref: str, retries: int, labels: Optional[dict] = None
) -> tuple[bool, str]:
    """Copy source->target. With labels, use `crane mutate` (rewrites the image
    config and pushes to the new tag); without, a plain `crane copy`."""
    if labels:
        cmd = ["crane", "mutate", source, "-t", target_ref]
        for k, v in labels.items():
            cmd += ["-l", f"{k}={v}"]
    else:
        cmd = ["crane", "copy", source, target_ref]
    log = []
    for attempt in range(1, retries + 2):
        rc, out = _run(cmd)
        log.append(out)
        if rc == 0:
            return True, "\n".join(log)
        if attempt > retries:
            return False, "\n".join(log)
        time.sleep(2 * attempt)
    return False, "\n".join(log)


def docker_login(creds: Optional[str]) -> None:
    if not creds:
        return
    user, _, token = creds.partition(":")
    proc = subprocess.run(
        ["docker", "login", REGISTRY, "-u", user, "--password-stdin"],
        input=token,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise SystemExit(f"docker login {REGISTRY} failed:\n{proc.stdout}")


def docker_exists(target_ref: str) -> bool:
    rc, _ = _run(["docker", "manifest", "inspect", target_ref])
    return rc == 0


def docker_copy(
    source: str, target_ref: str, retries: int, labels: Optional[dict] = None
) -> tuple[bool, str]:
    log = []
    for attempt in range(1, retries + 2):
        rc, out = _run(["docker", "pull", source])
        log.append(out)
        if rc == 0:
            break
        if attempt > retries:
            return False, "\n".join(log)
        time.sleep(2 * attempt)
    if labels:
        # add labels as a tiny metadata layer via `docker build FROM source`
        dockerfile = f"FROM {source}\n" + "".join(
            f"LABEL {k}={json.dumps(v)}\n" for k, v in labels.items()
        )
        rc, out = _run_stdin(["docker", "build", "-t", target_ref, "-"], dockerfile)
        log.append(out)
        if rc != 0:
            return False, "\n".join(log)
    else:
        rc, out = _run(["docker", "tag", source, target_ref])
        log.append(out)
        if rc != 0:
            return False, "\n".join(log)
    for attempt in range(1, retries + 2):
        rc, out = _run(["docker", "push", target_ref])
        log.append(out)
        if rc == 0:
            return True, "\n".join(log)
        if attempt > retries:
            return False, "\n".join(log)
        time.sleep(2 * attempt)
    return False, "\n".join(log)


# --------------------------------------------------------------------------- #
# visibility
# --------------------------------------------------------------------------- #
def _gh_api(
    path: str, token: str, *, method: str = "GET", payload: Optional[dict] = None
) -> tuple[int, Optional[dict]]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            **({"Content-Type": "application/json"} if data else {}),
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode() or "null"
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = None
        return e.code, body


def package_visibility(org: str, package_name: str, token: str) -> Optional[str]:
    """Return 'public'/'private'/'internal', or None if not found / no access.

    The container *package* name is the part below the org; ghcr maps the path
    ``a/b`` to a single package whose name is URL-encoded ``a%2Fb``.
    """
    encoded = urllib.request.quote(package_name, safe="")
    status, body = _gh_api(f"/orgs/{org}/packages/container/{encoded}", token)
    if status == 200 and isinstance(body, dict):
        return body.get("visibility")
    return None


def set_package_public(org: str, package_name: str, token: str) -> tuple[int, str]:
    """Best-effort flip of an org container package to public via the
    (undocumented) package PATCH endpoint.

    Returns (http_status, message). This is known to 404 for packages that are
    *not* connected to a repository (e.g. pushed with an external PAT); it has a
    real chance of succeeding when the package was pushed by the connected
    repo's GITHUB_TOKEN and ``token`` has admin on the package. There is no
    officially documented endpoint, so treat failures as "fall back to the UI".
    """
    encoded = urllib.request.quote(package_name, safe="")
    status, body = _gh_api(
        f"/orgs/{org}/packages/container/{encoded}",
        token,
        method="PATCH",
        payload={"visibility": "public"},
    )
    msg = ""
    if isinstance(body, dict):
        msg = body.get("message", "")
    return status, msg


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def normalize_repo_link(value: Optional[str]) -> Optional[str]:
    """Accept 'owner/repo' or a full URL; return a canonical https URL."""
    if not value:
        return None
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://github.com/{value.strip('/')}"


def resolve_creds(args) -> Optional[str]:
    user = args.ghcr_user or os.environ.get("GHCR_USER") or os.environ.get("GITHUB_ACTOR")
    token = args.ghcr_token or os.environ.get("GHCR_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if user and token:
        return f"{user}:{token}"
    return None


def gather_refs(args) -> list[S.ImageRef]:
    datasets = list(S.DATASETS) if args.dataset == "all" else [args.dataset]
    inst_filter = None
    if args.instance_id:
        inst_filter = set(args.instance_id)
    seen: set[str] = set()
    refs: list[S.ImageRef] = []
    for ds in datasets:
        for ref in S.iter_images(
            ds,
            split=args.split,
            limit=args.limit,
            arch=args.arch,
            instance_filter=inst_filter,
        ):
            tref = ref.target_ref(args.org, REGISTRY)
            if tref in seen:
                continue
            seen.add(tref)
            refs.append(ref)
    if args.shard:
        idx, total = (int(x) for x in args.shard.split("/", 1))
        if not (1 <= idx <= total):
            raise SystemExit(f"--shard {args.shard}: index must be in 1..{total}")
        # deterministic stride: enumeration order is stable, so shard i/n is a
        # disjoint, even slice across all targets.
        refs = refs[idx - 1 :: total]
    return refs


def copy_one(ref, args, creds, engine, manifest, repo_link) -> dict:
    target = ref.target_ref(args.org, REGISTRY)
    base = {
        "source": ref.source,
        "target": target,
        "dataset": ref.dataset,
        "instance_id": ref.instance_id,
        "repo_link": repo_link,
    }
    labels = {"org.opencontainers.image.source": repo_link} if repo_link else None

    if manifest.is_done(target, repo_link):
        return {**base, "status": "skip", "reason": "manifest"}

    # only trust a bare registry probe when we're not (re-)labeling — a present
    # image may lack the label we want to add.
    if args.skip_existing and not repo_link:
        if engine == "skopeo":
            exists = skopeo_exists(target, creds)
        elif engine == "crane":
            exists = crane_exists(target)
        else:
            exists = docker_exists(target)
        if exists:
            manifest.record({**base, "status": "ok", "reason": "already-present"})
            return {**base, "status": "skip", "reason": "registry"}

    if args.dry_run:
        return {**base, "status": "dry-run"}

    if engine == "skopeo":
        ok, out = skopeo_copy(ref.source, target, creds, args.retries)
    elif engine == "crane":
        ok, out = crane_copy(ref.source, target, args.retries, labels=labels)
    else:
        ok, out = docker_copy(ref.source, target, args.retries, labels=labels)

    rec = {**base, "status": "ok" if ok else "error"}
    if not ok:
        rec["log"] = out[-2000:]
    manifest.record(rec)
    return rec


def do_copy(args, refs, creds, engine, manifest, repo_link) -> dict:
    counts = {"ok": 0, "skip": 0, "error": 0, "dry-run": 0}
    errors: list[dict] = []
    total = len(refs)
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {
            pool.submit(copy_one, r, args, creds, engine, manifest, repo_link): r
            for r in refs
        }
        for fut in as_completed(futs):
            rec = fut.result()
            done += 1
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            if rec["status"] == "error":
                errors.append(rec)
            marker = {
                "ok": "✓",
                "skip": "·",
                "error": "✗",
                "dry-run": "→",
            }.get(rec["status"], "?")
            print(
                f"[{done}/{total}] {marker} {rec['target']}"
                + (f"  ({rec.get('reason')})" if rec.get("reason") else ""),
                flush=True,
            )
            if rec["status"] == "error":
                print(f"    {rec.get('log','').strip().splitlines()[-1:]}", flush=True)
    return {"counts": counts, "errors": errors}


def do_verify(args, refs, token) -> dict:
    if not token:
        print(
            "! visibility check skipped: no GitHub token "
            "(set --ghcr-token / GHCR_TOKEN / GITHUB_TOKEN)",
            file=sys.stderr,
        )
        return {}
    # unique package names (path below org)
    names = sorted({r.target_name for r in refs})
    vis: dict[str, Optional[str]] = {}
    private, missing = [], []
    with ThreadPoolExecutor(max_workers=min(8, args.jobs)) as pool:
        futs = {pool.submit(package_visibility, args.org, n, token): n for n in names}
        for fut in as_completed(futs):
            n = futs[fut]
            v = fut.result()
            vis[n] = v
            if v is None:
                missing.append(n)
            elif v != "public":
                private.append(n)
    print(f"\nvisibility: {len(names)} packages checked", flush=True)
    print(
        f"  public={sum(1 for v in vis.values() if v=='public')} "
        f"private/internal={len(private)} not-found/no-access={len(missing)}"
    )

    if private and args.set_public:
        print(f"\nattempting to set {len(private)} package(s) public …", flush=True)
        ok, fail = 0, []
        with ThreadPoolExecutor(max_workers=min(8, args.jobs)) as pool:
            futs = {pool.submit(set_package_public, args.org, n, token): n for n in private}
            for fut in as_completed(futs):
                n = futs[fut]
                status, msg = fut.result()
                if status in (200, 204):
                    ok += 1
                else:
                    fail.append((n, status, msg))
        print(f"  set-public: ok={ok} failed={len(fail)}", flush=True)
        if fail:
            s0 = fail[0][1]
            hint = (
                " (404 ⇒ package not connected to a repo / token lacks admin — "
                "push it from a GITHUB_TOKEN workflow in the connected repo)"
                if s0 == 404
                else ""
            )
            print(f"  first failure: HTTP {s0}{hint}", flush=True)
            for n, st, msg in fail[:20]:
                print(f"    {n}: HTTP {st} {msg}".rstrip(), flush=True)
        private = [n for n, *_ in fail]

    if private:
        print(
            "\n!! These packages are NOT public. There is no *documented* API to "
            "flip org package visibility. First enable public packages "
            "(Org → Settings → Packages → 'Package creation' → Public). Then either "
            "use --set-public (best-effort PATCH; works when the package was pushed "
            "by the connected repo's GITHUB_TOKEN) or flip each via its settings UI "
            "(Danger Zone):"
        )
        for n in private[:50]:
            print(f"   {REGISTRY}/{args.org}/{n}  ({vis.get(n)})")
        if len(private) > 50:
            print(f"   … and {len(private) - 50} more")
    return {"private": private, "missing": missing}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Mirror SWE-bench / R2E-Gym images to ghcr.io and verify public visibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(S.DATASETS) + ["all"],
        help="dataset to mirror (or 'all')",
    )
    p.add_argument("--org", default=DEFAULT_ORG, help=f"ghcr.io org (default {DEFAULT_ORG})")
    p.add_argument("--split", default=None, help="override the HF split")
    p.add_argument("--arch", default="x86_64", help="SWE-bench image arch (default x86_64)")
    p.add_argument("--limit", type=int, default=None, help="cap number of images per dataset")
    p.add_argument(
        "--instance-id",
        action="append",
        help="only mirror this instance id (repeatable)",
    )
    p.add_argument(
        "--engine",
        choices=["auto", "crane", "skopeo", "docker"],
        default="auto",
        help="copy engine (default auto: crane→skopeo→docker; crane→docker with --repo-link)",
    )
    p.add_argument(
        "--repo-link",
        default=None,
        help="link images to a GitHub repo (e.g. swe-images/swe-images or a full "
        "URL) via the org.opencontainers.image.source label",
    )
    p.add_argument(
        "--shard",
        default=None,
        metavar="i/n",
        help="process only shard i of n (1-based) — disjoint even slice of the "
        "target list, for fanning out across parallel Actions jobs",
    )
    p.add_argument("--jobs", type=int, default=4, help="concurrent copies (default 4)")
    p.add_argument("--retries", type=int, default=3, help="per-image retries (default 3)")
    p.add_argument(
        "--manifest",
        default=str(Path(__file__).parent / ".local" / "manifest.jsonl"),
        help="resumable JSONL manifest path",
    )
    p.add_argument(
        "--mapping-out",
        default=None,
        help="optional path to write a source→ghcr JSON mapping",
    )
    p.add_argument("--ghcr-user", default=None, help="GitHub user/org for push auth")
    p.add_argument("--ghcr-token", default=None, help="GitHub token (write:packages)")
    p.add_argument("--dry-run", action="store_true", help="enumerate only; copy nothing")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="probe the destination registry and skip images already present",
    )
    p.add_argument(
        "--set-public",
        action="store_true",
        help="after verifying, best-effort PATCH any private package to public "
        "(works when pushed by the connected repo's GITHUB_TOKEN; else 404s)",
    )
    p.add_argument("--no-verify", action="store_true", help="skip the visibility check")
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="only check visibility of the dataset's targets; copy nothing",
    )
    args = p.parse_args(argv)

    creds = resolve_creds(args)
    token = args.ghcr_token or os.environ.get("GHCR_TOKEN") or os.environ.get("GITHUB_TOKEN")
    repo_link = normalize_repo_link(args.repo_link)
    if repo_link:
        print(f"linking images to {repo_link} (org.opencontainers.image.source)", flush=True)

    print(f"enumerating {args.dataset} …", flush=True)
    refs = gather_refs(args)
    print(f"  {len(refs)} unique target images", flush=True)

    if args.mapping_out:
        mapping = {r.source: r.target_ref(args.org, REGISTRY) for r in refs}
        Path(args.mapping_out).write_text(json.dumps(mapping, indent=2))
        print(f"  wrote mapping → {args.mapping_out}", flush=True)

    if args.verify_only:
        do_verify(args, refs, token)
        return 0

    engine = (
        detect_engine(need_label=bool(repo_link))
        if args.engine == "auto"
        else args.engine
    )
    print(f"engine: {engine}", flush=True)
    if repo_link and engine == "skopeo":
        raise SystemExit(
            "skopeo cannot set labels — use --engine crane (or docker) with --repo-link"
        )

    if not args.dry_run:
        if engine == "docker":
            docker_login(creds)
        elif engine == "crane":
            crane_login(creds)
    if not creds and not args.dry_run:
        print(
            "! no ghcr credentials resolved — push will fail unless you have "
            "logged in already (set --ghcr-user/--ghcr-token or GHCR_USER/GHCR_TOKEN)",
            file=sys.stderr,
        )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(manifest_path)

    summary = do_copy(args, refs, creds, engine, manifest, repo_link)
    c = summary["counts"]
    print(
        f"\ncopy summary: ok={c.get('ok',0)} skip={c.get('skip',0)} "
        f"error={c.get('error',0)} dry-run={c.get('dry-run',0)}",
        flush=True,
    )

    if not args.dry_run and not args.no_verify:
        do_verify(args, refs, token)

    return 1 if summary["counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
