"""Enumerate the source container images for each supported SWE dataset.

Four families of images are handled:

* **SWE-bench** (`SWE-bench/SWE-bench_Verified`, `SWE-bench/SWE-bench`) — the
  official prebuilt per-instance images live on Docker Hub under the `swebench`
  namespace as ``swebench/sweb.eval.x86_64.<instance_id>:latest`` where the
  instance id is lower-cased and ``__`` is rewritten to ``_1776_`` (this is
  exactly what ``swebench``'s ``TestSpec.instance_image_key`` produces for a
  remote namespace, so we construct it directly and avoid needing the full repo
  version specs table).

* **R2E-Gym** (`R2E-Gym/R2E-Gym-V1`, `R2E-Gym/R2E-Gym-Lite`) — every row carries
  a ``docker_image`` column that is already a fully qualified Docker Hub
  reference, e.g. ``namanjain12/aiohttp_final:<commit>``.

* **SWE-smith** (`SWE-bench/SWE-smith`) — every row carries an ``image_name``
  column naming a Docker Hub repository (no tag) under the ``jyangballin``
  namespace, e.g. ``jyangballin/swesmith.x86_64.<repo>``; the tag defaults to
  ``latest``. Many rows share one image (one per training repo), so the ~59k
  rows collapse to a few hundred unique images after dedup.

* **SWE-rebench** (`nebius/SWE-rebench`) — a 21k-instance benchmark; only the
  ``filtered`` split (6.5k of 21k rows) carries a prebuilt image, in the
  ``swerebench`` Docker Hub namespace as ``swerebench/sweb.eval.x86_64.<id>``
  (no tag ⇒ ``latest``). The namespace is dropped on mirror, so targets are
  ``ghcr.io/<org>/sweb.eval.x86_64.<id>:latest`` — the same scheme as SWE-bench
  (and with no instance-id overlap).

* **Terminal-Bench 2.x** (``harborframework/terminal-bench-2.1`` and
  ``harborframework/terminal-bench-2.0``) — Harbor-style *file* datasets (one
  directory per task with a ``task.toml``), not load_dataset tables. Every
  task's ``[environment] docker_image`` is a fully qualified Docker Hub
  reference under the ``alexgshaw`` namespace with a date tag, e.g.
  ``alexgshaw/crack-7z-hash:20251031`` (89 tasks per release; all public,
  amd64-only, ~20 GiB compressed for 2.1). The source namespace is dropped and
  replaced by a ``terminal-bench/`` path prefix, so targets are
  ``ghcr.io/<org>/terminal-bench/<task>:<yyyymmdd>`` — no collision with the
  other datasets' targets, and identical refs across 2.0/2.1 de-duplicate.

Each entry resolves to an :class:`ImageRef` describing the public source ref and
the desired ghcr.io target path. The target *package name* drops the source
registry and its first namespace component, so:

    swebench/sweb.eval.x86_64.<id>:latest -> ghcr.io/<org>/sweb.eval.x86_64.<id>:latest
    namanjain12/aiohttp_final:<commit>    -> ghcr.io/<org>/aiohttp_final:<commit>
    jyangballin/swesmith.x86_64.<repo>    -> ghcr.io/<org>/swesmith.x86_64.<repo>:latest
    swerebench/sweb.eval.x86_64.<id>      -> ghcr.io/<org>/sweb.eval.x86_64.<id>:latest
    alexgshaw/<task>:<yyyymmdd>           -> ghcr.io/<org>/terminal-bench/<task>:<yyyymmdd>

This collapses overlapping datasets onto identical targets — SWE-bench_Verified
is a subset of SWE-bench, and R2E-Gym-Lite a subset of R2E-Gym-V1 — so the
manifest naturally de-duplicates shared images across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional

# Logical dataset name -> config. ``kind`` selects the enumeration strategy.
DATASETS: dict[str, dict] = {
    "swe-bench-verified": {
        "kind": "swebench",
        "hf": "SWE-bench/SWE-bench_Verified",
        "split": "test",
    },
    "swe-bench": {
        "kind": "swebench",
        "hf": "SWE-bench/SWE-bench",
        # the full set ships test/dev/train; default to test, override with --split
        "split": "test",
    },
    "swe-bench-multilingual": {
        "kind": "image",
        "hf": "SWE-bench/SWE-bench_Multilingual",
        # test split only (300 rows); each row carries a fully-qualified
        # 'image' ref like 'swebench/sweb.eval.x86_64.<instance>:latest', which
        # is the same naming the 'swebench' kind constructs from instance_id.
        # Mirroring via the image column therefore lands on the exact same
        # ghcr targets as SWE-bench, so overlapping instances de-duplicate.
        "split": "test",
        "field": "image",
    },
    "swe-rebench": {
        "kind": "image",
        "hf": "nebius/SWE-rebench",
        # 21k-instance benchmark; only the 'filtered' split (6542 of 21336 rows)
        # carries a prebuilt image, in the 'swerebench' Docker Hub namespace as
        # 'swerebench/sweb.eval.x86_64.<id>' (no tag => latest). The namespace
        # is dropped on mirror, so targets land on the same ghcr scheme as
        # SWE-bench; no instance-id overlap, so no collision or extra dedup.
        "split": "filtered",
        "field": "docker_image",
    },
    "r2e-gym-v1": {
        "kind": "image",
        "hf": "R2E-Gym/R2E-Gym-V1",
        "split": "train",
        "field": "docker_image",
    },
    "r2e-gym-lite": {
        "kind": "image",
        "hf": "R2E-Gym/R2E-Gym-Lite",
        "split": "train",
        "field": "docker_image",
    },
    "swe-smith": {
        "kind": "image",
        "hf": "SWE-bench/SWE-smith",
        # train only; the dataset is frozen (Dec 2025) but still pullable, and
        # the jyangballin/swesmith.* images on Docker Hub are public + latest.
        "split": "train",
        "field": "image_name",
        # one image per training repo, so key ImageRef.instance_id on the repo
        # (a stable 1:1 per-image id) rather than the per-task instance_id.
        "instance_field": "repo",
    },
    "terminal-bench-2.1": {
        "kind": "tbench",
        "hf": "harborframework/terminal-bench-2.1",
        # Harbor-style file dataset: tasks/<name>/task.toml with
        # [environment] docker_image = "alexgshaw/<task>:<yyyymmdd>" on Docker
        # Hub. 89 tasks, amd64-only. Tags are date-pinned, so 2.0/2.1 refs to
        # the same tag+repo collapse onto identical targets (dedup via the
        # target-ref set in publish.py).
        "tasks_glob": "tasks/*",
    },
    "terminal-bench-2.0": {
        "kind": "tbench",
        "hf": "harborframework/terminal-bench-2.0",
        # same content family; 2.0 keeps task dirs at the repo root and has no
        # dataset-viewer parquet conversion, hence the glob override.
        "tasks_glob": "*",
    },
}

DOCKER_HUB = "docker.io"


@dataclass(frozen=True)
class ImageRef:
    """A single image to mirror.

    ``source`` is a fully qualified pullable reference (registry/repo:tag).
    ``target_name``/``target_tag`` describe the path *below* the ghcr org.
    ``instance_id`` is informational (used for filtering / logging).
    """

    source: str
    target_name: str
    target_tag: str
    dataset: str
    instance_id: str

    def target_ref(self, org: str, registry: str = "ghcr.io") -> str:
        return f"{registry}/{org}/{self.target_name}:{self.target_tag}"


def _split_ref(ref: str) -> tuple[str, str, str]:
    """Split a docker reference into (registry, repository, tag).

    A leading host component is recognised only when it contains a ``.`` or a
    ``:`` (port) or equals ``localhost`` — matching Docker's own heuristic — so
    ``namanjain12/foo`` is treated as repo ``namanjain12/foo`` on the default
    registry rather than host ``namanjain12``.
    """
    registry = DOCKER_HUB
    body = ref
    head = ref.split("/", 1)[0]
    if "/" in ref and (("." in head) or (":" in head) or head == "localhost"):
        registry, body = ref.split("/", 1)
    if ":" in body.rsplit("/", 1)[-1]:
        repo, tag = body.rsplit(":", 1)
    else:
        repo, tag = body, "latest"
    return registry, repo, tag


def _target_name_from_repo(repo: str) -> str:
    """Drop the first repo component (the source user/org namespace) and
    normalise to a ghcr-legal lowercase path."""
    parts = repo.split("/")
    rest = "/".join(parts[1:]) if len(parts) > 1 else repo
    name = rest.lower()
    # ghcr accepts [a-z0-9._/-]; sanitise anything else to a dash.
    name = re.sub(r"[^a-z0-9._/-]", "-", name)
    return name


def _normalise_instance_id(instance_id: str) -> str:
    return instance_id.lower().replace("__", "_1776_")


def swebench_source(instance_id: str, arch: str = "x86_64") -> str:
    norm = _normalise_instance_id(instance_id)
    return f"{DOCKER_HUB}/swebench/sweb.eval.{arch}.{norm}:latest"


def _load_hf_ids(hf_id: str, split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(hf_id, split=split)
    return ds


def _iter_tbench(cfg: dict, dataset: str) -> Iterator[ImageRef]:
    """Yield ImageRefs for a Harbor-style Terminal-Bench file dataset.

    These repos are directories of tasks (``<tasks_glob>/<name>/task.toml``)
    rather than load_dataset tables, so we snapshot just the ``task.toml``
    files via huggingface_hub (no ``datasets`` dependency needed) and read each
    task's ``[environment] docker_image``.
    """
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    from huggingface_hub import snapshot_download

    tasks_glob = cfg.get("tasks_glob", "tasks/*")
    # fetch only the task.toml files — the repos also carry Dockerfiles,
    # encrypted solution artifacts and test bundles we do not need here.
    allow = [f"{tasks_glob}/task.toml"]
    root = Path(
        snapshot_download(repo_id=cfg["hf"], repo_type="dataset", allow_patterns=allow)
    )
    task_dirs = sorted(
        path for path in root.glob(tasks_glob) if (path / "task.toml").is_file()
    )
    if not task_dirs:
        raise RuntimeError(f"no task.toml found under {root}/{tasks_glob}")

    for task_dir in task_dirs:
        config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        src = (config.get("environment") or {}).get("docker_image")
        if not src:
            continue
        task_name = (config.get("task") or {}).get("name") or task_dir.name
        registry, repo, tag = _split_ref(src)
        # alexgshaw/<task>:<tag> -> terminal-bench/<task>:<tag>: drop the source
        # namespace and add a dataset prefix so shared-org targets can't
        # collide with the R2E/SWE-smith flat names.
        task_slug = repo.split("/", 1)[-1]
        yield ImageRef(
            source=f"{registry}/{repo}:{tag}",
            target_name=f"terminal-bench/{_target_name_from_repo(task_slug)}",
            target_tag=tag,
            dataset=dataset,
            instance_id=task_name,
        )


def iter_images(
    dataset: str,
    *,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    arch: str = "x86_64",
    instance_filter: Optional[set[str]] = None,
) -> Iterator[ImageRef]:
    """Yield :class:`ImageRef` for every image in ``dataset``."""
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}")
    cfg = DATASETS[dataset]
    use_split = split or cfg.get("split")

    if cfg["kind"] == "tbench":
        # _iter_tbench already yields ImageRefs; the row loop below would
        # re-derive one from dataset columns, so dispatch directly.
        refs: Iterator = _iter_tbench(cfg, dataset)
    else:
        refs = _load_hf_ids(cfg["hf"], use_split)

    count = 0
    for row in refs:
        if cfg["kind"] == "tbench":
            ref = row
        elif cfg["kind"] == "swebench":
            instance_id = row["instance_id"]
            source = swebench_source(instance_id, arch=arch)
            _, repo, tag = _split_ref(source)
            ref = ImageRef(
                source=source,
                target_name=_target_name_from_repo(repo),
                target_tag=tag,
                dataset=dataset,
                instance_id=instance_id,
            )
        elif cfg["kind"] == "image":
            src = row[cfg["field"]]
            if not src:
                continue
            # ``instance_field`` (when set) yields a stable per-image id for
            # datasets where many rows share one image (e.g. SWE-smith's
            # one-image-per-repo); otherwise fall back to the row's own id.
            instance_id = (
                row.get(cfg.get("instance_field"))
                or row.get("instance_id")
                or row.get("repo_name")
                or src
            )
            registry, repo, tag = _split_ref(src)
            source = f"{registry}/{repo}:{tag}"
            ref = ImageRef(
                source=source,
                target_name=_target_name_from_repo(repo),
                target_tag=tag,
                dataset=dataset,
                instance_id=str(instance_id),
            )
        else:  # pragma: no cover - guarded by DATASETS
            raise AssertionError(cfg["kind"])

        if instance_filter is not None and ref.instance_id not in instance_filter:
            continue
        yield ref
        count += 1
        if limit is not None and count >= limit:
            return
