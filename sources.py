"""Enumerate the source container images for each supported SWE dataset.

Two families of images are handled:

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

Each entry resolves to an :class:`ImageRef` describing the public source ref and
the desired ghcr.io target path. The target *package name* drops the source
registry and its first namespace component, so:

    swebench/sweb.eval.x86_64.<id>:latest -> ghcr.io/<org>/sweb.eval.x86_64.<id>:latest
    namanjain12/aiohttp_final:<commit>    -> ghcr.io/<org>/aiohttp_final:<commit>

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
    "r2e-gym-v1": {
        "kind": "r2e",
        "hf": "R2E-Gym/R2E-Gym-V1",
        "split": "train",
        "field": "docker_image",
    },
    "r2e-gym-lite": {
        "kind": "r2e",
        "hf": "R2E-Gym/R2E-Gym-Lite",
        "split": "train",
        "field": "docker_image",
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
    use_split = split or cfg["split"]
    rows = _load_hf_ids(cfg["hf"], use_split)

    count = 0
    for row in rows:
        if cfg["kind"] == "swebench":
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
        elif cfg["kind"] == "r2e":
            src = row[cfg["field"]]
            if not src:
                continue
            instance_id = row.get("instance_id") or row.get("repo_name") or src
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
