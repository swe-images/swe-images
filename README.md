# swe-images

Public mirror of the per-instance container images for **SWE-bench** and
**R2E-Gym**, hosted under `ghcr.io/swe-images/*`.

| dataset | source | mirrored as |
|---|---|---|
| `SWE-bench/SWE-bench_Verified` | `swebench/sweb.eval.x86_64.<id>:latest` | `ghcr.io/swe-images/sweb.eval.x86_64.<id>:latest` |
| `SWE-bench/SWE-bench` | `swebench/sweb.eval.x86_64.<id>:latest` | `ghcr.io/swe-images/sweb.eval.x86_64.<id>:latest` |
| `R2E-Gym/R2E-Gym-V1` | `<row.docker_image>` (Docker Hub) | `ghcr.io/swe-images/<name>:<commit>` |
| `R2E-Gym/R2E-Gym-Lite` | `<row.docker_image>` (Docker Hub) | `ghcr.io/swe-images/<name>:<commit>` |

```bash
docker pull ghcr.io/swe-images/sweb.eval.x86_64.astropy_1776_astropy-12907:latest
```

## How mirroring works

`publish.py` enumerates each dataset, copies every source image
registry→registry with `skopeo`, and (via `--set-public`) flips each package to
public. The [`mirror` workflow](.github/workflows/mirror.yml) runs it across N
parallel shards using this repo's `GITHUB_TOKEN`, which is what lets packages be
made public (see the comments at the top of the workflow).

Run it from the **Actions** tab → *mirror* → *Run workflow*:

- `dataset`: which set to mirror (or `all`)
- `shards`: parallel jobs (e.g. `8`)
- `limit`: cap per dataset for a smoke test (blank = everything)
- `mode`: `mirror` (copy + publish) or `flip-only` (just make already-pushed
  packages public)

Prerequisite (org owner, one-time): **Org → Settings → Packages → "Package
creation" → enable Public**.

## Building missing SWE-bench images

`publish.py` can only mirror images that already exist in a public source
registry. `build_swebench.py` covers the complementary case for rows that are
supported by the official SWE-bench Docker harness but whose image needs to be
built locally first.

`build_swebench.py` uses the official SWE-bench Docker harness to build those
instance images locally, tags them as `ghcr.io/swe-images/sweb.eval.*`, pushes
them with this repo's token, and then runs the same public-visibility check.
Unsupported rows are recorded in `.local/build-manifest.jsonl` and skipped.
This keeps the images benchmark-equivalent: rows whose repo/version is not
supported by the official harness are **not** replaced by a generic checkout
image under the `sweb.eval.*` name.

Smoke-test one official image:

```bash
python build_swebench.py \
  --dataset-name SWE-bench/SWE-bench \
  --split test \
  --instance-id django__django-10097 \
  --dry-run
```

Run from GitHub Actions: **Actions** → *build-swebench* → *Run workflow*.
Start with `instance_ids` or a small `limit`; full split builds need large
Docker disk and are better run on a self-hosted builder.

Note: the `SWE-bench/SWE-bench` `train` split is not buildable with official
`sweb.eval.*` semantics. Its rows use repos/versions that are not in
SWE-bench's official image-spec table, so this script reports them as
unsupported instead of publishing generic checkout-only images under benchmark
image names.

`publish.py` and `sources.py` are vendored from
`benchmaker/tools/datasets/publish_swe_images`.
