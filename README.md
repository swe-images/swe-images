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

`publish.py` and `sources.py` are vendored from
`benchmaker/tools/datasets/publish_swe_images`.
