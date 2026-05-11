# vLLM Wheel URL Lookup

vLLM publishes pre-built wheels to `wheels.vllm.ai` (an S3 bucket, PEP 503 Simple Repository — not browser-browsable).

## How to find wheel URLs for a given commit SHA

1. Find the commit SHA from the `vllm-project/vllm` GitHub repo (e.g. a release tag or specific commit).

2. To list available wheels, fetch the PEP 503 index page:
   ```
   https://wheels.vllm.ai/<commit-sha>/<variant>/vllm/
   ```
   Variants: `cpu`, `cu129`, `cu130`

3. The actual download URLs follow this pattern:
   ```
   https://wheels.vllm.ai/<commit-sha>/vllm-<version>%2B<variant>-cp38-abi3-manylinux_2_35_<arch>.whl
   ```
   `%2B` = URL-encoded `+`, `<arch>` = `x86_64` or `aarch64`.

## Notes
- Clicking S3 directories gives `NoSuchKey` errors — append `/vllm/` to reach the index.
- This project pins vLLM wheels in `services/uds_tokenizer/pyproject.toml` with platform markers for x86_64 and aarch64.
- When asked to find vLLM wheel URLs, use WebFetch on the index page to extract the links.
