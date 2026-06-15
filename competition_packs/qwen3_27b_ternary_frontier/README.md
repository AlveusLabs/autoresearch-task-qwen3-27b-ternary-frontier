# Qwen3.6 27B Mostly-Ternary Compression Frontier

This validator-owned pack measures hidden heldout PPL against true compression
under the Qwen3.6 27B ternary contract.

Primary submission surface:

- `artifact_uri` pointing at a compressed artifact in a public Hugging Face repo
  or another public HTTPS location
- `artifact_sha256` and `artifact_size_bytes` for the exact bytes validators
  must download

Recipe/code patches are optional metadata; coordinators store only URI/integrity
metadata.
Validators download, load, account for, and score the artifact directly.

Artifact accounting is validator-computed. The hard checks are:

- parameter count within the Qwen3.6 27B ternary-compatible cap
- compressed representation size within a 90% ternary plus 10% scaled q4 rescue budget
- rescue values submitted as signed q4 codes plus fp16 scales per 128-code group; missing or out-of-range q4 codes reject
- non-ternary q4 rescue fraction at or below `10%`

Miners can use residuals, sparse side tables, mixed ternary components, or other
designs as long as every extra component is declared in
`artifact.accounting.extra_entries` or a layer `extra_components` list so the
validator counts the parameters, bits, and rescue usage.
New best acceptance uses a `0.02`-nat PPL resolution before replacing the
incumbent on quality alone.
