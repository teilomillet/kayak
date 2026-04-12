# 2026-04-12 BrowseComp-Plus Gold Storage Encoding

## Goal

Close Phase I2 from
[docs/late_interaction_efficiency_roadmap.md](../late_interaction_efficiency_roadmap.md):

- measure one non-default compressed token storage path
- verify exact-search correctness after reload
- record bytes/vector, build cost, load cost, search latency, and retrieval
  quality on one hard public slice

## Files Added Or Changed

- `kayak/benchmarks/storage_encoding_json.mojo`
- `benchmarks/browsecomp_plus_gold_storage_encoding.mojo`
- `tests/test_storage_encoding_json.mojo`
- `pyproject.toml`

## Design Choice

Reason:
- the repo already supported `binary_f16_le` payloads in storage codecs and
  roundtrip tests
- the missing piece was benchmark evidence on a real hard-recall slice

Decision:
- benchmark the persisted packed-index encoding directly
- keep exact search unchanged after reload
- treat this as a storage tradeoff benchmark, not as an in-memory quantized
  search claim

That distinction matters because the current f16 path compresses the persisted
payload, then reloads it back into the native vector scalar for exact search.

## Verification Commands

```bash
pixi run test_storage
pixi run test_storage_encoding_json
pixi run bench_browsecomp_plus_gold_storage_encoding_raw
```

Artifact:

```text
.cache/kayak/browsecomp_plus_gold_storage_encoding.json
```

## Measurement Context

Slice:

- `family = "browsecomp_plus"`
- `slice = "browsecomp_plus_gold_slice"`
- `document_count = 90`
- `vector_count = 15756`
- `vector_dim = 128`

Measured encodings:

- `binary_le`
- `binary_f16_le`

Important boundary:
- this benchmark measures persisted packed-index encoding
- it does **not** claim a quantized in-memory exact scorer

## Result Snapshot

### Baseline `binary_le`

- `artifact_byte_size = 8,068,326`
- `artifact_bytes_per_vector = 512.0796`
- `mean_build_seconds = 0.004282`
- `mean_load_seconds = 0.014980`
- `mean_search_seconds = 0.0007467`
- `nDCG@10 = 0.2851`

### Compressed `binary_f16_le`

- `artifact_byte_size = 4,034,794`
- `artifact_bytes_per_vector = 256.0798`
- `mean_build_seconds = 0.004018`
- `mean_load_seconds = 0.017243`
- `mean_search_seconds = 0.0006690`
- `nDCG@10 = 0.2851`

## Interpretation

Verified on this slice:
- the f16 payload cuts persisted bytes/vector almost exactly in half
- retrieval quality did not change on this measured public slice
- search latency also did not regress after reload

Observed tradeoff:
- load time increased slightly:
  - about `14.98 ms` to `17.24 ms`
- that is consistent with decoding a smaller on-disk payload back into the
  native in-memory scalar type

## What This Verifies

Verified locally:
- Kayak already had one real compressed token storage path
- the repo can now quantify its storage-quality-latency tradeoff on a hard
  public slice
- Phase I2 is no longer an unmeasured claim

## What This Still Does Not Prove

Not verified by this work:
- anything close to `6 bytes/vector`
- an in-memory quantized exact scorer
- behavior on much larger corpora or cold-start environments beyond this host

## Takeaway

The repo should now speak precisely:
- one explicit compressed packed-index path is verified
- that path halves persisted bytes/vector on BrowseComp gold
- it does so without measured retrieval-quality loss on this slice

That is a real storage win, but it is still far from the stronger
`6 bytes/vector` thesis.
