# Kayak Python SDK

`kayak` is a Python SDK for late-interaction retrieval.

Its job is to make late interaction programmable in normal Python while keeping
query/document vector counts, layouts, and MaxSim semantics explicit.

Fundamentally, late interaction here means token-level MaxSim over explicit
query and document vector groups. The SDK does not hide that structure behind a
fake dense tensor API.

It gives you explicit objects for:
- queries
- query batches
- documents
- packed indexes
- text encoders
- stores
- candidate generators
- stage-2 operators
- search plans
- MaxSim scores
- top-k search hits

The API is designed to feel natural for NumPy and PyTorch users without hiding
the parts that matter in late interaction:
- query vector count stays explicit
- document vector count stays explicit
- layout stays explicit
- backend choice stays explicit

For the higher-level product positioning and the split between the open Python
SDK and the hosted engine, see
[docs/python_sdk_charter.md](../../docs/python_sdk_charter.md).
For the execution plan behind that position, see
[docs/python_sdk_roadmap.md](../../docs/python_sdk_roadmap.md).

## Install

With pip:

```bash
pip install kayak
```

With UV:

```bash
uv add kayak
```

With Pixi and PyPI:

```bash
pixi add --pypi kayak
```

## Optional Mojo Backend

The default Python SDK path uses the NumPy reference backend.

The Python package does not expose a separate "Mojo-mode" import surface.
You still write normal Python:
- `import kayak`
- build `query`, `documents`, and `index`
- opt into the Mojo backend explicitly on the operation call

You only need Mojo if you want the explicit exact CPU Mojo backend:
- `kayak.MOJO_EXACT_CPU_BACKEND`

Examples:

```bash
# global or activated environment
pip install kayak
mojo --version
```

```bash
# Pixi project
pixi add python=3.11 mojo
pixi add --pypi kayak
pixi run python app.py
```

Then select the backend explicitly:

```python
scores = kayak.maxsim(
    query,
    index,
    backend=kayak.MOJO_EXACT_CPU_BACKEND,
)
```

Kayak does not silently switch to the Mojo backend just because Mojo is
installed. The backend choice stays explicit.

If you are running inside an activated virtual environment or `pixi run
python`, Kayak first checks that active Python environment for a usable `mojo`
binary before falling back to `PATH`.

Current CLI discovery order for the Mojo backend:
- `KAYAK_MOJO_CLI`
- a usable `mojo` binary in the active Python environment
- `mojo` on `PATH`
- `pixi run mojo`

`KAYAK_MOJO_CLI` can be either a binary path or a full command prefix such as
`bash /full/path/to/run_mojo_with_wrapper.sh`.

If you do not pass `backend=kayak.MOJO_EXACT_CPU_BACKEND`, Kayak stays on the
NumPy reference backend and does not require Mojo.

Kayak wheels bundle the Mojo backend they were built with. If a
`mojo_exact_cpu` call reports a bundled-backend/compiler mismatch, upgrade
`kayak` and `mojo` together so both come from compatible releases.

## Text Encoders

Kayak's public core remains vector-first, but the SDK now exposes a small text
encoder contract for the common "I start from text" path.

Use the first-party ColBERT encoder when you want a ready-made text path:

```python
import kayak

encoder = kayak.open_encoder("colbert", model_name="colbert-ir/colbertv2.0")

query = encoder.encode_query("what tool installs Python and Mojo together?")
documents = encoder.encode_documents(
    ["doc-a", "doc-b"],
    [
        "Pixi can create one environment with Python, Mojo, and kayak.",
        "uv add kayak installs the Python package but not the Mojo CLI.",
    ],
)
index = documents.pack()
hits = kayak.search(query, index, k=2)
```

Use `CallableLateTextEncoder` when you already have your own text-to-token-vector
functions and only want them adapted to Kayak's public late-interaction types:

```python
import kayak

encoder = kayak.CallableLateTextEncoder(
    query_encoder=my_query_encoder,
    document_encoder=my_document_encoder,
)
```

The contract stays narrow:
- `encode_query(text)`
- `encode_document_vectors(text)`
- `encode_documents(doc_ids, texts)`

The factory is intentionally small:
- `open_encoder("colbert", model_name=...)`
- `open_encoder("callable", query_encoder=..., document_encoder=...)`
- `register_encoder(...)`

## Stores

Kayak search still operates on `LateIndex`, but the SDK now exposes one store
contract for persistence and materialization.

Use `open_store("kayak", path=...)` for the default local directory-backed
store:

```python
import kayak

store = kayak.open_store("kayak", path="./kayak-index")

documents = kayak.documents(
    ["doc-a", "doc-b"],
    [doc_a_vectors, doc_b_vectors],
    texts=["alpha", "beta"],
)
store.upsert(
    documents,
    metadata=[
        {"topic": "installation"},
        {"topic": "vector_db"},
    ],
)

index = store.load_index(
    where={"topic": "installation"},
    include_text=True,
)
```

Use `MemoryLateStore` when you want the same contract without persistence:

```python
store = kayak.MemoryLateStore()
store.upsert(documents)
index = store.load_index()
```

Use `open_store("lancedb", ...)` when you want Kayak to materialize search-ready
indexes from a LanceDB table while keeping persistence in the database:

```python
import kayak

store = kayak.open_store(
    "lancedb",
    path="./lancedb-store",
    table_name="docs",
)
store.upsert(documents, metadata=metadata_rows)
index = store.load_index(where={"topic": "installation"}, include_text=True)
```

The store contract is intentionally narrow:
- `upsert(...)`
- `delete(...)`
- `load_index(...)`
- `stats()`
- `capabilities()`

The factory is intentionally small:
- `open_store("kayak", path=...)`
- `open_store("memory")`
- `open_store("lancedb", path=..., table_name=...)`
- `register_store(...)`

## Core API

Create a query:

```python
import kayak
import numpy as np

query = kayak.query(
    np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
)
```

Attach query text only when a text-family stage-2 operator needs it:

```python
query = kayak.query(
    np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    text="founded in 1984 in a church artistic director",
)
```

Create a document collection:

```python
documents = kayak.documents(
    ["doc-a", "doc-b"],
    [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
    ],
)
```

If you want a text-family stage 2, attach document texts explicitly:

```python
documents = kayak.documents(
    ["doc-context", "doc-answer"],
    [
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
    ],
    texts=[
        "Gugulethu township logo emblem heritage schools history",
        "Zama Dance School was founded in 1984 in a church and the longest serving employee is the artistic director.",
    ],
)
```

Pack documents into an index:

```python
index = documents.pack()
```

Score with MaxSim:

```python
scores = kayak.maxsim(query, index)
```

Search:

```python
hits = kayak.search(query, index, k=2)
```

Create an explicit query batch without pretending it is one dense tensor:

```python
def dim128(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector

batch = kayak.query_batch(
    [
        np.stack([dim128(0), dim128(1)]),
        np.stack([dim128(0), dim128(1), dim128(2)]),
    ]
)

index = kayak.documents(
    ["doc-a", "doc-b"],
    [
        np.stack([dim128(0), dim128(1), dim128(2)]),
        np.stack([dim128(0), dim128(1)]),
    ],
).pack()

scores_batch = kayak.maxsim_batch(batch, index)
```

Stage 2 is explicit too. Exact full scan now defaults to a no-op stage 2 because
stage 1 is already exact:

```python
plan = kayak.exact_full_scan_search_plan(final_k=2, candidate_k=3)
result = kayak.search_with_plan(query, index, plan)

print(result.stage2.stage_name)  # noop_topk
print(result.hits)
```

Approximate stage 1 plus exact late interaction is still explicit:

```python
plan = kayak.document_proxy_search_plan(final_k=1, candidate_k=2)
result = kayak.search_with_plan(query, index, plan)

print(result.candidate_stage.hits)
print(result.stage2.stage_name)  # exact_late_interaction
print(result.stage2.materialized_artifacts[0].family)  # late_interaction
print(result.hits)
```

Text-family refinement is also explicit and requires both `query.text` and
document texts:

```python
plan = kayak.exact_full_scan_search_plan(
    final_k=1,
    candidate_k=2,
    stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
)
result = kayak.search_with_plan(query, index, plan)

print(result.stage2.stage_name)  # noop_topk
print(result.stage3_verifier.stage_name)  # clause_text
print(result.stage3_verifier.materialized_artifacts[0].family)  # document_text
print(result.hits)
```

Hybrid refinement stays explicit too. The default `document_proxy` plan already
uses exact late interaction as its stage-2 reference operator, so adding a
stage-3 verifier means specifying only the verifier:

```python
plan = kayak.document_proxy_search_plan(
    final_k=1,
    candidate_k=2,
    query_vector_budget=1,
    document_vector_budget=1,
    stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
)
result = kayak.search_with_plan(query, index, plan)

print(result.stage2.stage_name)  # exact_late_interaction
print(result.stage3_verifier.stage_name)  # clause_text
print([artifact.family for artifact in result.stage2.materialized_artifacts])
# ['late_interaction']
print([artifact.family for artifact in result.stage3_verifier.materialized_artifacts])
# ['document_text']
print(result.hits)
```

Stage-aware search plans are explicit too:

```python
plan = kayak.document_proxy_search_plan(final_k=1, candidate_k=2)
result = kayak.search_with_plan(query, index, plan)

print(result.candidate_stage.hits)
print(result.hits)
print(result.candidate_stage.profile.document_vector_count)
print(result.stage2.document_vector_count)
```

Current public stage-1 generators:
- `exact_full_scan`
- `document_proxy`

Current public staged refinement pieces:
- stage-2 reference operators:
  - `noop_topk`
  - `exact_late_interaction`
- stage-3 verifiers:
  - `none`
  - `clause_text`

That is an intentionally narrow first pass.
It gives Python users a real stage-aware primitive today without pretending the
full engine-native generator family or every future refinement operator is
already stable as public SDK surface.

## Layouts

Kayak keeps layout changes explicit.

`flat_dim128` and `hybrid_flat_dim128` require `vector_dim == 128`.

Example:

```python
import kayak
import numpy as np

def dim128(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector

query128 = kayak.query(np.stack([dim128(0), dim128(1)]))
documents128 = kayak.documents(
    ["doc-a", "doc-b"],
    [
        np.stack([dim128(0), dim128(1)]),
        np.stack([dim128(0), dim128(0)]),
    ],
)
index128 = documents128.pack()

flat_query = query128.to_layout("flat_dim128")
hybrid_index = index128.to_layout("hybrid_flat_dim128")

scores = kayak.maxsim(flat_query, hybrid_index)
```

## Backends

The package exposes two named backends:
- `kayak.NUMPY_REFERENCE_BACKEND`
- `kayak.MOJO_EXACT_CPU_BACKEND`

You can inspect backend availability explicitly:

```python
print(kayak.available_backends())
print(kayak.backend_info(kayak.MOJO_EXACT_CPU_BACKEND))
```

Example:

```python
scores = kayak.maxsim(
    query,
    index,
    backend=kayak.NUMPY_REFERENCE_BACKEND,
)
```

The NumPy backend is the safest default.

The Mojo exact CPU backend is the faster exact path when your environment has a
working Mojo installation:

```python
scores = kayak.maxsim(
    query,
    index,
    backend=kayak.MOJO_EXACT_CPU_BACKEND,
)
```

## Public Surface

Application code should import from `kayak`.

Main exports:
- `BackendInfo`
- `CandidateGenerator`
- `CandidateStageResult`
- `LateQuery`
- `LateQueryBatch`
- `LateDocuments`
- `LateIndex`
- `LateScores`
- `SearchHit`
- `SearchPlan`
- `SearchPlanResult`
- `SearchStageProfile`
- `StageArtifactMaterialization`
- `available_backends`
- `backend_info`
- `document_proxy_candidate_generator`
- `document_proxy_search_plan`
- `query`
- `query_batch`
- `documents`
- `exact_full_scan_candidate_generator`
- `exact_full_scan_search_plan`
- `generate_candidates`
- `packed_index`
- `hybrid_flat_dim128_index`
- `flat_query_dim128`
- `maxsim`
- `maxsim_batch`
- `search`
- `search_batch`
- `search_with_plan`

## Mental Model

Kayak is not a generic tensor library.

It is a late-interaction retrieval API with:
- ragged query and document vector counts
- explicit layout conversion
- exact MaxSim scoring
- explicit candidate-window selection before rescoring
- explicit search backends

That makes it suitable for code that wants retrieval semantics first, while
still fitting naturally into Python workflows built on NumPy or PyTorch.
