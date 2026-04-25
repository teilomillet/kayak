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

If you are coding at a REPL, in a notebook, or inside an editor console, start
with:

```python
import kayak

print(kayak.help())
print(kayak.doctor())
print(kayak.help("search"))
print(kayak.help("search_text"))
print(kayak.help("session"))
print(kayak.help("typing"))
print(kayak.available_encoder_kinds())
print(kayak.available_store_kinds())
print(kayak.help("TokenMatrixInput"))
print(kayak.help("mojo"))
print(kayak.help("stores"))
print(kayak.help(kayak.LateTextRetriever))
```

That help text is generated from the current public API, signatures, and
docstrings instead of a separate handwritten help registry.

When you want to verify the active environment before you code further, use:

```python
import kayak

print(kayak.doctor())
```

That report is factual. It tells you:
- which encoder and store kinds are currently registered
- which exact backend the high-level retriever would default to
- whether the Mojo bridge is available
- whether optional store adapter dependencies are importable

If you prefer normal Python introspection in an editor or REPL, the same
descriptions are available through public docstrings:

```python
import inspect
import kayak

print(inspect.signature(kayak.open_text_retriever))
print(inspect.getdoc(kayak.open_text_retriever))
print(inspect.getdoc(kayak.LateTextRetriever.search_text))
```

For the higher-level product positioning and the split between the open Python
SDK and the hosted engine, see
[docs/python_sdk_charter.md](../../docs/python_sdk_charter.md).
For the execution plan behind that position, see
[docs/python_sdk_roadmap.md](../../docs/python_sdk_roadmap.md).

## Install

Install the SDK with any Python package manager:

With UV:

```bash
uv add kayak
```

With pip:

```bash
pip install kayak
```

With Pixi and PyPI:

```bash
pixi add --pypi kayak
```

## Optional Mojo Backend

The default low-level Python SDK path uses the NumPy reference backend.

If you want the Mojo backend, the actual requirement is simple:

- install `kayak`
- make a usable `mojo` CLI visible to Kayak

If no usable `mojo` CLI is visible, the package still works and stays on
`kayak.NUMPY_REFERENCE_BACKEND`.

The Python package does not expose a separate "Mojo-mode" import surface.
You still write normal Python:
- `import kayak`
- build `query`, `documents`, and `index`
- opt into the Mojo backend explicitly on the operation call

You only need Mojo if you want the explicit exact CPU Mojo backend:
- `kayak.MOJO_EXACT_CPU_BACKEND`

Examples:

```bash
# any environment where mojo is already installed and discoverable
mojo --version
uv add kayak
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

Low-level operations do not silently switch to the Mojo backend just because
Mojo is installed. The backend choice stays explicit there.

The high-level `open_text_retriever(...)` workflow is different: it prefers
`kayak.MOJO_EXACT_CPU_BACKEND` automatically when the backend is actually
available.

Pixi is one easy way to create a Mojo-capable environment, but it is not a
requirement. UV, pip, or another environment manager work too if Kayak can find
the `mojo` CLI.

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

This is a plain-text convenience layer, not a document-intelligence pipeline.
Kayak does not own OCR, PDF layout recovery, table extraction, or answer
generation as part of this SDK surface.

There are two main user paths:

1. use the built-in ColBERT encoder when your checkpoint is already a ColBERT
   model on Hugging Face
2. use the callable encoder when you already have your own model methods and
   just want Kayak to wrap them into late-interaction objects

To inspect the currently registered public encoder kinds at runtime:

```python
import kayak

print(kayak.available_encoder_kinds())
```

Use the first-party ColBERT encoder when you want a ready-made text path:

```python
import kayak

encoder = kayak.open_encoder("colbert", model_name="colbert-ir/colbertv2.0")

query = encoder.encode_query("what keeps python mojo and kayak together?")
documents = encoder.encode_documents(
    ["doc-a", "doc-b"],
    [
        "One environment can keep Python, Mojo, and kayak together.",
        "Installing kayak alone adds the Python package but not the Mojo CLI.",
    ],
)
index = documents.pack()
hits = kayak.search(query, index, k=2)
```

`model_name` is the Hugging Face repo id for the ColBERT checkpoint.

Use `CallableLateTextEncoder` when you already have your own text-to-token-vector
functions or model methods and only want them adapted to Kayak's public
late-interaction types:

```python
import kayak

encoder = kayak.CallableLateTextEncoder(
    query_encoder=my_query_encoder,
    document_encoder=my_document_encoder,
)
```

That includes model-backed call sites such as:

```python
encoder = kayak.open_encoder(
    "callable",
    query_encoder=my_model.encode_query_tokens,
    document_encoder=my_model.encode_document_tokens,
)
```

If your model already exposes the usual method names, you can skip the wrapper
glue and pass the object directly:

```python
encoder = kayak.open_encoder("callable", model=my_model)
```

Or make the method names explicit when your model uses different names:

```python
encoder = kayak.open_encoder(
    "callable",
    model=my_model,
    query_method="query_tokens",
    document_method="document_tokens",
)
```

That is the intended path for non-ColBERT Hugging Face or custom models today.

The same shortcut works at the retriever layer, which is the simplest path when
you want one object for ingest and search:

```python
retriever = kayak.open_text_retriever(
    encoder=my_model,
    store="memory",
)
```

If the model uses different method names, keep the same retriever abstraction
and specify those names through `encoder_kwargs`:

```python
retriever = kayak.open_text_retriever(
    encoder=my_model,
    store="memory",
    encoder_kwargs={
        "query_method": "query_tokens",
        "document_method": "document_tokens",
    },
)
```

The contract stays narrow:
- `encode_query(text)`
- `encode_document_vectors(text)`
- `encode_documents(doc_ids, texts)`

If you are wrapping your own model, use the stable public typing aliases in
`kayak.typing` instead of importing types from the internal bridge package:

```python
from kayak.typing import DocIdsInput, DocTextsInput, TokenMatrixInput


class MyLateInteractionModel:
    def encode_query_tokens(self, text: str) -> TokenMatrixInput:
        ...

    def encode_document_tokens(self, text: str) -> TokenMatrixInput:
        ...


def encode_documents(
    model: MyLateInteractionModel,
    doc_ids: DocIdsInput,
    texts: DocTextsInput,
) -> list[TokenMatrixInput]:
    return [model.encode_document_tokens(str(text)) for text in texts]
```

That keeps editor help and annotations on the stable `kayak` public surface.
The same aliases are available in generated help:

```python
import kayak

print(kayak.help("typing"))
print(kayak.help("TokenMatrixInput"))
```

Current encoder behavior is intentionally simple:
- one query text at a time
- one document text at a time

If your model already has its own efficient batching path, keep that batching in
your wrapper and adapt it to Kayak through the callable encoder.

The factory is intentionally small:
- `open_encoder("colbert", model_name=...)`
- `open_encoder("callable", query_encoder=..., document_encoder=...)`
- `register_encoder(...)`

Examples:

- [`python/examples/colbert_hf_encoder.py`](../examples/colbert_hf_encoder.py)
- [`python/examples/byo_model_encoder.py`](../examples/byo_model_encoder.py)

## Text Retrievers

If you want one object that owns plain-text ingest, store materialization, and
search, use `LateTextRetriever`.

That is the highest-level public SDK shape today:

```python
import kayak

retriever = kayak.open_text_retriever(
    encoder="callable",
    store="kayak",
    encoder_kwargs={
        "query_encoder": my_query_encoder,
        "document_encoder": my_document_encoder,
    },
    store_kwargs={"path": "./kayak-index"},
)

retriever.upsert_texts(
    ["doc-a", "doc-b"],
    [
        "Pixi installs Python, Mojo, and kayak together.",
        "LanceDB can keep multivector rows on disk.",
    ],
    metadata=[
        {"topic": "installation"},
        {"topic": "storage"},
    ],
)

hits = retriever.search_text(
    "install python mojo together",
    k=2,
    where={"topic": "installation"},
)
```

`open_text_retriever(...)` prefers `kayak.MOJO_EXACT_CPU_BACKEND` automatically
when the active environment can actually run the Mojo backend. If Mojo is not
available, it falls back to `kayak.NUMPY_REFERENCE_BACKEND`. Pass
`backend=...` when you want to override that policy explicitly.

This path expects text strings. If your source data is PDFs, scans, tables, or
other mixed-layout documents, parse or extract them first and pass the resulting
text or token-level vectors into Kayak.

The retriever keeps the lower-level pieces injectable:
- pass your own encoder object
- pass your own store object
- or open both from the public factories

For most users this is the best mental model:

1. choose one encoder
2. choose one store
3. let the retriever own plain-text ingest plus search

The high-level contract stays narrow:
- `encode_query(text)`
- `encode_document_vectors(text)`
- `encode_documents(doc_ids, texts)`
- `upsert_texts(doc_ids, texts, metadata=None)`
- `delete(doc_ids)`
- `close()`
- `load_index(...)`
- `session(...)`
- `search_text(...)`
- `search_query(...)`
- `search_text_batch(...)`
- `search_query_batch(...)`
- `search_text_with_plan(...)`
- `search_query_with_plan(...)`

Use this when you want one object for normal text workflows.
Use raw encoders, stores, and `LateIndex` objects when you want lower-level
control over each step.

For repeated traffic against one stable slice:
- use `retriever.session(...)` when you want Kayak to load one exact slice once
  and keep one clean object for repeated search calls
- use `retriever.load_index(...)` when you want the raw reusable exact `LateIndex`
- use `retriever.search_text_batch(...)` when the queries still start as text
- use raw `query_batch(...)` and `search_batch(...)` when you already own the
  encoded queries

One reusable search session looks like this:

```python
session = retriever.session(
    where={"topic": "installation"},
    include_text=True,
)

hits = session.search_text("install python mojo together", k=2)
batch_hits = session.search_text_batch(
    [
        "install python mojo together",
        "storage rows",
    ],
    k=2,
)
```

If your application already owns the query fan-out, the same session object can
serve repeated calls from your own executor without reloading the slice:

```python
from concurrent.futures import ThreadPoolExecutor

session = retriever.session()
query_texts = ["install kayak", "exact search", "vector db storage"]

def run_query(text: str):
    return session.search_text(text, k=5)

with ThreadPoolExecutor(max_workers=3) as executor:
    hit_lists = list(executor.map(run_query, query_texts))
```

If you want one object that owns the encoder but you still want manual control
over persistence or search, the retriever also exposes side-effect-free
encoding helpers:

```python
encoded_query = retriever.encode_query("install python mojo together")
encoded_documents = retriever.encode_documents(
    ["doc-a", "doc-b"],
    [
        "Pixi installs Python, Mojo, and kayak together.",
        "LanceDB can keep multivector rows on disk.",
    ],
)
```

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

Optional database client packages stay separate from the core SDK:

- `uv add lancedb pyarrow` or `pixi add --pypi lancedb pyarrow`
- `uv add "psycopg[binary]" pgvector` or `pixi add --pypi "psycopg[binary]" pgvector`
- `uv add qdrant-client` or `pixi add --pypi qdrant-client`
- `uv add weaviate-client` or `pixi add --pypi weaviate-client`
- `uv add chromadb` or `pixi add --pypi chromadb`

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

Use the same store contract when your system of record is Postgres with
pgvector, Qdrant, Weaviate, or Chroma:

```python
pgvector_store = kayak.open_store(
    "pgvector",
    dsn="postgresql://postgres:postgres@127.0.0.1:5432/postgres",
    table_name="docs",
)

qdrant_store = kayak.open_store(
    "qdrant",
    client=my_qdrant_client,
    collection_name="docs",
)

weaviate_store = kayak.open_store(
    "weaviate",
    client=my_weaviate_client,
    collection_name="Doc",
    vector_name="colbert",
)

chroma_store = kayak.open_store(
    "chromadb",
    client=my_chroma_client,
    collection_name="docs",
)

for store in (pgvector_store, qdrant_store, weaviate_store, chroma_store):
    store.upsert(documents, metadata=metadata_rows)
    index = store.load_index(where={"topic": "installation"}, include_text=True)
```

Prefer the context-manager form for stores that may own client resources:

```python
with kayak.open_store("qdrant", client=my_qdrant_client, collection_name="docs") as store:
    store.upsert(documents, metadata=metadata_rows)
    index = store.load_index(include_text=True)
```

Store-specific filtering semantics are factual, not interchangeable:
- PgVector pushes simple scalar `where=` filters into Postgres JSONB
- Qdrant pushes simple scalar `where=` filters into Qdrant
- Chroma pushes simple scalar `where=` filters into Chroma
- Weaviate currently filters after collection iteration in the public adapter
- LanceDB currently filters after Arrow materialization in the public adapter
- PgVector stores the exact token matrix natively in Postgres as `vector(dim)[]`
- Chroma stores one pooled dense vector per document plus the exact token matrix in metadata

The store contract is intentionally narrow:
- `upsert(...)`
- `delete(...)`
- `load_index(...)`
- `close()`
- `stats()`
- `capabilities()`

The factory is intentionally small:
- `open_store("kayak", path=...)`
- `open_store("memory")`
- `open_store("lancedb", path=..., table_name=...)`
- `open_store("pgvector", dsn=... | connection=..., table_name=..., schema_name=...)`
- `open_store("qdrant", client=... | path=..., collection_name=...)`
- `open_store("weaviate", client=... | persistence_path=..., collection_name=..., vector_name=...)`
- `open_store("chromadb", client=... | path=..., collection_name=...)`
- `available_store_kinds()`
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
