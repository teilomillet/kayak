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

If you do not pass `backend=kayak.MOJO_EXACT_CPU_BACKEND`, Kayak stays on the
NumPy reference backend and does not require Mojo.

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

Candidate-window rescoring stays explicit:

```python
hits = kayak.search(query, index, k=10)
candidate_index = index.select([hit.doc_id for hit in hits])
candidate_scores = kayak.maxsim(query, candidate_index)
```

That is still late interaction as a primitive: explicit selection plus MaxSim,
not a hidden rerank mode.

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
- `LateQuery`
- `LateQueryBatch`
- `LateDocuments`
- `LateIndex`
- `LateScores`
- `SearchHit`
- `available_backends`
- `backend_info`
- `query`
- `query_batch`
- `documents`
- `packed_index`
- `hybrid_flat_dim128_index`
- `flat_query_dim128`
- `maxsim`
- `maxsim_batch`
- `search`
- `search_batch`

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
