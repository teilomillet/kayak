# Kayak Python SDK

`kayak` is a Python SDK for late-interaction retrieval.

It gives you explicit objects for:
- queries
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

## Layouts

Kayak keeps layout changes explicit.

Example:

```python
flat_query = query.to_layout("flat_dim128")
hybrid_index = index.to_layout("hybrid_flat_dim128")

scores = kayak.maxsim(flat_query, hybrid_index)
```

## Backends

The package exposes two named backends:
- `kayak.NUMPY_REFERENCE_BACKEND`
- `kayak.MOJO_EXACT_CPU_BACKEND`

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
- `LateQuery`
- `LateDocuments`
- `LateIndex`
- `LateScores`
- `SearchHit`
- `query`
- `documents`
- `packed_index`
- `hybrid_flat_dim128_index`
- `flat_query_dim128`
- `maxsim`
- `search`

## Mental Model

Kayak is not a generic tensor library.

It is a late-interaction retrieval API with:
- ragged query and document vector counts
- explicit layout conversion
- exact MaxSim scoring
- explicit search backends

That makes it suitable for code that wants retrieval semantics first, while
still fitting naturally into Python workflows built on NumPy or PyTorch.
