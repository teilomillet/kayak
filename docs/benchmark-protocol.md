# Reproduce a Kayak benchmark

**Current evidence: mock evaluator checks and a dataset audit. Full-model quality,
Mac/GPU latency, and accelerator memory remain unmeasured by this study.**
The [audit report](records/benchmark-audit.md) records the challenges and their limits.

## Run now: no model or GPU

From a checkout, install the base package and test tools:

```sh
uv venv .venv --allow-existing
uv pip install --python .venv/bin/python -e '.[test]'
uv run --no-sync -m benchmarks.mock_evaluation --output .benchmarks/mock-audit
uv run --no-sync kayak eval report .benchmarks/mock-audit/complete
uv run --no-sync pytest -q tests/test_eval_integrity.py tests/test_eval_measurement.py \
  tests/test_eval_report.py tests/test_dataset_audit.py
```

Installation may download Python packages. The mock campaign itself uses no
network, weights, PyTorch, or model service. Choose a fresh output directory for
each invocation. It deliberately creates `complete`, `failed`, and `interrupted`
runs; all three are retained. Expected choices, labels, and failure locations are
literal fixtures in [the script](../benchmarks/mock_evaluation.py). Their scores
check arithmetic and lifecycle behavior, not language understanding. Durations
and process memory vary across runs and are not inference measurements.

## Freeze the study before a real run

The initial question is narrow: how accurately does the pinned CLM classify the
official BANKING77 test texts using the fixed label descriptions, and what are
its sequential call durations on the recorded machine?

| Decision | Prespecified value |
| --- | --- |
| Model | Default pinned `Contrastive-LM/CLM-v0.1-8B` bundle; record returned model/encoder revisions and fingerprint |
| Data | BANKING77 at `57ec275d8078af65b7731c2a98be812d844a6d6b`, hash-checked locally |
| Development | 770 held-out original training rows; a 77-row balanced subset is the setup check |
| Primary evaluation | All 3,080 official test rows; no `--limit`, filtering, or relabeling |
| Request | Customer text plus the fixed question and all 77 label descriptions, in source order |
| Primary quality metric | First-attempt accuracy; failed or missing first attempts remain in the full denominator |
| Secondary quality | Top-5 accuracy, all-label macro F1, per-intent metrics, and confusion matrix |
| Initial Mac configuration | MPS, BF16, encoder batch size 1; a candidate configuration, not validated hardware support |
| Initial CUDA configuration | CUDA, BF16, encoder batch size 1; use only on compatible hardware |
| Execution | Three warmups, one measured call per example, order seed 42 |
| Timing | Synchronized sequential `decide`; exclude loading and evaluator bookkeeping |
| Failure policy | Retain failed/interrupted attempts and process runs; never replace them with only a successful rerun |

The exact question, partition recipe, candidate descriptions, and attribution are
in [evaluation.md](evaluation.md#dataset-and-task-contract). Repeated calls are
not independent quality samples. This zero-shot protocol is not directly
comparable to supervised or few-shot scores from the original paper.

Use development data to choose settings. Record those choices and any earlier
test exposure in the study notes before opening test scores. A configuration
change after inspecting test results makes that test developmental evidence;
declare it and obtain a new held-out dataset for a fresh generalization claim.
Do not drop the known normalized overlaps from the primary score. Any additional
filtered analysis needs its own frozen rule, IDs, denominator, and clear label.

## Later: commands for your Mac or GPU

These commands are prepared for the real run; they have not been validated on
your hardware. An 8B model requires substantial memory. Check the development
subset first and record allocation failures as outcomes. The optional
`--max-memory-gib` guard checks memory only after calls and cannot prevent OOM.

Use a clean committed checkout. Keep the source commit and resolved dependency
versions with the results so another operator can restore the same environment.
The run report records model identity and a hash of the installed Kayak source.

```sh
uv venv .venv --allow-existing
uv pip install --python .venv/bin/python -e '.[local]'
mkdir -p .benchmarks/study
git status --porcelain > .benchmarks/study/working-tree.txt
git rev-parse HEAD > .benchmarks/study/source-commit.txt
uv pip freeze --python .venv/bin/python --exclude-editable \
  > .benchmarks/study/python-packages.txt
uv run --no-sync kayak eval prepare banking77
uv run --no-sync -m benchmarks.audit_dataset > .benchmarks/study/dataset-audit.json
```

Check that `working-tree.txt` is empty before proceeding. Archive the source
commit and package list with the final artifacts. Record the machine's power
mode, whether it is plugged in, and other active workloads in a plain study note.
The automated metadata cannot establish a thermally stable, otherwise idle host.
It snapshots source and runtime settings before execution; it does not monitor
whether another thread or callback changes them during the run. This installation
route leaves no new `uv.lock` in the checkout. To recreate dependencies later,
install `python-packages.txt` in a fresh environment, then install the recorded
source checkout with `uv pip install --python .venv/bin/python --no-deps -e .`.

On Apple Silicon, start with this development subset:

```sh
if uv run --no-sync kayak eval run banking77 --split dev --limit 77 \
  --device mps --dtype bfloat16 --batch-size 1 --warmups 3 --repeats 1 --seed 42 \
  --output .benchmarks/study/dev-setup \
  > .benchmarks/study/dev-setup.stdout.log 2> .benchmarks/study/dev-setup.stderr.log
then
  printf '%s\n' 0 > .benchmarks/study/dev-setup.exit-code.txt
else
  printf '%s\n' "$?" > .benchmarks/study/dev-setup.exit-code.txt
fi
if [ -f .benchmarks/study/dev-setup/report.json ]; then
  uv run --no-sync kayak eval report .benchmarks/study/dev-setup \
    > .benchmarks/study/dev-setup/report.md
fi
```

For an NVIDIA GPU, replace `--device mps` with `--device cuda`. Do not silently
change precision, batching, or fallback settings on failure; record the failure
and freeze a revised development configuration first. Separate machines get
separate study directories and reports.

After confirming and freezing the configuration, run the full test split once:

```sh
if uv run --no-sync kayak eval run banking77 --split test \
  --device mps --dtype bfloat16 --batch-size 1 --warmups 3 --repeats 1 --seed 42 \
  --output .benchmarks/study/test-frozen \
  > .benchmarks/study/test-frozen.stdout.log 2> .benchmarks/study/test-frozen.stderr.log
then
  printf '%s\n' 0 > .benchmarks/study/test-frozen.exit-code.txt
else
  printf '%s\n' "$?" > .benchmarks/study/test-frozen.exit-code.txt
fi
if [ -f .benchmarks/study/test-frozen/report.json ]; then
  uv run --no-sync kayak eval report .benchmarks/study/test-frozen \
    > .benchmarks/study/test-frozen/report.md
fi
```

The test run makes 3,083 decisions including warmups, each with 77 candidates.
The current runtime encodes candidates per request; this can be lengthy. A
nonzero exit needs investigation even if an artifact exists. `eval report`
verifies and renders incomplete runs too, so its success is not a quality gate.
Read each saved exit code and stderr log before proceeding. Setup failures can
occur before `report.json` exists; these separate logs preserve them. Keep fresh
paths for subsequent attempts so reruns do not replace the failure logs.

## Evidence needed for stronger claims

A completed test run supports a descriptive score on this fixed dataset and
configuration. It does not establish calibrated probabilities, absence of
exposure during encoder or head training, production quality, or superiority
over Jev/Laya.
Those need separately defined tasks and equivalent measured baselines.

For an optimization claim, use the development split and choose the changed
factor before measurement. Run at least three fresh process pairs, alternating
baseline/candidate order; match each pair's dataset, seed, warmups, dependencies,
machine, and power state. Retain all pairs and setup failures. Report each run
and the observed range, not just the best speedup. Repeats inside a single loaded
model do not substitute for these process trials. Three pairs are a starting
observation, not a significance or power guarantee.

`kayak eval compare BASELINE CANDIDATE` discloses changed factors and withholds
speedups when recorded execution conditions are incompatible or unknown. Even
an eligible comparison does not control thermal state or host load. Keep exact
raw artifacts beside the generated Markdown and retain this protocol as it
stood before the measurements.
