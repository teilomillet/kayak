# Instructor Notes

This note records how to present the sequence without breaking the epistemic
standard.

## What To Emphasize

- the course is about debugging retrieval, not converting everyone to a new
  ideology
- exact late interaction is introduced first as a correctness anchor
- one-vector and chunked one-vector baselines are treated seriously because
  they are what many teams already use
- chunking is presented as a heuristic, not a moral failing
- the learner should feel like they are investigating their own system, not
  taking an exam in retrieval theory

## Backstage Versus Onstage

Keep this distinction explicit internally.

Backstage:

- evidence tables
- status labels
- rerun hooks
- scope warnings

Onstage:

- a concrete user pain
- a thing to try
- a result the learner can notice
- a small explanation of why it changed

Reason:
- the learner should experience the course as practical debugging
- the team still needs the backstage scaffold so the material stays honest

## Natural Learner Loop

Prefer this loop:

1. show a miss that feels real
2. let the learner try a comparison
3. name the variable that changed
4. show the narrow conclusion we can actually support
5. only then discuss broader limits

That is better than:

1. define every primitive up front
2. explain all evidence statuses
3. ask the learner to care about methodology before they care about the miss

Reason:
- most real users play first and theorize second
- the course should meet that behavior instead of fighting it

## What To Avoid Saying

Do not say:

- "chunking is wrong"
- "dense retrieval never works"
- "Kayak fixes RAG automatically"
- "late interaction is always better"

Reason:
- none of those claims are justified by the current repository evidence

## Safe Framing

Prefer:

- "Kayak makes the failure boundary explicit"
- "Kayak lets you compare simplifications against an exact reference path"
- "chunking can help when evidence is local and can hurt when evidence spans
  chunks"
- "a shortlist miss is different from a scorer miss"
- "let's see what changed when we removed this simplification"
- "now that we have seen the behavior, here is the narrow claim we can make"

## When To Surface Epistemic Language

Surface it explicitly when:

- a metric table appears
- a benchmark or judged slice is introduced
- the course is about to make a recommendation
- the course is drawing a boundary around what does not generalize

Keep it implicit when:

- the learner is exploring a toy failure
- the learner is comparing two visible outcomes
- the learner is building intuition from a local manipulation

Reason:
- rigor matters most at the moment a claim hardens
- constant explicit methodology language makes the experience feel school-like

## Best Teaching Query

For the real judged-slice module, prefer the cached `LIMIT-small` slice.

Reason:
- small enough to inspect manually
- strong enough to show a meaningful exact-versus-compressed gap

## When To Use The Toy Examples

Use the toy examples only to:

- make the retrieval geometry visible
- isolate one failure class at a time

Do not use them to:

- imply benchmark generalization
- make performance claims

## When To Show Real Numbers

Show real numbers when:

- comparing exact late interaction to one-vector or chunked one-vector
- discussing the stage-1 recall gap on named benchmark surfaces
- discussing current limits

Do not show real numbers without also saying:

- what slice they came from
- what baseline they compare against
- whether they are toy, judged-subset, or benchmark-trace evidence
