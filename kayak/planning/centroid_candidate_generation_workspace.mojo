# Reusable centroid-family stage-1 workspace.
# Owns reusable scratch for repeated centroid candidate generation over a loaded
# snapshot. It does not own planner policy or snapshot lifetime.

from .centroid_postings_blockmax_stage import MutableCentroidPostingBlockmaxScratch
from .centroid_primitives import MutableCentroidSegmentAccumulator


struct MutableCentroidCandidateGenerationWorkspace:
    var segment_accumulator: MutableCentroidSegmentAccumulator
    var blockmax_scratch: MutableCentroidPostingBlockmaxScratch

    def __init__(out self):
        self.segment_accumulator = MutableCentroidSegmentAccumulator()
        self.blockmax_scratch = MutableCentroidPostingBlockmaxScratch()
