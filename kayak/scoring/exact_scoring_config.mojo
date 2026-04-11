struct ExactScoringConfig(Copyable):
    var enable_parallel_scoring: Bool
    var enable_dim128_fast_path: Bool
    var enable_parallel_work_item_oversubscription: Bool
    var parallel_work_item_count_override: Int

    def __init__(out self):
        self.enable_parallel_scoring = True
        self.enable_dim128_fast_path = True
        self.enable_parallel_work_item_oversubscription = True
        self.parallel_work_item_count_override = 0
