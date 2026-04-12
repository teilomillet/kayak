from kayak.numeric import MetricScalar


struct FaithfulnessPolicy(Copyable):
    var kind: String

    def __init__(out self, var kind: String) raises:
        if (
            kind != "best_effort"
            and kind != "exact_stage1_required"
            and kind != "oracle_full_recall_required"
        ):
            raise Error("unknown faithfulness policy kind: " + kind)

        self.kind = kind^


def best_effort_faithfulness_policy() raises -> FaithfulnessPolicy:
    return FaithfulnessPolicy("best_effort")


def exact_stage1_required_faithfulness_policy() raises -> FaithfulnessPolicy:
    return FaithfulnessPolicy("exact_stage1_required")


def oracle_full_recall_required_faithfulness_policy() raises -> FaithfulnessPolicy:
    return FaithfulnessPolicy("oracle_full_recall_required")


def stage1_generator_is_exact(candidate_generator_kind: String) -> Bool:
    return candidate_generator_kind == "exact_full_scan"


def faithfulness_evidence_kind(
    candidate_generator_kind: String,
    observed_candidate_recall_at_final_k: MetricScalar,
) -> String:
    if stage1_generator_is_exact(candidate_generator_kind):
        return "exact_stage1"

    if observed_candidate_recall_at_final_k == MetricScalar(1.0):
        return "oracle_full_recall"

    return "oracle_recall_loss"


struct FaithfulnessAssessment(Copyable):
    var policy_kind: String
    var evidence_kind: String
    var stage1_is_exact: Bool
    var has_oracle_recall_measurement: Bool
    var observed_candidate_recall_at_final_k: MetricScalar
    var passes: Bool
    var message: String

    def __init__(
        out self,
        var policy_kind: String,
        var evidence_kind: String,
        stage1_is_exact: Bool,
        has_oracle_recall_measurement: Bool,
        observed_candidate_recall_at_final_k: MetricScalar,
        passes: Bool,
        var message: String,
    ):
        self.policy_kind = policy_kind^
        self.evidence_kind = evidence_kind^
        self.stage1_is_exact = stage1_is_exact
        self.has_oracle_recall_measurement = has_oracle_recall_measurement
        self.observed_candidate_recall_at_final_k = (
            observed_candidate_recall_at_final_k
        )
        self.passes = passes
        self.message = message^


def assess_faithfulness(
    read policy: FaithfulnessPolicy,
    candidate_generator_kind: String,
    observed_candidate_recall_at_final_k: MetricScalar,
) -> FaithfulnessAssessment:
    var stage1_is_exact = stage1_generator_is_exact(candidate_generator_kind)
    var evidence_kind = faithfulness_evidence_kind(
        candidate_generator_kind,
        observed_candidate_recall_at_final_k,
    )

    if policy.kind == "exact_stage1_required":
        if stage1_is_exact:
            return FaithfulnessAssessment(
                policy.kind.copy(),
                evidence_kind,
                stage1_is_exact,
                True,
                observed_candidate_recall_at_final_k,
                True,
                "faithfulness policy satisfied: stage 1 is exact",
            )

        return FaithfulnessAssessment(
            policy.kind.copy(),
            evidence_kind,
            stage1_is_exact,
            True,
            observed_candidate_recall_at_final_k,
            False,
            (
                "faithfulness policy violated: exact_stage1_required but candidate generator is "
                + candidate_generator_kind
            ),
        )

    if policy.kind == "oracle_full_recall_required":
        if (
            stage1_is_exact
            or observed_candidate_recall_at_final_k == MetricScalar(1.0)
        ):
            if stage1_is_exact:
                return FaithfulnessAssessment(
                    policy.kind.copy(),
                    evidence_kind,
                    stage1_is_exact,
                    True,
                    observed_candidate_recall_at_final_k,
                    True,
                    "faithfulness policy satisfied: exact stage 1 implies full recall at final_k"
                )

            return FaithfulnessAssessment(
                policy.kind.copy(),
                evidence_kind,
                stage1_is_exact,
                True,
                observed_candidate_recall_at_final_k,
                True,
                (
                    "faithfulness policy satisfied: exact oracle verified full recall at final_k"
                ),
            )

        return FaithfulnessAssessment(
            policy.kind.copy(),
            evidence_kind,
            stage1_is_exact,
            True,
            observed_candidate_recall_at_final_k,
            False,
            (
                "faithfulness policy violated: exact oracle observed recall loss at final_k with candidate recall "
                + String(observed_candidate_recall_at_final_k)
            ),
        )

    if stage1_is_exact:
        return FaithfulnessAssessment(
            policy.kind.copy(),
            evidence_kind,
            stage1_is_exact,
            True,
            observed_candidate_recall_at_final_k,
            True,
            "best_effort policy: stage 1 happens to be exact",
        )

    if observed_candidate_recall_at_final_k == MetricScalar(1.0):
        return FaithfulnessAssessment(
            policy.kind.copy(),
            evidence_kind,
            stage1_is_exact,
            True,
            observed_candidate_recall_at_final_k,
            True,
            (
                "best_effort policy: approximate stage 1 matched the exact oracle at final_k for this query"
            ),
        )

    return FaithfulnessAssessment(
        policy.kind.copy(),
        evidence_kind,
        stage1_is_exact,
        True,
        observed_candidate_recall_at_final_k,
        True,
        "best_effort policy: approximate stage 1 lost oracle recall at final_k for this query",
    )
