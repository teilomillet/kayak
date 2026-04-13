# Typed planner-selection decision semantics.

from kayak.collections.validation import require_non_empty_string


comptime SEARCH_PLAN_ORDER_POLICY_GOAL_DEFAULT = "goal_default"
comptime SEARCH_PLAN_ORDER_POLICY_PREFERRED_OVERRIDE = "preferred_override"
comptime SEARCH_PLAN_ORDER_POLICY_CONSTRAINT_OVERRIDE = "constraint_override"

comptime SEARCH_PLAN_SELECTION_CONSTRAINT_NONE = "none"
comptime SEARCH_PLAN_SELECTION_CONSTRAINT_UNSUPPORTED_FILTER = "unsupported_filter"
comptime SEARCH_PLAN_SELECTION_CONSTRAINT_EXACT_STAGE1_REQUIRED = (
    "exact_stage1_required"
)
comptime SEARCH_PLAN_SELECTION_CONSTRAINT_ORACLE_REQUIRES_DEBUG = (
    "oracle_requires_debug"
)

comptime SEARCH_PLAN_SELECTION_OUTCOME_SELECTED_AVAILABLE = (
    "selected_available_generator"
)
comptime SEARCH_PLAN_SELECTION_OUTCOME_EXACT_FALLBACK_UNAVAILABLE = (
    "exact_fallback_unavailable"
)


struct SearchPlanSelectionDecision(Copyable):
    var order_policy_kind: String
    var constraint_kind: String
    var outcome_kind: String
    var explanation: String

    def __init__(
        out self,
        var order_policy_kind: String,
        var constraint_kind: String,
        var outcome_kind: String,
        var explanation: String,
    ) raises:
        self.order_policy_kind = require_non_empty_string(
            order_policy_kind,
            "selection_decision.order_policy_kind",
        )
        self.constraint_kind = require_non_empty_string(
            constraint_kind,
            "selection_decision.constraint_kind",
        )
        self.outcome_kind = require_non_empty_string(
            outcome_kind,
            "selection_decision.outcome_kind",
        )
        self.explanation = require_non_empty_string(
            explanation,
            "selection_decision.explanation",
        )
