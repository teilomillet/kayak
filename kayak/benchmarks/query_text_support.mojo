from kayak.planning import SearchPlan


def judged_query_text_for_plan(read plan: SearchPlan, description: String) -> String:
    if plan.stage3_verifier.requires_query_text:
        return description.copy()
    return String()
