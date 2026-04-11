from std.collections import List

from .workload_profile import WorkloadProfile


def default_workload_profiles() -> List[WorkloadProfile]:
    var profiles = List[WorkloadProfile]()

    profiles.append(
        WorkloadProfile(
            "lotte",
            "forum_fast",
            "Proxy for domain-specific forum retrieval like LoTTE forum queries.",
            192,
            20,
            12,
            32,
            10,
        )
    )
    profiles.append(
        WorkloadProfile(
            "beir",
            "fact_fast",
            "Proxy for heterogeneous factual retrieval like SciFact or FiQA in BEIR.",
            224,
            14,
            8,
            32,
            10,
        )
    )
    profiles.append(
        WorkloadProfile(
            "msmarco",
            "passage_fast",
            "Proxy for short web-passage retrieval with compact documents.",
            256,
            10,
            6,
            32,
            10,
        )
    )
    profiles.append(
        WorkloadProfile(
            "bright",
            "reasoning_fast",
            "Proxy for reasoning-heavy retrieval with longer document representations.",
            128,
            28,
            14,
            32,
            10,
        )
    )
    profiles.append(
        WorkloadProfile(
            "miracl",
            "multilingual_fast",
            "Proxy for multilingual retrieval with moderate query and document fan-out.",
            192,
            16,
            10,
            32,
            10,
        )
    )

    return profiles^
