"""Generated public help for the Kayak Python SDK."""

from __future__ import annotations

from collections import defaultdict
import inspect
import sys
from types import ModuleType

from .encoders.registry import _ENCODER_FACTORIES
from .stores.registry import _STORE_FACTORIES
from . import typing as public_typing


CATEGORY_ORDER = (
    "Retrievers",
    "Encoders",
    "Stores",
    "Typing",
    "Core Operations",
    "Search Plans",
    "Backends",
    "Other Public API",
)


def _package_module() -> ModuleType:
    package = sys.modules.get("kayak")
    if package is None:  # pragma: no cover - defensive only
        import kayak as package  # type: ignore[no-redef]
    return package


def _public_members() -> list[tuple[str, object]]:
    package = _package_module()
    members: list[tuple[str, object]] = []
    for name in getattr(package, "PUBLIC_API", ()):
        members.append((name, getattr(package, name)))
    return members


def _public_method_members() -> list[tuple[str, object, str]]:
    methods: list[tuple[str, object, str]] = []
    for owner_name, obj in _public_members():
        if not inspect.isclass(obj):
            continue
        for method_name, method in _iter_public_class_members(obj):
            methods.append((method_name, method, owner_name))
    return methods


def _typing_alias_members() -> list[tuple[str, object]]:
    return [
        (name, getattr(public_typing, name))
        for name in getattr(public_typing, "__all__", ())
    ]


def _typing_alias_summary(name: str) -> str:
    return getattr(public_typing, "_ALIAS_SUMMARIES", {}).get(
        name,
        "Stable public type alias.",
    )


def _typing_alias_type_text(name: str, obj: object) -> str:
    return getattr(public_typing, "_ALIAS_TYPE_TEXT", {}).get(name, str(obj))


def _doc_summary(obj: object) -> str:
    doc = inspect.getdoc(obj)
    if not doc:
        return "No public docstring available yet."
    return next(
        (line.strip() for line in doc.splitlines() if line.strip()),
        "No public docstring available yet.",
    )


def _signature_text(
    obj: object, *, drop_bound_first_param: bool = False
) -> str | None:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    if drop_bound_first_param:
        parameters = list(signature.parameters.values())
        if parameters and parameters[0].name in {"self", "cls"}:
            signature = signature.replace(parameters=parameters[1:])
    return str(signature)


def _topic_name(obj: object) -> str:
    return getattr(obj, "__name__", obj.__class__.__name__)


def _category_for(name: str, obj: object) -> str:
    module = getattr(obj, "__module__", "")
    lowered = name.lower()

    if lowered.endswith("_backend") or lowered in {
        "backendinfo",
        "mojobridgeinfo",
        "available_backends",
        "backend_info",
        "mojo_bridge_info",
    }:
        return "Backends"

    if module.startswith("kayak.encoders"):
        return "Encoders"

    if module.startswith("kayak.stores"):
        return "Stores"

    if module.startswith("kayak.retrievers"):
        return "Retrievers"

    if module == "kayak.typing" or lowered == "typing":
        return "Typing"

    if any(
        token in lowered
        for token in (
            "plan",
            "candidate",
            "verifier",
            "operator",
            "semantics",
        )
    ):
        return "Search Plans"

    if lowered in {
        "query",
        "query_batch",
        "documents",
        "packed_index",
        "hybrid_flat_dim128_index",
        "flat_query_dim128",
        "latequery",
        "latequerybatch",
        "latedocuments",
        "lateindex",
        "latescores",
        "searchhit",
        "maxsim",
        "maxsim_batch",
        "search",
        "search_batch",
        "generate_candidates",
        "search_with_plan",
        "searchplanresult",
        "candidatestageresult",
        "searchstageprofile",
        "stageartifactmaterialization",
    }:
        return "Core Operations"

    return "Other Public API"


def _grouped_members() -> dict[str, list[tuple[str, object]]]:
    groups: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for name, obj in _public_members():
        groups[_category_for(name, obj)].append((name, obj))
    for members in groups.values():
        members.sort(key=lambda item: item[0].lower())
    ordered: dict[str, list[tuple[str, object]]] = {}
    for category in CATEGORY_ORDER:
        if category in groups:
            ordered[category] = groups[category]
    for category in sorted(groups):
        if category not in ordered:
            ordered[category] = groups[category]
    return ordered


def _is_constant(name: str, obj: object) -> bool:
    return name.isupper() and isinstance(obj, (str, int, float, bool))


def _registered_kind_lines(kind: str) -> list[str]:
    if kind == "Encoders":
        kinds = sorted(_ENCODER_FACTORIES)
        return [f"Registered kinds: {', '.join(kinds)}", ""]
    if kind == "Stores":
        kinds = sorted(_STORE_FACTORIES)
        return [f"Registered kinds: {', '.join(kinds)}", ""]
    return []


def _format_public_entry(name: str, obj: object) -> str:
    if _is_constant(name, obj):
        return f"- {name} = {obj!r}"
    signature = _signature_text(obj)
    call = name if signature is None else f"{name}{signature}"
    return f"- {call}\n  {_doc_summary(obj)}"


def _format_typing_alias_entry(name: str, obj: object) -> str:
    return (
        f"- {name} = {_typing_alias_type_text(name, obj)}\n"
        f"  {_typing_alias_summary(name)}"
    )


def _iter_public_class_members(
    cls: type[object],
) -> list[tuple[str, object]]:
    members: list[tuple[str, object]] = []
    for name, member in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(member, classmethod | staticmethod):
            members.append((name, member.__func__))
            continue
        if callable(member):
            members.append((name, member))
    return members


def _public_method_entries(cls: type[object]) -> list[str]:
    entries: list[str] = []
    for name, member in _iter_public_class_members(cls):
        signature = _signature_text(member, drop_bound_first_param=True)
        call = name if signature is None else f"{name}{signature}"
        entries.append(f"- {call}\n  {_doc_summary(member)}")
    return entries


def _related_topics_block(topics: list[str]) -> list[str]:
    unique_topics: list[str] = []
    for topic in topics:
        if topic not in unique_topics:
            unique_topics.append(topic)
    if not unique_topics:
        return []
    return [
        "",
        "Related topics:",
        *(f'- `kayak.help("{topic}")`' for topic in unique_topics),
    ]


def _related_topics_for_entry(name: str, obj: object) -> list[str]:
    category = _category_for(name, obj)
    if inspect.isclass(obj):
        return [category, *[method for method, _ in _iter_public_class_members(obj)[:4]]]
    siblings = [
        member_name
        for member_name, _ in _grouped_members().get(category, ())
        if member_name != name
    ]
    return [category, *siblings[:4]]


def _related_topics_for_method(
    method_name: str,
    owner_name: str,
    owner_cls: type[object],
) -> list[str]:
    method_tokens = set(method_name.split("_"))
    scored_siblings: list[tuple[int, int, str]] = []
    fallback_siblings: list[str] = []
    for index, (sibling_name, _) in enumerate(
        _iter_public_class_members(owner_cls)
    ):
        if sibling_name == method_name:
            continue
        fallback_siblings.append(sibling_name)
        overlap = len(method_tokens.intersection(sibling_name.split("_")))
        if overlap > 0:
            scored_siblings.append((-overlap, index, sibling_name))

    related = [owner_name]
    related.extend(
        sibling_name
        for _, _, sibling_name in sorted(scored_siblings)[:4]
    )
    for sibling_name in fallback_siblings:
        if sibling_name not in related:
            related.append(sibling_name)
        if len(related) >= 5:
            break
    return related[:5]


def _render_overview() -> str:
    package = _package_module()
    doc = inspect.getdoc(package) or "Kayak Python SDK."

    sections = [
        doc.splitlines()[0],
        "",
        'Use `kayak.help("search")`, `kayak.help("stores")`, `kayak.help("typing")`, `kayak.help("mojo")`, `kayak.help("doctor")`, `kayak.help("search_text")`, or `kayak.help(kayak.LateTextRetriever)`.',
        "",
    ]
    for category, members in _grouped_members().items():
        sections.append(f"{category}:")
        sections.extend(_registered_kind_lines(category))
        if category == "Typing":
            sections.extend(_format_public_entry(name, obj) for name, obj in members)
            sections.append("")
            sections.append("Stable aliases:")
            sections.extend(
                _format_typing_alias_entry(name, obj)
                for name, obj in _typing_alias_members()
            )
        else:
            sections.extend(_format_public_entry(name, obj) for name, obj in members)
        sections.append("")
    return "\n".join(sections).strip()


def _render_object(name: str, obj: object) -> str:
    lines = [name]

    if _is_constant(name, obj):
        lines.append(repr(obj))
        return "\n".join(lines).strip()

    signature = _signature_text(obj)
    if signature is not None:
        lines.append(signature)

    module = getattr(obj, "__module__", None)
    if module is None and inspect.ismodule(obj):
        module = getattr(obj, "__name__", None)
    if module:
        lines.append(f"module: {module}")

    doc = inspect.getdoc(obj)
    if doc:
        lines.append("")
        lines.append(doc)

    if inspect.isclass(obj):
        methods = _public_method_entries(obj)
        if methods:
            lines.append("")
            lines.append("Public methods:")
            lines.extend(methods)

    lines.extend(_related_topics_block(_related_topics_for_entry(name, obj)))

    return "\n".join(lines).strip()


def _render_method(name: str, obj: object, *, owner_name: str) -> str:
    lines = [f"{owner_name}.{name}"]

    signature = _signature_text(obj, drop_bound_first_param=True)
    if signature is not None:
        lines.append(signature)

    module = getattr(obj, "__module__", None)
    if module:
        lines.append(f"module: {module}")

    doc = inspect.getdoc(obj)
    if doc:
        lines.append("")
        lines.append(doc)

    owner = _package_module().__dict__.get(owner_name)
    if inspect.isclass(owner):
        lines.extend(
            _related_topics_block(
                _related_topics_for_method(name, owner_name, owner)
            )
        )

    return "\n".join(lines).strip()


def _render_typing_module() -> str:
    lines = ["typing", "module: kayak.typing", ""]

    doc = inspect.getdoc(public_typing)
    if doc:
        lines.append(doc)
        lines.append("")

    lines.append("Stable aliases:")
    lines.extend(
        _format_typing_alias_entry(name, obj)
        for name, obj in _typing_alias_members()
    )
    lines.extend(
        _related_topics_block(
            ["Typing", *[name for name, _ in _typing_alias_members()[:4]]]
        )
    )
    return "\n".join(lines).strip()


def _render_typing_alias(name: str, obj: object) -> str:
    lines = [
        name,
        _typing_alias_type_text(name, obj),
        "module: kayak.typing",
        "",
        _typing_alias_summary(name),
    ]
    sibling_names = [
        sibling_name
        for sibling_name, _ in _typing_alias_members()
        if sibling_name != name
    ]
    lines.extend(_related_topics_block(["typing", *sibling_names[:4]]))
    return "\n".join(lines).strip()


def _render_category(category: str, members: list[tuple[str, object]]) -> str:
    if category == "Typing":
        return _render_typing_module()
    lines = [category, ""]
    lines.extend(_registered_kind_lines(category))
    lines.extend(_format_public_entry(name, obj) for name, obj in members)
    lines.extend(_related_topics_block([name for name, _ in members[:4]]))
    return "\n".join(lines).strip()


def _normalize_topic(topic: str) -> str:
    return topic.strip().lower().replace(" ", "_")


def _topic_aliases() -> dict[str, str]:
    return {
        "retriever": "Retrievers",
        "retrieval": "Core Operations",
        "encoder": "Encoders",
        "store": "Stores",
        "type": "typing",
        "types": "typing",
        "annotations": "typing",
        "plan": "Search Plans",
        "plans": "Search Plans",
        "session": "LateTextSearchSession",
        "sessions": "LateTextSearchSession",
        "prepared": "LateTextSearchSession",
        "diagnostics": "doctor",
        "environment": "doctor",
        "backend": "Backends",
        "mojo": "Backends",
    }


def _render_matches(
    topic: str,
    matches: list[tuple[str, object]],
) -> str:
    lines = [f'Closest public matches for "{topic}"', ""]
    lines.extend(_format_public_entry(name, obj) for name, obj in matches[:12])
    return "\n".join(lines).strip()


def _render_method_matches(
    topic: str,
    matches: list[tuple[str, object, str]],
) -> str:
    lines = [f'Closest public methods for "{topic}"', ""]
    for method_name, method, owner_name in matches[:12]:
        signature = _signature_text(method, drop_bound_first_param=True)
        call = (
            f"{owner_name}.{method_name}"
            if signature is None
            else f"{owner_name}.{method_name}{signature}"
        )
        lines.append(f"- {call}\n  {_doc_summary(method)}")
    related_topics: list[str] = []
    for method_name, _obj, owner_name in matches[:12]:
        related_topics.append(owner_name)
        related_topics.append(f"{owner_name}.{method_name}")
    lines.extend(_related_topics_block(related_topics[:6]))
    return "\n".join(lines).strip()


def _render_typing_alias_matches(
    topic: str,
    matches: list[tuple[str, object]],
) -> str:
    lines = [f'Closest typing aliases for "{topic}"', ""]
    lines.extend(
        _format_typing_alias_entry(name, obj) for name, obj in matches[:12]
    )
    lines.extend(_related_topics_block(["typing"]))
    return "\n".join(lines).strip()


def help(topic: str | object | None = None) -> str:
    """Return generated public help from exported objects, signatures, and docstrings.

    The help text is derived from the current public API rather than a second
    handwritten registry, so it stays aligned with `kayak.PUBLIC_API`.
    """

    if topic is None:
        return _render_overview()

    if not isinstance(topic, str):
        name = _topic_name(topic)
        return _render_object(name, topic)

    normalized = _normalize_topic(topic)
    aliases = _topic_aliases()
    if normalized in aliases:
        normalized = _normalize_topic(aliases[normalized])
    public_members = _public_members()
    public_methods = _public_method_members()
    typing_aliases = _typing_alias_members()

    if normalized == "typing":
        return _render_typing_module()

    exact_typing_matches = [
        (name, obj)
        for name, obj in typing_aliases
        if name.lower() == normalized
    ]
    if exact_typing_matches:
        name, obj = exact_typing_matches[0]
        return _render_typing_alias(name, obj)

    exact_name_matches = [
        (name, obj)
        for name, obj in public_members
        if name.lower() == normalized
    ]
    if exact_name_matches:
        name, obj = exact_name_matches[0]
        return _render_object(name, obj)

    exact_method_matches = [
        (method_name, obj, owner_name)
        for method_name, obj, owner_name in public_methods
        if method_name.lower() == normalized
        or f"{owner_name.lower()}.{method_name.lower()}" == normalized
    ]
    if len(exact_method_matches) == 1:
        method_name, obj, owner_name = exact_method_matches[0]
        return _render_method(method_name, obj, owner_name=owner_name)
    if exact_method_matches:
        return _render_method_matches(topic, exact_method_matches)

    groups = _grouped_members()
    for category, members in groups.items():
        if _normalize_topic(category) == normalized:
            return _render_category(category, members)

    fuzzy_matches = [
        (name, obj)
        for name, obj in public_members
        if normalized in name.lower()
    ]
    if len(fuzzy_matches) == 1:
        name, obj = fuzzy_matches[0]
        return _render_object(name, obj)

    if fuzzy_matches:
        return _render_matches(topic, fuzzy_matches)

    fuzzy_typing_matches = [
        (name, obj)
        for name, obj in typing_aliases
        if normalized in name.lower()
    ]
    if len(fuzzy_typing_matches) == 1:
        name, obj = fuzzy_typing_matches[0]
        return _render_typing_alias(name, obj)
    if fuzzy_typing_matches:
        return _render_typing_alias_matches(topic, fuzzy_typing_matches)

    fuzzy_method_matches = [
        (method_name, obj, owner_name)
        for method_name, obj, owner_name in public_methods
        if normalized in method_name.lower()
        or normalized in f"{owner_name.lower()}.{method_name.lower()}"
    ]
    if len(fuzzy_method_matches) == 1:
        method_name, obj, owner_name = fuzzy_method_matches[0]
        return _render_method(method_name, obj, owner_name=owner_name)
    if fuzzy_method_matches:
        return _render_method_matches(topic, fuzzy_method_matches)

    available_topics = ", ".join(
        sorted(name for name, _ in public_members)[:16]
    )
    return (
        f'No public help topic named "{topic}".\n\n'
        f'Try one of the exported API names, a public method topic such as `"search_text"` or `"pack"`, or call `kayak.help()`.\n'
        f"Sample public topics: {available_topics}"
    )
