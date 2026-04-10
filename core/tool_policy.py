from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    read_only: bool = False
    mutating: bool = False
    destructive: bool = False
    requires_approval: bool = False
    networked: bool = False
    source: str = "local"
    impact_scope: str = "local_state"
    ui_kind: str = "generic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Dict[str, Any] | None,
        *,
        source: str = "local",
    ) -> "ToolMetadata":
        payload = dict(data or {})
        payload.setdefault("name", name)
        payload.setdefault("source", source)
        return cls(
            name=str(payload.get("name") or name),
            read_only=bool(payload.get("read_only", False)),
            mutating=bool(payload.get("mutating", False)),
            destructive=bool(payload.get("destructive", False)),
            requires_approval=bool(payload.get("requires_approval", False)),
            networked=bool(payload.get("networked", False)),
            source=str(payload.get("source") or source),
            impact_scope=str(payload.get("impact_scope") or ("unknown" if source == "mcp" else "local_state")),
            ui_kind=str(payload.get("ui_kind") or "generic"),
        )


def default_tool_metadata(name: str, source: str = "local") -> ToolMetadata:
    if source == "mcp":
        return ToolMetadata(
            name=name,
            read_only=False,
            mutating=True,
            destructive=False,
            requires_approval=True,
            networked=True,
            source="mcp",
            impact_scope="unknown",
            ui_kind="generic",
        )
    return ToolMetadata(
        name=name,
        read_only=True,
        networked=False,
        source=source,
        impact_scope="local_state",
        ui_kind="generic",
    )
