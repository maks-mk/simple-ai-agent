import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class SessionSnapshot:
    session_id: str
    thread_id: str
    checkpoint_backend: str
    checkpoint_target: str
    created_at: str
    updated_at: str
    approval_mode: str = "prompt"

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_active_session(self) -> Optional[SessionSnapshot]:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text("utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            payload.setdefault("approval_mode", "prompt")
            return SessionSnapshot(**payload)
        except TypeError:
            return None

    def save_active_session(self, snapshot: SessionSnapshot) -> None:
        snapshot.touch()
        self.path.write_text(json.dumps(asdict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")

    def new_session(self, checkpoint_backend: str, checkpoint_target: str) -> SessionSnapshot:
        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        return SessionSnapshot(
            session_id=session_id,
            thread_id=f"session_{session_id}",
            checkpoint_backend=checkpoint_backend,
            checkpoint_target=checkpoint_target,
            created_at=now,
            updated_at=now,
        )
