"""Pending-session markers, run locks and recovery of dead runs' sessions."""

from __future__ import annotations

import errno
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


_PENDING_SESSIONS_RELATIVE_DIR = Path("openviking") / "pending_sessions"
_RUN_LOCKS_RELATIVE_DIR = Path("openviking") / "runs"
_LEGACY_RECOVERY_LOCK_FILENAME = "legacy-recovery.lock"
_LOCK_BUSY_ERRNOS = {errno.EWOULDBLOCK, errno.EACCES, errno.EAGAIN}
