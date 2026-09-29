"""Turn upload, tracked workers, session commit and the exit commit."""

from __future__ import annotations

import atexit
import json
import threading
import weakref
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .connection import _CommitScope
from .deps import default_deps
from .host import env_var_enabled, spawn_context_thread
from .http import _TIMEOUT, _status_code_from_error, _VikingClient
from .log import get_logger
from .transcript import _message_text

logger = get_logger()


_SESSION_DRAIN_TIMEOUT = 10.0
_DEFERRED_COMMIT_TIMEOUT = (_TIMEOUT * 2) + 5.0
_SESSION_MESSAGE_BATCH_LIMIT = 100
_SYNC_TRACE_ENV = "HERMES_OPENVIKING_SYNC_TRACE"
# Host contexts that must not write into OpenViking. Fixed-prompt output from scheduled
# jobs, delegated subagents, and flush forks has no memory value and would spend server-side
# extraction budget. Hermes delivers the context to initialize(); recall/read paths are unchanged.
_NON_PRIMARY_AGENT_CONTEXTS = frozenset({"cron", "subagent", "flush"})


# atexit safety net: commit the current session even if shutdown_memory_provider
# never runs (gateway crash, exception in the session expiry watcher, ...).
# Only primary providers are held, and only weakly: a multiplexed gateway keeps one per
# profile, and a provider dropped by soft eviction must stay collectable. The hook is
# registered on the first primary initialize(), never at import.
_exit_registry: "weakref.WeakSet[OpenVikingMemoryProvider]" = weakref.WeakSet()  # noqa: F821
_exit_registry_lock = threading.Lock()
_exit_hook_registered = False
# The CLI's exit watchdog os._exit()s 30 s after cleanup starts
# (hermes:hermes_cli/cli_shutdown.py:51-99), and memory shutdown runs inside that window.
_EXIT_COMMIT_BUDGET = 20.0


def _register_for_exit(provider: "OpenVikingMemoryProvider") -> None:  # noqa: F821
    global _exit_hook_registered
    with _exit_registry_lock:
        if provider._shutting_down:
            return
        if not _exit_hook_registered:
            atexit.register(_atexit_commit_sessions)
            _exit_hook_registered = True
        _exit_registry.add(provider)


def _deregister_for_exit(provider: "OpenVikingMemoryProvider") -> None:  # noqa: F821
    with _exit_registry_lock:
        _exit_registry.discard(provider)


def _atexit_commit_sessions(budget: float = _EXIT_COMMIT_BUDGET):
    monotonic = default_deps().monotonic
    deadline = monotonic() + budget
    with _exit_registry_lock:
        providers = list(_exit_registry)
        _exit_registry.clear()
    for provider in providers:
        try:
            remaining = deadline - monotonic()
            if remaining <= 0:
                # No new commit once the budget is spent; the marker stays for the next run's recovery.
                logger.warning("OpenViking exit budget used up; leaving session %s pending", provider._session_id)
                continue
            with suppress(Exception):  # best-effort at shutdown time
                provider._end_session(min(_SESSION_DRAIN_TIMEOUT, remaining), deadline=deadline)
        finally:
            # ``finally`` (as on main): the run lock is released even when the commit
            # dies of a BaseException (KeyboardInterrupt during atexit).
            with suppress(Exception):
                provider._release_run_lock()


@dataclass
class _TurnUpload:
    """One turn's OpenViking upload: structured batches first, falling back to plain text
    on a first-batch failure, and to individual messages after a failed retry."""

    client: _VikingClient
    sid: str
    batch_messages: List[Dict[str, Any]]
    user_content: str
    assistant_content: str
    assistant_peer_id: str
    user_peer_id: str = ""
    next_index: int = 0

    def _trace(self, fmt: str, *args) -> None:
        if env_var_enabled(_SYNC_TRACE_ENV):
            logger.info("OpenViking sync_turn trace: " + fmt, *args)

    def post(self, client: _VikingClient) -> None:
        while self.next_index < len(self.batch_messages):
            batch_end = min(self.next_index + _SESSION_MESSAGE_BATCH_LIMIT, len(self.batch_messages))
            payload = {"messages": self.batch_messages[self.next_index:batch_end]}
            self._trace("POST /api/v1/sessions/%s/messages/batch range=%d:%d payload=%s",
                        self.sid, self.next_index, batch_end, json.dumps(payload, ensure_ascii=False))
            try:
                client.post(f"/api/v1/sessions/{self.sid}/messages/batch", payload)
            except Exception as batch_error:
                if self.next_index:
                    raise
                logger.warning("OpenViking structured sync failed; falling back to text sync: %s", batch_error)
                break
            self.next_index = batch_end
        if self.batch_messages and self.next_index == len(self.batch_messages):
            return
        # Plain-text fallback: one user + one assistant message.
        user_message: Dict[str, Any] = {"role": "user", "parts": [{"type": "text", "text": self.user_content[:4000]}]}
        if self.user_peer_id:
            user_message["peer_id"] = self.user_peer_id
        assistant_message: Dict[str, Any] = {"role": "assistant", "parts": [{"type": "text", "text": _message_text(self.assistant_content)[:4000]}]}
        if self.assistant_peer_id:
            assistant_message["peer_id"] = self.assistant_peer_id
        client.post(f"/api/v1/sessions/{self.sid}/messages/batch",
                    {"messages": [user_message, assistant_message]})

    def run(self) -> Optional[_VikingClient]:
        try:
            client = self.client
            self.post(client)
            return client
        except Exception as e:
            logger.debug("OpenViking sync_turn failed, retrying: %s", e)
        retry_client = None
        try:
            # The HTTP wrapper opens a new connection for every request. Keep
            # the original identity even if /reload happens during the retry.
            retry_client = self.client
            self.post(retry_client)
            return retry_client
        except Exception as retry_error:
            if retry_client is None or self.next_index >= len(self.batch_messages):
                logger.warning("OpenViking sync_turn failed: %s", retry_error)
                return
            logger.warning("OpenViking structured sync retry failed; writing %d remaining messages individually: %s",
                           len(self.batch_messages) - self.next_index, retry_error)
        try:
            path = f"/api/v1/sessions/{self.sid}/messages"
            for payload in self.batch_messages[self.next_index:]:
                self._trace("POST %s message_index=%d payload=%s", path, self.next_index, json.dumps(payload, ensure_ascii=False))
                retry_client.post(path, payload)
                self.next_index += 1
            return retry_client
        except Exception as fallback_error:
            logger.warning("OpenViking sync_turn failed during individual-message fallback: %s", fallback_error)


class SessionWriterMixin:
    """Methods of ``OpenVikingMemoryProvider`` moved here unchanged; mixed into that class."""

    def _spawn_tracked(self, name: str, body: Callable[[], None], lock: threading.Lock, workers: Callable[[], Set[threading.Thread]],
                       *, after_discard: Callable[[], None] = None, skip_if: Callable[[], bool] = None) -> None:
        """Daemon thread registered in ``workers()`` (evaluated under ``lock``) for the
        duration of ``body`` so shutdown / drains can join it."""

        def _run() -> None:
            try:
                body()
            finally:
                with lock:
                    workers().discard(thread)
                    if after_discard is not None:
                        after_discard()

        thread = spawn_context_thread(_run, name=name)
        with lock:
            if skip_if is not None and skip_if():
                return
            workers().add(thread)
            try:
                thread.start()
            except Exception as e:
                workers().discard(thread)
                logger.debug("OpenViking %s worker failed to start: %s", name, e)

    def _join_all(self, alive: Callable[[], List[threading.Thread]], timeout: float, *, slice_cap: Optional[float] = None) -> bool:
        """Join threads from ``alive()`` until none remain or the shared budget runs out."""
        monotonic = self._deps.monotonic
        deadline = monotonic() + timeout
        while True:
            workers = alive()
            if not workers:
                return True
            if deadline - monotonic() <= 0:
                return False
            for t in workers:
                slice_left = deadline - monotonic()
                if slice_left <= 0:
                    break
                t.join(timeout=min(slice_left, slice_cap) if slice_cap else slice_left)

    def _drain_finalizers(self, timeout: float) -> bool:
        """Join in-flight async session finalizers (shutdown/tests wait deterministically)."""
        def alive():
            with self._deferred_commit_lock:
                return [t for t in self._deferred_commit_threads if t.is_alive()]
        # Floor each join so a thread whose join() returns instantly while still alive can't hot-spin.
        return self._join_all(alive, timeout, slice_cap=0.05)

    def _drain_writers(self, sid: str, timeout: float) -> bool:
        """Join every in-flight writer for sid; False (budget exhausted) tells callers to skip the commit."""
        if not sid:
            return True

        def alive():
            with self._inflight_lock:
                return [t for t in self._inflight_writers.get(sid, ()) if t.is_alive()]
        return self._join_all(alive, timeout)

    # -- session commit / pending-session recovery --------------------------

    def _has_committed_session(self, sid: str, *, scope: Optional[_CommitScope] = None) -> bool:
        scope = scope or self._capture_commit_scope()
        with self._committed_session_lock:
            return sid in scope.committed

    def _mark_session_committed(self, sid: str, committed: bool = True, *, scope: Optional[_CommitScope] = None) -> None:
        """Latch (or, with ``committed=False``, re-arm) the per-sid commit guard. Re-arming is
        for in-place compression: it keeps the same live id, which would otherwise reject every later commit."""
        scope = scope or self._capture_commit_scope()
        with self._committed_session_lock:
            (scope.committed.add if committed else scope.committed.discard)(sid)

    def _claim_deferred_sid(self, sid: str, *, release: bool = False, scope: Optional[_CommitScope] = None) -> bool:
        """One finalizer per sid and connection generation; none after shutdown."""
        scope = scope or self._capture_commit_scope()
        with self._deferred_commit_lock:
            if release:
                scope.finalizing.discard(sid)
                return True
            if self._shutting_down or sid in scope.finalizing:
                return False
            scope.finalizing.add(sid)
            return True

    def _maybe_commit_live_session(self, sid: str, turn_count: int, threshold: int, client: _VikingClient,
                                   scope: _CommitScope) -> None:
        """Check after a successful upload; metadata failures must not replay the turn."""
        if self._shutting_down:
            return
        try:
            session = self._unwrap_result(client.get(f"/api/v1/sessions/{sid}"))
            if int(session.get("pending_tokens") or 0) >= threshold:
                self._finalize_session_async(sid, turn_count, context="after live token threshold", client=client, scope=scope)
        except Exception as e:
            logger.warning("OpenViking live commit check failed for %s: %s", sid, e)

    def _session_needs_commit(self, sid: str, turn_count: int, *, scope: Optional[_CommitScope] = None,
                              request_timeout: Optional[float] = None) -> bool:
        # The committed-guard wins over turn_count: a racing sync_turn can re-increment
        # _turn_count after a commit+reset.
        scope = scope or self._capture_commit_scope()
        if self._has_committed_session(sid, scope=scope):
            return False
        if turn_count > 0:
            return True
        try:
            timeout = {} if request_timeout is None else {"timeout": request_timeout}
            session = self._unwrap_result(scope.client.get(f"/api/v1/sessions/{sid}", **timeout))
            return isinstance(session, dict) and int(session.get("pending_tokens") or 0) > 0
        except Exception:
            return False

    def _commit_session(self, sid: str, turn_count: int, *, context: str, clear_missing: bool = False,
                        client: Optional[_VikingClient] = None, scope: Optional[_CommitScope] = None,
                        pending_path: Optional[Path] = None, request_timeout: Optional[float] = None) -> bool:
        scope = scope or self._capture_commit_scope()
        try:
            timeout = {} if request_timeout is None else {"timeout": request_timeout}
            (client or scope.client).post(f"/api/v1/sessions/{sid}/commit", {"keep_recent_count": 0}, **timeout)
            self._mark_session_committed(sid, scope=scope)
            self._clear_pending_session(sid, scope=scope, pending_path=pending_path)
            with self._session_state_lock:
                if self._session_id == sid and self._commit_scope is scope and self._client is scope.client:
                    self._turn_count = 0
            logger.info("OpenViking session %s committed %s (%d turns)", sid, context, turn_count)
            return True
        except Exception as e:
            if clear_missing and _status_code_from_error(e) == 404:
                self._clear_pending_session(sid, scope=scope, pending_path=pending_path)
                logger.debug("OpenViking pending session %s no longer exists; dropped marker", sid)
            else:
                logger.warning("OpenViking session commit failed for %s: %s", sid, e)
            return False

    def _finalize_session_async(self, sid: str, turn_count: int, *, context: str,
                                client: Optional[_VikingClient] = None, scope: Optional[_CommitScope] = None) -> None:
        """Drain the old session's writers and commit it on a daemon thread, so the
        multi-second drain + pending-token GET + commit POST never runs on the
        caller's command thread (on_session_switch). Deduped per sid and connection; no-op after shutdown."""
        scope = scope or self._capture_commit_scope()
        if not sid or not self._claim_deferred_sid(sid, scope=scope):
            return

        def _finalize() -> None:
            try:
                if self._shutting_down:
                    return
                # Drain before taking the write lock: queued uploads need that
                # lock to finish. A later writer re-arms the guard after this commit.
                if not self._drain_writers(sid, timeout=_DEFERRED_COMMIT_TIMEOUT):
                    logger.warning("OpenViking writer for %s still alive after drain — leaving session uncommitted", sid)
                    return
                with self._writer_commit_lock:
                    if not self._shutting_down and self._session_needs_commit(sid, turn_count, scope=scope):
                        self._commit_session(sid, turn_count, context=context, client=client, scope=scope)
            finally:
                self._claim_deferred_sid(sid, release=True, scope=scope)

        self._spawn_tracked(f"openviking-finalize-{sid}", _finalize, self._deferred_commit_lock, lambda: self._deferred_commit_threads)

    def _end_session(self, drain_timeout: float, *, deadline: Optional[float] = None) -> None:
        """Commit the current session; ``deadline`` (monotonic) caps every request at exit.
        Drops the atexit entry when nothing uncommitted is left behind."""
        if not self._writes_enabled:
            return
        if not self._ensure_client():
            return
        with self._session_state_lock:
            scope = self._capture_commit_scope()
            sid = self._session_id
        if not self._drain_writers(sid, timeout=drain_timeout):
            logger.warning("OpenViking writer for %s still alive after drain — skipping commit", sid)
            return

        def remaining() -> Optional[float]:
            return None if deadline is None else deadline - self._deps.monotonic()

        def expired() -> bool:
            return deadline is not None and remaining() <= 0

        # At exit the lock wait counts against the budget too: an older sid's finalizer or
        # recovery thread may hold it for a whole commit POST at the 30 s client default.
        if deadline is None:
            self._writer_commit_lock.acquire()
        elif not self._writer_commit_lock.acquire(timeout=max(0.0, remaining())):
            logger.warning("OpenViking exit budget used up waiting to commit; leaving session %s pending", sid)
            return
        try:
            with self._session_state_lock:
                turn_count = self._turn_count if self._session_id == sid and self._commit_scope is scope else 0
            if expired():
                return
            if self._session_needs_commit(sid, turn_count, scope=scope, request_timeout=remaining()):
                if expired() or not self._commit_session(
                        sid, turn_count, context="on session end", scope=scope, request_timeout=remaining()):
                    return
            # Under the write lock: a later writer re-registers before it marks its sid pending.
            if not self._has_uncommitted_data(scope):
                _deregister_for_exit(self)
        finally:
            self._writer_commit_lock.release()

    def _has_uncommitted_data(self, scope: _CommitScope) -> bool:
        """A writer still running, or a turn or pending marker of this connection generation not committed."""
        with self._inflight_lock:
            if any(t.is_alive() for group in self._inflight_writers.values() for t in group):
                return True
        with self._session_state_lock:
            return bool(scope.pending) or self._turn_count > 0
