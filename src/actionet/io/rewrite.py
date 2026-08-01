"""Shared crash-safe H5AD rewrite transaction."""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class FileFingerprint:
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: str) -> "FileFingerprint":
        stat = os.stat(path)
        return cls(
            inode=int(stat.st_ino),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )


@dataclass(frozen=True)
class CommitStats:
    """Timing breakdown for durable publication of one rewrite."""

    temp_fsync_seconds: float
    fingerprint_seconds: float
    source_close_seconds: float
    replace_seconds: float
    parent_fsync_seconds: float
    total_seconds: float


class RewriteTransaction:
    """Own a unique same-directory temp file and one atomic commit."""

    def __init__(self, source_path: str, destination_path: str):
        self.source_path = os.path.realpath(os.fspath(source_path))
        self.destination_path = os.path.realpath(os.fspath(destination_path))
        self.in_place = self.source_path == self.destination_path
        self.source_fingerprint = FileFingerprint.capture(self.source_path)

        destination_dir = os.path.dirname(self.destination_path) or "."
        fd, self.temp_path = tempfile.mkstemp(
            prefix=".actionet-rewrite-",
            suffix=".h5ad",
            dir=destination_dir,
        )
        os.close(fd)
        self._committed = False

    def __enter__(self) -> "RewriteTransaction":
        return self

    def _fsync_temp(self) -> None:
        descriptor = os.open(self.temp_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _fsync_parent_best_effort(self) -> None:
        directory = os.path.dirname(self.destination_path) or "."
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(descriptor)
            except OSError:
                pass
        finally:
            os.close(descriptor)

    def _copy_source_metadata_best_effort(self) -> None:
        """Preserve file mode (and, when possible, ownership) across replace.

        For in-place rewrites the intent is that the published H5AD retains
        the original file's access permissions. For cross-file rewrites we
        still prefer to mirror the source rather than the ``0600`` temp mode
        that ``mkstemp`` produced. Failures here are non-fatal: a permission
        or filesystem restriction must not abort a successful rewrite.
        """
        try:
            source_stat = os.stat(self.source_path)
        except OSError:
            return
        try:
            os.chmod(self.temp_path, stat.S_IMODE(source_stat.st_mode))
        except OSError:
            pass
        # ``os.chown`` typically requires elevated privileges when changing
        # the owning uid, so we only attempt it when the uid or gid differs
        # and swallow the resulting ``PermissionError`` if we lack them.
        try:
            temp_stat = os.stat(self.temp_path)
            if (
                temp_stat.st_uid != source_stat.st_uid
                or temp_stat.st_gid != source_stat.st_gid
            ):
                os.chown(
                    self.temp_path, source_stat.st_uid, source_stat.st_gid
                )
        except (OSError, AttributeError):
            pass

    def assert_source_unchanged(self) -> None:
        current = FileFingerprint.capture(self.source_path)
        if current != self.source_fingerprint:
            raise RuntimeError(
                "Backing file changed during rewrite; refusing atomic replacement"
            )

    def commit(
        self,
        *,
        close_source=None,
        restore_source=None,
    ) -> CommitStats:
        """Durably publish the temp file.

        For in-place operations, the original handle is closed only after the
        completed temp file is fsynced and its source fingerprint is verified.
        """
        commit_started = perf_counter()
        started = perf_counter()
        self._fsync_temp()
        temp_fsync_seconds = perf_counter() - started

        fingerprint_seconds = 0.0
        source_close_seconds = 0.0
        if self.in_place:
            started = perf_counter()
            self.assert_source_unchanged()
            fingerprint_seconds = perf_counter() - started
            if close_source is not None:
                started = perf_counter()
                try:
                    close_source()
                except Exception as close_error:
                    if restore_source is not None:
                        try:
                            restore_source()
                        except Exception as restore_error:
                            close_error.add_note(
                                "Additionally failed to restore the source "
                                f"handle: {restore_error}"
                            )
                    raise
                source_close_seconds = perf_counter() - started

        # ``mkstemp`` creates the file with mode 0600, so a straight
        # ``os.replace`` would strip any group/other read bits from the
        # published H5AD. Mirror the source's mode (and, when we have the
        # privilege, its ownership) before publication so shared HPC
        # directories keep their existing access permissions.
        self._copy_source_metadata_best_effort()

        started = perf_counter()
        try:
            os.replace(self.temp_path, self.destination_path)
        except Exception as replace_error:
            if self.in_place and restore_source is not None:
                try:
                    restore_source()
                except Exception as restore_error:
                    replace_error.add_note(
                        "Additionally failed to restore the source handle: "
                        f"{restore_error}"
                    )
            raise
        replace_seconds = perf_counter() - started
        # The rename is durable at this point; setting the flag before the
        # parent ``fsync`` prevents ``cleanup`` from unlinking a temp path
        # that a rare filesystem might still surface if the process is
        # interrupted between ``os.replace`` and directory ``fsync``.
        self._committed = True

        started = perf_counter()
        self._fsync_parent_best_effort()
        parent_fsync_seconds = perf_counter() - started
        return CommitStats(
            temp_fsync_seconds=temp_fsync_seconds,
            fingerprint_seconds=fingerprint_seconds,
            source_close_seconds=source_close_seconds,
            replace_seconds=replace_seconds,
            parent_fsync_seconds=parent_fsync_seconds,
            total_seconds=perf_counter() - commit_started,
        )

    def cleanup(self) -> None:
        if not self._committed and os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.cleanup()
        return False
