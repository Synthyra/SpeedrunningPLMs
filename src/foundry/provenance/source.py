"""What code a run executed and which files it left: a source revision over a named set of paths, and `SHA256SUMS`
over a folder in the format `sha256sum` writes and checks.

The revision's digest over the named paths stands on its own, so it works where git is absent or refuses the
directory (a shipped copy without `.git`, a container whose user does not own the checkout); git's commit and dirty
state join it when git answers, and git's own message is kept when it does not.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from foundry.digests import sha256_file


__all__ = [
    "SUMS_NAME", "GitState", "SourceRevision", "directory_digests", "git_state", "paths_sha256", "read_sha256sums",
    "source_revision", "write_sha256sums",
]

SUMS_NAME = "SHA256SUMS"
SKIPPED_PARTS = ("__pycache__", ".pytest_cache")  # bytecode and caches a run writes beside the source
GIT_TIMEOUT_SECONDS = 60.0
SUMS_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


class GitState(TypedDict):
    """git's view of the paths: the commit and the porcelain status lines, or git's own answer when it gave none."""

    commit: str | None
    dirty: bool | None
    status: list[str]
    answer: str | None


class SourceRevision(TypedDict):
    """A digest over the named paths' files, how many files it covers, and git's view of them."""

    paths: list[str]
    files: int
    paths_sha256: str
    git: GitState


def _files_under(root: Path, paths: Sequence[str]) -> list[Path]:
    """Every file at or under the named paths, relative to `root`, bytecode and caches left out; a missing path raises."""
    found: set[Path] = set()
    for name in paths:
        target = root / name
        if not target.exists():
            raise FileNotFoundError(f"{target} is not there; the source revision names it")
        candidates = [target] if target.is_file() else [path for path in target.rglob("*") if path.is_file()]
        for path in candidates:
            relative = path.relative_to(root)
            if not any(part in SKIPPED_PARTS for part in relative.parts) and path.suffix != ".pyc":
                found.add(relative)
    return sorted(found, key=lambda path: path.as_posix())


def paths_sha256(root: Path, paths: Sequence[str]) -> tuple[str, int]:
    """One digest over the files under the named paths (each file's SHA-256 and its relative path, in path order) and
    the file count."""
    files = _files_under(root, paths)
    digest = hashlib.sha256()
    for relative in files:
        digest.update(f"{sha256_file(root / relative)}  {relative.as_posix()}\n".encode())
    return digest.hexdigest(), len(files)


def git_state(root: Path, paths: Sequence[str], git: Sequence[str] = ("git",)) -> GitState:
    """The commit at `root` and whether the named paths differ from it; git's message when it is absent or refuses."""
    if shutil.which(git[0]) is None:
        return GitState(commit=None, dirty=None, status=[], answer=f"{git[0]} is not on PATH")
    head = subprocess.run([*git, "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS)
    if head.returncode != 0:
        return GitState(commit=None, dirty=None, status=[], answer=f"exit code {head.returncode}: {head.stderr.strip()[:500]}")
    status = subprocess.run([*git, "-C", str(root), "status", "--porcelain", "--", *paths], capture_output=True, text=True,
                            timeout=GIT_TIMEOUT_SECONDS)
    if status.returncode != 0:
        return GitState(commit=head.stdout.strip(), dirty=None, status=[],
                        answer=f"status exit code {status.returncode}: {status.stderr.strip()[:500]}")
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    return GitState(commit=head.stdout.strip(), dirty=bool(lines), status=lines, answer=None)


def source_revision(root: Path, paths: Sequence[str], git: Sequence[str] = ("git",)) -> SourceRevision:
    """The digest over the named paths under `root`, with git's commit and dirty state of them."""
    digest, count = paths_sha256(root, paths)
    return SourceRevision(paths=list(paths), files=count, paths_sha256=digest, git=git_state(root, paths, git))


def directory_digests(directory: Path, exclude: Sequence[str] = (SUMS_NAME,)) -> dict[str, str]:
    """Every file under `directory` by its relative POSIX path, with its SHA-256; top-level names in `exclude` skipped."""
    digests: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and relative not in exclude:
            digests[relative] = sha256_file(path)
    return dict(sorted(digests.items()))


def write_sha256sums(directory: Path) -> Path:
    """`SHA256SUMS` over every other file in `directory`, as `find . -type f | sort | xargs sha256sum` writes it."""
    lines = [f"{digest}  ./{relative}\n" for relative, digest in directory_digests(directory).items()]
    path = directory / SUMS_NAME
    path.write_bytes("".join(lines).encode("utf-8"))
    return path


def read_sha256sums(path: Path) -> dict[str, str]:
    """A `sha256sum` listing by relative path (a leading `./` removed); a line it cannot read raises, an escaped name
    (`sha256sum` starts the line with a backslash) included."""
    listed: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = SUMS_LINE.match(line)
        if match is None:
            raise ValueError(f"{path.name} line {number} is not `<sha256>  <path>`: {line[:200]!r}")
        name = match.group(2).removeprefix("./")
        if name in listed:
            raise ValueError(f"{path.name} lists {name} twice")
        listed[name] = match.group(1)
    return listed
