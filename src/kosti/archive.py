"""Bounded ZIP extraction; archives are always treated as untrusted input."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


@dataclass(frozen=True)
class ArchiveLimits:
    max_files: int = 2000
    max_file_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024

    def __post_init__(self):
        for name in ("max_files", "max_file_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class UnsafeArchiveError(ValueError):
    pass


def _is_system_junk(name):
    """File-manager metadata that is never an input image."""
    part = PurePosixPath(name)
    return ("__MACOSX" in part.parts or part.name == ".DS_Store"
            or part.name == "Thumbs.db" or part.name.startswith("._"))


def safe_extract_zip(archive, destination, limits=None):
    limits = limits or ArchiveLimits()
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted, created, total = [], [], 0
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > limits.max_files:
                raise UnsafeArchiveError("Archive entry limit exceeded")
            targets = set()
            file_targets = set()
            for member in members:
                name = member.filename
                part = PurePosixPath(name)
                if (not name or "\\" in name or "\x00" in name or part.is_absolute()
                        or ".." in part.parts or re.match(r"^[A-Za-z]:", name)):
                    raise UnsafeArchiveError("Unsafe archive path")
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise UnsafeArchiveError("Archive contains non-regular entry")
                target = root.joinpath(*part.parts)
                if not target.resolve().is_relative_to(root):
                    raise UnsafeArchiveError("Archive path escapes destination")
                if _is_system_junk(name):
                    continue
                if target in targets or target.exists():
                    raise UnsafeArchiveError("Archive contains duplicate or existing paths")
                targets.add(target)
                if not member.is_dir():
                    file_targets.add(target)
                if member.flag_bits & 1:
                    raise UnsafeArchiveError("Encrypted archives are unsupported")
                if member.file_size > limits.max_file_bytes:
                    raise UnsafeArchiveError("Archive file size limit exceeded")
                total += member.file_size
                if total > limits.max_total_bytes:
                    raise UnsafeArchiveError("Archive total size limit exceeded")
            # Reject file/directory collisions before writing, regardless of
            # member order (for example a file 'a' alongside 'a/image.dcm').
            if any(parent in file_targets for target in targets
                   for parent in target.parents if parent != root):
                raise UnsafeArchiveError("Archive contains conflicting file and directory paths")
            total = 0
            for member in members:
                if _is_system_junk(member.filename):
                    continue
                target = root.joinpath(*PurePosixPath(member.filename).parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                count = 0
                with bundle.open(member) as source, target.open("xb") as output:
                    created.append(target)
                    while chunk := source.read(1024 * 1024):
                        count += len(chunk)
                        total += len(chunk)
                        if count > limits.max_file_bytes or total > limits.max_total_bytes:
                            raise UnsafeArchiveError("Archive size limit exceeded while extracting")
                        output.write(chunk)
                extracted.append(target)
        return extracted
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
