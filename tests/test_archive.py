import stat
import zipfile
import pytest
from kosti.archive import ArchiveLimits, UnsafeArchiveError, safe_extract_zip


def archive(tmp_path, name, content=b"data"):
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(name, content)
    return path


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "C:/windows", "a\\..\\escape"])
def test_rejects_zip_slip(tmp_path, name):
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive(tmp_path, name), tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_rejects_symlinks(tmp_path):
    entry = zipfile.ZipInfo("link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive(tmp_path, entry, b"../escape"), tmp_path / "out")


def test_size_count_limits_and_valid_extract(tmp_path):
    source = archive(tmp_path, "nested/image", b"12345")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(source, tmp_path / "bad", ArchiveLimits(max_file_bytes=4))
    result = safe_extract_zip(source, tmp_path / "good")
    assert result[0].read_bytes() == b"12345"
    with zipfile.ZipFile(source, "a") as bundle:
        bundle.writestr("second", b"x")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(source, tmp_path / "count", ArchiveLimits(max_files=1))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_limit_validation(value):
    with pytest.raises(ValueError):
        ArchiveLimits(max_total_bytes=value)


def test_total_uncompressed_size_limit(tmp_path):
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("one", b"x" * 10000)
        bundle.writestr("two", b"x" * 10000)
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(path, tmp_path / "out", ArchiveLimits(max_total_bytes=15000))
    assert not list((tmp_path / "out").rglob("*"))


def test_duplicate_archive_names_rejected(tmp_path):
    path = archive(tmp_path, "image", b"one")
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(path, "a") as bundle:
            bundle.writestr("image", b"two")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(path, tmp_path / "out")


def test_failed_crc_removes_partial_output(tmp_path):
    path = archive(tmp_path, "image", b"unique contents")
    raw = path.read_bytes().replace(b"unique contents", b"broken contents", 1)
    path.write_bytes(raw)
    with pytest.raises(zipfile.BadZipFile):
        safe_extract_zip(path, tmp_path / "out")
    assert not (tmp_path / "out" / "image").exists()


@pytest.mark.parametrize("names", [("a", "a/image.dcm"), ("a/image.dcm", "a")])
def test_rejects_file_directory_conflicts_before_extracting(tmp_path, names):
    source = tmp_path / "conflict.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        for name in names:
            bundle.writestr(name, b"data")
    with pytest.raises(UnsafeArchiveError, match="conflicting"):
        safe_extract_zip(source, tmp_path / "out")
    assert not list((tmp_path / "out").rglob("*"))
