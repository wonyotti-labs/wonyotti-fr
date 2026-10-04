import ctypes
import errno
from pathlib import Path

import pytest

from wonyotti_fr.common import new_run, sha256
from wonyotti_fr.source_snapshot import copy_snapshot


def test_native_or_portable_snapshot_remains_independent_after_both_files_change(tmp_path):
    source, destination = tmp_path/'source.py', tmp_path/'snapshot.py'
    source.write_bytes(b'original source\n'*10000)
    expected = sha256(source)
    copy_snapshot(source, destination)
    assert sha256(destination) == expected and source.stat().st_ino != destination.stat().st_ino
    source.write_bytes(b'changed source')
    assert sha256(destination) == expected
    destination.write_bytes(b'changed snapshot')
    assert source.read_bytes() == b'changed source'


@pytest.mark.parametrize('code', [None, errno.EXDEV, errno.ENOTSUP, errno.ENOSYS])
def test_unavailable_clone_uses_independent_copy_without_overwriting(tmp_path, monkeypatch, code):
    def clone(_source, _destination, _flags):
        ctypes.set_errno(code)
        return -1
    monkeypatch.setattr('wonyotti_fr.source_snapshot._clonefile', lambda: None if code is None else clone)
    source, destination = tmp_path/'source.py', tmp_path/'snapshot.py'
    source.write_bytes(b'preserved')
    copy_snapshot(source, destination)
    assert destination.read_bytes() == b'preserved'
    source.write_bytes(b'changed')
    assert destination.read_bytes() == b'preserved'
    with pytest.raises(FileExistsError):
        copy_snapshot(source, destination)
    assert destination.read_bytes() == b'preserved'


@pytest.mark.parametrize('code', [errno.ENOSPC, errno.EACCES])
def test_clone_storage_or_permission_error_is_not_masked_by_another_copy(tmp_path, monkeypatch, code):
    def clone(_source, _destination, _flags):
        ctypes.set_errno(code)
        return -1
    monkeypatch.setattr('wonyotti_fr.source_snapshot._clonefile', lambda: clone)
    source, destination = tmp_path/'source.py', tmp_path/'snapshot.py'
    source.write_bytes(b'preserved')
    with pytest.raises(OSError) as error:
        copy_snapshot(source, destination)
    assert error.value.errno == code and not destination.exists()
    assert source.read_bytes() == b'preserved'


def test_source_and_destination_symlinks_cannot_replace_snapshot_contents(tmp_path):
    source, link = tmp_path/'source.py', tmp_path/'link.py'
    source.write_bytes(b'preserved')
    link.symlink_to(source)
    with pytest.raises(ValueError):
        copy_snapshot(link, tmp_path/'new.py')
    with pytest.raises(FileExistsError):
        copy_snapshot(source, link)
    assert source.read_bytes() == b'preserved'


def test_run_manifest_records_exact_complete_source_and_lock_bytes(tmp_path):
    import json

    run = new_run(tmp_path, 'snapshot-test', {'purpose': 'synthetic'})
    manifest = json.loads((run/'manifest.json').read_text())
    source = Path('src/wonyotti_fr')
    expected = {p.name: sha256(p) for p in source.glob('*.py')}
    assert manifest['source_sha256'] == expected
    assert {p.name: sha256(p) for p in (run/'code_snapshot').glob('*.py')} == expected
    assert sha256(run/'code_snapshot'/'uv.lock') == sha256(Path('uv.lock')) == manifest['uv_lock_sha256']
