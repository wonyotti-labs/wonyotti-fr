import json
import stat

import pytest

from wonyotti_fr.common import new_run, sha256
from wonyotti_fr.research_paths import (
    REPRODUCTION_DIRECTORY,
    REPRODUCTION_MARKER,
    reproduction_root,
)


def test_many_successive_reproductions_keep_one_depth_under_original_run(tmp_path):
    run = tmp_path/'original'
    run.mkdir()
    root = reproduction_root(run)
    assert root.parent == run and reproduction_root(run) == root
    for number in range(100):
        child = root/f'{number:03}-long-research-diagnosis-with-separate-identity'
        child.mkdir()
        assert reproduction_root(child) == root
        assert len(child.relative_to(run).parts) == 2
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root/REPRODUCTION_MARKER).stat().st_mode) == 0o600
    assert len(list(root.iterdir())) == 101


def test_an_unmarked_parent_name_does_not_move_reproductions_outside_the_run(tmp_path):
    run = tmp_path/REPRODUCTION_DIRECTORY/'original'
    run.mkdir(parents=True)
    root = reproduction_root(run)
    assert root == run/REPRODUCTION_DIRECTORY
    child = root/'child'
    child.mkdir()
    assert reproduction_root(child) == root


@pytest.mark.parametrize('damage', ['directory_link', 'marker_link', 'missing_marker', 'wrong_owner', 'wrong_format', 'parent_marker'])
def test_foreign_or_linked_reproduction_storage_is_rejected_without_overwriting(tmp_path, damage):
    run = tmp_path/'original'
    run.mkdir()
    root = reproduction_root(run)
    marker = root/REPRODUCTION_MARKER
    external = tmp_path/'external'
    external.mkdir()
    sentinel = external/'sentinel.json'
    sentinel.write_text('protected\n')
    original = sentinel.read_bytes()
    target = run
    if damage == 'directory_link':
        marker.unlink()
        root.rmdir()
        root.symlink_to(external, target_is_directory=True)
    elif damage == 'marker_link':
        marker.unlink()
        marker.symlink_to(sentinel)
    elif damage == 'missing_marker':
        marker.unlink()
    else:
        data = json.loads(marker.read_text())
        data['format' if damage == 'wrong_format' else 'owner'] = 'unexpected'
        marker.write_text(json.dumps(data))
        if damage == 'parent_marker':
            target = root/'child'
            target.mkdir()
    with pytest.raises(ValueError):
        reproduction_root(target)
    assert sentinel.read_bytes() == original


def test_actual_runs_keep_source_snapshots_and_explicit_links_at_bounded_depth(tmp_path):
    outer = new_run(tmp_path, 'outer', {'purpose': 'synthetic'})
    root = reproduction_root(outer)
    first = new_run(root, 'first', {'reference': str(outer), 'manifest_sha256': sha256(outer/'manifest.json')})
    second = new_run(reproduction_root(first), 'second', {'reference': str(first), 'manifest_sha256': sha256(first/'manifest.json')})
    assert first.parent == second.parent == root
    assert second.is_relative_to(outer)
    for run, previous in [(first, outer), (second, first)]:
        manifest = json.loads((run/'manifest.json').read_text())
        assert manifest['settings']['manifest_sha256'] == sha256(previous/'manifest.json')
        for name, digest in manifest['source_sha256'].items():
            assert sha256(run/'code_snapshot'/name) == digest
