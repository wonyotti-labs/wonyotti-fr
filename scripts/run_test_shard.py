"""전체 수집 결과를 파일 단위로 나눠 CI 검사 누락을 방지한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest


def partition_nodes(nodes: list[tuple[str, str]], count: int) -> list[list[str]]:
    if type(count) is not int or count < 1 or not nodes:
        raise ValueError('검사 분할 수·수집 결과 오류')
    if len({node for node, _ in nodes}) != len(nodes):
        raise ValueError('중복 검사 식별자')
    groups: list[list[str]] = [[] for _ in range(count)]
    for node, filename in nodes:
        path = PurePosixPath(filename)
        if not node or path.is_absolute() or '..' in path.parts or not filename:
            raise ValueError('검사 식별자·상대 경로 오류')
        # 같은 파일의 모듈 공유 자료를 중복 생성하지 않도록 파일 전체를 배정한다.
        owner = int.from_bytes(hashlib.sha256(filename.encode('utf-8')).digest(), 'big') % count
        groups[owner].append(node)
    return groups


class ShardSelection:
    def __init__(self, index: int, count: int, manifest: Path):
        if type(index) is not int or type(count) is not int or not 0 <= index < count:
            raise ValueError('검사 분할 번호 오류')
        self.index, self.count, self.manifest = index, count, manifest

    def pytest_collection_modifyitems(self, config, items):
        nodes = [(item.nodeid, item.path.relative_to(config.rootpath).as_posix()) for item in items]
        groups = partition_nodes(nodes, self.count)
        chosen = set(groups[self.index])
        if not chosen:
            raise pytest.UsageError('선택된 검사가 없는 분할')
        selected = [item for item in items if item.nodeid in chosen]
        deselected = [item for item in items if item.nodeid not in chosen]
        report = {'index': self.index, 'count': self.count, 'collected': len(nodes),
            'selected': len(selected), 'nodes': nodes, 'selected_nodeids': groups[self.index]}
        with self.manifest.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        reporter = config.pluginmanager.get_plugin('terminalreporter')
        if reporter:
            reporter.write_line(f'전체 {len(nodes)}개 중 분할 {self.index + 1}/{self.count}: {len(selected)}개')
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def main() -> int:
    parser = argparse.ArgumentParser(description='파일 단위의 전체 pytest 분할 실행')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    plugin = ShardSelection(args.index, args.count, args.manifest)
    options = ['--strict-config', '--strict-markers']
    if args.collect_only:
        options.append('--collect-only')
    return int(pytest.main(options, plugins=[plugin]))


if __name__ == '__main__':
    raise SystemExit(main())
