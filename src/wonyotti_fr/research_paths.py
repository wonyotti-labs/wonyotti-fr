from __future__ import annotations

import json
import os
from pathlib import Path

REPRODUCTION_DIRECTORY = 'reference-runs'
REPRODUCTION_MARKER = '.owner.json'


def checked_reproduction_root(root):
    marker = root/REPRODUCTION_MARKER
    if root.is_symlink() or not root.is_dir() or marker.is_symlink() or not marker.is_file():
        raise ValueError('재현 저장소의 디렉터리·소유 기록 오류')
    expected = {'format': 'flat_reproductions_v1', 'owner': str(root.parent.resolve())}
    if json.loads(marker.read_text(encoding='utf-8')) != expected:
        raise ValueError('재현 저장소의 소유 실행 불일치')
    return root


def reproduction_root(run: Path) -> Path:
    run = Path(run)
    if run.is_symlink() or not run.is_dir():
        raise ValueError('재현할 실행 디렉터리 오류')
    parent = run.parent
    marker = parent/REPRODUCTION_MARKER
    if parent.name == REPRODUCTION_DIRECTORY and (marker.exists() or marker.is_symlink()):
        # 연속 재현을 최초 실행 아래의 형제 폴더에 저장해 경로 깊이를 제한한다.
        return checked_reproduction_root(parent)
    root = run/REPRODUCTION_DIRECTORY
    if root.exists() or root.is_symlink():
        return checked_reproduction_root(root)
    root.mkdir(mode=0o700)
    descriptor = os.open(root/REPRODUCTION_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        json.dump({'format': 'flat_reproductions_v1', 'owner': str(run.resolve())}, handle, ensure_ascii=False)
        handle.write('\n')
    return root
