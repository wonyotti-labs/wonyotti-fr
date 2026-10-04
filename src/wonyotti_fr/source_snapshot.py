from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
from functools import cache
from pathlib import Path


@cache
def _clonefile():
    if sys.platform != 'darwin':
        return None
    function = getattr(ctypes.CDLL(None, use_errno=True), 'clonefile', None)
    if function is not None:
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        function.restype = ctypes.c_int
    return function


def copy_snapshot(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError('코드 스냅샷 원본은 일반 파일이어야 합니다.')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(errno.EEXIST, '기존 코드 스냅샷 덮어쓰기 거부', str(destination))
    clone = _clonefile()
    if clone is not None:
        # 저장 블록만 공유하며 이후 원본 수정은 기존 스냅샷에 전파하지 않는다.
        ctypes.set_errno(0)
        if clone(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return
        error = ctypes.get_errno()
        if error not in {errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}:
            raise OSError(error, '코드 스냅샷 복제 실패', str(destination))
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=1024*1024)
