"""Git에 포함된 비공개 데이터와 비정상 파일을 검사한다."""
import subprocess
import sys
from pathlib import PurePosixPath


def inspect_index():
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    violations = []
    forbidden_dirs = {"data", "artifacts", "runs", ".venv", ".idea", "aoa_public_2021-12-31_with_letter"}
    for name in filter(None, names):
        path = PurePosixPath(name)
        synthetic = name.startswith(("tests/fixtures/", "examples/synthetic/"))
        blocked = bool(set(path.parts) & forbidden_dirs) or name.startswith("research/private/")
        blocked |= path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example")
        blocked |= any(name.endswith(suffix) for suffix in [".parquet", ".arrow", ".zip", ".csv.gz", ".pkl", ".pickle", ".joblib", ".db"])
        blocked |= path.suffix in {".csv", ".tsv"} and not synthetic
        if blocked:
            violations.append(name)
    if violations:
        print("공개 정책에 맞지 않는 추적 파일:\n" + "\n".join(violations), file=sys.stderr)
        return 1
    print("추적 파일에 원본·가공 데이터·실험 출력·환경 파일이 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(inspect_index())
