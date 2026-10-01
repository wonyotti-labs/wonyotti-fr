# Wonyotti, For Real?

공개 거래 기록에서 매매 행동을 연구하고, 추정한 전략을 과거 시장에서 검증하는 Python 프로젝트다. 현재는 **오프라인 연구용**이다. 실제 주문·API 키·자금 이체 기능은 없다.

기록을 설명하는 모델과 수익성 있는 전략은 다르다. 첫 보유 방향 모사 후보는 다년 평가에서 수익성 있는 전략으로 채택하지 않았다. 실패 결과와 실행 조건도 로컬 연구 기록에 보존한다. 봇 성과로 원자료의 진위, 개인의 전체 재산, 실제 전략을 입증한다고 주장하지 않는다.

- [전체 로드맵](ROADMAP.md)
- [진행 상태와 남은 문제](docs/STATUS.md)
- [연구 방법과 단위](docs/METHODOLOGY.md)
- [데이터 관리 원칙](DATA_POLICY.md)
- [보안 정책](SECURITY.md)

## 설치와 빠른 실행

Python 3.12와 [uv](https://docs.astral.sh/uv/getting-started/installation/)를 사용한다. 의존성은 `uv.lock`으로 고정한다.

```sh
uv sync --locked
uv run wonyotti demo
uv run pytest
uv run ruff check .
```

`demo`는 직접 생성한 합성 시세만 사용하며 네트워크나 원본 거래 내역이 필요하지 않다. 완료 후 출력되는 `REPORT.md` 경로에서 결과를 확인한다. 데모 수익은 실제 전략 성과가 아니다.

## 원본을 가진 사용자의 연구 흐름

```text
부모 폴더/
├── aoa_public_2021-12-31_with_letter/  # 로컬 입력
└── wonyotti-fr/                      # 이 저장소
```

```sh
# 전체 입력 감사와 XBTUSD 포지션 복원을 실행한다.
uv run wonyotti audit --source ../aoa_public_2021-12-31_with_letter --timezone UTC

# 공개 HTTPS 자료를 받아 공식 체크섬을 검증한다.
uv run wonyotti market --start 2019-09 --end 2025-12

# 월별 자료의 내부 결측을 일별 자료로 보완한다. 출력은 새 폴더여야 한다.
uv run wonyotti market-repair --market data/market --output data/market-complete
```

아래 `artifacts/감사실행ID`, `artifacts/연구실행ID`를 앞선 명령이 출력한 폴더로 바꾼다. 시장 자료의 UTC와 달리 원본 기록의 UTC는 독립적으로 확인하지 못한 가정이다.

```sh
uv run wonyotti study --audit-run artifacts/감사실행ID
uv run wonyotti research --audit-run artifacts/감사실행ID --market data/market-complete
uv run wonyotti robustness --research-run artifacts/연구실행ID
uv run wonyotti replay --research-run artifacts/연구실행ID --symbol BTCUSDT --start 2022-01-01 --end 2026-01-01
```

날짜 구간은 시작 포함·종료 미포함이다. 기본 연구는 2020년 학습, 2021년 BTC 설정 선택, 2022~2025년 BTC·ETH·SOL 평가다. 평가 결과를 관찰한 뒤 수정한 규칙은 같은 기간을 다시 미사용 최종 평가라고 부를 수 없다.

공식 월별 시세는 BTC·ETH의 2020년 이후와 SOL의 2020년 9월 이후를 확보했다. 수집 범위의 실제 시작·누락·일별 보완은 시장 매니페스트에 남긴다. 시장 명령은 여러 봉 간격을 지원하지만 현재 전략·시뮬레이터는 15분봉만 지원한다.

## 생성되는 자료

| 명령 | 주요 결과 |
| --- | --- |
| `audit` | 원본 해시·품질 감사, XBTUSD 행동·포지션·실현손익, 지갑 대조 |
| `market`, `market-repair` | 공식 아카이브 캐시, 검증한 시세·펀딩, 누락·보완 출처 |
| `study` | 보유 시간·추가 진입 상태·손실 집중도·진입 직전 시장 맥락 |
| `research` | 후보 계획, 고정 모델·선택 설정, 비교 전략, 비용 스트레스, 그림 |
| `robustness` | 고정 후보의 연도별 재시작, 30일 블록 재표집 |
| `replay`, `demo` | 시간순 모의 체결, 잔고 곡선, 종료 상태, 회계 대조 |

실험은 매번 새 폴더를 생성한다. `artifacts/`의 결과, `data/`의 시장 자료, 원본 CSV, 학습 모델은 Git에서 제외한다. 공개 저장소만 복제해 원본 기반 학습을 재현하려면 별도로 이용 권한이 있는 원본을 준비해야 한다. 합성 예제와 테스트는 원본 없이 실행할 수 있다.

오프라인 재생은 종료 시 청산한다. 실행 중 체크포인트 복원, 실시간 모의매매, 원거래소 지정가 대기열, 모든 계약의 손익 복원은 아직 지원하지 않는다.

## 개발과 공개 범위

```sh
# 선택한 로컬 저장소에서 데이터 유입 방지 훅을 활성화한다.
git config core.hooksPath .githooks
uv run python scripts/check_public_tree.py
uv run pip-audit --skip-editable
```

CI는 합성 데이터만 사용한다. 원본 내역이나 키를 이슈·PR·CI 로그에 붙이지 않는다. 기여 방식과 커밋 형식은 [CONTRIBUTING.md](CONTRIBUTING.md)를 따른다.

범용 코드와 직접 작성한 문서는 [MIT License](LICENSE)를 적용한다. 제3자 원본·서한·시장 자료·원본으로 학습한 모델 및 로컬 파생 결과는 이 코드 라이선스의 적용 대상이 아니다. 원자료의 조건은 [DATA_POLICY.md](DATA_POLICY.md)에 별도로 기록한다. 이 프로젝트는 원자료 공개자의 공식 프로젝트나 보증을 의미하지 않는다.
