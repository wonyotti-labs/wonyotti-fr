# Wonyotti, For Real?

공개 거래 기록에서 매매 행동을 연구하고, 추정한 전략을 과거 시장에서 검증하는 Python 프로젝트다. 현재는 **오프라인 연구용**이며 실제 주문·API 키·자금 이체 기능은 없다.

기록을 설명하는 모델과 수익성 있는 전략은 다르다. 첫 보유 방향 모사 후보와 사건별 후보의 개발 결과는 수익성을 확보하지 못했다. 실패 결과와 실행 조건도 로컬 연구 기록에 보존한다. 공개 익명 체결의 일부 일치나 봇의 성과로 개인의 전체 재산·계정 소유·실제 전략을 입증하지 않는다.

- [전체 로드맵](ROADMAP.md)
- [진행 상태와 남은 문제](docs/STATUS.md)
- [연구 방법과 단위](docs/METHODOLOGY.md)
- [사건별 실험 사전 계획](docs/EXPERIMENT_V2.md)
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

`demo`는 합성 시세만 사용하며 네트워크나 원본이 필요하지 않다. 데모 수익은 실제 전략 성과가 아니다. 각 명령이 출력하는 보고서 경로에서 결과를 확인한다.

## 원본 감사와 외부 대조

```text
부모 폴더/
├── aoa_public_2021-12-31_with_letter/  # 로컬 입력
└── wonyotti-fr/                      # 이 저장소
```

```sh
uv run wonyotti audit --source ../aoa_public_2021-12-31_with_letter --timezone UTC
uv run wonyotti verify --audit-run artifacts/감사실행ID
uv run wonyotti portfolio --audit-run artifacts/감사실행ID
```

명령의 `감사실행ID` 등은 앞 단계가 출력한 폴더로 바꾼다. UTC 연결은 공식 체결 표본으로 확인했다. 전체 체결과 계정 소유의 인증은 아니며 원본의 같은 시각 순서는 별도 가정이다. 전체 계약은 현재 상품 사양 대신 원본 정산 통화와 비용으로 복원한다.

## 사건별 연구

```sh
uv run wonyotti bitmex-history --start 2018-03-01 --end 2022-01-01
uv run wonyotti market --interval 5m --start 2019-09 --end 2025-12 --output data/market-5m
uv run wonyotti market-repair --interval 5m --market data/market-5m --output data/market-5m-complete
uv run wonyotti event-study --audit-run artifacts/감사실행ID
uv run wonyotti event-select --study-run artifacts/사건학습ID --market data/market-5m-complete
uv run wonyotti event-evaluate --selection-run artifacts/선택실행ID --market data/market-5m-complete --period observed
uv run wonyotti event-walkforward --study-run artifacts/사건학습ID --audit-run artifacts/감사실행ID
uv run wonyotti event-diagnose --study-run artifacts/사건학습ID --evaluation-runs artifacts/완료한평가ID
```

독립 주문의 진입·추가·축소·청산 의도를 학습한다. 같은 주문의 부분 체결은 새 판단으로 반복 학습하지 않는다. 원거래소의 봉 시작 가격은 이전 종가이므로 특징 계산에만 쓰고, 체결 평가는 별도 거래소 시세를 사용한다.

v2는 2018~2019년 학습, 2020년 설정 선택, 2021년 모사 평가다. 2022~2025년은 이미 관찰한 탐색·이전 실험이다. 후보를 고정한 뒤에만 새 구간을 수집한다.

```sh
uv run wonyotti market --interval 5m --start 2025-12 --end 2026-08 --output data/market-5m-2026-new
uv run wonyotti event-evaluate --selection-run artifacts/선택실행ID --market data/market-5m-2026-new --period new
```

위 `new`는 v2가 처음 열었던 2026년 1~8월 구간을 뜻한다. 재실행이나 새 전략에 대해 다시 미사용 평가라고 부르면 안 된다. 2025년 12월은 지표 준비 구간이다. 날짜는 시작 포함·종료 미포함이며, 월별 수집의 `--end`는 해당 월 포함이다.

## 중단과 복원이 가능한 오프라인 봇

```sh
uv run wonyotti event-replay --selection-run artifacts/선택실행ID --market data/market-5m-complete --start 2020-01-01 --end 2020-02-01 --journal artifacts/bot.sqlite --max-bars 300
uv run wonyotti event-replay --selection-run artifacts/선택실행ID --market data/market-5m-complete --start 2020-01-01 --end 2020-02-01 --journal artifacts/bot.sqlite --verify-memory
```

첫 실행은 처리 봉 수만 제한하며 열린 포지션을 유지한다. 다음 실행은 같은 모델·시세·설정·소스의 저널에서 이어서 처리한다. 전체 기간 끝에서만 비용을 내고 청산한다. `--verify-memory`는 처음부터 한 번에 처리한 잔고·체결·상태와 대조한다. 긴 기간에서는 추가 시간과 메모리가 든다.

`--halt --max-bars 0`은 수동 중지 의도를 저장한다. 다음 유효 시세를 처리할 때 청산하고 재진입을 막는다. 같은 사건의 재전달은 중복 체결하지 않고, 같은 ID의 다른 시세는 거부한다. 해시 저널은 로컬 손상 탐지용이며 외부 서명이나 계정 인증은 아니다. 소스 변경 후에는 기존 저널을 다른 프로그램으로 이어서 처리하지 않고 새 실행을 만들거나 보존한 코드 스냅샷을 사용한다.

## 기존 v1 연구와 출력

15분봉을 쓰는 기존 `study`, `research`, `robustness`, `replay`, `demo`도 유지한다. v1과 사건별 v2의 모델·시세 간격을 섞지 않는다.

| 명령 | 주요 결과 |
| --- | --- |
| `audit`, `portfolio`, `verify` | 원본 감사, 계약별 손익·지갑 대조, 공식 체결 표본 대조 |
| `market`, `market-repair`, `bitmex-history` | 시세·펀딩·출처·해시·누락과 보완 이력 |
| `event-study`, `event-select` | 사건별 학습 자료·모델·시간순 모사 평가·고정 후보 선택 |
| `event-evaluate` | 다년·다시장 성과, 위험 규칙·비용·지연 비교, 그림, 채택 판단 |
| `event-walkforward`, `event-diagnose` | 확장 학습의 모사 성능, 원본 행동 빈도와 봇의 가격 손익·비용 분해 |
| `event-replay` | 영속 저널, 중단·재개 상태, 단일 실행 대조 |
| `study`, `research`, `robustness`, `replay` | v1 행동 연구·방향 모사·비용 비교·재표집·오프라인 재생 |

실험마다 새 폴더를 만든다. 원본 CSV, `artifacts/` 결과, `data/` 시장 자료, 학습 모델은 Git에서 제외한다. 원본 기반 연구를 재현하려면 별도로 이용 권한이 있는 원본이 필요하다. 합성 예제와 테스트는 원본 없이 실행할 수 있다. 실제 호가 대기열·시장 충격·실시간 모의매매는 구현하지 않았다.

## 개발과 공개 범위

```sh
git config core.hooksPath .githooks
uv run python scripts/check_public_tree.py
uv run pip-audit --skip-editable
```

CI는 합성 데이터만 사용한다. 원본·키를 이슈·PR·CI 로그에 넣지 않는다. [기여 안내](CONTRIBUTING.md)에 따라 영어 prefix와 한글 명사형 종결로 커밋하고 바로 푸시한다.

범용 코드와 직접 작성한 문서는 [MIT License](LICENSE)를 적용한다. 제3자 원본·서한·시장 자료·학습 모델·로컬 파생 결과는 적용 대상이 아니다. 원자료 조건은 [DATA_POLICY.md](DATA_POLICY.md)에 기록한다. 이 프로젝트는 원자료 공개자의 공식 프로젝트나 보증을 의미하지 않는다.
