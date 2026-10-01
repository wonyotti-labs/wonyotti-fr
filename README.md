# Wonyotti, For Real?

공개 거래 기록에서 매매 행동을 연구하고, 추정한 전략을 과거 시장에서 검증하는 Python 프로젝트다. 현재는 **오프라인 연구용**이며 실제 주문·API 키·자금 이체 기능은 없다.

기록을 설명하는 모델과 수익성 있는 전략은 다르다. 첫 보유 방향 모사 후보와 사건별 후보의 개발 결과는 수익성을 확보하지 못했다. 실패 결과와 실행 조건도 로컬 연구 기록에 보존한다. 공개 익명 체결의 일부 일치나 봇의 성과로 개인의 전체 재산·계정 소유·실제 전략을 입증하지 않는다.

- [전체 로드맵](ROADMAP.md)
- [진행 상태와 남은 문제](docs/STATUS.md)
- [연구 방법과 단위](docs/METHODOLOGY.md)
- [사건별 실험 사전 계획](docs/EXPERIMENT_V2.md)
- [행동 빈도 반영 실험 계획](docs/EXPERIMENT_V3.md)
- [노출 확대 시점·방향 분리 계획](docs/EXPERIMENT_V4.md)
- [보유 시간·비용을 고려한 진입 계획](docs/EXPERIMENT_V5.md)
- [최초 체결 시점 해상도 연구](docs/EXPERIMENT_V6.md)
- [요구 사항별 검증 기록](docs/VERIFICATION.md)
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
uv run wonyotti context-study --audit-run artifacts/감사실행ID --study-run artifacts/사건학습ID
```

독립 주문의 진입·추가·축소·청산 의도를 학습한다. 같은 주문의 부분 체결은 새 판단으로 반복 학습하지 않는다. 원거래소의 봉 시작 가격은 이전 종가이므로 특징 계산에만 쓰고, 체결 평가는 별도 거래소 시세를 사용한다.

v2는 2018~2019년 학습, 2020년 설정 선택, 2021년 모사 평가다. 2022~2025년은 이미 관찰한 탐색·이전 실험이다. 후보를 고정한 뒤에만 새 구간을 수집한다.

```sh
uv run wonyotti market --interval 5m --start 2025-12 --end 2026-08 --output data/market-5m-2026-new
uv run wonyotti event-evaluate --selection-run artifacts/선택실행ID --market data/market-5m-2026-new --period new
```

위 `new`는 v2가 처음 열었던 2026년 1~8월 구간을 뜻한다. 재실행이나 새 전략에 대해 다시 미사용 평가라고 부르면 안 된다. 2025년 12월은 지표 준비 구간이다. 날짜는 시작 포함·종료 미포함이며, 월별 수집의 `--end`는 해당 월 포함이다.

## 학습 행동 빈도를 반영한 v3 연구

```sh
uv run wonyotti frequency-select --study-run artifacts/사건학습ID --audit-run artifacts/감사실행ID --v2-selection-run artifacts/선택실행ID
uv run wonyotti frequency-evaluate --selection-run artifacts/빈도선택ID --market data/market-5m-complete --period observed
uv run wonyotti frequency-evaluate --selection-run artifacts/빈도선택ID --market data/market-5m-2026-new --period seen_2026
```

v3는 기존 모델의 계수에 학습 구간의 행동 빈도를 반영한 점수 후보를 비교한다. 일부 기간의 매매 비용과 손실은 줄었지만 개발·확인 조건을 통과하지 못했다. `frequency-evaluate --period new`는 선행 검증 조건을 다시 계산하며, 실패한 후보로 새 기간을 열지 않는다. v3의 `new`는 2026년 9월이며 v3 단계에서는 수집하지 않았다. 이후 v5에서 개봉한 이력은 아래에 구분한다.

## 노출 확대를 학습하는 v4 연구

```sh
uv run wonyotti execution-study --audit-run artifacts/감사실행ID --study-run artifacts/사건학습ID
uv run wonyotti expansion-select --audit-run artifacts/감사실행ID --study-run artifacts/사건학습ID
uv run wonyotti expansion-evaluate --selection-run artifacts/노출확대선택ID --v3-selection-run artifacts/빈도선택ID --market data/market-5m-complete --period observed
uv run wonyotti expansion-evaluate --selection-run artifacts/노출확대선택ID --v3-selection-run artifacts/빈도선택ID --market data/market-5m-2026-new --period seen_2026
```

새 진입과 추가 진입의 시점·방향을 분리해 학습한다. 로지스틱 회귀와 경사 부스팅을 숫자 JSON으로 저장하며 실행 코드가 포함된 모델 파일은 읽지 않는다. 최초 체결 이후의 가격 분석은 설명 연구이며 신호 입력으로 사용하지 않는다. v4도 개발·2021년 조건을 통과하지 못했다. 같은 `event-replay`와 `engine-stress` 명령으로 고정 후보를 검증할 수 있다. v3 이후 후보의 새 기간 개봉 조건은 재생 명령에도 적용한다.

## 보유 시간·비용 예측 v5 연구

```sh
uv run wonyotti edge-select --audit-run artifacts/감사실행ID --study-run artifacts/사건학습ID --v4-selection-run artifacts/노출확대선택ID
uv run wonyotti edge-evaluate --selection-run artifacts/비용예측선택ID --v4-selection-run artifacts/노출확대선택ID --market data/market-5m-complete --period observed
uv run wonyotti edge-evaluate --selection-run artifacts/비용예측선택ID --v4-selection-run artifacts/노출확대선택ID --market data/market-5m-2026-new --period seen_2026
```

개발·2021년 선행 조건을 통과한 뒤 2026년 9월을 열었다. 월별 시세가 아직 없어 공식 일별 자료로 보완한 경로는 다음과 같다. 보완 전 실패 실행도 보존한다.

```sh
uv run wonyotti market --interval 5m --start 2026-08 --end 2026-09 --output data/market-5m-2026-september
uv run wonyotti market-repair --interval 5m --market data/market-5m-2026-september --output data/market-5m-2026-september-complete --start 2026-08-01 --end 2026-10-01
uv run wonyotti edge-evaluate --selection-run artifacts/비용예측선택ID --v4-selection-run artifacts/노출확대선택ID --market data/market-5m-2026-september-complete --period new
uv run wonyotti edge-diagnose --selection-run artifacts/비용예측선택ID --audit-run artifacts/감사실행ID --study-run artifacts/사건학습ID --recent-market data/market-5m-2026-new --new-market data/market-5m-2026-september-complete
```

v5도 다른 시장 손실과 거래 표본 부족으로 채택하지 않았다. 9월 무거래를 수익성 증거로 해석하지 않는다. 이후 후보의 평가에서 9월을 다시 미사용 구간으로 부르면 안 된다. 후속 연구는 `bitmex-history --interval 1m --output data/bitmex-history-1m`으로 별도 자료를 수집해 체결 시점의 차이부터 검증한다.

```sh
uv run wonyotti bitmex-history --interval 1m --output data/bitmex-history-1m
uv run wonyotti timing-study --audit-run artifacts/감사실행ID --study-run artifacts/사건학습ID --minute-history data/bitmex-history-1m
```

`timing-study`는 두 간격의 수집이 완료된 후 실행한다. 같은 주문의 직전 확정 가격과 이후 가격을 연결하고, 1분봉을 집계한 값과 공식 5분 자료를 대조한다. 미완성 봉·값 차이·연결 실패를 각각 남긴다. 전체 기간의 집계와 같은 주문 연결을 완료했다. 시간 해상도를 높인 것 자체를 수익성 확보로 해석하지 않는다. 후속 구현은 [v7 진입 대기 계획](docs/EXPERIMENT_V7.md)에 따른다.

실행용 1분 자료에서 집계 불일치가 발견되면 `minute-repair --market <1분 자료> --feature-market <5분 자료> --output <새 폴더>`로 공식 개별 체결을 대조한다. 원체결 집계가 공식 5분 가격·거래량·건수와 일치해야 한다. 수정할 분봉에 ID 공백이 있으면 별도 집계 체결에 모든 시장 체결이 정확히 한 번씩 연결되고 가격·수량·방향·시각이 일치해야 한다. 원본·전후 값·체크섬·실패를 보존하며 이 검사가 모든 분봉의 정확성을 인증하는 것은 아니다.

공식 5분봉도 원체결과 다를 때는 `paired-repair --market <1분 자료> --feature-market <5분 자료> --output <새 1분 폴더> --feature-output <새 5분 폴더>`를 사용한다. 불일치 구간 전체의 원체결과 별도 집계 체결이 일치해야 두 해상도를 각각 복원한다. 모든 검사를 마치기 전 출력은 실행 입력으로 사용할 수 없다. 날짜 경계·대조 실패는 중단하며 [자료 검증 범위](docs/EXPERIMENT_V7.md)를 함께 확인한다.

```bash
uv run wonyotti pullback-select --v4-selection-run <v4 선택 폴더> \
  --market <대조를 마친 1분 자료> --feature-market <기존 5분 자료>
```

`pullback-select`는 확정된 5분 특징과 1분 종가로 여섯 후보를 비교한다. 선택은 2020년, 고정 후 확인은 2021년이다. 대기 상태도 저장하며 지정가 체결·리베이트를 가정하지 않는다. 후속 다년 평가와 실제 프로세스 종료 복원 검증은 진행 중이다.


## 중단과 복원이 가능한 오프라인 봇

```sh
uv run wonyotti event-replay --selection-run artifacts/선택실행ID --market data/market-5m-complete --start 2020-01-01 --end 2020-02-01 --journal artifacts/bot.sqlite --max-bars 300
uv run wonyotti event-replay --selection-run artifacts/선택실행ID --market data/market-5m-complete --start 2020-01-01 --end 2020-02-01 --journal artifacts/bot.sqlite --verify-memory
```

첫 실행은 처리 봉 수만 제한하며 열린 포지션을 유지한다. 다음 실행은 같은 모델·시세·설정·소스의 저널에서 이어서 처리한다. 전체 기간 끝에서만 비용을 내고 청산한다. `--verify-memory`는 처음부터 한 번에 처리한 잔고·체결·상태와 대조한다. 긴 기간에서는 추가 시간과 메모리가 든다.

`--halt --max-bars 0`은 수동 중지 의도를 저장한다. 다음 유효 시세를 처리할 때 청산하고 재진입을 막는다. 같은 사건의 재전달은 중복 체결하지 않고, 같은 ID의 다른 시세는 거부한다. 해시 저널은 로컬 손상 탐지용이며 외부 서명이나 계정 인증은 아니다. 소스 변경 후에는 기존 저널을 다른 프로그램으로 이어서 처리하지 않고 새 실행을 만들거나 보존한 코드 스냅샷을 사용한다.

```sh
uv run wonyotti engine-stress --selection-run artifacts/빈도선택ID
```

`engine-stress`는 과거 급변 시세에서 검증용 하위 프로세스를 커밋 직전에 강제 종료하고 복구 결과를 대조한다. 기존 봇 프로세스나 사용자 작업을 종료하지 않는다. 테스트용 저널과 결과도 새 로컬 실행 폴더에 남긴다.

## 기존 v1 연구와 출력

15분봉을 쓰는 기존 `study`, `research`, `robustness`, `replay`, `demo`도 유지한다. v1과 사건별 v2의 모델·시세 간격을 섞지 않는다.

| 명령 | 주요 결과 |
| --- | --- |
| `audit`, `portfolio`, `verify` | 원본 감사, 계약별 손익·지갑 대조, 공식 체결 표본 대조 |
| `market`, `market-repair`, `bitmex-history` | 시세·펀딩·출처·해시·누락과 보완 이력 |
| `event-study`, `event-select` | 사건별 학습 자료·모델·시간순 모사 평가·고정 후보 선택 |
| `event-evaluate` | 다년·다시장 성과, 위험 규칙·비용·지연 비교, 그림, 채택 판단 |
| `event-walkforward`, `event-diagnose` | 확장 학습의 모사 성능, 원본 행동 빈도와 봇의 가격 손익·비용 분해 |
| `context-study` | 원거래소 전체 기간의 독립 주문 맥락·지정가/유동성 비용·추가 진입별 손익 |
| `frequency-select`, `frequency-evaluate` | 행동 빈도 반영 후보·신뢰도 진단·고정 비교·새 구간 개봉 조건 |
| `execution-study`, `expansion-select`, `expansion-evaluate` | 최초 체결 이후 가격·비용, 노출 확대 시점·방향 모델, 다년 비교 |
| `edge-select`, `edge-evaluate`, `edge-diagnose` | 보유 시간·비용 예측 후보, 단계별 신호 표본, 후속 기간 비교 |
| `timing-study` | 같은 주문의 1분·5분 연결과 집계 일치·누락 비교 |
| `event-replay` | 영속 저널, 중단·재개 상태, 단일 실행 대조 |
| `engine-stress` | 실제 프로세스 종료 복구와 급변·중복·누락·수동 중지 검사 |
| `study`, `research`, `robustness`, `replay` | v1 행동 연구·방향 모사·비용 비교·재표집·오프라인 재생 |

실험마다 새 폴더를 만든다. 원본 CSV, `artifacts/` 결과, `data/` 시장 자료, 학습 모델은 Git에서 제외한다. 원본 기반 연구를 재현하려면 별도로 이용 권한이 있는 원본이 필요하다. 합성 예제와 테스트는 원본 없이 실행할 수 있다. 실제 호가 대기열·시장 충격·실시간 모의매매는 구현하지 않았다.

사건별 백테스트 출력은 작은 묶음으로 저장한다. `storage.json`에 출력별 최대 대기 행 수, `config.json`에 해당 실행의 시간 간격·위험 설정을 남긴다. 입력 시세와 특징 캐시는 별도 메모리를 사용한다. 실패하면 부분 출력과 `failure.json`을 보존하고 성공 지표를 생성하지 않는다. 기존 다년 실행과 모든 잔고·체결·최종 상태를 대조했다.

## 개발과 공개 범위

```sh
git config core.hooksPath .githooks
uv run python scripts/check_public_tree.py
uv run pip-audit --skip-editable
```

CI는 합성 데이터만 사용한다. 원본·키를 이슈·PR·CI 로그에 넣지 않는다. [기여 안내](CONTRIBUTING.md)에 따라 영어 prefix와 한글 명사형 종결로 커밋하고 바로 푸시한다.

범용 코드와 직접 작성한 문서는 [MIT License](LICENSE)를 적용한다. 제3자 원본·서한·시장 자료·학습 모델·로컬 파생 결과는 적용 대상이 아니다. 원자료 조건은 [DATA_POLICY.md](DATA_POLICY.md)에 기록한다. 이 프로젝트는 원자료 공개자의 공식 프로젝트나 보증을 의미하지 않는다.
