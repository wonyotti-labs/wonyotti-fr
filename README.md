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
- [확정 종가에 따른 진입 대기 계획](docs/EXPERIMENT_V7.md)
- [실행 순손익에 맞춘 진입 필터 계획](docs/EXPERIMENT_V8.md)
- [원본 수익 구조와 관리 정책 복원 계획](docs/EXPERIMENT_V9.md)
- [분별 관리 사건과 행동별 판단](docs/EXPERIMENT_V10.md)
- [관리 방식의 시기 변화](docs/EXPERIMENT_V11.md)
- [보유 가격 경로 복원](docs/EXPERIMENT_V12.md)
- [청산과 방향 반전의 의미 대조](docs/EXPERIMENT_V13.md)
- [행동 빈도 누적 대조](docs/EXPERIMENT_V14.md)
- [전체 거래 순손익 진입 필터](docs/EXPERIMENT_V15.md)
- [중첩 정답 가중치](docs/EXPERIMENT_V16.md)
- [활동 관문과 신규 진입 기회 분리](docs/EXPERIMENT_V17.md)
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

한 시장의 자료 검증 실패가 다른 시장의 독립 검증을 막지 않도록 `paired-repair --symbols ETHUSDT SOLUSDT`처럼 범위를 지정할 수 있다. 출력 매니페스트는 선택한 시장만 포함한다. 제외한 시장의 실패나 전체 평가 미완료를 성공으로 바꾸지는 않는다.

집계에 연결되지 않는 체결이 수정할 필요 없는 봉에만 있으면 `--verify-unchanged-minutes`로 봉 값의 추가 대조를 요청할 수 있다. 원체결 집계·기존 입력·공식 일별·월별 자료의 아홉 값이 정확히 같아야 하며 해당 분봉은 바꾸지 않는다. 수정할 분봉의 체결 검증은 그대로 필요하다. 행 단위 미연결 상태와 봉 값 검증은 결과에서 분리하며 [검증 근거와 한계](docs/EXPERIMENT_V7.md)를 보존한다.

```bash
uv run wonyotti pullback-select --v4-selection-run <v4 선택 폴더> \
  --market <대조를 마친 1분 자료> --feature-market <기존 5분 자료>
```

`pullback-select`는 확정된 5분 특징과 1분 종가로 여섯 후보를 비교한다. 선택은 2020년, 고정 후 확인은 2021년이다. 대기 상태도 저장하며 지정가 체결·리베이트를 가정하지 않는다. 개발·확인 및 2022~2025년·2026년의 세 시장 연속 운용에서 손실을 기록했다. 진입 대기·보유 중 실제 프로세스 종료 복원과 한 달 중단·재개 대조를 통과했다.

```sh
uv run wonyotti pullback-evaluate --selection-run <v7 선택 폴더> \
  --market <전체 기간 1분 자료> --feature-market <전체 기간 5분 자료> --period observed
uv run wonyotti pullback-evaluate --selection-run <v7 선택 폴더> \
  --market <검증된 1분 자료> --feature-market <검증된 5분 자료> --period seen_2026
uv run wonyotti pullback-evaluate --selection-run <v7 선택 폴더> \
  --market <2022~2024년 1분 자료> --feature-market <2022~2024년 5분 자료> --period verified_2022_2024
```

`observed`는 원래 계획한 2022~2025년 전체 기간이고 `seen_2026`은 2026년 1~9월이다. 두 기간의 세 시장·여섯 조건 및 12개 연간 자본 초기화를 완료했다. 모두 이미 관찰한 구간이다. BTC의 미연결 체결 10건은 보존하고, 미수정 분봉의 추가 대조를 근거로 봉 기반 전체 평가를 수행했다. 해당 체결의 개별 진위나 누락 원인은 미확정이다. 시장별로 다른 검증 폴더를 사용할 때는 `--symbols BTCUSDT` 또는 `--symbols ETHUSDT SOLUSDT`로 평가 대상을 지정한다.

`verified_2022_2024`는 자료 검증 실패 뒤 성과 확인 전에 고정했던 추가 범위다. 이 결과와 원래 전체 기간 결과를 각각 보존한다. 고정 후보·즉시 진입·현금·비용 2·3배·추가 1분 지연과 거래 회계·진입 대기·조건부 재표집을 기록한다. 기간별 자료를 만들 때는 `paired-repair --start 2021-12-01 --end 2025-01-01`과 이미 발견한 오류의 `--extra-targets`를 사용한다.


## 실행 순손익을 학습하는 v8 연구

```sh
uv run wonyotti net-edge-select --v7-selection-run <v7 선택 폴더> \
  --market <2020~2021년 1분 자료> --feature-market <2020~2021년 5분 자료> \
  --confirmation-market <2022년 1분 자료> --confirmation-features <2022년 5분 자료>
uv run wonyotti net-edge-evaluate --selection-run <v8 선택 폴더> \
  --market <전체 기간 1분 자료> --feature-market <전체 기간 5분 자료> --period observed
uv run wonyotti net-edge-evaluate --selection-run <v8 선택 폴더> \
  --market <2026년 1분 자료> --feature-market <2026년 5분 자료> --period seen_2026
```

확정 종가로 제시된 기회를 같은 실행 엔진의 다음 시가·손절·수수료·슬리피지·펀딩으로 평가해 순손익을 학습한다. 기존 v7 신호·대기·위험 설정은 고정한다. 2020년 학습, 네 후보의 2021년 선택, 선택 후 2022년 확인 순서다. 기회마다 독립 자본을 가정하므로 겹치는 기회 수를 독립 거래 표본 수로 해석하지 않는다. 손실 정답도 보존한다.

v8의 `observed`는 **2023~2025년**이며 v7과 시작 연도가 다르다. `seen_2026`은 2026년 1~9월이다. 명령은 공통 평가기를 사용하며 `verified_2022_2024`는 v8에서 거부한다. 시장별 입력 폴더가 다르면 `--symbols`로 분리한다. 두 기간·세 시장의 여섯 조건과 아홉 연간 초기화를 완료했다. BTC 일부 기본 조건은 양수였지만 비용 증가·표본 부족과 다른 시장의 장기 손실로 채택하지 않았다. 모든 구간은 이미 관찰했으며 새 미사용 평가가 아니다.

필터는 판단 시점의 확정 특징으로 예상 순손익을 계산한다. 거절하면 해당 대기를 끝낸다. 필터 제거 비교는 기존 v7이며 최근 세 시장의 전체 실행 결과가 이전 v7과 정확히 같다. 영속 재생과 대기·보유 중 실제 프로세스 종료 복원도 검증했다.

## 원본 수익 구조와 관리 정책 v9 연구

```sh
uv run wonyotti structure-study --audit-run <감사 폴더> --study-run <사건 학습 폴더>
uv run wonyotti lifecycle-select --v7-selection-run <v7 선택 폴더> \
  --audit-run <감사 폴더> --study-run <사건 학습 폴더> \
  --market <2021년 1분 자료> --feature-market <2021년 5분 자료> \
  --confirmation-market <2022년 1분 자료> --confirmation-features <2022년 5분 자료>
uv run wonyotti lifecycle-evaluate --selection-run <v9 선택 폴더> \
  --market <검증한 다년 1분 자료> --feature-market <검증한 다년 5분 자료> --period observed
uv run wonyotti lifecycle-evaluate --selection-run <v9 선택 폴더> \
  --market <2026년 1분 자료> --feature-market <2026년 5분 자료> --period seen_2026
```

`structure-study`는 원거래소의 전체 5분·1분 자료를 사용한다. 원본 행동을 고정한 여덟 조건에서 추가 제거·30분 제한·가격·비용 변경의 영향을 분리한다. 미래 원본 행동을 알고 재생한 설명 실험이며 실행 가능한 봇 성과가 아니다. 원래 규모의 BTC 손익, 첫 주문당 정규화, 연도별 결과를 함께 기록한다. 가격 대조는 원본 사건·펀딩 시각을 유지하므로 실제 지연에 따른 펀딩 변화까지 재현하지 않는다.

실행 후보는 v7 진입 규칙을 유지하고 원본의 보유·추가·축소·청산을 2018~2020년에서 학습한다. 30분 강제 종료를 없애고 위험 한도 안에서 추가를 허용한다. 관리 판단은 확정 5분 특징과 현재 봇 상태를 사용하며 다음 1분 시가에 실행한다. 추가·축소 크기는 학습 주문 비율의 제한된 고정 기준으로 원본 레버리지나 상태별 수량 전략을 복원한 것이 아니다.

네 후보를 2021년에서 선택하고 고정 후 2022년에서 확인한다. v9의 평가 기간은 v8과 같고 시장별 입력이 다르면 `--symbols`로 분리한다. 비교 조건은 고정 후보·기존 v7·현금·추가 제거·30분 제한·비용 2배·3배·추가 1분 지연이다. 보유 상태가 달라지면 같은 진입 규칙도 실행 경로가 달라진다. 이미 관찰한 기간의 비교이며 `verified_2022_2024`는 거부한다.

원본 수익 구조를 훼손한 단순화는 계량화했지만 새 후보도 청산·축소 모사가 약하고 2022년 확인과 일부 장기 시장에서 실패했다. 보유 정답이 대부분인 전체 정확도를 모사 성공으로 해석하지 않는다. 일부 양수 구간만으로 수익성 전략으로 채택하지 않았다. 입력 지문·실패·개별 결과는 로컬에 보존한다.

## 관리 사건·가격 경로·반전 연구

v10은 다음 1분의 독립 추가·축소·청산을 복수 정답으로 보존하고 행동별 모델과 별도 문턱을 학습한다. 원본 반전 주문의 청산 대상은 새 포지션이 아니라 직전 포지션이다. v11은 다른 설정을 유지한 채 최근 학습·문턱 조정 기간을 사용한다. 두 후보 모두 확인 및 전체 조건에서 수익성 기준에 미달했다.

```sh
uv run wonyotti action-labels --audit-run <감사 폴더> --study-run <사건 연구 폴더> \
  --history <원거래소 5분 자료> --minute-history <원거래소 1분 자료>
uv run wonyotti action-select --v9-selection-run <v9 선택 폴더> --labels-run <관리 정답 폴더> \
  --regime recent --market <2021년 1분 자료> --feature-market <2021년 5분 자료> \
  --confirmation-market <2022년 1분 자료> --confirmation-features <2022년 5분 자료>
```

`--regime original`은 v10, `recent`는 v11이다. v12는 과거 보유 종가의 최선·최악·반납 폭을 추가한다. `path-labels` 출력과 `--regime path`를 함께 사용한다. 모델 형식과 학습 자료는 기존 후보와 분리하며, 추가 진입으로 평균가가 바뀌면 당시까지 관찰한 극값을 새 평균가로 계산한다. 가격 경로는 관리 대기 중에도 갱신하고 저널에 저장한다.

```sh
uv run wonyotti path-labels --labels-run <관리 정답 폴더>
uv run wonyotti action-select --v9-selection-run <v9 선택 폴더> --labels-run <가격 경로 정답 폴더> \
  --regime path --market <2021년 1분 자료> --feature-market <2021년 5분 자료> \
  --confirmation-market <2022년 1분 자료> --confirmation-features <2022년 5분 자료>
uv run wonyotti reversal-select --path-selection-run <v12 선택 폴더> \
  --market <2021년 1분 자료> --feature-market <2021년 5분 자료> \
  --confirmation-market <2022년 1분 자료> --confirmation-features <2022년 5분 자료>
```

v13은 v12의 고정 모델·문턱·신규 진입·위험을 유지하고 모델 청산을 반대 방향 진입으로 해석하는 단일 실행 가설이다. 위험 청산은 평탄 상태를 유지한다. 원본의 소수 평탄 청산을 따로 분류하지 못한다는 한계를 함께 기록한다.

```sh
uv run wonyotti action-evaluate --selection-run <선택 폴더> \
  --market <평가 1분 자료> --feature-market <평가 5분 자료> --period observed
uv run wonyotti action-evaluate --selection-run <선택 폴더> \
  --market <2026년 1분 자료> --feature-market <2026년 5분 자료> --period seen_2026
```

v14는 고정 관리 모델의 행동 점수를 시간에 걸쳐 누적한다. 행동별 누적 배율은 2020년 보정 구간의 실제 사건 수와 점수 합으로 고정한다. 청산·축소·추가의 빈도 조절만으로 수익성을 확보하지 못했다.

```sh
uv run wonyotti rate-select --path-selection-run <v12 선택 폴더> \
  --market <2021년 1분 자료> --feature-market <2021년 5분 자료> \
  --confirmation-market <2022년 1분 자료> --confirmation-features <2022년 5분 자료>
```

v15는 같은 v14 관리로 진입부터 자연 청산까지 재생한 순손익을 학습한다. 손실과 미확정 거래를 원장에 보존한다. 단일 Ridge alpha 100·8bp 진입 기준이며 관리 중에는 필터를 적용하지 않는다. 2021년은 전체 시스템 학습 구간이고 같은 기간의 재생은 일반화 성과가 아니다. 2022년 확인 기준을 통과했지만 전체 후속 평가에서 수익성 기준에 미달했다.

```sh
uv run wonyotti lifecycle-edge-labels --rate-selection-run <v14 선택 폴더> \
  --market <2021년 1분 자료> --feature-market <2021년 5분 자료>
uv run wonyotti lifecycle-edge-select --rate-selection-run <v14 선택 폴더> \
  --labels-run <전체 거래 정답 폴더> \
  --market <2021년 1분 자료> --feature-market <2021년 5분 자료> \
  --confirmation-market <2022년 1분 자료> --confirmation-features <2022년 5분 자료>
```

v15 후속 평가는 같은 `action-evaluate` 명령을 사용한다. 기존 여덟 조건에 진입 필터만 제거한 `unfiltered_v14`를 추가한다. 독립 초기 계좌의 학습 정답과 연속 계좌의 실행 결과, 겹친 기회와 독립 표본을 구분한다.

v16은 같은 정답의 중첩 정도로 표준화·회귀 가중치를 조정한다. 위 `lifecycle-edge-select`에 `--overlap-weighted`를 추가한다. v15·v16 각각 학습 적합·확인·54개 후속 조건·9개 독립 연도, 총 65개 실행을 대조했지만 수익성 전략으로 채택하지 않았다.

v17은 기존 방향 판단과 대기·관리를 유지하고 신규 진입에 적용하던 노출 확대 활동 관문을 제거한다. 정답 생성에는 `--direction-only`, 학습에는 `--direction-only --overlap-weighted`를 추가해야 한다. 이전 기회 집합의 정답을 잘못 전달하면 거부한다. 후속 평가에는 원래 v7·v14와 새 기회에서 순손익 필터만 제거한 `unfiltered_direction_only`를 구분한 열 조건이 있다. 확장 기회 전체의 정답 생성·기존 정답 불변·가중치 및 모델 재생 검증을 완료했다. 확인 선행 조건 통과 후 세 시장·열 조건·독립 연도를 포함한 71개 실행을 검증했지만 후속 손실로 채택하지 않았다.

[v18](docs/EXPERIMENT_V18.md)은 원래 v14 누적 관리가 선택한 청산을 반전으로 실행하는 단일 대조다. `rate-reversal-select --rate-selection-run <v14 선택 폴더>`와 기존 학습·확인 시세 인자를 사용한다. 모델·빈도 배율은 재학습하지 않고 위험 청산은 평탄 상태를 유지한다. 반전 체결 전에는 기존 누적 상태를 보존하며 실제 새 포지션 체결 뒤 초기화한다. 원본의 평탄 청산까지 구분한 전략 복제는 아니다. 확인 실패 후 사전 계획한 7개 실행과 실제 복원을 검증했지만 수익성 기준에 미달했다.


무거래 분봉은 거래 활동과 함께 엔진에 전달한다. 해당 봉에서는 체결하지 않고 기존 주문과 위험 청산을 다음 거래 가능 시점까지 보존한다. 펀딩·시간·평가 손익은 계속 반영한다. 종료 봉이 무거래이고 포지션이 남으면 임의 가격으로 완료하지 않는다. 첫 확장 정답의 실패와 부분 결과는 보존했으며 실제 문제 사례를 수정 후 재생했다.

v12~v17의 2022년 확인이 사전 조건에 미달하면 계획에 따라 `--symbols BTCUSDT --diagnostic-only`로 고정 후보와 독립 연도 진단만 수행한다. 이 경우 다른 시장·비용 배수·추가 지연을 검사했다고 표시하지 않는다. 확인을 통과한 후보에는 이 축소 옵션을 허용하지 않는다. 이미 관찰한 시기의 반복 실험이며 수익성 문제를 해결할 때까지 다음 원인을 연구한다.

## 중단과 복원이 가능한 오프라인 봇

```sh
uv run wonyotti event-replay --selection-run artifacts/선택실행ID --market data/market-5m-complete --start 2020-01-01 --end 2020-02-01 --journal artifacts/bot.sqlite --max-bars 300
uv run wonyotti event-replay --selection-run artifacts/선택실행ID --market data/market-5m-complete --start 2020-01-01 --end 2020-02-01 --journal artifacts/bot.sqlite --verify-memory
```

첫 실행은 처리 봉 수만 제한하며 열린 포지션을 유지한다. 다음 실행은 같은 모델·시세·설정·소스의 저널에서 이어서 처리한다. 전체 기간 끝에서만 비용을 내고 청산한다. `--verify-memory`는 처음부터 한 번에 처리한 잔고·체결·상태와 대조한다. 긴 기간에서는 추가 시간과 메모리가 든다.

v7~v17은 `--market`에 1분 자료를 지정하고 `--feature-market`에 별도 5분 특징 자료를 함께 지정한다. 두 입력의 해시를 저널에 묶는다. 진입 대기·만료 상태도 재개하며, 다음처럼 실제 종료 복구를 검증할 수 있다.

```sh
uv run wonyotti engine-stress --selection-run <v7 선택 폴더> \
  --market <개발 1분 자료> --feature-market <개발 5분 자료> \
  --start 2020-03-10 --end 2020-03-14
```

v8~v17도 같은 두 명령에 해당 선택 폴더를 지정한다. v15~v17의 재생은 전체 시스템 학습 시작인 2021년 이후만 허용한다. 완료한 복원 검사는 2021년 개발 구간 첫 거래의 월을 사용했으며, 신호 조건을 바꿔 대기·보유 상태를 만들지 않았다.

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
| `minute-repair`, `paired-repair` | 원체결 대조·두 해상도 복원·전후 값·실패·범위별 자료 |
| `pullback-select`, `pullback-evaluate` | 확정 종가의 진입 대기 후보·고정 비교·대기 사건·실제 가격 차이 |
| `net-edge-select`, `net-edge-evaluate` | 같은 실행 엔진의 순손익 학습·시간순 선택·필터 제거·비용·지연 비교 |
| `structure-study` | 원본 회계·추가·보유 제한·가격·비용의 여덟 설명 대조 |
| `lifecycle-select`, `lifecycle-evaluate` | 보유·추가·축소·청산 학습·고정 후보·관리 조건 제거·다년 비교 |
| `action-labels`, `path-labels` | 분별 복수 정답·연결 원장·과거 보유 가격 경로 |
| `action-select`, `action-evaluate` | 행동별 문턱·시기 분리·관리 가격 경로·고정 조건 비교 |
| `reversal-select` | 고정 v12 모델의 평탄 청산과 방향 반전 실행 대조 |
| `rate-select` | 고정 관리 모델의 행동별 빈도 누적 대조 |
| `lifecycle-edge-labels`, `lifecycle-edge-select` | 전체 거래 순손익 정답·손실 원장·고정 진입 필터 학습과 확인 |
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

[v19](docs/EXPERIMENT_V19.md)의 수량 진단은 `inventory-study --audit-run <원본 감사 폴더> --labels-run <action-labels 출력>`으로 실행한다. `--bot-runs <개별 봇 실행 폴더> ...`로 보유 수량 상태를 비교할 수 있다. 과거 최대 수량 대비 잔량, 축소 요청·실제 체결 크기, 겹친 주문, 원본 BTC 손익을 연결하며 새 모델이나 수익 문턱을 선택하지 않는다. 원본 수량이 적게 남았다는 이유로 회계에서 삭제하지 않는다.

수량 관리 학습은 `inventory-labels --path-labels-run <path-labels 출력> --inventory-study-run <수량 진단 출력>` 뒤 `inventory-select --rate-selection-run <v14 선택> --labels-run <수량 정답 출력>`으로 진행한다. 선택 명령에는 기존과 같은 개발·확인 시장 경로가 필요하다. 관리 정답은 그대로 유지하고, 실제 체결량과 분 경계 보유량을 연결할 수 있는 주문만 축소 크기 회귀에 사용한다. 예측 축소 비율은 주문과 함께 저장하며 고정 축소 대조와 성과로 재선택하지 않는다.

[v20](docs/EXPERIMENT_V20.md)은 같은 `inventory-select` 명령에 `--latest-source`를 추가한다. 관리·축소 크기 학습을 2020년~2021년 상반기, 빈도 보정을 2021년 하반기로 옮긴다. 이 경우 2021년 재생은 학습 적합 진단이며 시간순 성과로 보고하지 않는다.

[v21](docs/EXPERIMENT_V21.md)의 입력 준비는 `minute-inventory-labels --labels-run <수량 정답 출력> --minute-history <원본 1분 시세> --history <원본 5분 시세>`로 실행한다. 현재까지 확정된 1분·3분 수익률, 1분 가격 범위, 직전 60분 대비 거래량을 기존 관리 정답에 연결한다. 원본 정답과 시간순 학습·보정 행은 보존한다.

분봉 입력 모델의 고정 비교는 `minute-inventory-select --inventory-selection-run <v20 선택> --labels-run <분봉 입력 정답>`과 개발·확인 시장 경로로 실행한다. 기존 v20을 별도 고정 모델로 보존하며, 확인을 통과하면 원래 v14·v20과 고정 축소 대조를 포함한 열한 조건을 평가한다.

[v22](docs/EXPERIMENT_V22.md)의 추가 주문 순효과 정답은 `addition-effect-labels --selection-run <v21 선택> --market <2021년 1분 시세> --feature-market <5분 특징 시세>`로 생성한다. 기존 연속 계좌를 그대로 복원한 뒤 추가 요청 시점의 동일 계좌에서 해당 추가만 유지하거나 취소한다. 두 경로의 현재 포지션이 자연 종료한 후 비용·펀딩을 포함한 현금 차이를 학습 목표로 저장한다. 손실·0효과·경계 미확정도 보존하며 미래 결과를 당시 판단 입력으로 사용하지 않는다. 이 정답은 시뮬레이터 안의 단일 주문 대조이며 수익성 검증 결과가 아니다.

추가 순효과 모델은 `addition-effect-select --selection-run <v21 선택> --labels-run <추가 순효과 정답>`과 개발·확인 시장 경로로 학습한다. 같은 원래 포지션의 총 비중을 동일하게 맞춘 단일 모델을 사용하며, 예측 순효과가 양수일 때 기존 추가 요청을 유지한다. 거절한 요청의 관리 대기는 그대로 보존한다. 확인 통과 시 원래 v21을 포함한 열한 조건을 비교하고, 실패 시 사전 고정한 BTC 진단을 수행한다.

[v23](docs/EXPERIMENT_V23.md)은 `recent-entry-select --selection-run <v21 선택> --audit-run <원본 감사> --study-run <사건 연구> --history <원본 5분 시세>`와 개발·확인 시장 경로로 실행한다. 2020~2021년의 경계 에피소드와 양쪽 24시간을 제거한 원본으로 활동·방향 모델만 다시 학습한다. 기존 모형·학습 분위·방향 문턱을 유지하며 관리·수량·위험과 원래 진입 모델을 사용하는 비교 정책은 보존한다.

[v24](docs/EXPERIMENT_V24.md)의 청산 정답은 `realized-exit-labels --labels-run <v21 분봉 입력 정답> --audit-run <원본 감사>`로 생성한다. 실제 수량 원장에서 보유가 끝나거나 반전한 사건을 분 경계 직전의 같은 포지션에 연결한다. 이어서 `realized-exit-select --selection-run <v21 선택> --labels-run <실제 종료 정답>`과 개발·확인 시장 경로로 청산 모델만 다시 학습한다. 비청산 입력·모델·문턱·빈도와 원래 축소 크기가 일치하지 않으면 비교를 중단한다. 마지막 부분 체결은 실행 결과이며 판단 시각의 복원으로 해석하지 않는다.

[v25](docs/EXPERIMENT_V25.md)은 `new-position-select --selection-run <v23 선택> --audit-run <원본 감사> --study-run <사건 연구> --history <원본 5분 시세>`와 개발·확인 시장 경로로 실행한다. 실제 신규 보유·반전의 첫 체결을 진입 활동 정답으로 삼고 기존 포지션의 추가 매수는 활동 목표에서 제외한다. 원자료와 별도 사건 원장은 보존한다. 방향·관리·축소·위험은 고정하며 새 방향 모델의 지원 기준을 낮추지 않는다. 전체 비교에는 원래 v23을 포함하고 v20 단독 조건을 대신한다.

[v26](docs/EXPERIMENT_V26.md)의 빈도 보정은 `position-prior-select --selection-run <v25 선택>`과 개발·확인 시장 경로로 실행한다. 원래 확대 학습과 신규 포지션 정답의 롱·숏 오즈 차이만 방향 절편에 더한다. 원래 방향 학습 행·모델·집계를 대조하며 활동·방향 계수·관리·위험과 문턱은 고정한다. 조건부 특징 분포의 동일성을 입증한 것은 아니므로 정확한 확률 보정으로 해석하지 않는다.

[v27](docs/EXPERIMENT_V27.md)은 `position-direction-select --selection-run <v26 선택> --audit-run <원본 감사> --study-run <사건 연구> --history <원본 5분 시세>`와 개발·확인 시장 경로로 실행한다. 기존 최소 표본 기준을 충족하는 2018~2021년의 실제 신규 보유·반전 사건으로 방향 모델만 직접 학습한다. 활동·관리·축소·위험은 고정하고 이전 절편 보정은 새 모델에 더하지 않는다. 목표와 기간을 함께 바꾼 비교이며 활동이 많은 과거 연도의 표본 집중도 함께 기록한다.

[v28](docs/EXPERIMENT_V28.md)의 사전 방향 진단은 `direction-model-diagnose --audit-run <원본 감사> --study-run <사건 연구> --history <원본 5분 시세>`로 실행한다. 2018~2020년 실제 신규 방향으로 두 고정 모형을 학습하고 2021년 행동 예측을 비교한다. 양쪽 시간·포지션 경계를 제거하며 상수 기준과 로그 손실을 대조한다. 사전 개선 조건을 통과하기 전에는 새 매매 후보를 만들지 않는다. 이 진단은 이미 관찰한 원본 내부 비교이며 수익성 검증이 아니다.

방향 진단의 사전 기준을 통과하면 `boosted-direction-select --selection-run <v27 선택> --diagnosis-run <시간순 방향 진단> --audit-run <원본 감사> --study-run <사건 연구> --history <원본 5분 시세>`와 개발·확인 시장 경로로 후속 후보를 학습한다. 진단의 원본·기간·지원·판단과 출력 지문을 확인하고 v27 학습 행이 정확히 같은지 대조한다. 활동·관리·위험은 유지하며 원래 v26·v27을 포함한 열한 조건 또는 확인 실패 시 고정 BTC 진단으로 평가한다.

[v29](docs/EXPERIMENT_V29.md)의 최초 노출 분리는 `probe-entry-select --selection-run <v28 선택>`과 개발·확인 시장 경로로 실행한다. 최초 투입만 줄인 후보, 원래 v28, 최초 투입과 전체 한도를 함께 줄인 대조를 고정해 비교한다. 모델 재학습과 위험 비율 검색은 하지 않으며 원래 비교 정책은 원래 위험 설정으로 재생한다.

[v30](docs/EXPERIMENT_V30.md)의 관리 진단은 `management-model-diagnose --selection-run <v21 선택> --labels-run <v21 분봉 입력 정답>`으로 실행한다. 같은 시간순 학습·진단 행에서 기존 관리 모델을 재현하고 세 행동의 히스토그램 부스팅과 상수 기준을 비교한다. 고정 모형의 숫자 내보내기를 학습·진단 전체에서 대조하고 세 행동 모두의 사전 개선 조건을 요구한다. 이 단계에는 매매 실행이나 수익성 채택이 없다.

[v31](docs/EXPERIMENT_V31.md)은 `current-state-select --selection-run <v29 선택>`과 개발·확인 시장 경로로 실행한다. 같은 모델·위험·문턱·배율을 유지하고 누적 점수 대신 현재 점수로 관리 요청을 판단한다. 원래 v29 전체 출력을 대조하며 실제 복원 기록에 이전 정책의 누적량이 섞이면 중단한다. 관리 부스팅의 진단 실패는 그대로 보존한다.


[v32](docs/EXPERIMENT_V32.md)의 빈도 보정 진단은 `management-calibration-diagnose --selection-run <v21 선택> --labels-run <v21 정답> --diagnosis-run <v30 진단>`으로 실행한다. 2020년 학습·2021년 상반기 절편 보정·하반기 진단을 분리한다. 두 모형의 보정 전후와 기존 모델·상수를 대조하며 세 행동 모두의 개선 조건을 요구한다. 매매 후보 실행은 별도 계획 전까지 포함하지 않는다.


[v33](docs/EXPERIMENT_V33.md)의 과거 주문 맥락은 `order-history-diagnose --selection-run <v21 선택> --labels-run <v21 정답> --audit-run <원본 감사>`로 진단한다. 같은 포지션에서 판단 시각보다 앞선 독립 증가·축소 첫 체결만 이용해 여섯 입력을 추가한다. 기존 행·정답·모형 종류·시간 구간을 보존하며 세 행동 전체의 개선 조건을 요구한다. 이 단계에는 매매 실행이 없다.


[v34](docs/EXPERIMENT_V34.md)는 `order-history-boost-diagnose --history-run <v33 진단>`으로 실행한다. v33의 50개 입력과 모든 학습·진단 행을 유지하고 고정 부스팅 하나를 기존 두 로지스틱과 비교한다. v33 재학습·기준 예측을 먼저 재현하며 세 행동 모두 두 기준의 개선 조건을 통과해야 한다.


[v35](docs/EXPERIMENT_V35.md)는 `history-calibration-diagnose --history-run <v33 진단> --calibration-run <v32 진단>`으로 실행한다. v32의 세 시간 구간과 모든 기존 열을 유지한 채 검증된 과거 주문 특징 여섯 개를 연결한다. 새 보정 부스팅을 기존 v21·새 보정 로지스틱·v32 보정 부스팅 모두와 비교하며 마지막 구간의 정답은 학습에 사용하지 않는다.

[v36](docs/EXPERIMENT_V36.md)은 `history-state-select --selection-run <v31 선택> --diagnosis-run <v35 진단>`과 개발·확인 시장 경로로 실행한다. 사전 진단을 통과한 관리 모형과 절편을 그대로 복사하고, 상반기 F점수로 문턱을 고정한다. 봇 자신의 체결만 과거 이력에 반영하며 기존 진입·축소 크기·위험을 유지한다. 원래 v31 대조·실제 종료 복원·다년 후속 검증을 포함하고 실거래 연결은 제외한다.

[v37](docs/EXPERIMENT_V37.md)은 `first-management-diagnose --selection-run <v36 선택> --diagnosis-run <v35 진단>`으로 실행한다. 원래 청산·반복 행동 문턱을 유지하고 첫 추가·축소만 상반기 정답으로 분리한다. 하반기 첫 행동 재현율·F점수와 전체 F점수의 사전 조건을 두 행동 모두 통과해야 후속 매매 연구를 허용한다. 원본 상태 고정 진단이며 봇 수익률을 실행하지 않는다.

[v38](docs/EXPERIMENT_V38.md)은 `first-management-direct-diagnose --diagnosis-run <v37 진단>`으로 실행한다. 같은 상반기 첫 문턱·모델을 재현하고 첫 추가·축소에만 가산을 제거한다. 청산·반복 요청과 기존 진단 기준은 보존하며, 통과하더라도 별도 매매 계획 전에는 봇에 적용하지 않는다.

[v39](docs/EXPERIMENT_V39.md)은 `first-state-select --selection-run <v36 선택> --diagnosis-run <v38 진단>`과 개발·확인 시장 경로로 실행한다. 원래 모든 모델·위험을 유지하고, 실제 체결 이력이 없는 첫 추가·축소에만 검증된 문턱을 직접 적용한다. 청산·반복 요청의 기존 배율을 보존하고 원래 v36 전체 출력과 복원을 대조한다.

[v55](docs/EXPERIMENT_V55.md)는 `policy-outcome-labels --selection-run <v54 선택> --market <2021년 1분 시세> --features <5분 특징 시세>`로 현재 정책의 모든 독립 진입 손익을 생성한다. 중단한 실행은 같은 입력과 `--resume-run <정답 출력>`으로 재개한다. 완료 기회는 원자적으로 보존하며 입력·정책·코드가 바뀐 재개를 거부한다. `--max-opportunities`는 새로 계산할 기회 수만 제한하며 일부 완료 결과는 학습에 사용할 수 없다.

[v56](docs/EXPERIMENT_V56.md)는 `policy-entry-select --selection-run <v54 선택> --labels <완료한 v55 정답> --market <2021년 1분 시세> --features <5분 특징 시세> --confirmation-market <2022년 1분 시세> --confirmation-features <확인 5분 특징 시세>`로 실행한다. 현재 관리 정책의 전체 순손익을 가중 회귀로 학습해 신규 진입만 판단한다. 손실을 포함한 전체 정답과 읽기 전용 원장을 대조하고 겹친 구간의 비중을 조정한다. 관리·위험은 유지하며 필터 제거 대조가 원래 v54 전체 결과와 일치해야 한다. 학습 완료나 합성 검사 통과를 수익성 입증으로 표시하지 않는다.

[v57](docs/EXPERIMENT_V57.md)은 `entry-regression-diagnose --selection-run <v56 선택>`으로 실행한다. 같은 현재 정책 정답을 시간순으로 나누고 경계를 넘는 정답을 배제한 뒤, 각 구간 안에서만 중첩 가중치를 다시 계산한다. 앞 구간에서 학습한 가중 회귀·고정 부스팅·상수를 후속 구간에서 비교하며 모든 배제 사유와 방향·월별 결과를 보존한다. 진단 조건을 통과해도 실제 계좌 수익성이나 봇 적용을 뜻하지 않는다.

[v58](docs/EXPERIMENT_V58.md)은 `entry-stop-diagnose --diagnosis-run <v57 진단>`으로 실행한다. 원래 진단의 모든 행·가중치·모델·예측을 먼저 재현한다. 손절 확률 하나와 앞 구간의 손절·비손절 평균 손익을 결합해 세 금액 기준과 손절 빈도 상수에 비교한다. 실제 종료 사유는 정답에만 쓰고, 같은 집단 안 손익 크기를 상수로 근사한 한계를 기록한다.

[v59](docs/EXPERIMENT_V59.md)는 `close-effect-labels --selection-run <v54 선택> --market <2021년 1분 시세> --features <5분 특징 시세>`로 보유 유지와 청산의 순효과 정답을 생성한다. 같은 계좌 상태에서 다음 청산만 비교하고, 원래 전체 경로와 실제 판단 입력을 대조한다. `--max-opportunities`는 새 기회 처리 한도, `--resume-run`은 같은 입력의 재개다. 부분 완료나 양수 정답을 실행 가능한 새 봇 수익으로 표시하지 않는다.

[v60](docs/EXPERIMENT_V60.md)은 `close-learning-diagnose --labels <완료한 v59 정답>`으로 실행한다. 원래 전체 계좌·실제 입력·청산 현금 흐름과 저장 연결을 먼저 검증한다. 포지션과 시간을 분리하고 포지션별 총 비중을 같게 만든 뒤 청산 순효과의 회귀·부스팅·상수를 비교한다. 원래 청산도 학습·오차에서 보존하며, 여러 조기 청산 선택의 평균을 계좌 수익으로 합산하지 않는다.

[v61](docs/EXPERIMENT_V61.md)은 `close-calibration-diagnose --diagnosis-run <v60 진단>`으로 실행한다. 기존 진단 전체를 먼저 재현하고 앞 학습·중간 보정·뒤 진단을 분리한다. 단일 부스팅 금액에 순위를 뒤집지 않는 선형 보정을 적용하며 미보정 모델·Ridge·학습 및 보정 평균 상수와 비교한다. 후속 진단으로 보정 계수나 문턱을 고르지 않는다.

[v62](docs/EXPERIMENT_V62.md)는 `close-economic-diagnose --diagnosis-run <v60 진단>`으로 실행한다. 원래 진단 전체를 재현하고 현재 순자산 대비 보유 명목가치와 비용 포함 추정 청산 순손익 두 입력만 추가한다. 기존 행·50개 입력·정답·포지션 가중치·대조 모델은 보존한다. 실제 다음 체결 가격이나 미래 손익을 새 입력에 넣지 않는다.

[v63](docs/EXPERIMENT_V63.md)은 `first-close-diagnose --diagnosis-run <v62 진단>`으로 실행한다. 같은 원래 포지션에서 고정된 두 모델의 최초 추가 청산 선택만 기록하고 선택하지 않은 포지션은 효과 0으로 보존한다. 첫 이용 가능 판단의 순자산으로 비교 기준을 맞추고 같은 주간 블록으로 불확실성을 계산한다. 청산 뒤 재진입·계좌 복리·수익성 채택은 이 진단에 포함하지 않는다.

[v64](docs/EXPERIMENT_V64.md)은 `continuation-diagnose --diagnosis-run <v63 진단>`으로 실행한다. 원래 봇이 현재 예약한 유지·청산·축소·추가 요청을 명시적으로 구분하고 기존 52개 입력·행·가중치·대조 모델을 보존한다. 미래 체결 결과를 입력에 사용하지 않는다. 행 단위 오차·선택 효과와 포지션당 최초 효과를 함께 검사하고 같은 주간 블록으로 불확실성을 기록한다.

[v65](docs/EXPERIMENT_V65.md)은 `exposure-close-diagnose --diagnosis-run <v64 진단>`으로 실행한다. 현재 노출로 정답을 나누고 학습 가중치에 노출 제곱을 반영해 원래 계좌 오차의 상대 비중을 유지한다. 예측을 계좌 기준으로 복원한 뒤 기존 가중치·최초 선택·주간 비교와 열네 조건을 평가한다.

[v66](docs/EXPERIMENT_V66.md)은 `close-capacity-diagnose --diagnosis-run <v65 진단>`으로 실행한다. 앞 학습을 다시 시간순으로 나눠 고정한 네 복잡도를 비교하고, 내부 선택 오차가 가장 작은 설정 하나만 전체 앞 학습에 재적합한다. 마지막 진단은 선택에 사용하지 않으며 기존 모든 대조와 최초 효과·주간 비교·열여섯 조건을 보존한다.

[v67](docs/EXPERIMENT_V67.md)은 `weekly-close-diagnose --diagnosis-run <v66 진단>`으로 실행한다. 각 주 시작 이틀 전까지 종료가 확인된 결과만 누적해 같은 모델을 갱신한다. 첫 학습의 기존 모델 동일성, 13개 학습 가용 시각·포지션 분리·예측 연결, 기존 전체 평가 행과 열여덟 조건을 검증한다. 고정 봇 경로의 주간 예측 연구이며 연속 자체 상태의 온라인 매매 성과가 아니다.

[v68](docs/EXPERIMENT_V68.md)은 `first-close-margin-diagnose --diagnosis-run <v67 진단>`으로 실행한다. 기존 반년 학습 모델을 고정하고 다음 분기의 최초 청산 효과로 여섯 실행 문턱 중 하나를 선택한다. 지원·양수 효과·0bp 대비 개선이 없으면 새 마지막 진단을 차단한다. 모델 점수와 회귀 오차를 유지하며 마지막 최초 효과·주간 구간·열 가지 조건을 검증한다.

[v69](docs/EXPERIMENT_V69.md)은 `close-utility-diagnose --diagnosis-run <v68 진단>`으로 실행한다. 잘못된 청산 선택의 손익 크기를 분류 비용에 반영하고 고정 점수 0.5를 기준으로 선택한다. 0 효과 행을 포함한 원래 평가 자료를 보존하며 비용 지표·원래 가중 기회손실·최초 효과·주간 비교와 열네 조건을 검증한다. 출력은 비용 가중 선택 점수이며 수익 확률이나 예상 손익이 아니다.

[v70](docs/EXPERIMENT_V70.md)은 `close-context-diagnose --diagnosis-run <v69 진단>`으로 실행한다. 원래 56개 입력에 확정된 4·16·64일 시장 상태 여덟 개를 추가하고 같은 청산 비용·고정 점수 기준으로 비교한다. 원래 시세와 입력·손익 원장·학습 상수를 보존하며 두 주간 비교와 열아홉 조건을 검증한다.

[v71](docs/EXPERIMENT_V71.md)은 `close-flow-diagnose --diagnosis-run <v70 진단>`으로 실행한다. 기존 시세의 매수 체결 불균형과 포지션 방향을 반영한 열 개 입력을 추가한다. 원래 64개 입력과 비용·판정을 보존하며 동일한 추출의 세 주간 비교와 스물네 조건으로 평가한다.

[v72](docs/EXPERIMENT_V72.md)은 `weekly-flow-diagnose --diagnosis-run <v71 진단>`으로 실행한다. 매주 시작 이틀 전에 종료된 거래만 누적해 같은 74개 입력의 비용 분류기를 갱신한다. 첫 주 모델·비용·상수의 기존 동일성, 13개 학습과 예측 연결, 주간 상수 대조와 네 주간 비교·서른두 조건을 검증한다.

[v73](docs/EXPERIMENT_V73.md)은 `stopping-close-diagnose --diagnosis-run <v72 진단>`으로 실행한다. 같은 포지션을 학습에서 제외한 보조 정책의 미래 첫 청산 현금으로 정답을 만들고, 원래 고정 학습의 입력·행·비중을 유지한 후보 하나를 적합한다. 기존 자연 종료 대비 손익과 다섯 주간 비교·서른일곱 조건으로 평가하며 미래 값은 정답에만 사용한다.

[v74](docs/EXPERIMENT_V74.md)는 `minute-close-labels --labels-run <v59 원장>`으로 분별 청산 기회를 복원한다. 기존 5분 원장의 전체 행·정답을 그대로 대조하고, 누락된 분별 시점을 같은 계좌 상태·비용으로 계산한다. `--max-opportunities`와 `--resume`으로 중단·재개할 수 있으며 새 모델이나 수익성 평가는 포함하지 않는다.

[v75](docs/EXPERIMENT_V75.md)는 `minute-close-diagnose --labels-run <v74 원장> --diagnosis-run <v71 진단>`으로 실행한다. 확정 시세로 분별 입력을 연결하고 기존 모델을 재현한 뒤, 기존·새 학습 모델의 5분·1분 정책을 같은 포지션과 기준 순자산에서 비교한다. 후보는 새 1분 정책 하나이며 열다섯 조건과 기존 실패를 보존한다.

[v76](docs/EXPERIMENT_V76.md)은 `visited-close-diagnose --diagnosis-run <v75 진단>`으로 실행한다. 자기 포지션을 제외한 다섯 보조 모델의 최초 청산까지 학습 기여를 맞추고, 기존 전체 진단 원장에서 후보 하나와 기존 정책·상수를 비교한다. 원래 정답·학습 제외 행·평가 비중·실패 기록과 스무 조건을 보존한다.

[v77](docs/EXPERIMENT_V77.md)은 `retained-weight-close-diagnose --diagnosis-run <v76 진단>`으로 실행한다. 같은 보조 점수·방문 구간에서 유지한 원래 행의 상대 비중만 보존해 구간별 재비중의 영향을 분리한다. 기존 일곱 정책과 실패를 재현하고 후보·새 상수를 더한 아홉 정책과 스물네 조건을 비교한다.

[v78](docs/EXPERIMENT_V78.md)은 `close-threshold-diagnose --diagnosis-run <v77 진단>`으로 실행한다. 앞 학습 하나와 별도 보정 구간에서 일곱 비용 점수 문턱을 비교한다. 최초 청산 손익이 기본 문턱·무조건 첫 청산보다 높고 양수인 후보만 마지막 진단으로 넘어간다. 적격 후보가 없으면 새 진단 예측을 차단하고 실패·모든 보정 후보·기존 아홉 정책을 보존한다. 선택 성공 시 열세 정책과 열두 조건을 평가한다.

[v79](docs/EXPERIMENT_V79.md)은 `early-stopping-close-diagnose --diagnosis-run <v78 내부 선택 실패 진단>`으로 실행한다. 같은 앞 학습에서 포지션을 제외한 다섯 보조 모델의 미래 첫 청산 현금을 구성하고 현재 청산과 비교한 정답으로 후보 하나를 학습한다. 고정 0.5 문턱의 후보가 내부 보정을 통과한 경우에만 열네 정책·여섯 주간 비교·열두 조건을 평가한다. 기존 실패와 원래 모든 행을 보존하며 마지막 자료로 문턱이나 모델을 다시 선택하지 않는다.

[v80](docs/EXPERIMENT_V80.md)은 `stopping-regression-diagnose --diagnosis-run <v79 내부 보정 실패 진단>`으로 실행한다. 동일한 미래 첫 청산 현금 정답과 전체 행 비중으로 회귀기 하나를 학습하며 0 정답도 포함한다. 예측은 계좌 bp이고 문턱은 0bp다. 다섯 내부 보정 조건을 모두 통과한 경우에만 열다섯 정책·일곱 주간 비교·열세 조건을 평가하며 기존 실패를 유지한다.

[v81](docs/EXPERIMENT_V81.md)은 `first-opportunity-diagnose --diagnosis-run <v80 실패 진단> --verification <독립 검산 기록 JSON> --verification-sha256 <신뢰한 검산 기록 SHA-256>`으로 실행한다. 포지션의 첫 적격 기회만 동일 비중으로 학습하고 한 번만 판단한다. 기존 검산 근거를 지문으로 재사용하며 여덟 보정 정책·여섯 조건을 기록한다. 마지막 진단과 연속 매매 적용은 이번 단계에 포함하지 않는다.

[v82](docs/EXPERIMENT_V82.md)은 `first-linear-diagnose --diagnosis-run <v81 실패 진단> --verification <독립 검산 기록 JSON> --verification-sha256 <신뢰한 검산 기록 SHA-256>`으로 실행한다. 같은 첫 기회·현금·비용으로 표준화한 L2 로지스틱 모델 하나를 비교한다. 기존 여덟 정책을 보존하고 아홉 정책·일곱 내부 조건을 기록하며 마지막 진단과 매매 적용은 수행하지 않는다.
