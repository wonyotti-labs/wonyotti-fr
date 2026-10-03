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

v17은 기존 방향 판단과 대기·관리를 유지하고 신규 진입에 적용하던 노출 확대 활동 관문을 제거한다. 정답 생성에는 `--direction-only`, 학습에는 `--direction-only --overlap-weighted`를 추가해야 한다. 이전 기회 집합의 정답을 잘못 전달하면 거부한다. 후속 평가에는 원래 v7·v14와 새 기회에서 순손익 필터만 제거한 `unfiltered_direction_only`를 구분한 열 조건이 있다. 현재 확장 기회의 전체 거래를 계산 중이다.

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
