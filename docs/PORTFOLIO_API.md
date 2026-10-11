# 동시 계약 계좌 평가 API

v96의 가격 처리·계좌 평가·독립 검산은 `wonyotti_fr` 패키지에서 호출한다. 이 단계는 원본 계좌의 BTC 가격 민감도를 재구성하며 새로운 주문을 생성하지 않는다. 역사적 계약 승수는 관찰한 체결에서 구하고 전체 체결 원가와 대조한다. 현재 거래소 상품 명세를 과거 계약에 소급하지 않는다.

## 가격 준비와 검산

`portfolio_prices.required_prices(executions, points)`는 각 조회 시각보다 앞선 체결로 계약별 보유 수량을 구하고 필요한 가격 경계를 반환한다. 담보 환산용 XBTUSD를 포함한다. `price_requests(needed)`는 BTC 외 계약의 연속 구간을 최대 1,000봉씩 나눈다. `retained_xbt_prices(bars, points)`는 기존 BTC 5분 봉의 정확한 경계를 선택하며 누락을 채우지 않는다.

요청 목록의 JSON과 신뢰한 SHA-256을 준비한 뒤 다음과 같이 공개 과거 종가를 수집한다. 같은 출력 폴더와 같은 요청 목록으로 중단된 수집을 재개할 수 있다. 완료된 원응답은 지문을 확인하고 재사용한다.

```sh
wonyotti portfolio-prices-collect --requests <요청_JSON> --requests-sha256 <신뢰한_SHA256> --output <새_로컬_수집_폴더>
```

수집 결과의 `prices.parquet`는 `symbol`, UTC `end`, `close`, `support` 열을 갖는다. 응답이 없거나 유효한 종가가 없으면 해당 행과 미지원 사유를 남긴다. 수집 원장의 종가는 실제 거래가 없는 봉에서도 존재할 수 있다. `portfolio_price_activity.activity_record`와 `boundary_activity`로 필요한 모든 계약의 거래 수·수량을 확인하고 무거래 경계를 별도로 표시한다.

`portfolio-prices-verify --collection <v96_수집_근거_폴더> --final-sha256 <collection_final_verification.json의_신뢰한_SHA256>`는 v96의 전체 근거 구조를 검산한다. 이 구조는 수집 하위 폴더 외에 `needed.parquet`, `requests.json`, `retained_xbt_prices.parquet`, `retained_xbt_reference.json`, `all_prices.parquet`, `price_preparation.json`, `collection_final_verification.json`을 포함한다. 단독 수집 명령의 출력만으로는 이 전체 검산을 실행할 수 없다. 원본에서 정한 필요 범위·보존 BTC·수집 원응답의 연결을 각각 준비해야 한다.

## 계좌 평가와 독립 대조

실행 원장에는 `time`, `symbol`, `exectype`, `side`, `lastqty`, `lastpx`, `execcost`, `execcomm`, `currency`, `settlcurrency`가 필요하다. 수량·원가는 원본 정수 단위를 유지한다. 입출금 원장 형식은 v95 `collateral_direction`과 같으며 `RealisedPNL`을 외부 입출금으로 더하지 않는다. `points`는 중복 없는 UTC 조회 시각이다. `prices`는 보존 BTC와 수집 가격을 결합한 위 형식이다.

```python
from wonyotti_fr.portfolio_valuation import contract_specs, portfolio_ledger
from wonyotti_fr.portfolio_valuation_reference import independent_contract_check, independent_values
from wonyotti_fr.portfolio_valuation_checks import verify_values

specs = contract_specs(executions, prices.symbol.unique())
independent_contract_check(executions, specs)
for extra_days in (0, 1):
    ledger = portfolio_ledger(executions, wallet, points, prices, specs, extra_days=extra_days)
    reference = independent_values(executions, wallet, points, prices, specs, extra_days=extra_days)
    proof = verify_values(ledger, reference)
```

기본·확장 입출금 날짜 구간은 시각이 없는 원본에 대한 가정이다. 평가에는 수수료·펀딩과 열린 동시 계약을 포함한다. BTC 부분 민감도는 다른 USD·USDT 가격과 BTC 표시 계약의 기준 가격을 고정한 값이다. 전체 다자산 수익률이나 BTC 전용 봇의 정답과 같다고 가정하지 않는다. 거래 종가는 당시 마크 가격을 복원한 값이 아니며 ±1% 가격 대조도 보장된 오차 한계가 아니다. 가격 누락·지원되지 않는 계약·날짜 불확실성을 그대로 유지한다.

검산된 전체 행과 월별 결과는 [v96 결과](EXPERIMENT_V96_RESULTS.md)에 보존한다. API 등록과 기존 결과 재현은 수익성 채택을 의미하지 않는다.

## 순노출 정답과 회귀 진단

[v97](EXPERIMENT_V97.md)의 `net_exposure_targets.net_exposure_targets(source, comparisons, part)`는 원본의 현재·다음 경계를 담보 포함 BTC 부분 노출 정답에 연결한다. `source`는 v44의 원래 행과 14개 확정 시장 입력, `comparisons`는 위 평가 행에 거래 활동을 연결한 v96의 확장 입출금 날짜 구간, `part`는 `training` 또는 `diagnosis`다. 날짜 불확실·무거래 가격·비양수 계좌는 행을 보존한 채 미지원 사유로 남긴다.

`net_exposure_model.NetExposureModel.fit(x, y, validation)`과 `fit_ridge`는 고정한 부스팅·선형 대조를 학습한다. `x`는 `event_features.MARKET_FEATURES` 순서의 시장 입력만 사용하며 원본 현재 노출을 섞지 않는다. 모형은 `to_dict`와 `from_dict`의 숫자 트리로 저장·복원한다. `net_exposure_reference`의 `independent_target_check`·`independent_model_check`·`independent_metrics`는 정답 연결·독립 재학습·전체 예측과 지표를 검산한다. 현재 원본 노출 유지 대조는 실행 가능한 봇의 입력이 아니다.

2020년 학습·2021년 진단의 기존 실패와 모든 월을 보존하며, 예측 오차 감소를 수익성으로 판정하지 않는다. 기존 모형의 실제 계좌 적용과 손실은 [v98 결과](EXPERIMENT_V98_RESULTS.md)에 별도로 기록한다.

## 노출 변경과 자체 계좌 실행

[v98](EXPERIMENT_V98.md)의 `exposure_change_targets.change_targets`는 현재·다음 순노출 정답의 차이와 양쪽 지원 여부를 보존한다. `exposure_change_model.ExposureChangeModel`은 시장 입력 14개와 현재 노출 한 개를 받으며, `predict_one`으로 동일한 숫자 트리를 단일 상태에서 계산한다. 이 변경량은 가격·펀딩 등에 의한 변화도 포함한다. 체결 의도와 같다고 가정하지 않으며 후속 v99·v100의 정답 분해 결과도 함께 확인해야 한다.

`exposure_policy.ExposurePolicy`의 `change`·`absolute`는 각각 변경량·절대 노출 모형을 받는다. `change`는 봇 자신의 잔고·수량·확정 종가로 현재 노출을 구한다. `constant` 대조는 `constant_exposure`를 반드시 외부에서 전달한다. 학습 자료에서 얻은 상수나 모형을 공개 소스에 포함하지 않는다. 공개 테스트는 합성 상수를 사용한다.

`exposure_engine.ExposureConfig`에 [실행 명세](EXPERIMENT_V98_EXECUTION.md)의 비용·낙폭·보유·대기 설정을 명시한 뒤 `exposure_backtest.backtest(frame, policy, config, output)`으로 새 로컬 출력 폴더에 실행한다. 입력은 `minute_data.prepare_minute_period`가 반환하는 시세·펀딩·확정 특징과 같은 형식이다. `ExposureEngine`은 다음 시가 체결·부분 감소·반전·위험 청산·무거래 대기·JSON 상태 복원을 처리한다.

`exposure_backtest.verify_path(frame, kind, model_dict, config, output, review, constant_exposure=...)`는 모든 분봉·체결·거래·목표·최종 잔고를 별도 수량 방정식과 현금 흐름으로 검산한다. `model_dict`는 모형의 `to_dict()`이며 상수 대조는 `None`이다. 상수 대조의 값은 검산에도 동일하게 명시해야 하며 다른 정책에는 생략한다. 저장된 정책 설정과 다르면 검산을 거부한다. 이 실행기는 오프라인 연구용이며 실제 주문 연결은 없다.

## 가격·자금·체결의 변경량 분해

[v99](EXPERIMENT_V99.md)의 `exposure_components.account_snapshots`는 모든 원본 사건을 조회 시각 직전까지 누적한다. `price_lookup`에 실제 가격과 거래 활동을 함께 전달하고, `decompose(source, snapshots, book, specs)`로 v98 원본 행을 분해한다. 가격·외부 입출금·펀딩·수수료·체결 순으로 평가하며 각 단계의 평가액·BTC 민감도·노출과 다섯 기여분·사건 수·미지원 사유를 보존한다. 체결이 없는 구간의 가격 변화도 기록한다.

`exposure_components_reference.reference_snapshots`·`reference_prices`·`verify_decomposition`은 별도 현금·평균 원가·가격 차분으로 모든 단계와 사건 수를 대조한다. `independent_statistics`와 `summarize_components`는 전체·월별 분모와 기여분을 비교한다. 이 분해는 고정한 단계 순서의 회계 항등식이며 유일한 인과 기여도가 아니다. 가격이 없는 행도 보존하고, v100 가격 보완 전후의 자료를 구분한다.

## 희소 체결 행동과 자체 계좌 정책

[v100](EXPERIMENT_V100.md)의 `sparse_exposure.SparseExposureModel.fit(x, y, validation)`은 감소·거래 없음·증가 분류기와 방향별 크기 회귀를 함께 학습한다. `x`는 시장 특징 14개와 현재 노출, `y`는 지원되는 회계 분해의 `execution_change`다. 학습·진단의 원본 상태와 연속 실행의 봇 상태를 구분하며, 모형과 학습 상수는 로컬 숫자 JSON으로 저장한다. `sparse_exposure_reference.independent_refit`와 `reference_metrics`는 전체 트리·예측·지표를 별도 계산한다.

`sparse_exposure_policy.SparseExposurePolicy(model)`을 `sparse_exposure_backtest.backtest(frame, policy, config, output)`에 전달한다. 최대 확률 동률과 거래 없음·숫자상 0 크기는 `None`을 반환해 수량을 유지한다. 방향 행동은 자체 계좌 노출에 예측 변경량을 더하고 [-0.5, 0.5]로 제한한다. 위험 청산·다음 시가 체결·비용·펀딩은 `ExposureEngine`을 사용한다.

`sparse_exposure_backtest.verify_path(frame, "sparse_execution", model.to_dict(), config, output, review)`는 전체 분봉뿐 아니라 각 판단 시각의 확률·크기·행동·목표·미지원 사유를 독립 재생한다. 입력과 새 출력 폴더의 규약은 v98과 같다. 이 API 등록은 후보 채택이 아니다. [기존 결과](EXPERIMENT_V100_RESULTS.md)의 두 자체 계좌는 거래가 없어 수익성 기준을 충족하지 못했다.
