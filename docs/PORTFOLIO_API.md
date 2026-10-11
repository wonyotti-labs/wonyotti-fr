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
