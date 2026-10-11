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

## 과거 체결 이력의 예측 진단

[v101](EXPERIMENT_V101.md)의 `sparse_history.history_features(frame, source)`는 각 현재 시각에 이미 끝난 5분 창 12개만 조회한다. 양방향별 최근 행동의 존재·경과 시간·창 수를 여섯 특징으로 만든다. 창이 없거나 과거 분해가 미지원이면 0으로 채우지 않고 미지원 사유를 보존한다. 현재 정답 창과 미래 창은 특징에 포함하지 않는다.

`HistoryClassifier.fit`는 `sparse_history.FEATURES` 순서의 시장·현재 노출·과거 이력 21개 입력을 받는다. `training_support`·`admission`은 고정한 표본 수와 동일 행 대조·최근 무체결 구간의 양방향 적중 조건을 검사한다. `sparse_history_reference.verify_history`·`independent_refit`·`reference_admission`은 모든 과거 참조·후보 및 대조 트리·채택 조건을 독립 확인한다.

이 진단에는 원본 계좌 상태와 원본 과거 체결이 들어간다. 최근 1시간 무체결 구간은 봇의 실제 첫 진입과 같지 않다. [기존 실패](EXPERIMENT_V101_RESULTS.md)를 보존하며 자체 계좌 실행에 연결하지 않는다.

## 계약별 체결 기여분

[v102](EXPERIMENT_V102.md)의 `contract_components.decompose_contracts(source, events, wallet, book, specs)`는 기존 분해 행과 정규화한 원본 사건·지갑, `price_lookup`의 가격·활동 사전, 계약 명세를 받는다. 전체 원본 행에 세 계약군 기여분과 검산 잔차를 붙인 표, 사건이 있는 계약별 원가·순수량·평가액·민감도·기여분 표를 반환한다. 수수료와 펀딩을 이중 반영하지 않으며 순수량이 0인 왕복 거래도 보존한다.

`contract_components_reference.verify_contracts(source, saved, details, events, wallet, prices, activity, specs)`는 별도 정수 사건 누적·유리수 계좌·가격 차분으로 전체 결과를 대조한다. `summarize_contracts`·`reference_summary`는 전체·월별 계약군 분포와 부호 교차표를 독립 비교한다. 동일한 최종 평가액을 분모로 쓰는 회계 배분이며, 실제 반사실 계좌나 인과 기여도·새 매매 정답으로 자동 채택하지 않는다.

## 기록된 변경량의 자체 계좌 진단

[v103](EXPERIMENT_V103.md)의 `recorded_change_backtest.backtest(frame, source, config, output, target=..., timing=...)`는 공개된 계좌 기록의 변경량을 현재 봇 노출에 더하는 설명용 재생이다. `source`는 v102 전체 원본 행이며 추가 열·미지원·0 변경 행까지 저장한다. 표적은 `execution_change` 또는 `xbtusd_execution_change`, 공급 시각은 `window_start` 또는 `window_end`다. 창 시작은 `historical_lookahead=True`를 명시해야 하며 300초의 정보 선행을 판단 기록에 남긴다. 창 종료도 타인의 과거 기록을 공급받는 진단으로, 실시간 예측 전략이 아니다.

출력에는 `source.parquet`·`decisions.parquet`·지원과 호출 차이를 집계한 `coverage.json`, 기존 `ExposureEngine`의 `account/`가 포함된다. 0 변경·미지원·정확한 창 없음은 수량을 유지한다. 가격 손절 0 설정에서도 나머지 위험·시간·기간 종료 규칙은 그대로 적용된다.

`recorded_change_reference.verify_path(frame, source, config, output, review, target=..., timing=..., historical_lookahead=...)`는 별도 시각 검색·정책 계산과 현금 흐름 계좌로 모든 판단·분봉·체결·거래·지표, 두 중간 JSON 상태와 최종 상태를 대조한다. 전체 원본과 파일 목록·해시도 확인한다. 합성 검사의 통과와 실제 여덟 경로의 결과는 구분하며, 수익성 채택은 이 API의 기능이 아니다.

## 행동 창과 시장 관측 시각 정렬

[v104](EXPERIMENT_V104.md)의 `market_alignment.align_market(source, features, first, last)`는 v102 전체 원본 행을 보존하며 창 시작·종료 시각의 시장 특징을 정확히 연결한다. `first`·`last`는 UTC 경계이며 끝은 제외한다. `verify_alignment`는 별도 이진 검색으로 모든 연결·지원·사유를 대조한다. `aligned_matrix(frame, timing)`는 두 시각 모두 지원되는 동일 행의 시장 특징 14개만 반환한다.

`market_change_model.MarketChangeModel.fit(x, y, validation, timing=...)`는 XBTUSD 변경량의 세 행동 분류와 방향별 크기를 학습한다. `timing`은 `window_start` 또는 `window_end`이며 숫자 모형에 저장한다. 현재·종료 원본 노출이나 체결 이력을 받지 않는다. 원본 신호 없이 확정 시장 특징만 받는 `market_change_backtest.backtest(frame, model, config, output)`과 `verify_path(frame, model.to_dict(), config, output, review)`로 자체 계좌를 실행·검산한다.

검산은 14개 시장 열만 참조하도록 모형을 검증한 뒤 기존 독립 희소 계좌 검산기를 사용한다. 두 중간 JSON 계좌 상태와 최종 상태도 별도 방정식으로 대조한다. 공유 검산기의 중간 상태 검사는 명시적으로 전달하는 콜백으로 추가했으며 기존 호출의 출력 규약은 유지했다. 원본 신호 공급 진단과 달리 실행 시 원본 자료를 읽지 않는다. 구현 검사와 실제 수익성 검증은 구분한다.
# 희소 행동의 학습·선택 분리 API

`weighted_action_model.WeightedMarketClassifier.fit(x, labels, validation)`은 학습 클래스 빈도로만 역빈도 가중치를 계산한다. `WeightedActionModel`은 기존 크기 모형과 A~D 조건을 조립하며, JSON에 학습 형식·클래스 수·조건을 보존한다. B·C의 선택 점수는 원래 모집단의 발생 확률로 해석하지 않는다.

`weighted_action_evaluation.evaluate`는 네 조건의 행동·변경량과 A·D·빈도 대조의 Brier 점수·고정 신뢰도 구간을 평가한다. `weighted_action_backtest.backtest`와 `verify_path`는 자체 계좌 실행 및 별도 트리·계좌 재생을 제공한다. 실행 중 원본 신호나 원본 계좌는 입력하지 않는다. v105의 규약과 실제 성과는 별도 실험 기록을 따른다.

## 신규 진입과 관측 경계 지원

`entry_boundary_support.transition_ledger(events, actions)`는 정수 수량·행동·에피소드를 재생하고, 실제 무포지션 진입의 직전 공백과 1분·5분 경계 상태를 기록한다. 경계와 같은 시각의 체결은 직전 상태에 포함하지 않는다. `entry_support`는 신규 진입의 확정 시장 특징과 기존 정답 창을 정확한 시각으로 연결하고, `summarize_support`는 전체·연도·월별 사건·주문·지원 수를 반환한다.

`entry_boundary_reference`의 원본 정수 집계·별도 수량 재생·이진 검색·요약으로 모든 행을 검산한다. 최초 기록 이전의 무포지션 길이는 미상이며, 펀딩은 수량이나 공백 시작을 변경하지 않는다. 이 API는 기술 통계용이며 원본 상태를 실행 봇에 제공하지 않는다.

## 원거래소 전체 기간의 상태별 행동 정답

`state_action_windows.requested_orders(executions, ledger)`는 정상 XBTUSD 주문의 최초 체결에서 관측된 요청 수량과 직전 포지션으로 요청 행동을 구분한다. 요청 반전과 첫 부분 체결의 축소는 서로 다른 의미로 보존한다. `action_windows(features, ledger, orders, episodes)`는 모든 5분 창과 실제 체결·독립 요청의 두 원장을 반환한다. 창 시작의 엄격한 이전 상태, 시작·종료 특징, 다중 행동 수와 양성 여부, 시간·에피소드 배제 사유를 보존한다.

`state_action_windows_reference.verify_requests`·`verify_windows`·`independent_summary`는 원본 순회·별도 이진 검색·수동 집계로 모든 주문·창·특징·상태·정답·요약을 대조한다. 같은 시각의 후속 주문도 순서로 구분한다. 원본 상태는 오프라인 정답 구축용이며 실행 봇의 입력이 아니다. 새 모형이나 실제 체결 가격은 이 API의 범위에 포함하지 않는다.

`summarize_request_sequence`와 `independent_request_sequence`는 첫 독립 주문 뒤에 오는 같은 에피소드의 관리 요청을 전체·지원 구간으로 나누어 학습·진단·연도·월별로 집계한다. 동일 시각의 후속 주문도 포함하며 전체 정답 표를 다시 선택하거나 변경하지 않는다.

## 첫 주문과 복수 관리 정답의 같은 입력 비교

`paired_management.management_frame`은 v107 전체 표를 보존하며 엄격한 이전 원가·보유 시간·추가 주문 수와 창 시작의 원거래소 시세로 32개 입력을 구성한다. 첫 정상 주문과 모든 같은 에피소드 요청의 정답을 각각 보존한다. `split_support`는 학습·문턱 보정·진단을 24시간과 현재·관련 에피소드 경계로 분리한다.

`PairedManagementModel`은 고정된 독립 이진 부스팅 3개, `PairedManagementLinear`는 같은 입력의 표준화 선형 대조다. 선형 합산의 재현성을 위해 연속 C 배열과 단일 계산 스레드를 사용한다. `direct_requests`는 공통 활동 관문 없이 청산·축소·추가 요청과 우선순위 행동을 반환한다. 평가·문턱과 별도 원가 재생·전체 트리 및 선형 재학습은 `paired_management_evaluation`·`paired_management_reference`에 제공한다. 원본 상태 진단용이며 실제 매매 연결은 없다.

선형 대조의 요청 검산은 저장된 세 출력 동시 행렬 연산을 따른다. 라이브러리의 독립 단일 출력 확률은 1e-12 이내 수치 일치를 확인하며, 그 작은 차이로 문턱과 같은 요청을 다시 분류하지 않는다. 숫자 모형·문턱 자체를 반올림하거나 변경하지 않는다.


## 고정 관리 정책의 입력 지원 범위

`management_support.ManagementInputTrace`는 부모의 실제 관리 입력을 관찰하고 같은 판단을 반환한다. `training_profile`과 `SupportCounts`는 학습 경계와 비교 집계를 분리한다. `python -m wonyotti_fr.management_support_research --reference <로컬 출처 JSON> --sha256 <SHA-256>`은 고정 v48 두 계좌의 기존 출력 전체와 비교하고 새 폴더에 입력 기록·집계·지문을 저장한다. 새 모델 학습이나 실거래 기능은 없다. [계획](EXPERIMENT_V113.md)과 [결과](EXPERIMENT_V113_RESULTS.md)를 참조한다.


## 관리 점수와 반복 문턱의 고정 비교

`python -m wonyotti_fr.management_score_research --reference <v113 검산 JSON> --sha256 <SHA-256>`은 원본 세 구간과 두 봇의 봉인 입력에서 같은 모델을 재계산한다. 반복 가산의 유무, 행동 우선순위·수량·대기 영향을 분리하며 새 모델이나 계좌 수익을 만들지 않는다. 별도 말단 경로·문턱·집계 검산을 포함한다. [계획](EXPERIMENT_V114.md)과 [결과](EXPERIMENT_V114_RESULTS.md)를 참조한다.


## 관리 모델의 거래소 시장 입력 정렬

`management_market.connect_market`은 기존 50개 입력 중 시장 관측 36개만 교체하고 원본 상태·정답을 보존한다. `python -m wonyotti_fr.management_market_research --reference <고정 출처 JSON> --sha256 <SHA-256>`은 원래 BitMEX 입력을 재현하고 Binance 입력의 시각·공동 지원을 별도 계산으로 대조한다. 모든 원본 행과 제외 사유를 저장한다. 현재 명령은 [v116 계획](EXPERIMENT_V116.md)의 입력 검산 단계이며 새 모델 적합이나 매매 실행을 수행하지 않는다.


## 반복 관리 가산 제거의 자체 계좌

`RepeatDirectPolicy`는 고정 v48의 반복 추가·축소 문턱만 직접 적용한다. `ReferenceRepeatPolicy`는 원래 부모의 문턱 호출을 감싸 별도 계좌 검산에 사용한다. `python -m wonyotti_fr.repeat_direct_research --reference <고정 입력 JSON> --sha256 <SHA-256>`은 v114의 선행 통과를 확인하고 다섯 계좌와 비용·위험·집중 집계를 새 폴더에 저장한다. [계획](EXPERIMENT_V115.md)과 [결과](EXPERIMENT_V115_RESULTS.md)를 참조한다.


## 같은 행의 거래소 관리 모형 학습

`python -m wonyotti_fr.management_market_learning --reference <v116 입력 검산 JSON> --sha256 <SHA-256>`은 검산된 공동 지원 행에서 원본·정렬 부스팅과 정렬 선형 모형을 적합한다. 상반기 절편·문턱 보정 뒤 하반기와 월별 지표를 평가하며, 별도 재학습의 384개 트리·선형 계수·전체 확률·선택·판정을 대조한다. 진단 자료는 모형·절편 적합에 사용하지 않는다. 기존 모델과 원본 입력·정답·제외 행은 보존하며 자체 계좌 매매 실행은 포함하지 않는다.


## 원본 관리 행동의 체결 방향

`python -m wonyotti_fr.management_flow_research --reference <v116 입력·모형 완료 참조 JSON> --sha256 <신뢰한 SHA-256>`은 원본 전체 행과 기존 50개 입력을 보존하고 열두 체결 입력·두 해상도 매수량·지원 및 독립 창 합을 검산한다. 실패 시 새 폴더에 원인을 기록한다.

`python -m wonyotti_fr.management_flow_learning --reference <입력 완료 검산 JSON> --sha256 <신뢰한 SHA-256>`은 전체 지원·원본 연결을 확인한 뒤 고정 부스팅·선형을 학습하고 원본 v35와 대조한다. 모든 확률·절편·문턱·월별 지표·18개 조건을 검산하며 자기 계좌나 실제 주문은 실행하지 않는다. [계획](EXPERIMENT_V117.md)과 [입력 결과](EXPERIMENT_V117_INPUTS.md), [모형 결과](EXPERIMENT_V117_RESULTS.md)를 참조한다.


## 기존 진입 대기와 보유 시간

`python -m wonyotti_fr.entry_path_research --reference <v115 완료 검산 JSON> --sha256 <신뢰한 SHA-256>`은 기존 다섯 계좌를 읽고 대기·실제 진입·보유 분말·전체 월을 별도 계산으로 대조한다. 원래 계좌 파일은 변경하지 않고 새 폴더에 사본과 집계·검산 증거를 저장한다. 새 모델이나 계좌를 실행하지 않는다. [계획](EXPERIMENT_V118.md)과 [결과](EXPERIMENT_V118_RESULTS.md)를 참조한다.


## 첫 추가·축소의 분리 학습

`python -m wonyotti_fr.first_state_learning --reference <기존 경로·원본 모형 완료 참조 JSON> --sha256 <신뢰한 SHA-256>`은 기존 전체 분할을 보존하고 첫 상태 두 모형·보정·문턱을 비교한다. 독립 재적합·모든 확률·월별 지표·보존 요청과 21개 조건을 검산한다. 기존 `first_state_research` 계좌 실행과 구분하며 새 계좌나 실제 주문은 실행하지 않는다. [계획](EXPERIMENT_V119.md)과 [결과](EXPERIMENT_V119_RESULTS.md)를 참조한다.


## 추가·축소 체결 이력 조합

`python -m wonyotti_fr.joint_history_learning --reference <v119 완료 검산 JSON> --sha256 <신뢰한 SHA-256>`은 원본 50개에 네 이력 조합 표시를 추가한 전체 모델을 비교한다. 세 분할·모든 확률·첫/전체 문턱·네 조합·월별 지표·25개 조건을 독립 검산하며 실제 주문이나 새 계좌는 실행하지 않는다. [계획](EXPERIMENT_V120.md)과 [결과](EXPERIMENT_V120_RESULTS.md)를 참조한다.


## 관리 모델의 시간순 내부 크기 선택

`python -m wonyotti_fr.management_capacity_learning --reference <v120 완료 검산 JSON> --sha256 <신뢰한 SHA-256>`은 2020년 내부에서만 네 고정 크기를 비교하고 하나를 선택해 전체 2020년으로 재학습한다. 원래 모델과 선형 대조, 2021년 보정·진단, 모든 월·첫/반복·네 이력 조합을 보존한다. 내부 1,920개 트리와 선택한 전체 모델을 별도 재학습·숫자 추론으로 대조하며 내부 분할의 제외 행·네 후보의 전체 점수·25개 후속 조건을 기록한다. 실제 계좌나 주문은 실행하지 않는다. [계획](EXPERIMENT_V121.md)과 [결과](EXPERIMENT_V121_RESULTS.md)를 참조한다.


## 관리 행동의 이진 빈도 균형 학습

`python -m wonyotti_fr.balanced_management_learning --reference <v121 완료 검산 JSON> --sha256 <신뢰한 SHA-256>`은 원래 50개 입력과 깊이 2·256회를 유지하고 각 행동의 2020년 양성·음성 학습 비중만 균등화한다. 원래 빈도로 보정·평가하며 대조의 전체 확률·요청·월별 지표를 재현한다. 행별 가중치와 새 768개 트리·모든 확률·첫/전체 문턱·25개 조건을 독립 검산한다. 실제 계좌나 주문은 실행하지 않는다. [계획](EXPERIMENT_V122.md)과 [결과](EXPERIMENT_V122_RESULTS.md)를 참조한다.


## 일간 변동성 단위의 보유 가격 경로

`python -m wonyotti_fr.volatility_path_learning --reference <v122 완료 검산 JSON> --sha256 <신뢰한 SHA-256>`은 기존 50개 입력에 방향 보정 로그 가격 경로를 일간 변동성으로 나눈 네 값을 추가한다. 같은 크기의 무가중 256회 모델과 새 54개 입력 선형 대조를 비교하고 기존 세 모형의 전체 확률·요청·문턱·월 지표를 재현한다. 새 입력·768개 트리·선형 계수·학습 전용 변동성 구간·31개 조건을 독립 검산하며 실제 계좌나 주문은 실행하지 않는다. [계획](EXPERIMENT_V123.md)과 [결과](EXPERIMENT_V123_RESULTS.md)를 참조한다.


## 보유 극값 갱신 후 경과 시간

`python -m wonyotti_fr.extrema_age_learning --reference <v123 완료 검산 JSON> --sha256 <신뢰한 SHA-256>`은 전체 원본 분별 경로에서 최선·최악 종가의 엄격한 갱신 후 경과 시간을 복원한다. 기존 50개 입력과 정답을 보존하고 두 시간 입력·52개 입력 모형·선형 대조를 비교한다. 전체 경로의 배치/상태 갱신, 원래 세 모형의 점수·요청·월 지표, 새 모델·시간 구간·31개 조건을 검산한다. 실제 계좌나 주문은 실행하지 않는다. [계획](EXPERIMENT_V124.md)과 [결과](EXPERIMENT_V124_RESULTS.md)를 참조한다.

## v125 고정 관리 요청의 시간 진단

`request_timing.audit_action(frame, request, label)`은 UTC 분 경계·연속 포지션과 이진 요청·정답을 검증하고 정확한 분 혼동 수, FP의 앞뒤 14분 분류, 정확한 분을 먼저 고정한 최대 일대일 대응, 3분 간격 군집을 반환한다. 군집은 실제 대기 정책 재생이 아니다. `request_timing_reference.verify_action`은 별도 증대 경로 최대 대응과 직접 시간 차이·순회로 모든 집계 및 원장을 검사한다.

```sh
python -m wonyotti_fr.request_timing_research --reference <v124_완료_증거_JSON> --sha256 <신뢰한_SHA256> --output <로컬_출력_폴더>
```

실행은 새 폴더에서만 출력하며 v124·교사 자료의 봉인과 원본 행·기존 문턱 통과·혼동 수, 실행 전후 입력·코드 지문을 검증한다. 학습·주문 실행·수익성 판정은 포함하지 않는다.

## v126 최근 관리 행동 부재 상태의 분리 학습

`quiet_management.quiet_mask`는 기존 과거 15분 독립 첫 체결 로그 횟수에서 행동별 quiet 상태를 고르고 유한·비음수·정수 복원을 검증한다. `fit_quiet_models(training, calibration)`은 진단 자료를 받지 않고 축소·추가의 256회 부스팅·선형, 절편·문턱을 적합한다. `quiet_predictions`는 quiet만 새 모델로 바꾸고 청산과 recent 상태의 기존 native 점수·요청을 보존한다. capacity 대조도 quiet에서만 교체한다.

```sh
python -m wonyotti_fr.quiet_management_learning --reference <v125_완료_증거_JSON> --sha256 <신뢰한_SHA256> --output <로컬_출력_폴더>
```

실행은 원본 분할·파일 봉인과 v125→v124 출처를 확인하고, 새 512개 트리·선형·모든 확률·월별 지표·27개 조건을 독립 검산한다. 새 계좌 실행과 주문 연결은 포함하지 않는다.

## v127 최초·재시작 상태별 보정

`quiet_state_calibration.fit_state_calibration(calibration)`은 v126 고정 점수의 first/restart 각각에서 부스팅·선형 절편과 문턱을 계산한다. 문턱만 분리하는 대조도 함께 반환한다. `state_predictions`는 quiet의 보정 점수와 요청만 바꾸고 recent·청산·기타 대조를 유지한다. `quiet_state_reference`는 별도 절편·문턱 계산, 전체 확률·요청·보존·평균·31개 판정을 검산한다.

```sh
python -m wonyotti_fr.quiet_state_learning --reference <v126_완료_증거_JSON> --sha256 <신뢰한_SHA256> --output <로컬_출력_폴더>
```

실행은 v126 모델을 복원해 전체 원시·기존 보정 점수와 요청·월별 지표를 재현한다. 새 모델 학습·자체 계좌 실행·실거래 연결은 포함하지 않는다.

## v128 최초·재시작 관리 모델의 개별 학습

`management_experts.fit_experts(training, calibration)`은 축소·추가의 first/restart 각각에서 부스팅·선형·절편·문턱·상태별 빈도 상수를 적합한다. `load_expert`는 행동·상태·모델 형식의 연결을 검증한다. `expert_predictions`는 quiet에 해당 모델을 적용하고 청산·recent 및 native/capacity 대조를 보존한다.

```sh
python -m wonyotti_fr.management_experts_learning --reference <v127_완료_증거_JSON> --sha256 <신뢰한_SHA256> --output <로컬_출력_폴더>
```

실행은 v126/v127 모델·보정·전체 예측·월별 결과를 재현하고 새 1,024개 트리·4개 선형·모든 지표·31개 조건을 독립 검산한다. 자체 계좌와 실거래 연결은 포함하지 않는다.

## v129 관리 점수의 동점 순위와 문턱 이동

`management_rank_audit.tail_cutoffs(scores)`는 정답 없이 고정 비율의 상위 점수 절단점을 계산한다. `audit_scores(frame, threshold, calibration_cutoffs)`는 기존 요청·양성 동점 순위·고정/사후 절단점·포지션 집중도와 로컬 원장을 반환한다.

```sh
python -m wonyotti_fr.management_rank_research --reference <v128_완료_증거_JSON> --sha256 <신뢰한_SHA256> --output <로컬_출력_폴더>
```

`management_rank_reference.verify_score_audit`는 별도 점수 빈도·누적 수·혼동 집계·분위수 보간으로 대조한다. 사후 절단점은 미래 점수를 포함하는 진단이며 실행 정책으로 사용하지 않는다.
