# 실험 v139 결과: 관리 요청과 후속 행동의 일대일 대응

[사전 계획](EXPERIMENT_V139.md)에 따라 기존 요청과 같은 포지션·집단의 보존된 양성 분을 미래 방향으로만 연결했다. 새 모델·문턱·매매 정책·계좌는 없다. 수익성 문제는 미해결이다.

## 해석

한도 전 재시작 추가의 실패는 작은 시점 차이만으로 설명되지 않았다. 하반기 v137 기존 요청은 19분이며, 정확한 1분 대응은 1분, 이후 15분 안의 일대일 대응은 2분이다. 대응의 선행 시간은 0분과 2분이고 17분은 보존된 목표와 연결되지 않았다. 목표 35분 중 포착은 2분(5.7143%), 요청 대비 대응은 10.5263%다. 19개 요청의 15분 관측 창은 모두 원래 표에 있어 이 집단에서 관측 경계 부족으로 대응이 줄어든 것은 아니다. 이 결과를 전체 실제 주문이나 운용 수익의 확정 오신호 판정으로 확대하지 않는다.

공통 오신호 예산 요청에서는 하반기 두 모델 모두 요청 1분·목표 35분·15분 대응 0분이다. v138에서 사라진 후보의 새 적중을 시간 허용 범위로 복구하지 못했다. 하반기 기준 모델의 기존 요청도 1분 중 대응 0분이다. 요청 시점을 바꾸거나 미래 원본 행동을 운용 중 기다려서 선택한 정책은 아니다.

하반기 후보의 기존 요청·15분 대응·미연결은 7월 4/0/4, 8월 3/1/2, 9월 12/1/11, 10월 0/0/0이다. 공통 예산 조건은 9월에만 요청 1분이 남고 대응은 0분이다. 11·12월은 해당 집단의 행·목표가 없으며 이를 성공한 무거래 성과로 취급하지 않는다. 두 하반기 대응은 각각 같은 월 안에서 연결됐다.

중복을 허용하면 앞선 기간의 대응이 과장된다. 같은 집단의 상반기 후보 기존 요청 148분 중 이후 목표가 하나라도 있는 요청은 43분이지만, 일대일 대응은 12분이다. 목표 60분 중 포착 12분이며 미연결 요청은 136분이다. 기준 기존 요청은 18분 중 6분 대응이다. 공통 예산 조건은 후보 17분 중 3분, 기준 18분 중 6분 대응이다. 학습 기간 후보 기존 요청도 237분 중 목표가 있는 요청은 89분이지만 일대일 대응은 43분이다. 동일 행동 앞 반복 요청을 각각 독립 적중으로 세지 않아야 한다.

다른 상태의 선행 대응은 남아 있다. 하반기 quiet 축소의 기존 요청·15분 대응은 기준 16/4, 후보 118/11이며 미연결은 12분과 107분이다. 공통 예산 조건은 기준 17/4, 후보 33/6으로 후보의 포착 증가와 요청 증가가 함께 나타났다. quiet 추가의 기존 요청·대응은 기준 12/4, 후보 70/6이고, 공통 예산 조건은 기준 22/4, 후보 23/4다. 시간 범위를 넓혔다는 이유로 후보가 같은 비용에서 더 잘 구분한다고 판단할 수 없다. 상태·횟수 집단은 겹치므로 이 수들을 더하지 않는다.

v137의 미채택과 v138의 낮은 오신호 수준에서 새 적중 미유지 결론을 변경하지 않는다. 이번 범위에서 단순한 평가 시간 허용은 핵심 재시작 추가 문제를 해소하지 못했다. 정답 범위를 넓혀 성적을 다시 이름 붙이는 방법을 해결책으로 채택하지 않는다. 다음 학습 변경은 시간 이동보다 낮은 오신호 수준에서 양성과 음성을 구분하는 능력을 개선하는 가설에 근거해야 한다. 그 변경도 별도 계획과 시간순 비교가 필요하며, 새 실행 손익 검증과 v10 최종 기준을 대체할 수 없다.

## 한도 전 재시작 추가의 전체 기간

| 기간·월 | 행동/상태/횟수 | 모델 | 요청 조건 | 범위 분 | 지원 | 요청 분 | 기존 정확 적중 | 목표 분 | 대응 요청 | 포착 목표 | 미연결 요청 | 미포착 목표 | 요청 대비 대응 | 목표 포착 비율 | 목표가 있는 요청 | 요청이 있는 목표 | 관측 창 완전 | 기간 끝 초과 | 선행 0분부터 순서대로 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training | increase__restart__below | v127 | original | 1 | supported | 37 | 7 | 214 | 7 | 7 | 30 | 207 | 0.1891891892 | 0.03271028037 | 7 | 7 | 37 | 0 | 7 |
| training | increase__restart__below | v127 | original | 15 | supported | 37 | 7 | 214 | 15 | 15 | 22 | 199 | 0.4054054054 | 0.07009345794 | 19 | 15 | 37 | 0 | 5,6,0,2,0,0,1,0,0,0,0,0,0,1,0 |
| training | increase__restart__below | v127 | budget_matched | 1 | supported | 38 | 7 | 214 | 7 | 7 | 31 | 207 | 0.1842105263 | 0.03271028037 | 7 | 7 | 38 | 0 | 7 |
| training | increase__restart__below | v127 | budget_matched | 15 | supported | 38 | 7 | 214 | 15 | 15 | 23 | 199 | 0.3947368421 | 0.07009345794 | 19 | 15 | 38 | 0 | 5,6,0,2,0,0,1,0,0,0,0,0,0,1,0 |
| training | increase__restart__below | v137 | original | 1 | supported | 237 | 29 | 214 | 29 | 29 | 208 | 185 | 0.1223628692 | 0.1355140187 | 29 | 29 | 237 | 0 | 29 |
| training | increase__restart__below | v137 | original | 15 | supported | 237 | 29 | 214 | 43 | 43 | 194 | 171 | 0.1814345992 | 0.2009345794 | 89 | 43 | 234 | 0 | 15,4,4,1,7,1,0,0,0,2,0,1,1,1,6 |
| training | increase__restart__below | v137 | budget_matched | 1 | supported | 23 | 6 | 214 | 6 | 6 | 17 | 208 | 0.2608695652 | 0.02803738318 | 6 | 6 | 23 | 0 | 6 |
| training | increase__restart__below | v137 | budget_matched | 15 | supported | 23 | 6 | 214 | 13 | 13 | 10 | 201 | 0.5652173913 | 0.06074766355 | 14 | 13 | 22 | 0 | 6,2,1,1,0,0,1,0,0,0,0,1,0,1,0 |
| calibration | increase__restart__below | v127 | original | 1 | supported | 18 | 3 | 60 | 3 | 3 | 15 | 57 | 0.1666666667 | 0.05 | 3 | 3 | 18 | 0 | 3 |
| calibration | increase__restart__below | v127 | original | 15 | supported | 18 | 3 | 60 | 6 | 6 | 12 | 54 | 0.3333333333 | 0.1 | 10 | 6 | 18 | 0 | 1,1,0,0,1,1,0,1,0,1,0,0,0,0,0 |
| calibration | increase__restart__below | v127 | budget_matched | 1 | supported | 18 | 3 | 60 | 3 | 3 | 15 | 57 | 0.1666666667 | 0.05 | 3 | 3 | 18 | 0 | 3 |
| calibration | increase__restart__below | v127 | budget_matched | 15 | supported | 18 | 3 | 60 | 6 | 6 | 12 | 54 | 0.3333333333 | 0.1 | 10 | 6 | 18 | 0 | 1,1,0,0,1,1,0,1,0,1,0,0,0,0,0 |
| calibration | increase__restart__below | v137 | original | 1 | supported | 148 | 7 | 60 | 7 | 7 | 141 | 53 | 0.0472972973 | 0.1166666667 | 7 | 7 | 148 | 0 | 7 |
| calibration | increase__restart__below | v137 | original | 15 | supported | 148 | 7 | 60 | 12 | 12 | 136 | 48 | 0.08108108108 | 0.2 | 43 | 12 | 148 | 0 | 1,1,1,3,0,0,0,1,0,1,0,1,1,1,1 |
| calibration | increase__restart__below | v137 | budget_matched | 1 | supported | 17 | 2 | 60 | 2 | 2 | 15 | 58 | 0.1176470588 | 0.03333333333 | 2 | 2 | 17 | 0 | 2 |
| calibration | increase__restart__below | v137 | budget_matched | 15 | supported | 17 | 2 | 60 | 3 | 3 | 14 | 57 | 0.1764705882 | 0.05 | 5 | 3 | 17 | 0 | 0,2,0,0,0,0,0,0,0,1,0,0,0,0,0 |
| diagnosis | increase__restart__below | v127 | original | 1 | supported | 1 | 0 | 35 | 0 | 0 | 1 | 35 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| diagnosis | increase__restart__below | v127 | original | 15 | supported | 1 | 0 | 35 | 0 | 0 | 1 | 35 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis | increase__restart__below | v127 | budget_matched | 1 | supported | 1 | 0 | 35 | 0 | 0 | 1 | 35 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| diagnosis | increase__restart__below | v127 | budget_matched | 15 | supported | 1 | 0 | 35 | 0 | 0 | 1 | 35 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis | increase__restart__below | v137 | original | 1 | supported | 19 | 1 | 35 | 1 | 1 | 18 | 34 | 0.05263157895 | 0.02857142857 | 1 | 1 | 19 | 0 | 1 |
| diagnosis | increase__restart__below | v137 | original | 15 | supported | 19 | 1 | 35 | 2 | 2 | 17 | 33 | 0.1052631579 | 0.05714285714 | 2 | 2 | 19 | 0 | 1,0,1,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis | increase__restart__below | v137 | budget_matched | 1 | supported | 1 | 0 | 35 | 0 | 0 | 1 | 35 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| diagnosis | increase__restart__below | v137 | budget_matched | 15 | supported | 1 | 0 | 35 | 0 | 0 | 1 | 35 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |

## 같은 집단의 모든 월

월별 대응 요청은 요청 시작 월, 포착 목표는 목표 발생 월에 배정했다. 월별 두 수는 다를 수 있으며 전체 기간에서만 같다. 먼저 전체 기간에서 연결한 뒤 나눴다.

| 기간·월 | 행동/상태/횟수 | 모델 | 요청 조건 | 범위 분 | 지원 | 요청 분 | 기존 정확 적중 | 목표 분 | 대응 요청 | 포착 목표 | 미연결 요청 | 미포착 목표 | 요청 대비 대응 | 목표 포착 비율 | 목표가 있는 요청 | 요청이 있는 목표 | 관측 창 완전 | 기간 끝 초과 | 선행 0분부터 순서대로 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training/2020-01 | increase__restart__below | v127 | original | 1 | supported | 5 | 1 | 37 | 1 | 1 | 4 | 36 | 0.2 | 0.02702702703 | 1 | 1 | 5 | 0 | 1 |
| training/2020-02 | increase__restart__below | v127 | original | 1 | supported | 1 | 0 | 11 | 0 | 0 | 1 | 11 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-03 | increase__restart__below | v127 | original | 1 | supported | 8 | 2 | 22 | 2 | 2 | 6 | 20 | 0.25 | 0.09090909091 | 2 | 2 | 8 | 0 | 2 |
| training/2020-04 | increase__restart__below | v127 | original | 1 | supported | 2 | 0 | 8 | 0 | 0 | 2 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| training/2020-05 | increase__restart__below | v127 | original | 1 | supported | 4 | 0 | 14 | 0 | 0 | 4 | 14 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| training/2020-06 | increase__restart__below | v127 | original | 1 | supported | 3 | 0 | 13 | 0 | 0 | 3 | 13 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| training/2020-07 | increase__restart__below | v127 | original | 1 | supported | 1 | 0 | 9 | 0 | 0 | 1 | 9 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-08 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 31 | 0 | 0 | 0 | 31 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase__restart__below | v127 | original | 1 | supported | 12 | 4 | 32 | 4 | 4 | 8 | 28 | 0.3333333333 | 0.125 | 4 | 4 | 12 | 0 | 4 |
| training/2020-12 | increase__restart__below | v127 | original | 1 | supported | 1 | 0 | 17 | 0 | 0 | 1 | 17 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-01 | increase__restart__below | v127 | original | 15 | supported | 5 | 1 | 37 | 2 | 2 | 3 | 35 | 0.4 | 0.05405405405 | 2 | 2 | 5 | 0 | 1,1,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-02 | increase__restart__below | v127 | original | 15 | supported | 1 | 0 | 11 | 0 | 0 | 1 | 11 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-03 | increase__restart__below | v127 | original | 15 | supported | 8 | 2 | 22 | 6 | 6 | 2 | 16 | 0.75 | 0.2727272727 | 6 | 6 | 8 | 0 | 2,2,0,1,0,0,1,0,0,0,0,0,0,0,0 |
| training/2020-04 | increase__restart__below | v127 | original | 15 | supported | 2 | 0 | 8 | 0 | 0 | 2 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-05 | increase__restart__below | v127 | original | 15 | supported | 4 | 0 | 14 | 0 | 0 | 4 | 14 | 0 | 0 | 0 | 0 | 4 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-06 | increase__restart__below | v127 | original | 15 | supported | 3 | 0 | 13 | 2 | 2 | 1 | 11 | 0.6666666667 | 0.1538461538 | 3 | 2 | 3 | 0 | 0,1,0,0,0,0,0,0,0,0,0,0,0,1,0 |
| training/2020-07 | increase__restart__below | v127 | original | 15 | supported | 1 | 0 | 9 | 1 | 1 | 0 | 8 | 1 | 0.1111111111 | 1 | 1 | 1 | 0 | 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-08 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 31 | 0 | 0 | 0 | 31 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-09 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-10 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-11 | increase__restart__below | v127 | original | 15 | supported | 12 | 4 | 32 | 4 | 4 | 8 | 28 | 0.3333333333 | 0.125 | 7 | 4 | 12 | 0 | 2,1,0,1,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-12 | increase__restart__below | v127 | original | 15 | supported | 1 | 0 | 17 | 0 | 0 | 1 | 17 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-01 | increase__restart__below | v127 | budget_matched | 1 | supported | 5 | 1 | 37 | 1 | 1 | 4 | 36 | 0.2 | 0.02702702703 | 1 | 1 | 5 | 0 | 1 |
| training/2020-02 | increase__restart__below | v127 | budget_matched | 1 | supported | 1 | 0 | 11 | 0 | 0 | 1 | 11 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-03 | increase__restart__below | v127 | budget_matched | 1 | supported | 9 | 2 | 22 | 2 | 2 | 7 | 20 | 0.2222222222 | 0.09090909091 | 2 | 2 | 9 | 0 | 2 |
| training/2020-04 | increase__restart__below | v127 | budget_matched | 1 | supported | 2 | 0 | 8 | 0 | 0 | 2 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| training/2020-05 | increase__restart__below | v127 | budget_matched | 1 | supported | 4 | 0 | 14 | 0 | 0 | 4 | 14 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| training/2020-06 | increase__restart__below | v127 | budget_matched | 1 | supported | 3 | 0 | 13 | 0 | 0 | 3 | 13 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| training/2020-07 | increase__restart__below | v127 | budget_matched | 1 | supported | 1 | 0 | 9 | 0 | 0 | 1 | 9 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-08 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 31 | 0 | 0 | 0 | 31 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase__restart__below | v127 | budget_matched | 1 | supported | 12 | 4 | 32 | 4 | 4 | 8 | 28 | 0.3333333333 | 0.125 | 4 | 4 | 12 | 0 | 4 |
| training/2020-12 | increase__restart__below | v127 | budget_matched | 1 | supported | 1 | 0 | 17 | 0 | 0 | 1 | 17 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-01 | increase__restart__below | v127 | budget_matched | 15 | supported | 5 | 1 | 37 | 2 | 2 | 3 | 35 | 0.4 | 0.05405405405 | 2 | 2 | 5 | 0 | 1,1,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-02 | increase__restart__below | v127 | budget_matched | 15 | supported | 1 | 0 | 11 | 0 | 0 | 1 | 11 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-03 | increase__restart__below | v127 | budget_matched | 15 | supported | 9 | 2 | 22 | 6 | 6 | 3 | 16 | 0.6666666667 | 0.2727272727 | 6 | 6 | 9 | 0 | 2,2,0,1,0,0,1,0,0,0,0,0,0,0,0 |
| training/2020-04 | increase__restart__below | v127 | budget_matched | 15 | supported | 2 | 0 | 8 | 0 | 0 | 2 | 8 | 0 | 0 | 0 | 0 | 2 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-05 | increase__restart__below | v127 | budget_matched | 15 | supported | 4 | 0 | 14 | 0 | 0 | 4 | 14 | 0 | 0 | 0 | 0 | 4 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-06 | increase__restart__below | v127 | budget_matched | 15 | supported | 3 | 0 | 13 | 2 | 2 | 1 | 11 | 0.6666666667 | 0.1538461538 | 3 | 2 | 3 | 0 | 0,1,0,0,0,0,0,0,0,0,0,0,0,1,0 |
| training/2020-07 | increase__restart__below | v127 | budget_matched | 15 | supported | 1 | 0 | 9 | 1 | 1 | 0 | 8 | 1 | 0.1111111111 | 1 | 1 | 1 | 0 | 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-08 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 31 | 0 | 0 | 0 | 31 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-09 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-10 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-11 | increase__restart__below | v127 | budget_matched | 15 | supported | 12 | 4 | 32 | 4 | 4 | 8 | 28 | 0.3333333333 | 0.125 | 7 | 4 | 12 | 0 | 2,1,0,1,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-12 | increase__restart__below | v127 | budget_matched | 15 | supported | 1 | 0 | 17 | 0 | 0 | 1 | 17 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-01 | increase__restart__below | v137 | original | 1 | supported | 24 | 9 | 37 | 9 | 9 | 15 | 28 | 0.375 | 0.2432432432 | 9 | 9 | 24 | 0 | 9 |
| training/2020-02 | increase__restart__below | v137 | original | 1 | supported | 4 | 1 | 11 | 1 | 1 | 3 | 10 | 0.25 | 0.09090909091 | 1 | 1 | 4 | 0 | 1 |
| training/2020-03 | increase__restart__below | v137 | original | 1 | supported | 68 | 6 | 22 | 6 | 6 | 62 | 16 | 0.08823529412 | 0.2727272727 | 6 | 6 | 68 | 0 | 6 |
| training/2020-04 | increase__restart__below | v137 | original | 1 | supported | 14 | 0 | 8 | 0 | 0 | 14 | 8 | 0 | 0 | 0 | 0 | 14 | 0 | 0 |
| training/2020-05 | increase__restart__below | v137 | original | 1 | supported | 25 | 1 | 14 | 1 | 1 | 24 | 13 | 0.04 | 0.07142857143 | 1 | 1 | 25 | 0 | 1 |
| training/2020-06 | increase__restart__below | v137 | original | 1 | supported | 6 | 1 | 13 | 1 | 1 | 5 | 12 | 0.1666666667 | 0.07692307692 | 1 | 1 | 6 | 0 | 1 |
| training/2020-07 | increase__restart__below | v137 | original | 1 | supported | 9 | 0 | 9 | 0 | 0 | 9 | 9 | 0 | 0 | 0 | 0 | 9 | 0 | 0 |
| training/2020-08 | increase__restart__below | v137 | original | 1 | supported | 16 | 1 | 31 | 1 | 1 | 15 | 30 | 0.0625 | 0.03225806452 | 1 | 1 | 16 | 0 | 1 |
| training/2020-09 | increase__restart__below | v137 | original | 1 | supported | 7 | 0 | 9 | 0 | 0 | 7 | 9 | 0 | 0 | 0 | 0 | 7 | 0 | 0 |
| training/2020-10 | increase__restart__below | v137 | original | 1 | supported | 4 | 0 | 11 | 0 | 0 | 4 | 11 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| training/2020-11 | increase__restart__below | v137 | original | 1 | supported | 35 | 7 | 32 | 7 | 7 | 28 | 25 | 0.2 | 0.21875 | 7 | 7 | 35 | 0 | 7 |
| training/2020-12 | increase__restart__below | v137 | original | 1 | supported | 25 | 3 | 17 | 3 | 3 | 22 | 14 | 0.12 | 0.1764705882 | 3 | 3 | 25 | 0 | 3 |
| training/2020-01 | increase__restart__below | v137 | original | 15 | supported | 24 | 9 | 37 | 11 | 11 | 13 | 26 | 0.4583333333 | 0.2972972973 | 15 | 11 | 24 | 0 | 6,1,2,0,0,0,0,0,0,0,0,1,0,0,1 |
| training/2020-02 | increase__restart__below | v137 | original | 15 | supported | 4 | 1 | 11 | 1 | 1 | 3 | 10 | 0.25 | 0.09090909091 | 1 | 1 | 4 | 0 | 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-03 | increase__restart__below | v137 | original | 15 | supported | 68 | 6 | 22 | 10 | 10 | 58 | 12 | 0.1470588235 | 0.4545454545 | 23 | 10 | 65 | 0 | 2,1,0,0,2,1,0,0,0,1,0,0,0,0,3 |
| training/2020-04 | increase__restart__below | v137 | original | 15 | supported | 14 | 0 | 8 | 0 | 0 | 14 | 8 | 0 | 0 | 0 | 0 | 14 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-05 | increase__restart__below | v137 | original | 15 | supported | 25 | 1 | 14 | 1 | 1 | 24 | 13 | 0.04 | 0.07142857143 | 1 | 1 | 25 | 0 | 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-06 | increase__restart__below | v137 | original | 15 | supported | 6 | 1 | 13 | 2 | 2 | 4 | 11 | 0.3333333333 | 0.1538461538 | 6 | 2 | 6 | 0 | 0,1,0,0,0,0,0,0,0,0,0,0,0,1,0 |
| training/2020-07 | increase__restart__below | v137 | original | 15 | supported | 9 | 0 | 9 | 1 | 1 | 8 | 8 | 0.1111111111 | 0.1111111111 | 8 | 1 | 9 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 |
| training/2020-08 | increase__restart__below | v137 | original | 15 | supported | 16 | 1 | 31 | 3 | 3 | 13 | 28 | 0.1875 | 0.09677419355 | 5 | 3 | 16 | 0 | 0,1,0,0,2,0,0,0,0,0,0,0,0,0,0 |
| training/2020-09 | increase__restart__below | v137 | original | 15 | supported | 7 | 0 | 9 | 0 | 0 | 7 | 9 | 0 | 0 | 0 | 0 | 7 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-10 | increase__restart__below | v137 | original | 15 | supported | 4 | 0 | 11 | 1 | 1 | 3 | 10 | 0.25 | 0.09090909091 | 1 | 1 | 4 | 0 | 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0 |
| training/2020-11 | increase__restart__below | v137 | original | 15 | supported | 35 | 7 | 32 | 9 | 9 | 26 | 23 | 0.2571428571 | 0.28125 | 19 | 9 | 35 | 0 | 4,0,1,0,3,0,0,0,0,0,0,0,0,0,1 |
| training/2020-12 | increase__restart__below | v137 | original | 15 | supported | 25 | 3 | 17 | 4 | 4 | 21 | 13 | 0.16 | 0.2352941176 | 10 | 4 | 25 | 0 | 1,0,1,1,0,0,0,0,0,0,0,0,1,0,0 |
| training/2020-01 | increase__restart__below | v137 | budget_matched | 1 | supported | 1 | 0 | 37 | 0 | 0 | 1 | 37 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-02 | increase__restart__below | v137 | budget_matched | 1 | supported | 2 | 0 | 11 | 0 | 0 | 2 | 11 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| training/2020-03 | increase__restart__below | v137 | budget_matched | 1 | supported | 7 | 2 | 22 | 2 | 2 | 5 | 20 | 0.2857142857 | 0.09090909091 | 2 | 2 | 7 | 0 | 2 |
| training/2020-04 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase__restart__below | v137 | budget_matched | 1 | supported | 1 | 0 | 14 | 0 | 0 | 1 | 14 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-06 | increase__restart__below | v137 | budget_matched | 1 | supported | 1 | 0 | 13 | 0 | 0 | 1 | 13 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| training/2020-07 | increase__restart__below | v137 | budget_matched | 1 | supported | 2 | 0 | 9 | 0 | 0 | 2 | 9 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| training/2020-08 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 31 | 0 | 0 | 0 | 31 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase__restart__below | v137 | budget_matched | 1 | supported | 6 | 3 | 32 | 3 | 3 | 3 | 29 | 0.5 | 0.09375 | 3 | 3 | 6 | 0 | 3 |
| training/2020-12 | increase__restart__below | v137 | budget_matched | 1 | supported | 3 | 1 | 17 | 1 | 1 | 2 | 16 | 0.3333333333 | 0.05882352941 | 1 | 1 | 3 | 0 | 1 |
| training/2020-01 | increase__restart__below | v137 | budget_matched | 15 | supported | 1 | 0 | 37 | 0 | 0 | 1 | 37 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-02 | increase__restart__below | v137 | budget_matched | 15 | supported | 2 | 0 | 11 | 0 | 0 | 2 | 11 | 0 | 0 | 0 | 0 | 2 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-03 | increase__restart__below | v137 | budget_matched | 15 | supported | 7 | 2 | 22 | 6 | 6 | 1 | 16 | 0.8571428571 | 0.2727272727 | 6 | 6 | 6 | 0 | 2,2,0,1,0,0,1,0,0,0,0,0,0,0,0 |
| training/2020-04 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-05 | increase__restart__below | v137 | budget_matched | 15 | supported | 1 | 0 | 14 | 0 | 0 | 1 | 14 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-06 | increase__restart__below | v137 | budget_matched | 15 | supported | 1 | 0 | 13 | 1 | 1 | 0 | 12 | 1 | 0.07692307692 | 1 | 1 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0 |
| training/2020-07 | increase__restart__below | v137 | budget_matched | 15 | supported | 2 | 0 | 9 | 1 | 1 | 1 | 8 | 0.5 | 0.1111111111 | 2 | 1 | 2 | 0 | 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-08 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 31 | 0 | 0 | 0 | 31 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-09 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-10 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-11 | increase__restart__below | v137 | budget_matched | 15 | supported | 6 | 3 | 32 | 3 | 3 | 3 | 29 | 0.5 | 0.09375 | 3 | 3 | 6 | 0 | 3,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| training/2020-12 | increase__restart__below | v137 | budget_matched | 15 | supported | 3 | 1 | 17 | 2 | 2 | 1 | 15 | 0.6666666667 | 0.1176470588 | 2 | 2 | 3 | 0 | 1,0,0,0,0,0,0,0,0,0,0,1,0,0,0 |
| calibration/2021-01 | increase__restart__below | v127 | original | 1 | supported | 10 | 3 | 14 | 3 | 3 | 7 | 11 | 0.3 | 0.2142857143 | 3 | 3 | 10 | 0 | 3 |
| calibration/2021-02 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase__restart__below | v127 | original | 1 | supported | 6 | 0 | 18 | 0 | 0 | 6 | 18 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| calibration/2021-06 | increase__restart__below | v127 | original | 1 | supported | 2 | 0 | 15 | 0 | 0 | 2 | 15 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| calibration/2021-01 | increase__restart__below | v127 | original | 15 | supported | 10 | 3 | 14 | 3 | 3 | 7 | 11 | 0.3 | 0.2142857143 | 7 | 3 | 10 | 0 | 1,0,0,0,1,0,0,1,0,0,0,0,0,0,0 |
| calibration/2021-02 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-03 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-04 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-05 | increase__restart__below | v127 | original | 15 | supported | 6 | 0 | 18 | 2 | 2 | 4 | 16 | 0.3333333333 | 0.1111111111 | 2 | 2 | 6 | 0 | 0,1,0,0,0,1,0,0,0,0,0,0,0,0,0 |
| calibration/2021-06 | increase__restart__below | v127 | original | 15 | supported | 2 | 0 | 15 | 1 | 1 | 1 | 14 | 0.5 | 0.06666666667 | 1 | 1 | 2 | 0 | 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0 |
| calibration/2021-01 | increase__restart__below | v127 | budget_matched | 1 | supported | 10 | 3 | 14 | 3 | 3 | 7 | 11 | 0.3 | 0.2142857143 | 3 | 3 | 10 | 0 | 3 |
| calibration/2021-02 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase__restart__below | v127 | budget_matched | 1 | supported | 6 | 0 | 18 | 0 | 0 | 6 | 18 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| calibration/2021-06 | increase__restart__below | v127 | budget_matched | 1 | supported | 2 | 0 | 15 | 0 | 0 | 2 | 15 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| calibration/2021-01 | increase__restart__below | v127 | budget_matched | 15 | supported | 10 | 3 | 14 | 3 | 3 | 7 | 11 | 0.3 | 0.2142857143 | 7 | 3 | 10 | 0 | 1,0,0,0,1,0,0,1,0,0,0,0,0,0,0 |
| calibration/2021-02 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-03 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-04 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-05 | increase__restart__below | v127 | budget_matched | 15 | supported | 6 | 0 | 18 | 2 | 2 | 4 | 16 | 0.3333333333 | 0.1111111111 | 2 | 2 | 6 | 0 | 0,1,0,0,0,1,0,0,0,0,0,0,0,0,0 |
| calibration/2021-06 | increase__restart__below | v127 | budget_matched | 15 | supported | 2 | 0 | 15 | 1 | 1 | 1 | 14 | 0.5 | 0.06666666667 | 1 | 1 | 2 | 0 | 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0 |
| calibration/2021-01 | increase__restart__below | v137 | original | 1 | supported | 69 | 3 | 14 | 3 | 3 | 66 | 11 | 0.04347826087 | 0.2142857143 | 3 | 3 | 69 | 0 | 3 |
| calibration/2021-02 | increase__restart__below | v137 | original | 1 | supported | 3 | 0 | 9 | 0 | 0 | 3 | 9 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| calibration/2021-03 | increase__restart__below | v137 | original | 1 | supported | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase__restart__below | v137 | original | 1 | supported | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase__restart__below | v137 | original | 1 | supported | 59 | 3 | 18 | 3 | 3 | 56 | 15 | 0.05084745763 | 0.1666666667 | 3 | 3 | 59 | 0 | 3 |
| calibration/2021-06 | increase__restart__below | v137 | original | 1 | supported | 17 | 1 | 15 | 1 | 1 | 16 | 14 | 0.05882352941 | 0.06666666667 | 1 | 1 | 17 | 0 | 1 |
| calibration/2021-01 | increase__restart__below | v137 | original | 15 | supported | 69 | 3 | 14 | 4 | 4 | 65 | 10 | 0.05797101449 | 0.2857142857 | 15 | 4 | 69 | 0 | 1,0,0,0,0,0,0,1,0,1,0,0,0,0,1 |
| calibration/2021-02 | increase__restart__below | v137 | original | 15 | supported | 3 | 0 | 9 | 1 | 1 | 2 | 8 | 0.3333333333 | 0.1111111111 | 1 | 1 | 3 | 0 | 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-03 | increase__restart__below | v137 | original | 15 | supported | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-04 | increase__restart__below | v137 | original | 15 | supported | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-05 | increase__restart__below | v137 | original | 15 | supported | 59 | 3 | 18 | 5 | 5 | 54 | 13 | 0.08474576271 | 0.2777777778 | 15 | 5 | 59 | 0 | 0,1,0,2,0,0,0,0,0,0,0,1,0,1,0 |
| calibration/2021-06 | increase__restart__below | v137 | original | 15 | supported | 17 | 1 | 15 | 2 | 2 | 15 | 13 | 0.1176470588 | 0.1333333333 | 12 | 2 | 17 | 0 | 0,0,0,1,0,0,0,0,0,0,0,0,1,0,0 |
| calibration/2021-01 | increase__restart__below | v137 | budget_matched | 1 | supported | 10 | 1 | 14 | 1 | 1 | 9 | 13 | 0.1 | 0.07142857143 | 1 | 1 | 10 | 0 | 1 |
| calibration/2021-02 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase__restart__below | v137 | budget_matched | 1 | supported | 5 | 1 | 18 | 1 | 1 | 4 | 17 | 0.2 | 0.05555555556 | 1 | 1 | 5 | 0 | 1 |
| calibration/2021-06 | increase__restart__below | v137 | budget_matched | 1 | supported | 2 | 0 | 15 | 0 | 0 | 2 | 15 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| calibration/2021-01 | increase__restart__below | v137 | budget_matched | 15 | supported | 10 | 1 | 14 | 1 | 1 | 9 | 13 | 0.1 | 0.07142857143 | 2 | 1 | 10 | 0 | 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-02 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-03 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-04 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-05 | increase__restart__below | v137 | budget_matched | 15 | supported | 5 | 1 | 18 | 1 | 1 | 4 | 17 | 0.2 | 0.05555555556 | 2 | 1 | 5 | 0 | 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| calibration/2021-06 | increase__restart__below | v137 | budget_matched | 15 | supported | 2 | 0 | 15 | 1 | 1 | 1 | 14 | 0.5 | 0.06666666667 | 1 | 1 | 2 | 0 | 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0 |
| diagnosis/2021-07 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 14 | 0 | 0 | 0 | 14 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase__restart__below | v127 | original | 1 | supported | 1 | 0 | 10 | 0 | 0 | 1 | 10 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| diagnosis/2021-10 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase__restart__below | v127 | original | 1 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-08 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 14 | 0 | 0 | 0 | 14 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-09 | increase__restart__below | v127 | original | 15 | supported | 1 | 0 | 10 | 0 | 0 | 1 | 10 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-10 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-11 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-12 | increase__restart__below | v127 | original | 15 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-07 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 14 | 0 | 0 | 0 | 14 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase__restart__below | v127 | budget_matched | 1 | supported | 1 | 0 | 10 | 0 | 0 | 1 | 10 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| diagnosis/2021-10 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase__restart__below | v127 | budget_matched | 1 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-08 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 14 | 0 | 0 | 0 | 14 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-09 | increase__restart__below | v127 | budget_matched | 15 | supported | 1 | 0 | 10 | 0 | 0 | 1 | 10 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-10 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-11 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-12 | increase__restart__below | v127 | budget_matched | 15 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-07 | increase__restart__below | v137 | original | 1 | supported | 4 | 0 | 9 | 0 | 0 | 4 | 9 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| diagnosis/2021-08 | increase__restart__below | v137 | original | 1 | supported | 3 | 1 | 14 | 1 | 1 | 2 | 13 | 0.3333333333 | 0.07142857143 | 1 | 1 | 3 | 0 | 1 |
| diagnosis/2021-09 | increase__restart__below | v137 | original | 1 | supported | 12 | 0 | 10 | 0 | 0 | 12 | 10 | 0 | 0 | 0 | 0 | 12 | 0 | 0 |
| diagnosis/2021-10 | increase__restart__below | v137 | original | 1 | supported | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase__restart__below | v137 | original | 1 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase__restart__below | v137 | original | 1 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase__restart__below | v137 | original | 15 | supported | 4 | 0 | 9 | 0 | 0 | 4 | 9 | 0 | 0 | 0 | 0 | 4 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-08 | increase__restart__below | v137 | original | 15 | supported | 3 | 1 | 14 | 1 | 1 | 2 | 13 | 0.3333333333 | 0.07142857143 | 1 | 1 | 3 | 0 | 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-09 | increase__restart__below | v137 | original | 15 | supported | 12 | 0 | 10 | 1 | 1 | 11 | 9 | 0.08333333333 | 0.1 | 1 | 1 | 12 | 0 | 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-10 | increase__restart__below | v137 | original | 15 | supported | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-11 | increase__restart__below | v137 | original | 15 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-12 | increase__restart__below | v137 | original | 15 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-07 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 14 | 0 | 0 | 0 | 14 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase__restart__below | v137 | budget_matched | 1 | supported | 1 | 0 | 10 | 0 | 0 | 1 | 10 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| diagnosis/2021-10 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 없음 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase__restart__below | v137 | budget_matched | 1 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 9 | 0 | 0 | 0 | 9 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-08 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 14 | 0 | 0 | 0 | 14 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-09 | increase__restart__below | v137 | budget_matched | 15 | supported | 1 | 0 | 10 | 0 | 0 | 1 | 10 | 0 | 0 | 0 | 0 | 1 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-10 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 없음 | 0 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-11 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |
| diagnosis/2021-12 | increase__restart__below | v137 | budget_matched | 15 | supported | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 없음 | 없음 | 0 | 0 | 0 | 0 | 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 |

## 검산과 해석 범위

[모든 집단과 월](EXPERIMENT_V139_TABLES.md)에 18집단·두 모델·두 요청 조건·두 범위의 432개 기간 조건과 3,456개 월 조건을 보존했다. 그룹은 겹치며 합산하지 않는다. 미지원 공통 문턱은 성적 없음이다.

원본 1,018,460행과 기존 예측·횟수·공통 예산 요청의 연결을 대조했다. 저장된 모델 출처와 지문을 확인했으며 모델을 다시 학습하거나 새로 추론한 성과는 아니다. 1분 대응 수·미연결·미포착은 기존 TP/FP/FN과 같았다. 15분의 일대일 최대 대응 수는 독립 이분 그래프 증대 경로 계산과 모두 일치했다. 개별 대응은 가장 이른 요청 순으로 고정했고, 최대 대응 중 정확한 동일 분 연결을 우선한 것은 아니다. 기존 정확 적중 열은 원래 요청과 정답의 교집합이다.

15분 범위는 요청 t에 대해 목표 경계 e가 t 이상이고 t+15분 미만인 경우다. 미래 원본 행동은 사후 대응에만 사용했다. 목표는 기존 표의 양성 분이며 주문 수·실제 종료 수가 아니다. 보존 표의 경계가 빠진 관측 창을 따로 표시했으며 미연결을 새로운 확정 오신호로 재분류하지 않았다. 범위를 넓혀 늘어난 대응은 그 자체로 예측력이나 수익 개선을 입증하지 않는다.

신규 46개 포함 관련 213개 검사가 11.72초에 통과했고 Ruff도 통과했다. 최초 신규 40개는 4.73초에 통과했고 import 정렬 한 건을 수정했다. 모든 3×3 이분 그래프 512개와 작은 시간 대응 160개를 완전 탐색 최대 수와 대조했다. 전체 검사 모음이나 최신 원격 CI 통과를 주장하지 않는다. 실제 실행은 한 번이며 종료 코드 0이다.

모든 원장 저장 후 재읽기와 실행 전후 입력·코드·계획 지문을 확인했다. v137 24/40 미달, v138 진단, 이전 계좌 손실·표본 부족과 v10 최종 요구는 그대로다. 모든 기간은 이미 관찰했으며 새로운 미관찰 검증이 아니다. 원본·개별 대응·모형은 로컬 전용이고 실제 거래 연결은 없다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m wonyotti_fr.management_temporal_research \
  --reference artifacts/v138-final-verification.json \
  --sha256 e30f92116b6ed7664a608de3425a3807db5f4fc6445a2e2e17721860ed8346a5
```

선행 로컬 자료가 있는 환경에서만 실행한다.

| 근거 | SHA-256 |
| --- | --- |
| 출력 봉인 | 1eb6a1a6caa40eb2a22059995417d9fbdedcb491c4ecb455bec6ed6c11b8dfb2 |
| 요약 | 424e34c3b9b792fc62ea1fc1944fa443f4eb76bd22e6bf4c395f475414930c83 |
| 검사 증거 | 9b9f4cc3cbeb6898cd9a3937d3161a5de3071f993646326be9fa391ee8a59949 |
| 완료 검산 | 81c5c69b244f2ddbb09e2d439ed94fed88d27982436af7d251246fde448c99b9 |
