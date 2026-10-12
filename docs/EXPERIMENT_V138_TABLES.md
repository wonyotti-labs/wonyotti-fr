# 실험 v138 전체 오신호 한도 집계

[결과와 해석](EXPERIMENT_V138_RESULTS.md). original은 기존 요청, budget_matched는 상반기 공통 FP 한도의 고정 문턱 요청이다. 집단 중복·빈 집단·미지원 상태를 보존했다. FPR은 음성이 없으면 없음이다.

## 모든 집단의 지원

| 기간·월 | 행동 | 상태 | 횟수 | 행 | 양성 | 음성 | 포지션 | 양성 포지션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training | reduce | quiet | all | 497219 | 744 | 496475 | 270 | 202 |
| training | reduce | quiet | below | 203394 | 244 | 203150 | 270 | 164 |
| training | reduce | quiet | capped | 293825 | 500 | 293325 | 96 | 93 |
| training | reduce | first | all | 136178 | 194 | 135984 | 267 | 194 |
| training | reduce | first | below | 121200 | 161 | 121039 | 267 | 161 |
| training | reduce | first | capped | 14978 | 33 | 14945 | 33 | 33 |
| training | reduce | restart | all | 361041 | 550 | 360491 | 158 | 120 |
| training | reduce | restart | below | 82194 | 83 | 82111 | 115 | 56 |
| training | reduce | restart | capped | 278847 | 467 | 278380 | 90 | 86 |
| training | increase | quiet | all | 494681 | 827 | 493854 | 265 | 178 |
| training | increase | quiet | below | 201412 | 366 | 201046 | 260 | 176 |
| training | increase | quiet | capped | 293269 | 461 | 292808 | 98 | 72 |
| training | increase | first | all | 65424 | 152 | 65272 | 227 | 152 |
| training | increase | first | below | 65424 | 152 | 65272 | 227 | 152 |
| training | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training | increase | restart | all | 429257 | 675 | 428582 | 177 | 127 |
| training | increase | restart | below | 135988 | 214 | 135774 | 159 | 115 |
| training | increase | restart | capped | 293269 | 461 | 292808 | 98 | 72 |
| calibration | reduce | quiet | all | 250047 | 347 | 249700 | 89 | 77 |
| calibration | reduce | quiet | below | 74721 | 66 | 74655 | 89 | 49 |
| calibration | reduce | quiet | capped | 175326 | 281 | 175045 | 42 | 42 |
| calibration | reduce | first | all | 73521 | 77 | 73444 | 89 | 77 |
| calibration | reduce | first | below | 57078 | 49 | 57029 | 89 | 49 |
| calibration | reduce | first | capped | 16443 | 28 | 16415 | 28 | 28 |
| calibration | reduce | restart | all | 176526 | 270 | 176256 | 63 | 50 |
| calibration | reduce | restart | below | 17643 | 17 | 17626 | 34 | 16 |
| calibration | reduce | restart | capped | 158883 | 253 | 158630 | 40 | 39 |
| calibration | increase | quiet | all | 249583 | 335 | 249248 | 88 | 68 |
| calibration | increase | quiet | below | 73836 | 120 | 73716 | 86 | 66 |
| calibration | increase | quiet | capped | 175747 | 215 | 175532 | 42 | 33 |
| calibration | increase | first | all | 25355 | 60 | 25295 | 76 | 60 |
| calibration | increase | first | below | 25355 | 60 | 25295 | 76 | 60 |
| calibration | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| calibration | increase | restart | all | 224228 | 275 | 223953 | 71 | 51 |
| calibration | increase | restart | below | 48481 | 60 | 48421 | 58 | 40 |
| calibration | increase | restart | capped | 175747 | 215 | 175532 | 42 | 33 |
| diagnosis | reduce | quiet | all | 249701 | 140 | 249561 | 33 | 29 |
| diagnosis | reduce | quiet | below | 70516 | 28 | 70488 | 33 | 15 |
| diagnosis | reduce | quiet | capped | 179185 | 112 | 179073 | 16 | 16 |
| diagnosis | reduce | first | all | 79164 | 29 | 79135 | 33 | 29 |
| diagnosis | reduce | first | below | 52443 | 15 | 52428 | 33 | 15 |
| diagnosis | reduce | first | capped | 26721 | 14 | 26707 | 14 | 14 |
| diagnosis | reduce | restart | all | 170537 | 111 | 170426 | 26 | 25 |
| diagnosis | reduce | restart | below | 18073 | 13 | 18060 | 12 | 9 |
| diagnosis | reduce | restart | capped | 152464 | 98 | 152366 | 16 | 16 |
| diagnosis | increase | quiet | all | 249378 | 146 | 249232 | 33 | 26 |
| diagnosis | increase | quiet | below | 70097 | 58 | 70039 | 33 | 26 |
| diagnosis | increase | quiet | capped | 179281 | 88 | 179193 | 16 | 11 |
| diagnosis | increase | first | all | 22135 | 23 | 22112 | 28 | 23 |
| diagnosis | increase | first | below | 22135 | 23 | 22112 | 28 | 23 |
| diagnosis | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| diagnosis | increase | restart | all | 227243 | 123 | 227120 | 28 | 20 |
| diagnosis | increase | restart | below | 47962 | 35 | 47927 | 25 | 18 |
| diagnosis | increase | restart | capped | 179281 | 88 | 179193 | 16 | 11 |
| training/2020-01 | reduce | quiet | all | 41634 | 76 | 41558 | 35 | 24 |
| training/2020-01 | reduce | quiet | below | 19488 | 29 | 19459 | 35 | 19 |
| training/2020-01 | reduce | quiet | capped | 22146 | 47 | 22099 | 12 | 11 |
| training/2020-01 | reduce | first | all | 6566 | 24 | 6542 | 35 | 24 |
| training/2020-01 | reduce | first | below | 6157 | 19 | 6138 | 35 | 19 |
| training/2020-01 | reduce | first | capped | 409 | 5 | 404 | 5 | 5 |
| training/2020-01 | reduce | restart | all | 35068 | 52 | 35016 | 21 | 13 |
| training/2020-01 | reduce | restart | below | 13331 | 10 | 13321 | 16 | 7 |
| training/2020-01 | reduce | restart | capped | 21737 | 42 | 21695 | 11 | 10 |
| training/2020-01 | increase | quiet | all | 41197 | 100 | 41097 | 34 | 20 |
| training/2020-01 | increase | quiet | below | 19153 | 55 | 19098 | 34 | 20 |
| training/2020-01 | increase | quiet | capped | 22044 | 45 | 21999 | 12 | 8 |
| training/2020-01 | increase | first | all | 2862 | 18 | 2844 | 29 | 18 |
| training/2020-01 | increase | first | below | 2862 | 18 | 2844 | 29 | 18 |
| training/2020-01 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | restart | all | 38335 | 82 | 38253 | 23 | 18 |
| training/2020-01 | increase | restart | below | 16291 | 37 | 16254 | 23 | 18 |
| training/2020-01 | increase | restart | capped | 22044 | 45 | 21999 | 12 | 8 |
| training/2020-02 | reduce | quiet | all | 40200 | 88 | 40112 | 13 | 9 |
| training/2020-02 | reduce | quiet | below | 2767 | 10 | 2757 | 12 | 8 |
| training/2020-02 | reduce | quiet | capped | 37433 | 78 | 37355 | 7 | 7 |
| training/2020-02 | reduce | first | all | 766 | 8 | 758 | 12 | 8 |
| training/2020-02 | reduce | first | below | 766 | 8 | 758 | 12 | 8 |
| training/2020-02 | reduce | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | restart | all | 39434 | 80 | 39354 | 8 | 7 |
| training/2020-02 | reduce | restart | below | 2001 | 2 | 1999 | 6 | 1 |
| training/2020-02 | reduce | restart | capped | 37433 | 78 | 37355 | 7 | 7 |
| training/2020-02 | increase | quiet | all | 39784 | 113 | 39671 | 12 | 8 |
| training/2020-02 | increase | quiet | below | 2638 | 18 | 2620 | 11 | 7 |
| training/2020-02 | increase | quiet | capped | 37146 | 95 | 37051 | 7 | 6 |
| training/2020-02 | increase | first | all | 390 | 7 | 383 | 11 | 7 |
| training/2020-02 | increase | first | below | 390 | 7 | 383 | 11 | 7 |
| training/2020-02 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | restart | all | 39394 | 106 | 39288 | 8 | 7 |
| training/2020-02 | increase | restart | below | 2248 | 11 | 2237 | 7 | 6 |
| training/2020-02 | increase | restart | capped | 37146 | 95 | 37051 | 7 | 6 |
| training/2020-03 | reduce | quiet | all | 42075 | 136 | 41939 | 52 | 39 |
| training/2020-03 | reduce | quiet | below | 6518 | 43 | 6475 | 51 | 31 |
| training/2020-03 | reduce | quiet | capped | 35557 | 93 | 35464 | 17 | 17 |
| training/2020-03 | reduce | first | all | 3630 | 37 | 3593 | 50 | 37 |
| training/2020-03 | reduce | first | below | 3431 | 30 | 3401 | 50 | 30 |
| training/2020-03 | reduce | first | capped | 199 | 7 | 192 | 7 | 7 |
| training/2020-03 | reduce | restart | all | 38445 | 99 | 38346 | 27 | 22 |
| training/2020-03 | reduce | restart | below | 3087 | 13 | 3074 | 17 | 9 |
| training/2020-03 | reduce | restart | capped | 35358 | 86 | 35272 | 15 | 15 |
| training/2020-03 | increase | quiet | all | 41971 | 122 | 41849 | 49 | 30 |
| training/2020-03 | increase | quiet | below | 6348 | 47 | 6301 | 45 | 29 |
| training/2020-03 | increase | quiet | capped | 35623 | 75 | 35548 | 19 | 14 |
| training/2020-03 | increase | first | all | 1611 | 25 | 1586 | 39 | 25 |
| training/2020-03 | increase | first | below | 1611 | 25 | 1586 | 39 | 25 |
| training/2020-03 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-03 | increase | restart | all | 40360 | 97 | 40263 | 30 | 20 |
| training/2020-03 | increase | restart | below | 4737 | 22 | 4715 | 22 | 16 |
| training/2020-03 | increase | restart | capped | 35623 | 75 | 35548 | 19 | 14 |
| training/2020-04 | reduce | quiet | all | 42463 | 41 | 42422 | 14 | 12 |
| training/2020-04 | reduce | quiet | below | 17075 | 11 | 17064 | 14 | 8 |
| training/2020-04 | reduce | quiet | capped | 25388 | 30 | 25358 | 6 | 5 |
| training/2020-04 | reduce | first | all | 3748 | 11 | 3737 | 12 | 11 |
| training/2020-04 | reduce | first | below | 3302 | 8 | 3294 | 12 | 8 |
| training/2020-04 | reduce | first | capped | 446 | 3 | 443 | 3 | 3 |
| training/2020-04 | reduce | restart | all | 38715 | 30 | 38685 | 12 | 7 |
| training/2020-04 | reduce | restart | below | 13773 | 3 | 13770 | 8 | 3 |
| training/2020-04 | reduce | restart | capped | 24942 | 27 | 24915 | 6 | 4 |
| training/2020-04 | increase | quiet | all | 42320 | 46 | 42274 | 14 | 13 |
| training/2020-04 | increase | quiet | below | 16933 | 19 | 16914 | 14 | 13 |
| training/2020-04 | increase | quiet | capped | 25387 | 27 | 25360 | 6 | 3 |
| training/2020-04 | increase | first | all | 3558 | 11 | 3547 | 12 | 11 |
| training/2020-04 | increase | first | below | 3558 | 11 | 3547 | 12 | 11 |
| training/2020-04 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | restart | all | 38762 | 35 | 38727 | 13 | 6 |
| training/2020-04 | increase | restart | below | 13375 | 8 | 13367 | 12 | 6 |
| training/2020-04 | increase | restart | capped | 25387 | 27 | 25360 | 6 | 3 |
| training/2020-05 | reduce | quiet | all | 43723 | 53 | 43670 | 14 | 11 |
| training/2020-05 | reduce | quiet | below | 12331 | 19 | 12312 | 14 | 9 |
| training/2020-05 | reduce | quiet | capped | 31392 | 34 | 31358 | 5 | 5 |
| training/2020-05 | reduce | first | all | 7117 | 11 | 7106 | 14 | 11 |
| training/2020-05 | reduce | first | below | 6107 | 9 | 6098 | 14 | 9 |
| training/2020-05 | reduce | first | capped | 1010 | 2 | 1008 | 2 | 2 |
| training/2020-05 | reduce | restart | all | 36606 | 42 | 36564 | 10 | 9 |
| training/2020-05 | reduce | restart | below | 6224 | 10 | 6214 | 8 | 7 |
| training/2020-05 | reduce | restart | capped | 30382 | 32 | 30350 | 5 | 5 |
| training/2020-05 | increase | quiet | all | 43714 | 48 | 43666 | 14 | 10 |
| training/2020-05 | increase | quiet | below | 12321 | 21 | 12300 | 14 | 10 |
| training/2020-05 | increase | quiet | capped | 31393 | 27 | 31366 | 5 | 4 |
| training/2020-05 | increase | first | all | 7381 | 7 | 7374 | 11 | 7 |
| training/2020-05 | increase | first | below | 7381 | 7 | 7374 | 11 | 7 |
| training/2020-05 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | restart | all | 36333 | 41 | 36292 | 10 | 8 |
| training/2020-05 | increase | restart | below | 4940 | 14 | 4926 | 10 | 8 |
| training/2020-05 | increase | restart | capped | 31393 | 27 | 31366 | 5 | 4 |
| training/2020-06 | reduce | quiet | all | 42630 | 36 | 42594 | 14 | 11 |
| training/2020-06 | reduce | quiet | below | 16645 | 18 | 16627 | 13 | 11 |
| training/2020-06 | reduce | quiet | capped | 25985 | 18 | 25967 | 5 | 4 |
| training/2020-06 | reduce | first | all | 6120 | 11 | 6109 | 13 | 11 |
| training/2020-06 | reduce | first | below | 6120 | 11 | 6109 | 13 | 11 |
| training/2020-06 | reduce | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | all | 36510 | 25 | 36485 | 8 | 5 |
| training/2020-06 | reduce | restart | below | 10525 | 7 | 10518 | 6 | 3 |
| training/2020-06 | reduce | restart | capped | 25985 | 18 | 25967 | 5 | 4 |
| training/2020-06 | increase | quiet | all | 42522 | 42 | 42480 | 15 | 10 |
| training/2020-06 | increase | quiet | below | 16588 | 22 | 16566 | 14 | 10 |
| training/2020-06 | increase | quiet | capped | 25934 | 20 | 25914 | 5 | 3 |
| training/2020-06 | increase | first | all | 1174 | 9 | 1165 | 13 | 9 |
| training/2020-06 | increase | first | below | 1174 | 9 | 1165 | 13 | 9 |
| training/2020-06 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | restart | all | 41348 | 33 | 41315 | 10 | 6 |
| training/2020-06 | increase | restart | below | 15414 | 13 | 15401 | 9 | 6 |
| training/2020-06 | increase | restart | capped | 25934 | 20 | 25914 | 5 | 3 |
| training/2020-07 | reduce | quiet | all | 44179 | 27 | 44152 | 8 | 7 |
| training/2020-07 | reduce | quiet | below | 25583 | 8 | 25575 | 8 | 5 |
| training/2020-07 | reduce | quiet | capped | 18596 | 19 | 18577 | 3 | 3 |
| training/2020-07 | reduce | first | all | 17551 | 7 | 17544 | 8 | 7 |
| training/2020-07 | reduce | first | below | 17337 | 5 | 17332 | 8 | 5 |
| training/2020-07 | reduce | first | capped | 214 | 2 | 212 | 2 | 2 |
| training/2020-07 | reduce | restart | all | 26628 | 20 | 26608 | 6 | 5 |
| training/2020-07 | reduce | restart | below | 8246 | 3 | 8243 | 4 | 2 |
| training/2020-07 | reduce | restart | capped | 18382 | 17 | 18365 | 3 | 3 |
| training/2020-07 | increase | quiet | all | 44066 | 30 | 44036 | 8 | 7 |
| training/2020-07 | increase | quiet | below | 25467 | 14 | 25453 | 8 | 7 |
| training/2020-07 | increase | quiet | capped | 18599 | 16 | 18583 | 3 | 3 |
| training/2020-07 | increase | first | all | 11344 | 5 | 11339 | 6 | 5 |
| training/2020-07 | increase | first | below | 11344 | 5 | 11339 | 6 | 5 |
| training/2020-07 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | restart | all | 32722 | 25 | 32697 | 7 | 6 |
| training/2020-07 | increase | restart | below | 14123 | 9 | 14114 | 6 | 5 |
| training/2020-07 | increase | restart | capped | 18599 | 16 | 18583 | 3 | 3 |
| training/2020-08 | reduce | quiet | all | 43683 | 58 | 43625 | 34 | 24 |
| training/2020-08 | reduce | quiet | below | 22761 | 29 | 22732 | 34 | 21 |
| training/2020-08 | reduce | quiet | capped | 20922 | 29 | 20893 | 11 | 10 |
| training/2020-08 | reduce | first | all | 15986 | 23 | 15963 | 32 | 23 |
| training/2020-08 | reduce | first | below | 15123 | 21 | 15102 | 32 | 21 |
| training/2020-08 | reduce | first | capped | 863 | 2 | 861 | 3 | 2 |
| training/2020-08 | reduce | restart | all | 27697 | 35 | 27662 | 24 | 16 |
| training/2020-08 | reduce | restart | below | 7638 | 8 | 7630 | 22 | 7 |
| training/2020-08 | reduce | restart | capped | 20059 | 27 | 20032 | 10 | 10 |
| training/2020-08 | increase | quiet | all | 43344 | 71 | 43273 | 34 | 24 |
| training/2020-08 | increase | quiet | below | 22493 | 50 | 22443 | 34 | 24 |
| training/2020-08 | increase | quiet | capped | 20851 | 21 | 20830 | 11 | 5 |
| training/2020-08 | increase | first | all | 9382 | 19 | 9363 | 28 | 19 |
| training/2020-08 | increase | first | below | 9382 | 19 | 9363 | 28 | 19 |
| training/2020-08 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | all | 33962 | 52 | 33910 | 25 | 16 |
| training/2020-08 | increase | restart | below | 13111 | 31 | 13080 | 24 | 16 |
| training/2020-08 | increase | restart | capped | 20851 | 21 | 20830 | 11 | 5 |
| training/2020-09 | reduce | quiet | all | 42111 | 65 | 42046 | 17 | 13 |
| training/2020-09 | reduce | quiet | below | 11717 | 18 | 11699 | 16 | 12 |
| training/2020-09 | reduce | quiet | capped | 30394 | 47 | 30347 | 4 | 4 |
| training/2020-09 | reduce | first | all | 10548 | 13 | 10535 | 17 | 13 |
| training/2020-09 | reduce | first | below | 10218 | 12 | 10206 | 16 | 12 |
| training/2020-09 | reduce | first | capped | 330 | 1 | 329 | 1 | 1 |
| training/2020-09 | reduce | restart | all | 31563 | 52 | 31511 | 8 | 7 |
| training/2020-09 | reduce | restart | below | 1499 | 6 | 1493 | 6 | 3 |
| training/2020-09 | reduce | restart | capped | 30064 | 46 | 30018 | 4 | 4 |
| training/2020-09 | increase | quiet | all | 41959 | 70 | 41889 | 17 | 12 |
| training/2020-09 | increase | quiet | below | 11605 | 20 | 11585 | 16 | 11 |
| training/2020-09 | increase | quiet | capped | 30354 | 50 | 30304 | 4 | 4 |
| training/2020-09 | increase | first | all | 1140 | 11 | 1129 | 15 | 11 |
| training/2020-09 | increase | first | below | 1140 | 11 | 1129 | 15 | 11 |
| training/2020-09 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | restart | all | 40819 | 59 | 40760 | 12 | 7 |
| training/2020-09 | increase | restart | below | 10465 | 9 | 10456 | 11 | 6 |
| training/2020-09 | increase | restart | capped | 30354 | 50 | 30304 | 4 | 4 |
| training/2020-10 | reduce | quiet | all | 44016 | 34 | 43982 | 23 | 15 |
| training/2020-10 | reduce | quiet | below | 26913 | 12 | 26901 | 23 | 9 |
| training/2020-10 | reduce | quiet | capped | 17103 | 22 | 17081 | 9 | 9 |
| training/2020-10 | reduce | first | all | 26408 | 13 | 26395 | 23 | 13 |
| training/2020-10 | reduce | first | below | 18845 | 9 | 18836 | 23 | 9 |
| training/2020-10 | reduce | first | capped | 7563 | 4 | 7559 | 4 | 4 |
| training/2020-10 | reduce | restart | all | 17608 | 21 | 17587 | 12 | 9 |
| training/2020-10 | reduce | restart | below | 8068 | 3 | 8065 | 7 | 3 |
| training/2020-10 | reduce | restart | capped | 9540 | 18 | 9522 | 8 | 8 |
| training/2020-10 | increase | quiet | all | 43781 | 39 | 43742 | 23 | 14 |
| training/2020-10 | increase | quiet | below | 26805 | 21 | 26784 | 22 | 13 |
| training/2020-10 | increase | quiet | capped | 16976 | 18 | 16958 | 9 | 7 |
| training/2020-10 | increase | first | all | 7195 | 10 | 7185 | 18 | 10 |
| training/2020-10 | increase | first | below | 7195 | 10 | 7185 | 18 | 10 |
| training/2020-10 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | restart | all | 36586 | 29 | 36557 | 14 | 10 |
| training/2020-10 | increase | restart | below | 19610 | 11 | 19599 | 10 | 6 |
| training/2020-10 | increase | restart | capped | 16976 | 18 | 16958 | 9 | 7 |
| training/2020-11 | reduce | quiet | all | 31161 | 82 | 31079 | 39 | 28 |
| training/2020-11 | reduce | quiet | below | 18421 | 35 | 18386 | 39 | 23 |
| training/2020-11 | reduce | quiet | capped | 12740 | 47 | 12693 | 15 | 13 |
| training/2020-11 | reduce | first | all | 12976 | 24 | 12952 | 39 | 24 |
| training/2020-11 | reduce | first | below | 12426 | 21 | 12405 | 39 | 21 |
| training/2020-11 | reduce | first | capped | 550 | 3 | 547 | 4 | 3 |
| training/2020-11 | reduce | restart | all | 18185 | 58 | 18127 | 18 | 15 |
| training/2020-11 | reduce | restart | below | 5995 | 14 | 5981 | 13 | 9 |
| training/2020-11 | reduce | restart | capped | 12190 | 44 | 12146 | 13 | 12 |
| training/2020-11 | increase | quiet | all | 30705 | 106 | 30599 | 38 | 28 |
| training/2020-11 | increase | quiet | below | 18132 | 55 | 18077 | 37 | 27 |
| training/2020-11 | increase | quiet | capped | 12573 | 51 | 12522 | 15 | 14 |
| training/2020-11 | increase | first | all | 5535 | 23 | 5512 | 32 | 23 |
| training/2020-11 | increase | first | below | 5535 | 23 | 5512 | 32 | 23 |
| training/2020-11 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | restart | all | 25170 | 83 | 25087 | 24 | 19 |
| training/2020-11 | increase | restart | below | 12597 | 32 | 12565 | 21 | 16 |
| training/2020-11 | increase | restart | capped | 12573 | 51 | 12522 | 15 | 14 |
| training/2020-12 | reduce | quiet | all | 39344 | 48 | 39296 | 18 | 12 |
| training/2020-12 | reduce | quiet | below | 23175 | 12 | 23163 | 17 | 8 |
| training/2020-12 | reduce | quiet | capped | 16169 | 36 | 16133 | 7 | 7 |
| training/2020-12 | reduce | first | all | 24762 | 12 | 24750 | 18 | 12 |
| training/2020-12 | reduce | first | below | 21368 | 8 | 21360 | 17 | 8 |
| training/2020-12 | reduce | first | capped | 3394 | 4 | 3390 | 4 | 4 |
| training/2020-12 | reduce | restart | all | 14582 | 36 | 14546 | 9 | 8 |
| training/2020-12 | reduce | restart | below | 1807 | 4 | 1803 | 4 | 2 |
| training/2020-12 | reduce | restart | capped | 12775 | 32 | 12743 | 6 | 6 |
| training/2020-12 | increase | quiet | all | 39318 | 40 | 39278 | 18 | 11 |
| training/2020-12 | increase | quiet | below | 22929 | 24 | 22905 | 17 | 10 |
| training/2020-12 | increase | quiet | capped | 16389 | 16 | 16373 | 7 | 4 |
| training/2020-12 | increase | first | all | 13852 | 7 | 13845 | 13 | 7 |
| training/2020-12 | increase | first | below | 13852 | 7 | 13845 | 13 | 7 |
| training/2020-12 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | restart | all | 25466 | 33 | 25433 | 12 | 10 |
| training/2020-12 | increase | restart | below | 9077 | 17 | 9060 | 10 | 8 |
| training/2020-12 | increase | restart | capped | 16389 | 16 | 16373 | 7 | 4 |
| calibration/2021-01 | reduce | quiet | all | 40342 | 81 | 40261 | 32 | 25 |
| calibration/2021-01 | reduce | quiet | below | 17639 | 23 | 17616 | 32 | 18 |
| calibration/2021-01 | reduce | quiet | capped | 22703 | 58 | 22645 | 13 | 13 |
| calibration/2021-01 | reduce | first | all | 15833 | 25 | 15808 | 32 | 25 |
| calibration/2021-01 | reduce | first | below | 12404 | 18 | 12386 | 32 | 18 |
| calibration/2021-01 | reduce | first | capped | 3429 | 7 | 3422 | 7 | 7 |
| calibration/2021-01 | reduce | restart | all | 24509 | 56 | 24453 | 22 | 16 |
| calibration/2021-01 | reduce | restart | below | 5235 | 5 | 5230 | 13 | 5 |
| calibration/2021-01 | reduce | restart | capped | 19274 | 51 | 19223 | 13 | 13 |
| calibration/2021-01 | increase | quiet | all | 40191 | 75 | 40116 | 33 | 22 |
| calibration/2021-01 | increase | quiet | below | 17428 | 33 | 17395 | 33 | 22 |
| calibration/2021-01 | increase | quiet | capped | 22763 | 42 | 22721 | 13 | 11 |
| calibration/2021-01 | increase | first | all | 5617 | 19 | 5598 | 27 | 19 |
| calibration/2021-01 | increase | first | below | 5617 | 19 | 5598 | 27 | 19 |
| calibration/2021-01 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | increase | restart | all | 34574 | 56 | 34518 | 24 | 15 |
| calibration/2021-01 | increase | restart | below | 11811 | 14 | 11797 | 18 | 10 |
| calibration/2021-01 | increase | restart | capped | 22763 | 42 | 22721 | 13 | 11 |
| calibration/2021-02 | reduce | quiet | all | 39227 | 60 | 39167 | 14 | 12 |
| calibration/2021-02 | reduce | quiet | below | 14727 | 7 | 14720 | 14 | 5 |
| calibration/2021-02 | reduce | quiet | capped | 24500 | 53 | 24447 | 7 | 7 |
| calibration/2021-02 | reduce | first | all | 11594 | 11 | 11583 | 13 | 11 |
| calibration/2021-02 | reduce | first | below | 8837 | 5 | 8832 | 13 | 5 |
| calibration/2021-02 | reduce | first | capped | 2757 | 6 | 2751 | 6 | 6 |
| calibration/2021-02 | reduce | restart | all | 27633 | 49 | 27584 | 11 | 7 |
| calibration/2021-02 | reduce | restart | below | 5890 | 2 | 5888 | 5 | 1 |
| calibration/2021-02 | reduce | restart | capped | 21743 | 47 | 21696 | 7 | 6 |
| calibration/2021-02 | increase | quiet | all | 39150 | 60 | 39090 | 13 | 9 |
| calibration/2021-02 | increase | quiet | below | 14601 | 17 | 14584 | 13 | 9 |
| calibration/2021-02 | increase | quiet | capped | 24549 | 43 | 24506 | 7 | 5 |
| calibration/2021-02 | increase | first | all | 7439 | 8 | 7431 | 12 | 8 |
| calibration/2021-02 | increase | first | below | 7439 | 8 | 7431 | 12 | 8 |
| calibration/2021-02 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | restart | all | 31711 | 52 | 31659 | 9 | 7 |
| calibration/2021-02 | increase | restart | below | 7162 | 9 | 7153 | 6 | 5 |
| calibration/2021-02 | increase | restart | capped | 24549 | 43 | 24506 | 7 | 5 |
| calibration/2021-03 | reduce | quiet | all | 44177 | 26 | 44151 | 2 | 1 |
| calibration/2021-03 | reduce | quiet | below | 3652 | 0 | 3652 | 2 | 0 |
| calibration/2021-03 | reduce | quiet | capped | 40525 | 26 | 40499 | 1 | 1 |
| calibration/2021-03 | reduce | first | all | 2892 | 1 | 2891 | 1 | 1 |
| calibration/2021-03 | reduce | first | below | 2867 | 0 | 2867 | 1 | 0 |
| calibration/2021-03 | reduce | first | capped | 25 | 1 | 24 | 1 | 1 |
| calibration/2021-03 | reduce | restart | all | 41285 | 25 | 41260 | 2 | 1 |
| calibration/2021-03 | reduce | restart | below | 785 | 0 | 785 | 1 | 0 |
| calibration/2021-03 | reduce | restart | capped | 40500 | 25 | 40475 | 1 | 1 |
| calibration/2021-03 | increase | quiet | all | 44162 | 23 | 44139 | 2 | 1 |
| calibration/2021-03 | increase | quiet | below | 3607 | 4 | 3603 | 2 | 1 |
| calibration/2021-03 | increase | quiet | capped | 40555 | 19 | 40536 | 1 | 1 |
| calibration/2021-03 | increase | first | all | 2539 | 1 | 2538 | 1 | 1 |
| calibration/2021-03 | increase | first | below | 2539 | 1 | 2538 | 1 | 1 |
| calibration/2021-03 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | all | 41623 | 22 | 41601 | 2 | 1 |
| calibration/2021-03 | increase | restart | below | 1068 | 3 | 1065 | 2 | 1 |
| calibration/2021-03 | increase | restart | capped | 40555 | 19 | 40536 | 1 | 1 |
| calibration/2021-04 | reduce | quiet | all | 42516 | 36 | 42480 | 3 | 3 |
| calibration/2021-04 | reduce | quiet | below | 2228 | 2 | 2226 | 2 | 1 |
| calibration/2021-04 | reduce | quiet | capped | 40288 | 34 | 40254 | 2 | 2 |
| calibration/2021-04 | reduce | first | all | 3363 | 2 | 3361 | 2 | 2 |
| calibration/2021-04 | reduce | first | below | 2225 | 1 | 2224 | 2 | 1 |
| calibration/2021-04 | reduce | first | capped | 1138 | 1 | 1137 | 1 | 1 |
| calibration/2021-04 | reduce | restart | all | 39153 | 34 | 39119 | 3 | 3 |
| calibration/2021-04 | reduce | restart | below | 3 | 1 | 2 | 1 | 1 |
| calibration/2021-04 | reduce | restart | capped | 39150 | 33 | 39117 | 2 | 2 |
| calibration/2021-04 | increase | quiet | all | 42576 | 31 | 42545 | 3 | 3 |
| calibration/2021-04 | increase | quiet | below | 2198 | 3 | 2195 | 2 | 2 |
| calibration/2021-04 | increase | quiet | capped | 40378 | 28 | 40350 | 2 | 2 |
| calibration/2021-04 | increase | first | all | 4 | 2 | 2 | 2 | 2 |
| calibration/2021-04 | increase | first | below | 4 | 2 | 2 | 2 | 2 |
| calibration/2021-04 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | all | 42572 | 29 | 42543 | 3 | 3 |
| calibration/2021-04 | increase | restart | below | 2194 | 1 | 2193 | 1 | 1 |
| calibration/2021-04 | increase | restart | capped | 40378 | 28 | 40350 | 2 | 2 |
| calibration/2021-05 | reduce | quiet | all | 43202 | 75 | 43127 | 23 | 22 |
| calibration/2021-05 | reduce | quiet | below | 26564 | 25 | 26539 | 22 | 18 |
| calibration/2021-05 | reduce | quiet | capped | 16638 | 50 | 16588 | 9 | 9 |
| calibration/2021-05 | reduce | first | all | 22026 | 21 | 22005 | 22 | 21 |
| calibration/2021-05 | reduce | first | below | 21113 | 18 | 21095 | 22 | 18 |
| calibration/2021-05 | reduce | first | capped | 913 | 3 | 910 | 3 | 3 |
| calibration/2021-05 | reduce | restart | all | 21176 | 54 | 21122 | 18 | 14 |
| calibration/2021-05 | reduce | restart | below | 5451 | 7 | 5444 | 13 | 7 |
| calibration/2021-05 | reduce | restart | capped | 15725 | 47 | 15678 | 9 | 9 |
| calibration/2021-05 | increase | quiet | all | 43226 | 68 | 43158 | 22 | 19 |
| calibration/2021-05 | increase | quiet | below | 26427 | 34 | 26393 | 21 | 18 |
| calibration/2021-05 | increase | quiet | capped | 16799 | 34 | 16765 | 9 | 8 |
| calibration/2021-05 | increase | first | all | 5124 | 16 | 5108 | 18 | 16 |
| calibration/2021-05 | increase | first | below | 5124 | 16 | 5108 | 18 | 16 |
| calibration/2021-05 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | restart | all | 38102 | 52 | 38050 | 20 | 14 |
| calibration/2021-05 | increase | restart | below | 21303 | 18 | 21285 | 19 | 13 |
| calibration/2021-05 | increase | restart | capped | 16799 | 34 | 16765 | 9 | 8 |
| calibration/2021-06 | reduce | quiet | all | 40583 | 69 | 40514 | 20 | 18 |
| calibration/2021-06 | reduce | quiet | below | 9911 | 9 | 9902 | 20 | 7 |
| calibration/2021-06 | reduce | quiet | capped | 30672 | 60 | 30612 | 12 | 12 |
| calibration/2021-06 | reduce | first | all | 17813 | 17 | 17796 | 19 | 17 |
| calibration/2021-06 | reduce | first | below | 9632 | 7 | 9625 | 19 | 7 |
| calibration/2021-06 | reduce | first | capped | 8181 | 10 | 8171 | 10 | 10 |
| calibration/2021-06 | reduce | restart | all | 22770 | 52 | 22718 | 12 | 12 |
| calibration/2021-06 | reduce | restart | below | 279 | 2 | 277 | 4 | 2 |
| calibration/2021-06 | reduce | restart | capped | 22491 | 50 | 22441 | 10 | 10 |
| calibration/2021-06 | increase | quiet | all | 40278 | 78 | 40200 | 20 | 18 |
| calibration/2021-06 | increase | quiet | below | 9575 | 29 | 9546 | 18 | 16 |
| calibration/2021-06 | increase | quiet | capped | 30703 | 49 | 30654 | 12 | 8 |
| calibration/2021-06 | increase | first | all | 4632 | 14 | 4618 | 16 | 14 |
| calibration/2021-06 | increase | first | below | 4632 | 14 | 4618 | 16 | 14 |
| calibration/2021-06 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | increase | restart | all | 35646 | 64 | 35582 | 18 | 14 |
| calibration/2021-06 | increase | restart | below | 4943 | 15 | 4928 | 15 | 11 |
| calibration/2021-06 | increase | restart | capped | 30703 | 49 | 30654 | 12 | 8 |
| diagnosis/2021-07 | reduce | quiet | all | 42679 | 28 | 42651 | 8 | 7 |
| diagnosis/2021-07 | reduce | quiet | below | 18073 | 10 | 18063 | 8 | 4 |
| diagnosis/2021-07 | reduce | quiet | capped | 24606 | 18 | 24588 | 3 | 3 |
| diagnosis/2021-07 | reduce | first | all | 14413 | 7 | 14406 | 8 | 7 |
| diagnosis/2021-07 | reduce | first | below | 9405 | 4 | 9401 | 8 | 4 |
| diagnosis/2021-07 | reduce | first | capped | 5008 | 3 | 5005 | 3 | 3 |
| diagnosis/2021-07 | reduce | restart | all | 28266 | 21 | 28245 | 7 | 7 |
| diagnosis/2021-07 | reduce | restart | below | 8668 | 6 | 8662 | 4 | 4 |
| diagnosis/2021-07 | reduce | restart | capped | 19598 | 15 | 19583 | 3 | 3 |
| diagnosis/2021-07 | increase | quiet | all | 42564 | 33 | 42531 | 8 | 6 |
| diagnosis/2021-07 | increase | quiet | below | 18025 | 14 | 18011 | 8 | 6 |
| diagnosis/2021-07 | increase | quiet | capped | 24539 | 19 | 24520 | 3 | 2 |
| diagnosis/2021-07 | increase | first | all | 3107 | 5 | 3102 | 6 | 5 |
| diagnosis/2021-07 | increase | first | below | 3107 | 5 | 3102 | 6 | 5 |
| diagnosis/2021-07 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | all | 39457 | 28 | 39429 | 7 | 5 |
| diagnosis/2021-07 | increase | restart | below | 14918 | 9 | 14909 | 7 | 5 |
| diagnosis/2021-07 | increase | restart | capped | 24539 | 19 | 24520 | 3 | 2 |
| diagnosis/2021-08 | reduce | quiet | all | 43613 | 51 | 43562 | 10 | 8 |
| diagnosis/2021-08 | reduce | quiet | below | 4991 | 5 | 4986 | 9 | 3 |
| diagnosis/2021-08 | reduce | quiet | capped | 38622 | 46 | 38576 | 6 | 6 |
| diagnosis/2021-08 | reduce | first | all | 9966 | 7 | 9959 | 9 | 7 |
| diagnosis/2021-08 | reduce | first | below | 4080 | 3 | 4077 | 9 | 3 |
| diagnosis/2021-08 | reduce | first | capped | 5886 | 4 | 5882 | 4 | 4 |
| diagnosis/2021-08 | reduce | restart | all | 33647 | 44 | 33603 | 8 | 7 |
| diagnosis/2021-08 | reduce | restart | below | 911 | 2 | 909 | 3 | 1 |
| diagnosis/2021-08 | reduce | restart | capped | 32736 | 42 | 32694 | 6 | 6 |
| diagnosis/2021-08 | increase | quiet | all | 43461 | 60 | 43401 | 10 | 8 |
| diagnosis/2021-08 | increase | quiet | below | 4782 | 20 | 4762 | 9 | 7 |
| diagnosis/2021-08 | increase | quiet | capped | 38679 | 40 | 38639 | 6 | 5 |
| diagnosis/2021-08 | increase | first | all | 1045 | 6 | 1039 | 8 | 6 |
| diagnosis/2021-08 | increase | first | below | 1045 | 6 | 1039 | 8 | 6 |
| diagnosis/2021-08 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | all | 42416 | 54 | 42362 | 8 | 7 |
| diagnosis/2021-08 | increase | restart | below | 3737 | 14 | 3723 | 7 | 6 |
| diagnosis/2021-08 | increase | restart | capped | 38679 | 40 | 38639 | 6 | 5 |
| diagnosis/2021-09 | reduce | quiet | all | 42563 | 38 | 42525 | 14 | 12 |
| diagnosis/2021-09 | reduce | quiet | below | 9766 | 11 | 9755 | 13 | 6 |
| diagnosis/2021-09 | reduce | quiet | capped | 32797 | 27 | 32770 | 6 | 6 |
| diagnosis/2021-09 | reduce | first | all | 14355 | 11 | 14344 | 13 | 11 |
| diagnosis/2021-09 | reduce | first | below | 7708 | 6 | 7702 | 13 | 6 |
| diagnosis/2021-09 | reduce | first | capped | 6647 | 5 | 6642 | 5 | 5 |
| diagnosis/2021-09 | reduce | restart | all | 28208 | 27 | 28181 | 10 | 10 |
| diagnosis/2021-09 | reduce | restart | below | 2058 | 5 | 2053 | 4 | 4 |
| diagnosis/2021-09 | reduce | restart | capped | 26150 | 22 | 26128 | 6 | 6 |
| diagnosis/2021-09 | increase | quiet | all | 42487 | 36 | 42451 | 14 | 12 |
| diagnosis/2021-09 | increase | quiet | below | 9612 | 20 | 9592 | 13 | 11 |
| diagnosis/2021-09 | increase | quiet | capped | 32875 | 16 | 32859 | 6 | 4 |
| diagnosis/2021-09 | increase | first | all | 2603 | 10 | 2593 | 11 | 10 |
| diagnosis/2021-09 | increase | first | below | 2603 | 10 | 2593 | 11 | 10 |
| diagnosis/2021-09 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | restart | all | 39884 | 26 | 39858 | 13 | 8 |
| diagnosis/2021-09 | increase | restart | below | 7009 | 10 | 6999 | 10 | 6 |
| diagnosis/2021-09 | increase | restart | capped | 32875 | 16 | 32859 | 6 | 4 |
| diagnosis/2021-10 | reduce | quiet | all | 44478 | 12 | 44466 | 3 | 2 |
| diagnosis/2021-10 | reduce | quiet | below | 23417 | 1 | 23416 | 3 | 1 |
| diagnosis/2021-10 | reduce | quiet | capped | 21061 | 11 | 21050 | 2 | 2 |
| diagnosis/2021-10 | reduce | first | all | 18173 | 2 | 18171 | 3 | 2 |
| diagnosis/2021-10 | reduce | first | below | 16981 | 1 | 16980 | 3 | 1 |
| diagnosis/2021-10 | reduce | first | capped | 1192 | 1 | 1191 | 1 | 1 |
| diagnosis/2021-10 | reduce | restart | all | 26305 | 10 | 26295 | 2 | 2 |
| diagnosis/2021-10 | reduce | restart | below | 6436 | 0 | 6436 | 1 | 0 |
| diagnosis/2021-10 | reduce | restart | capped | 19869 | 10 | 19859 | 2 | 2 |
| diagnosis/2021-10 | increase | quiet | all | 44484 | 8 | 44476 | 3 | 2 |
| diagnosis/2021-10 | increase | quiet | below | 23408 | 3 | 23405 | 3 | 2 |
| diagnosis/2021-10 | increase | quiet | capped | 21076 | 5 | 21071 | 2 | 1 |
| diagnosis/2021-10 | increase | first | all | 1110 | 1 | 1109 | 2 | 1 |
| diagnosis/2021-10 | increase | first | below | 1110 | 1 | 1109 | 2 | 1 |
| diagnosis/2021-10 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | all | 43374 | 7 | 43367 | 2 | 2 |
| diagnosis/2021-10 | increase | restart | below | 22298 | 2 | 22296 | 2 | 2 |
| diagnosis/2021-10 | increase | restart | capped | 21076 | 5 | 21071 | 2 | 1 |
| diagnosis/2021-11 | reduce | quiet | all | 43101 | 7 | 43094 | 2 | 2 |
| diagnosis/2021-11 | reduce | quiet | below | 14269 | 1 | 14268 | 2 | 1 |
| diagnosis/2021-11 | reduce | quiet | capped | 28832 | 6 | 28826 | 1 | 1 |
| diagnosis/2021-11 | reduce | first | all | 22257 | 2 | 22255 | 2 | 2 |
| diagnosis/2021-11 | reduce | first | below | 14269 | 1 | 14268 | 2 | 1 |
| diagnosis/2021-11 | reduce | first | capped | 7988 | 1 | 7987 | 1 | 1 |
| diagnosis/2021-11 | reduce | restart | all | 20844 | 5 | 20839 | 1 | 1 |
| diagnosis/2021-11 | reduce | restart | below | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | capped | 20844 | 5 | 20839 | 1 | 1 |
| diagnosis/2021-11 | increase | quiet | all | 43068 | 8 | 43060 | 2 | 1 |
| diagnosis/2021-11 | increase | quiet | below | 14270 | 1 | 14269 | 2 | 1 |
| diagnosis/2021-11 | increase | quiet | capped | 28798 | 7 | 28791 | 1 | 1 |
| diagnosis/2021-11 | increase | first | all | 14270 | 1 | 14269 | 2 | 1 |
| diagnosis/2021-11 | increase | first | below | 14270 | 1 | 14269 | 2 | 1 |
| diagnosis/2021-11 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | all | 28798 | 7 | 28791 | 1 | 1 |
| diagnosis/2021-11 | increase | restart | below | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | capped | 28798 | 7 | 28791 | 1 | 1 |
| diagnosis/2021-12 | reduce | quiet | all | 33267 | 4 | 33263 | 1 | 1 |
| diagnosis/2021-12 | reduce | quiet | below | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | capped | 33267 | 4 | 33263 | 1 | 1 |
| diagnosis/2021-12 | reduce | first | all | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | below | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | capped | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | all | 33267 | 4 | 33263 | 1 | 1 |
| diagnosis/2021-12 | reduce | restart | below | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | capped | 33267 | 4 | 33263 | 1 | 1 |
| diagnosis/2021-12 | increase | quiet | all | 33314 | 1 | 33313 | 1 | 1 |
| diagnosis/2021-12 | increase | quiet | below | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | capped | 33314 | 1 | 33313 | 1 | 1 |
| diagnosis/2021-12 | increase | first | all | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | first | below | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | first | capped | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | all | 33314 | 1 | 33313 | 1 | 1 |
| diagnosis/2021-12 | increase | restart | below | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | capped | 33314 | 1 | 33313 | 1 | 1 |

## 모든 집단의 점수와 요청

| 기간·월 | 행동 | 상태 | 횟수 | 모델 | 요청 | 문턱 지원 | 미지원 행 | 행 | 양성 | 로그 손실 | 평균 정밀도 | Brier | TP/FP/FN | 요청 수 | 정밀도 | 재현율 | F | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training | reduce | quiet | all | v127 | original | supported | 0 | 497219 | 744 | 0.008972666966 | 0.05049760965 | 0.001464298419 | 40/147/704 | 187 | 0.2139037433 | 0.05376344086 | 0.08592910849 | 0.0002960874163 |
| training | reduce | quiet | all | v127 | budget_matched | supported | 0 | 497219 | 744 | 0.008972666966 | 0.05049760965 | 0.001464298419 | 48/205/696 | 253 | 0.1897233202 | 0.06451612903 | 0.0962888666 | 0.0004129110227 |
| training | reduce | quiet | all | v137 | original | supported | 0 | 497219 | 744 | 0.009012218678 | 0.05098945613 | 0.00146477817 | 77/459/667 | 536 | 0.1436567164 | 0.1034946237 | 0.1203125 | 0.0009245178508 |
| training | reduce | quiet | all | v137 | budget_matched | supported | 0 | 497219 | 744 | 0.009012218678 | 0.05098945613 | 0.00146477817 | 40/188/704 | 228 | 0.1754385965 | 0.05376344086 | 0.08230452675 | 0.0003786696208 |
| training | reduce | quiet | below | v127 | original | supported | 0 | 203394 | 244 | 0.007249105944 | 0.0549585653 | 0.001177384367 | 15/60/229 | 75 | 0.2 | 0.06147540984 | 0.09404388715 | 0.0002953482648 |
| training | reduce | quiet | below | v127 | budget_matched | supported | 0 | 203394 | 244 | 0.007249105944 | 0.0549585653 | 0.001177384367 | 19/98/225 | 117 | 0.1623931624 | 0.07786885246 | 0.1052631579 | 0.0004824021659 |
| training | reduce | quiet | below | v137 | original | supported | 0 | 203394 | 244 | 0.007285520396 | 0.0614145714 | 0.001178223041 | 27/135/217 | 162 | 0.1666666667 | 0.1106557377 | 0.1330049261 | 0.0006645335959 |
| training | reduce | quiet | below | v137 | budget_matched | supported | 0 | 203394 | 244 | 0.007285520396 | 0.0614145714 | 0.001178223041 | 14/42/230 | 56 | 0.25 | 0.05737704918 | 0.09333333333 | 0.0002067437854 |
| training | reduce | quiet | capped | v127 | original | supported | 0 | 293825 | 500 | 0.0101657648 | 0.05224255277 | 0.001662908465 | 25/87/475 | 112 | 0.2232142857 | 0.05 | 0.08169934641 | 0.0002965993352 |
| training | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 293825 | 500 | 0.0101657648 | 0.05224255277 | 0.001662908465 | 37/136/463 | 173 | 0.2138728324 | 0.074 | 0.1099554235 | 0.0004636495355 |
| training | reduce | quiet | capped | v137 | original | supported | 0 | 293825 | 500 | 0.01020748821 | 0.05112779531 | 0.001663139759 | 50/324/450 | 374 | 0.1336898396 | 0.1 | 0.114416476 | 0.001104576835 |
| training | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 293825 | 500 | 0.01020748821 | 0.05112779531 | 0.001663139759 | 25/132/475 | 157 | 0.1592356688 | 0.05 | 0.07610350076 | 0.0004500127845 |
| training | reduce | first | all | v127 | original | supported | 0 | 136178 | 194 | 0.008709330099 | 0.05926525347 | 0.001402265767 | 15/63/179 | 78 | 0.1923076923 | 0.07731958763 | 0.1102941176 | 0.0004632897988 |
| training | reduce | first | all | v127 | budget_matched | supported | 0 | 136178 | 194 | 0.008709330099 | 0.05926525347 | 0.001402265767 | 15/64/179 | 79 | 0.1898734177 | 0.07731958763 | 0.1098901099 | 0.0004706436051 |
| training | reduce | first | all | v137 | original | supported | 0 | 136178 | 194 | 0.008861800599 | 0.07469226174 | 0.001405730384 | 22/130/172 | 152 | 0.1447368421 | 0.1134020619 | 0.1271676301 | 0.0009559948229 |
| training | reduce | first | all | v137 | budget_matched | supported | 0 | 136178 | 194 | 0.008861800599 | 0.07469226174 | 0.001405730384 | 7/21/187 | 28 | 0.25 | 0.03608247423 | 0.06306306306 | 0.0001544299329 |
| training | reduce | first | below | v127 | original | supported | 0 | 121200 | 161 | 0.00818639628 | 0.05728688225 | 0.001308736787 | 12/46/149 | 58 | 0.2068965517 | 0.07453416149 | 0.1095890411 | 0.0003800427961 |
| training | reduce | first | below | v127 | budget_matched | supported | 0 | 121200 | 161 | 0.00818639628 | 0.05728688225 | 0.001308736787 | 12/47/149 | 59 | 0.2033898305 | 0.07453416149 | 0.1090909091 | 0.000388304596 |
| training | reduce | first | below | v137 | original | supported | 0 | 121200 | 161 | 0.008310435208 | 0.08300305423 | 0.001312207915 | 16/90/145 | 106 | 0.1509433962 | 0.09937888199 | 0.1198501873 | 0.0007435619924 |
| training | reduce | first | below | v137 | budget_matched | supported | 0 | 121200 | 161 | 0.008310435208 | 0.08300305423 | 0.001312207915 | 4/8/157 | 12 | 0.3333333333 | 0.0248447205 | 0.04624277457 | 6.609439933e-05 |
| training | reduce | first | capped | v127 | original | supported | 0 | 14978 | 33 | 0.01294084158 | 0.08748910701 | 0.002159089929 | 3/17/30 | 20 | 0.15 | 0.09090909091 | 0.1132075472 | 0.001137504182 |
| training | reduce | first | capped | v127 | budget_matched | supported | 0 | 14978 | 33 | 0.01294084158 | 0.08748910701 | 0.002159089929 | 3/17/30 | 20 | 0.15 | 0.09090909091 | 0.1132075472 | 0.001137504182 |
| training | reduce | first | capped | v137 | original | supported | 0 | 14978 | 33 | 0.0133233766 | 0.08977367256 | 0.002162501861 | 6/40/27 | 46 | 0.1304347826 | 0.1818181818 | 0.1518987342 | 0.002676480428 |
| training | reduce | first | capped | v137 | budget_matched | supported | 0 | 14978 | 33 | 0.0133233766 | 0.08977367256 | 0.002162501861 | 2/13/31 | 15 | 0.1333333333 | 0.06060606061 | 0.08333333333 | 0.0008698561392 |
| training | reduce | restart | all | v127 | original | supported | 0 | 361041 | 550 | 0.00907199277 | 0.05291859133 | 0.00148769599 | 25/84/525 | 109 | 0.2293577982 | 0.04545454545 | 0.07587253414 | 0.000233015526 |
| training | reduce | restart | all | v127 | budget_matched | supported | 0 | 361041 | 550 | 0.00907199277 | 0.05291859133 | 0.00148769599 | 25/85/525 | 110 | 0.2272727273 | 0.04545454545 | 0.07575757576 | 0.0002357895204 |
| training | reduce | restart | all | v137 | original | supported | 0 | 361041 | 550 | 0.0090689536 | 0.05373578455 | 0.001487049905 | 55/329/495 | 384 | 0.1432291667 | 0.1 | 0.1177730193 | 0.0009126441437 |
| training | reduce | restart | all | v137 | budget_matched | supported | 0 | 361041 | 550 | 0.0090689536 | 0.05373578455 | 0.001487049905 | 22/85/528 | 107 | 0.2056074766 | 0.04 | 0.06697108067 | 0.0002357895204 |
| training | reduce | restart | below | v127 | original | supported | 0 | 82194 | 83 | 0.00586701493 | 0.07478648804 | 0.0009836973186 | 3/14/80 | 17 | 0.1764705882 | 0.03614457831 | 0.06 | 0.0001705009073 |
| training | reduce | restart | below | v127 | budget_matched | supported | 0 | 82194 | 83 | 0.00586701493 | 0.07478648804 | 0.0009836973186 | 3/14/80 | 17 | 0.1764705882 | 0.03614457831 | 0.06 | 0.0001705009073 |
| training | reduce | restart | below | v137 | original | supported | 0 | 82194 | 83 | 0.00577422182 | 0.08319846559 | 0.0009806542812 | 11/45/72 | 56 | 0.1964285714 | 0.1325301205 | 0.1582733813 | 0.0005480386306 |
| training | reduce | restart | below | v137 | budget_matched | supported | 0 | 82194 | 83 | 0.00577422182 | 0.08319846559 | 0.0009806542812 | 2/2/81 | 4 | 0.5 | 0.02409638554 | 0.04597701149 | 2.435727247e-05 |
| training | reduce | restart | capped | v127 | original | supported | 0 | 278847 | 467 | 0.0100167042 | 0.05212284567 | 0.001636256552 | 22/70/445 | 92 | 0.2391304348 | 0.04710920771 | 0.07871198569 | 0.0002514548459 |
| training | reduce | restart | capped | v127 | budget_matched | supported | 0 | 278847 | 467 | 0.0100167042 | 0.05212284567 | 0.001636256552 | 22/71/445 | 93 | 0.2365591398 | 0.04710920771 | 0.07857142857 | 0.000255047058 |
| training | reduce | restart | capped | v137 | original | supported | 0 | 278847 | 467 | 0.01004012124 | 0.05098270961 | 0.001636317002 | 44/284/423 | 328 | 0.1341463415 | 0.09421841542 | 0.1106918239 | 0.001020188232 |
| training | reduce | restart | capped | v137 | budget_matched | supported | 0 | 278847 | 467 | 0.01004012124 | 0.05098270961 | 0.001636317002 | 20/81/447 | 101 | 0.198019802 | 0.04282655246 | 0.07042253521 | 0.0002909691788 |
| training | increase | quiet | all | v127 | original | supported | 0 | 494681 | 827 | 0.009945453247 | 0.05107181369 | 0.001631600091 | 28/96/799 | 124 | 0.2258064516 | 0.0338573156 | 0.1058201058 | 0.0001943894349 |
| training | increase | quiet | all | v127 | budget_matched | supported | 0 | 494681 | 827 | 0.009945453247 | 0.05107181369 | 0.001631600091 | 35/113/792 | 148 | 0.2364864865 | 0.0423216445 | 0.1233262861 | 0.000228812564 |
| training | increase | quiet | all | v137 | original | supported | 0 | 494681 | 827 | 0.009936606287 | 0.06270235966 | 0.001628536235 | 81/625/746 | 706 | 0.1147308782 | 0.09794437727 | 0.1109285127 | 0.001265556217 |
| training | increase | quiet | all | v137 | budget_matched | supported | 0 | 494681 | 827 | 0.009936606287 | 0.06270235966 | 0.001628536235 | 38/110/789 | 148 | 0.2567567568 | 0.04594921403 | 0.1338971106 | 0.0002227378942 |
| training | increase | quiet | below | v127 | original | supported | 0 | 201412 | 366 | 0.00990579738 | 0.07765842556 | 0.001747240715 | 23/84/343 | 107 | 0.214953271 | 0.06284153005 | 0.144836272 | 0.0004178148284 |
| training | increase | quiet | below | v127 | budget_matched | supported | 0 | 201412 | 366 | 0.00990579738 | 0.07765842556 | 0.001747240715 | 25/79/341 | 104 | 0.2403846154 | 0.06830601093 | 0.1598465473 | 0.0003929448982 |
| training | increase | quiet | below | v137 | original | supported | 0 | 201412 | 366 | 0.00992920459 | 0.1007538293 | 0.001740896714 | 48/237/318 | 285 | 0.1684210526 | 0.131147541 | 0.1593625498 | 0.001178834695 |
| training | increase | quiet | below | v137 | budget_matched | supported | 0 | 201412 | 366 | 0.00992920459 | 0.1007538293 | 0.001740896714 | 28/70/338 | 98 | 0.2857142857 | 0.07650273224 | 0.1846965699 | 0.0003481790237 |
| training | increase | quiet | capped | v127 | original | supported | 0 | 293269 | 461 | 0.009972688201 | 0.03060658468 | 0.00155218014 | 5/12/456 | 17 | 0.2941176471 | 0.01084598698 | 0.04725897921 | 4.098248682e-05 |
| training | increase | quiet | capped | v127 | budget_matched | supported | 0 | 293269 | 461 | 0.009972688201 | 0.03060658468 | 0.00155218014 | 5/13/456 | 18 | 0.2777777778 | 0.01084598698 | 0.0469043152 | 4.439769405e-05 |
| training | increase | quiet | capped | v137 | original | supported | 0 | 293269 | 461 | 0.009941689643 | 0.03115668474 | 0.001551369031 | 33/388/428 | 421 | 0.0783847981 | 0.0715835141 | 0.07692307692 | 0.001325100407 |
| training | increase | quiet | capped | v137 | budget_matched | supported | 0 | 293269 | 461 | 0.009941689643 | 0.03115668474 | 0.001551369031 | 12/56/449 | 68 | 0.1764705882 | 0.02603036876 | 0.08185538881 | 0.0001912516051 |
| training | increase | first | all | v127 | original | supported | 0 | 65424 | 152 | 0.01080013585 | 0.1258302208 | 0.002173067053 | 16/54/136 | 70 | 0.2285714286 | 0.1052631579 | 0.1851851852 | 0.000827307268 |
| training | increase | first | all | v127 | budget_matched | supported | 0 | 65424 | 152 | 0.01080013585 | 0.1258302208 | 0.002173067053 | 16/54/136 | 70 | 0.2285714286 | 0.1052631579 | 0.1851851852 | 0.000827307268 |
| training | increase | first | all | v137 | original | supported | 0 | 65424 | 152 | 0.01056230462 | 0.1743167881 | 0.002145563541 | 19/29/133 | 48 | 0.3958333333 | 0.125 | 0.2761627907 | 0.000444294644 |
| training | increase | first | all | v137 | budget_matched | supported | 0 | 65424 | 152 | 0.01056230462 | 0.1743167881 | 0.002145563541 | 19/42/133 | 61 | 0.3114754098 | 0.125 | 0.2398989899 | 0.0006434612085 |
| training | increase | first | below | v127 | original | supported | 0 | 65424 | 152 | 0.01080013585 | 0.1258302208 | 0.002173067053 | 16/54/136 | 70 | 0.2285714286 | 0.1052631579 | 0.1851851852 | 0.000827307268 |
| training | increase | first | below | v127 | budget_matched | supported | 0 | 65424 | 152 | 0.01080013585 | 0.1258302208 | 0.002173067053 | 16/54/136 | 70 | 0.2285714286 | 0.1052631579 | 0.1851851852 | 0.000827307268 |
| training | increase | first | below | v137 | original | supported | 0 | 65424 | 152 | 0.01056230462 | 0.1743167881 | 0.002145563541 | 19/29/133 | 48 | 0.3958333333 | 0.125 | 0.2761627907 | 0.000444294644 |
| training | increase | first | below | v137 | budget_matched | supported | 0 | 65424 | 152 | 0.01056230462 | 0.1743167881 | 0.002145563541 | 19/42/133 | 61 | 0.3114754098 | 0.125 | 0.2398989899 | 0.0006434612085 |
| training | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training | increase | restart | all | v127 | original | supported | 0 | 429257 | 675 | 0.009815189199 | 0.03445183418 | 0.001549073924 | 12/42/663 | 54 | 0.2222222222 | 0.01777777778 | 0.06734006734 | 9.799758273e-05 |
| training | increase | restart | all | v127 | budget_matched | supported | 0 | 429257 | 675 | 0.009815189199 | 0.03445183418 | 0.001549073924 | 12/44/663 | 56 | 0.2142857143 | 0.01777777778 | 0.06674082314 | 0.0001026641343 |
| training | increase | restart | all | v137 | original | supported | 0 | 429257 | 675 | 0.009841242233 | 0.03542380754 | 0.001549734971 | 62/596/613 | 658 | 0.09422492401 | 0.09185185185 | 0.09374055035 | 0.001390632364 |
| training | increase | restart | all | v137 | budget_matched | supported | 0 | 429257 | 675 | 0.009841242233 | 0.03542380754 | 0.001549734971 | 18/73/657 | 91 | 0.1978021978 | 0.02666666667 | 0.08662175168 | 0.0001703291319 |
| training | increase | restart | below | v127 | original | supported | 0 | 135988 | 214 | 0.009475530005 | 0.04308138989 | 0.001542375121 | 7/30/207 | 37 | 0.1891891892 | 0.03271028037 | 0.09668508287 | 0.0002209554112 |
| training | increase | restart | below | v127 | budget_matched | supported | 0 | 135988 | 214 | 0.009475530005 | 0.04308138989 | 0.001542375121 | 7/31/207 | 38 | 0.1842105263 | 0.03271028037 | 0.0956284153 | 0.0002283205916 |
| training | increase | restart | below | v137 | original | supported | 0 | 135988 | 214 | 0.009624619358 | 0.04813913152 | 0.001546210988 | 29/208/185 | 237 | 0.1223628692 | 0.1355140187 | 0.1247848537 | 0.001531957518 |
| training | increase | restart | below | v137 | budget_matched | supported | 0 | 135988 | 214 | 0.009624619358 | 0.04813913152 | 0.001546210988 | 6/17/208 | 23 | 0.2608695652 | 0.02803738318 | 0.09803921569 | 0.0001252080663 |
| training | increase | restart | capped | v127 | original | supported | 0 | 293269 | 461 | 0.009972688201 | 0.03060658468 | 0.00155218014 | 5/12/456 | 17 | 0.2941176471 | 0.01084598698 | 0.04725897921 | 4.098248682e-05 |
| training | increase | restart | capped | v127 | budget_matched | supported | 0 | 293269 | 461 | 0.009972688201 | 0.03060658468 | 0.00155218014 | 5/13/456 | 18 | 0.2777777778 | 0.01084598698 | 0.0469043152 | 4.439769405e-05 |
| training | increase | restart | capped | v137 | original | supported | 0 | 293269 | 461 | 0.009941689643 | 0.03115668474 | 0.001551369031 | 33/388/428 | 421 | 0.0783847981 | 0.0715835141 | 0.07692307692 | 0.001325100407 |
| training | increase | restart | capped | v137 | budget_matched | supported | 0 | 293269 | 461 | 0.009941689643 | 0.03115668474 | 0.001551369031 | 12/56/449 | 68 | 0.1764705882 | 0.02603036876 | 0.08185538881 | 0.0001912516051 |
| calibration | reduce | quiet | all | v127 | original | supported | 0 | 250047 | 347 | 0.008837873749 | 0.02326799619 | 0.001370939028 | 20/189/327 | 209 | 0.0956937799 | 0.05763688761 | 0.07194244604 | 0.0007569082899 |
| calibration | reduce | quiet | all | v127 | budget_matched | supported | 0 | 250047 | 347 | 0.008837873749 | 0.02326799619 | 0.001370939028 | 16/188/331 | 204 | 0.07843137255 | 0.04610951009 | 0.05807622505 | 0.0007529034842 |
| calibration | reduce | quiet | all | v137 | original | supported | 0 | 250047 | 347 | 0.008941560803 | 0.02007423102 | 0.001372707106 | 39/732/308 | 771 | 0.05058365759 | 0.1123919308 | 0.06976744186 | 0.002931517821 |
| calibration | reduce | quiet | all | v137 | budget_matched | supported | 0 | 250047 | 347 | 0.008941560803 | 0.02007423102 | 0.001372707106 | 15/186/332 | 201 | 0.07462686567 | 0.04322766571 | 0.05474452555 | 0.0007448938726 |
| calibration | reduce | quiet | below | v127 | original | supported | 0 | 74721 | 66 | 0.005913867265 | 0.01407576374 | 0.0008773286766 | 4/50/62 | 54 | 0.07407407407 | 0.06060606061 | 0.06666666667 | 0.0006697475052 |
| calibration | reduce | quiet | below | v127 | budget_matched | supported | 0 | 74721 | 66 | 0.005913867265 | 0.01407576374 | 0.0008773286766 | 1/50/65 | 51 | 0.01960784314 | 0.01515151515 | 0.01709401709 | 0.0006697475052 |
| calibration | reduce | quiet | below | v137 | original | supported | 0 | 74721 | 66 | 0.006047032215 | 0.01368052271 | 0.0008781583224 | 12/275/54 | 287 | 0.04181184669 | 0.1818181818 | 0.06798866856 | 0.003683611279 |
| calibration | reduce | quiet | below | v137 | budget_matched | supported | 0 | 74721 | 66 | 0.006047032215 | 0.01368052271 | 0.0008781583224 | 2/49/64 | 51 | 0.03921568627 | 0.0303030303 | 0.03418803419 | 0.0006563525551 |
| calibration | reduce | quiet | capped | v127 | original | supported | 0 | 175326 | 281 | 0.01008403626 | 0.02578032376 | 0.001581307478 | 16/139/265 | 155 | 0.1032258065 | 0.05693950178 | 0.07339449541 | 0.0007940815219 |
| calibration | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 175326 | 281 | 0.01008403626 | 0.02578032376 | 0.001581307478 | 15/139/266 | 154 | 0.0974025974 | 0.05338078292 | 0.06896551724 | 0.0007940815219 |
| calibration | reduce | quiet | capped | v137 | original | supported | 0 | 175326 | 281 | 0.01017516033 | 0.02265596268 | 0.001583475501 | 27/457/254 | 484 | 0.05578512397 | 0.09608540925 | 0.07058823529 | 0.002610757234 |
| calibration | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 175326 | 281 | 0.01017516033 | 0.02265596268 | 0.001583475501 | 14/139/267 | 153 | 0.09150326797 | 0.04982206406 | 0.06451612903 | 0.0007940815219 |
| calibration | reduce | first | all | v127 | original | supported | 0 | 73521 | 77 | 0.00691939759 | 0.01480799793 | 0.001039711847 | 6/108/71 | 114 | 0.05263157895 | 0.07792207792 | 0.06282722513 | 0.001470508142 |
| calibration | reduce | first | all | v127 | budget_matched | supported | 0 | 73521 | 77 | 0.00691939759 | 0.01480799793 | 0.001039711847 | 6/108/71 | 114 | 0.05263157895 | 0.07792207792 | 0.06282722513 | 0.001470508142 |
| calibration | reduce | first | all | v137 | original | supported | 0 | 73521 | 77 | 0.006928936713 | 0.01437423221 | 0.001039648222 | 15/473/62 | 488 | 0.03073770492 | 0.1948051948 | 0.05309734513 | 0.00644028103 |
| calibration | reduce | first | all | v137 | budget_matched | supported | 0 | 73521 | 77 | 0.006928936713 | 0.01437423221 | 0.001039648222 | 4/108/73 | 112 | 0.03571428571 | 0.05194805195 | 0.04232804233 | 0.001470508142 |
| calibration | reduce | first | below | v127 | original | supported | 0 | 57078 | 49 | 0.005571432234 | 0.02342199372 | 0.00084886395 | 4/47/45 | 51 | 0.07843137255 | 0.08163265306 | 0.08 | 0.0008241421031 |
| calibration | reduce | first | below | v127 | budget_matched | supported | 0 | 57078 | 49 | 0.005571432234 | 0.02342199372 | 0.00084886395 | 4/47/45 | 51 | 0.07843137255 | 0.08163265306 | 0.08 | 0.0008241421031 |
| calibration | reduce | first | below | v137 | original | supported | 0 | 57078 | 49 | 0.005571313518 | 0.02419412152 | 0.0008480110854 | 11/240/38 | 251 | 0.0438247012 | 0.2244897959 | 0.07333333333 | 0.004208385208 |
| calibration | reduce | first | below | v137 | budget_matched | supported | 0 | 57078 | 49 | 0.005571313518 | 0.02419412152 | 0.0008480110854 | 4/47/45 | 51 | 0.07843137255 | 0.08163265306 | 0.08 | 0.0008241421031 |
| calibration | reduce | first | capped | v127 | original | supported | 0 | 16443 | 28 | 0.0115985417 | 0.009838446134 | 0.001702195352 | 2/61/26 | 63 | 0.03174603175 | 0.07142857143 | 0.04395604396 | 0.003716113311 |
| calibration | reduce | first | capped | v127 | budget_matched | supported | 0 | 16443 | 28 | 0.0115985417 | 0.009838446134 | 0.001702195352 | 2/61/26 | 63 | 0.03174603175 | 0.07142857143 | 0.04395604396 | 0.003716113311 |
| calibration | reduce | first | capped | v137 | original | supported | 0 | 16443 | 28 | 0.01164160573 | 0.008847527052 | 0.001704871384 | 4/233/24 | 237 | 0.01687763713 | 0.1428571429 | 0.03018867925 | 0.01419433445 |
| calibration | reduce | first | capped | v137 | budget_matched | supported | 0 | 16443 | 28 | 0.01164160573 | 0.008847527052 | 0.001704871384 | 0/61/28 | 61 | 0 | 0 | 0 | 0.003716113311 |
| calibration | reduce | restart | all | v127 | original | supported | 0 | 176526 | 270 | 0.009636896475 | 0.02609804916 | 0.001508891247 | 14/81/256 | 95 | 0.1473684211 | 0.05185185185 | 0.07671232877 | 0.0004595588235 |
| calibration | reduce | restart | all | v127 | budget_matched | supported | 0 | 176526 | 270 | 0.009636896475 | 0.02609804916 | 0.001508891247 | 14/81/256 | 95 | 0.1473684211 | 0.05185185185 | 0.07671232877 | 0.0004595588235 |
| calibration | reduce | restart | all | v137 | original | supported | 0 | 176526 | 270 | 0.009779795034 | 0.02187264089 | 0.001511422209 | 24/259/246 | 283 | 0.08480565371 | 0.08888888889 | 0.08679927667 | 0.001469453522 |
| calibration | reduce | restart | all | v137 | budget_matched | supported | 0 | 176526 | 270 | 0.009779795034 | 0.02187264089 | 0.001511422209 | 9/81/261 | 90 | 0.1 | 0.03333333333 | 0.05 | 0.0004595588235 |
| calibration | reduce | restart | below | v127 | original | supported | 0 | 17643 | 17 | 0.007021700777 | 0.006713372602 | 0.0009694167376 | 0/3/17 | 3 | 0 | 0 | 0 | 0.000170203109 |
| calibration | reduce | restart | below | v127 | budget_matched | supported | 0 | 17643 | 17 | 0.007021700777 | 0.006713372602 | 0.0009694167376 | 0/3/17 | 3 | 0 | 0 | 0 | 0.000170203109 |
| calibration | reduce | restart | below | v137 | original | supported | 0 | 17643 | 17 | 0.007586060259 | 0.006409130084 | 0.0009756895811 | 1/35/16 | 36 | 0.02777777778 | 0.05882352941 | 0.03773584906 | 0.001985702939 |
| calibration | reduce | restart | below | v137 | budget_matched | supported | 0 | 17643 | 17 | 0.007586060259 | 0.006409130084 | 0.0009756895811 | 0/3/17 | 3 | 0 | 0 | 0 | 0.000170203109 |
| calibration | reduce | restart | capped | v127 | original | supported | 0 | 158883 | 253 | 0.009927298203 | 0.02830229671 | 0.001568796642 | 14/78/239 | 92 | 0.152173913 | 0.05533596838 | 0.08115942029 | 0.0004917102692 |
| calibration | reduce | restart | capped | v127 | budget_matched | supported | 0 | 158883 | 253 | 0.009927298203 | 0.02830229671 | 0.001568796642 | 14/78/239 | 92 | 0.152173913 | 0.05533596838 | 0.08115942029 | 0.0004917102692 |
| calibration | reduce | restart | capped | v137 | original | supported | 0 | 158883 | 253 | 0.01002339606 | 0.02538476215 | 0.00157091209 | 23/224/230 | 247 | 0.09311740891 | 0.09090909091 | 0.092 | 0.001412091029 |
| calibration | reduce | restart | capped | v137 | budget_matched | supported | 0 | 158883 | 253 | 0.01002339606 | 0.02538476215 | 0.00157091209 | 10/78/243 | 88 | 0.1136363636 | 0.0395256917 | 0.05865102639 | 0.0004917102692 |
| calibration | increase | quiet | all | v127 | original | supported | 0 | 249583 | 335 | 0.008708550346 | 0.02032431885 | 0.001328861107 | 8/80/327 | 88 | 0.09090909091 | 0.02388059701 | 0.05822416303 | 0.0003209654641 |
| calibration | increase | quiet | all | v127 | budget_matched | supported | 0 | 249583 | 335 | 0.008708550346 | 0.02032431885 | 0.001328861107 | 7/80/328 | 87 | 0.08045977011 | 0.02089552239 | 0.05124450952 | 0.0003209654641 |
| calibration | increase | quiet | all | v137 | original | supported | 0 | 249583 | 335 | 0.008754560587 | 0.02471176308 | 0.001326365507 | 23/484/312 | 507 | 0.04536489152 | 0.06865671642 | 0.04866694879 | 0.001941841058 |
| calibration | increase | quiet | all | v137 | budget_matched | supported | 0 | 249583 | 335 | 0.008754560587 | 0.02471176308 | 0.001326365507 | 7/80/328 | 87 | 0.08045977011 | 0.02089552239 | 0.05124450952 | 0.0003209654641 |
| calibration | increase | quiet | below | v127 | original | supported | 0 | 73836 | 120 | 0.009503196062 | 0.03874747161 | 0.001592689632 | 7/47/113 | 54 | 0.1296296296 | 0.05833333333 | 0.1041666667 | 0.0006375820717 |
| calibration | increase | quiet | below | v127 | budget_matched | supported | 0 | 73836 | 120 | 0.009503196062 | 0.03874747161 | 0.001592689632 | 4/47/116 | 51 | 0.07843137255 | 0.03333333333 | 0.06172839506 | 0.0006375820717 |
| calibration | increase | quiet | below | v137 | original | supported | 0 | 73836 | 120 | 0.009564688627 | 0.04972165193 | 0.00158759701 | 12/169/108 | 181 | 0.06629834254 | 0.1 | 0.07109004739 | 0.002292582343 |
| calibration | increase | quiet | below | v137 | budget_matched | supported | 0 | 73836 | 120 | 0.009564688627 | 0.04972165193 | 0.00158759701 | 6/47/114 | 53 | 0.1132075472 | 0.05 | 0.09036144578 | 0.0006375820717 |
| calibration | increase | quiet | capped | v127 | original | supported | 0 | 175747 | 215 | 0.008374698496 | 0.01156189755 | 0.001218019711 | 1/33/214 | 34 | 0.02941176471 | 0.004651162791 | 0.01424501425 | 0.0001879999088 |
| calibration | increase | quiet | capped | v127 | budget_matched | supported | 0 | 175747 | 215 | 0.008374698496 | 0.01156189755 | 0.001218019711 | 1/33/214 | 34 | 0.02941176471 | 0.004651162791 | 0.01424501425 | 0.0001879999088 |
| calibration | increase | quiet | capped | v137 | original | supported | 0 | 175747 | 215 | 0.008414204201 | 0.01054978874 | 0.001216615189 | 11/315/204 | 326 | 0.03374233129 | 0.0511627907 | 0.0362080316 | 0.001794544584 |
| calibration | increase | quiet | capped | v137 | budget_matched | supported | 0 | 175747 | 215 | 0.008414204201 | 0.01054978874 | 0.001216615189 | 0/33/215 | 33 | 0 | 0 | 0 | 0.0001879999088 |
| calibration | increase | first | all | v127 | original | supported | 0 | 25355 | 60 | 0.01260381919 | 0.05295754754 | 0.002306345029 | 4/32/56 | 36 | 0.1111111111 | 0.06666666667 | 0.09803921569 | 0.001265072149 |
| calibration | increase | first | all | v127 | budget_matched | supported | 0 | 25355 | 60 | 0.01260381919 | 0.05295754754 | 0.002306345029 | 4/32/56 | 36 | 0.1111111111 | 0.06666666667 | 0.09803921569 | 0.001265072149 |
| calibration | increase | first | all | v137 | original | supported | 0 | 25355 | 60 | 0.01228187255 | 0.07989790626 | 0.002282474985 | 5/28/55 | 33 | 0.1515151515 | 0.08333333333 | 0.1302083333 | 0.00110693813 |
| calibration | increase | first | all | v137 | budget_matched | supported | 0 | 25355 | 60 | 0.01228187255 | 0.07989790626 | 0.002282474985 | 5/32/55 | 37 | 0.1351351351 | 0.08333333333 | 0.1201923077 | 0.001265072149 |
| calibration | increase | first | below | v127 | original | supported | 0 | 25355 | 60 | 0.01260381919 | 0.05295754754 | 0.002306345029 | 4/32/56 | 36 | 0.1111111111 | 0.06666666667 | 0.09803921569 | 0.001265072149 |
| calibration | increase | first | below | v127 | budget_matched | supported | 0 | 25355 | 60 | 0.01260381919 | 0.05295754754 | 0.002306345029 | 4/32/56 | 36 | 0.1111111111 | 0.06666666667 | 0.09803921569 | 0.001265072149 |
| calibration | increase | first | below | v137 | original | supported | 0 | 25355 | 60 | 0.01228187255 | 0.07989790626 | 0.002282474985 | 5/28/55 | 33 | 0.1515151515 | 0.08333333333 | 0.1302083333 | 0.00110693813 |
| calibration | increase | first | below | v137 | budget_matched | supported | 0 | 25355 | 60 | 0.01228187255 | 0.07989790626 | 0.002282474985 | 5/32/55 | 37 | 0.1351351351 | 0.08333333333 | 0.1201923077 | 0.001265072149 |
| calibration | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration | increase | restart | all | v127 | original | supported | 0 | 224228 | 275 | 0.008268085545 | 0.01359127453 | 0.001218330287 | 4/48/271 | 52 | 0.07692307692 | 0.01454545455 | 0.04140786749 | 0.0002143306855 |
| calibration | increase | restart | all | v127 | budget_matched | supported | 0 | 224228 | 275 | 0.008268085545 | 0.01359127453 | 0.001218330287 | 4/48/271 | 52 | 0.07692307692 | 0.01454545455 | 0.04140786749 | 0.0002143306855 |
| calibration | increase | restart | all | v137 | original | supported | 0 | 224228 | 275 | 0.008355703197 | 0.01211661108 | 0.001218251642 | 18/456/257 | 474 | 0.03797468354 | 0.06545454545 | 0.04145555044 | 0.002036141512 |
| calibration | increase | restart | all | v137 | budget_matched | supported | 0 | 224228 | 275 | 0.008355703197 | 0.01211661108 | 0.001218251642 | 2/48/273 | 50 | 0.04 | 0.007272727273 | 0.02105263158 | 0.0002143306855 |
| calibration | increase | restart | below | v127 | original | supported | 0 | 48481 | 60 | 0.007881606175 | 0.02832797889 | 0.001219456146 | 3/15/57 | 18 | 0.1666666667 | 0.05 | 0.1136363636 | 0.0003097829454 |
| calibration | increase | restart | below | v127 | budget_matched | supported | 0 | 48481 | 60 | 0.007881606175 | 0.02832797889 | 0.001219456146 | 3/15/57 | 18 | 0.1666666667 | 0.05 | 0.1136363636 | 0.0003097829454 |
| calibration | increase | restart | below | v137 | original | supported | 0 | 48481 | 60 | 0.008143632987 | 0.02474467529 | 0.001224183898 | 7/141/53 | 148 | 0.0472972973 | 0.1166666667 | 0.0536809816 | 0.002911959687 |
| calibration | increase | restart | below | v137 | budget_matched | supported | 0 | 48481 | 60 | 0.008143632987 | 0.02474467529 | 0.001224183898 | 2/15/58 | 17 | 0.1176470588 | 0.03333333333 | 0.078125 | 0.0003097829454 |
| calibration | increase | restart | capped | v127 | original | supported | 0 | 175747 | 215 | 0.008374698496 | 0.01156189755 | 0.001218019711 | 1/33/214 | 34 | 0.02941176471 | 0.004651162791 | 0.01424501425 | 0.0001879999088 |
| calibration | increase | restart | capped | v127 | budget_matched | supported | 0 | 175747 | 215 | 0.008374698496 | 0.01156189755 | 0.001218019711 | 1/33/214 | 34 | 0.02941176471 | 0.004651162791 | 0.01424501425 | 0.0001879999088 |
| calibration | increase | restart | capped | v137 | original | supported | 0 | 175747 | 215 | 0.008414204201 | 0.01054978874 | 0.001216615189 | 11/315/204 | 326 | 0.03374233129 | 0.0511627907 | 0.0362080316 | 0.001794544584 |
| calibration | increase | restart | capped | v137 | budget_matched | supported | 0 | 175747 | 215 | 0.008414204201 | 0.01054978874 | 0.001216615189 | 0/33/215 | 33 | 0 | 0 | 0 | 0.0001879999088 |
| diagnosis | reduce | quiet | all | v127 | original | supported | 0 | 249701 | 140 | 0.004035078862 | 0.02301394822 | 0.0005556348156 | 2/14/138 | 16 | 0.125 | 0.01428571429 | 0.02564102564 | 5.609850898e-05 |
| diagnosis | reduce | quiet | all | v127 | budget_matched | supported | 0 | 249701 | 140 | 0.004035078862 | 0.02301394822 | 0.0005556348156 | 1/16/139 | 17 | 0.05882352941 | 0.007142857143 | 0.0127388535 | 6.411258169e-05 |
| diagnosis | reduce | quiet | all | v137 | original | supported | 0 | 249701 | 140 | 0.00409578802 | 0.01542067829 | 0.0005568691009 | 6/112/134 | 118 | 0.05084745763 | 0.04285714286 | 0.04651162791 | 0.0004487880719 |
| diagnosis | reduce | quiet | all | v137 | budget_matched | supported | 0 | 249701 | 140 | 0.00409578802 | 0.01542067829 | 0.0005568691009 | 2/31/138 | 33 | 0.06060606061 | 0.01428571429 | 0.02312138728 | 0.000124218127 |
| diagnosis | reduce | quiet | below | v127 | original | supported | 0 | 70516 | 28 | 0.002918751591 | 0.05103508686 | 0.0003924257968 | 2/8/26 | 10 | 0.2 | 0.07142857143 | 0.1052631579 | 0.0001134944955 |
| diagnosis | reduce | quiet | below | v127 | budget_matched | supported | 0 | 70516 | 28 | 0.002918751591 | 0.05103508686 | 0.0003924257968 | 2/10/26 | 12 | 0.1666666667 | 0.07142857143 | 0.1 | 0.0001418681194 |
| diagnosis | reduce | quiet | below | v137 | original | supported | 0 | 70516 | 28 | 0.002937600668 | 0.01668542649 | 0.0003953043406 | 3/61/25 | 64 | 0.046875 | 0.1071428571 | 0.0652173913 | 0.0008653955283 |
| diagnosis | reduce | quiet | below | v137 | budget_matched | supported | 0 | 70516 | 28 | 0.002937600668 | 0.01668542649 | 0.0003953043406 | 1/19/27 | 20 | 0.05 | 0.03571428571 | 0.04166666667 | 0.0002695494269 |
| diagnosis | reduce | quiet | capped | v127 | original | supported | 0 | 179185 | 112 | 0.0044743954 | 0.01675566456 | 0.0006198636694 | 0/6/112 | 6 | 0 | 0 | 0 | 3.350588866e-05 |
| diagnosis | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 179185 | 112 | 0.0044743954 | 0.01675566456 | 0.0006198636694 | 0/5/112 | 5 | 0 | 0 | 0 | 2.792157388e-05 |
| diagnosis | reduce | quiet | capped | v137 | original | supported | 0 | 179185 | 112 | 0.004551578065 | 0.01789479793 | 0.0006204508775 | 3/51/109 | 54 | 0.05555555556 | 0.02678571429 | 0.03614457831 | 0.0002848000536 |
| diagnosis | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 179185 | 112 | 0.004551578065 | 0.01789479793 | 0.0006204508775 | 1/16/111 | 17 | 0.05882352941 | 0.008928571429 | 0.01550387597 | 8.934903643e-05 |
| diagnosis | reduce | first | all | v127 | original | supported | 0 | 79164 | 29 | 0.002688274329 | 0.03577018209 | 0.0003630033795 | 1/9/28 | 10 | 0.1 | 0.03448275862 | 0.05128205128 | 0.0001137297024 |
| diagnosis | reduce | first | all | v127 | budget_matched | supported | 0 | 79164 | 29 | 0.002688274329 | 0.03577018209 | 0.0003630033795 | 1/9/28 | 10 | 0.1 | 0.03448275862 | 0.05128205128 | 0.0001137297024 |
| diagnosis | reduce | first | all | v137 | original | supported | 0 | 79164 | 29 | 0.002698131625 | 0.01918216765 | 0.0003639071933 | 4/60/25 | 64 | 0.0625 | 0.1379310345 | 0.08602150538 | 0.000758198016 |
| diagnosis | reduce | first | all | v137 | budget_matched | supported | 0 | 79164 | 29 | 0.002698131625 | 0.01918216765 | 0.0003639071933 | 0/9/29 | 9 | 0 | 0 | 0 | 0.0001137297024 |
| diagnosis | reduce | first | below | v127 | original | supported | 0 | 52443 | 15 | 0.002161060093 | 0.04442815186 | 0.0002834786399 | 1/5/14 | 6 | 0.1666666667 | 0.06666666667 | 0.09523809524 | 9.536888685e-05 |
| diagnosis | reduce | first | below | v127 | budget_matched | supported | 0 | 52443 | 15 | 0.002161060093 | 0.04442815186 | 0.0002834786399 | 1/5/14 | 6 | 0.1666666667 | 0.06666666667 | 0.09523809524 | 9.536888685e-05 |
| diagnosis | reduce | first | below | v137 | original | supported | 0 | 52443 | 15 | 0.002131963904 | 0.02417672308 | 0.0002838319438 | 2/42/13 | 44 | 0.04545454545 | 0.1333333333 | 0.06779661017 | 0.0008010986496 |
| diagnosis | reduce | first | below | v137 | budget_matched | supported | 0 | 52443 | 15 | 0.002131963904 | 0.02417672308 | 0.0002838319438 | 0/3/15 | 3 | 0 | 0 | 0 | 5.722133211e-05 |
| diagnosis | reduce | first | capped | v127 | original | supported | 0 | 26721 | 14 | 0.003722992197 | 0.03571003861 | 0.0005190797208 | 0/4/14 | 4 | 0 | 0 | 0 | 0.0001497734676 |
| diagnosis | reduce | first | capped | v127 | budget_matched | supported | 0 | 26721 | 14 | 0.003722992197 | 0.03571003861 | 0.0005190797208 | 0/4/14 | 4 | 0 | 0 | 0 | 0.0001497734676 |
| diagnosis | reduce | first | capped | v137 | original | supported | 0 | 26721 | 14 | 0.003809300137 | 0.02237669666 | 0.000521063973 | 2/18/12 | 20 | 0.1 | 0.1428571429 | 0.1176470588 | 0.0006739806043 |
| diagnosis | reduce | first | capped | v137 | budget_matched | supported | 0 | 26721 | 14 | 0.003809300137 | 0.02237669666 | 0.000521063973 | 0/4/14 | 4 | 0 | 0 | 0 | 0.0001497734676 |
| diagnosis | reduce | restart | all | v127 | original | supported | 0 | 170537 | 111 | 0.004660271249 | 0.02418244158 | 0.0006450551468 | 1/5/110 | 6 | 0.1666666667 | 0.009009009009 | 0.01709401709 | 2.933824651e-05 |
| diagnosis | reduce | restart | all | v127 | budget_matched | supported | 0 | 170537 | 111 | 0.004660271249 | 0.02418244158 | 0.0006450551468 | 1/5/110 | 6 | 0.1666666667 | 0.009009009009 | 0.01709401709 | 2.933824651e-05 |
| diagnosis | reduce | restart | all | v137 | original | supported | 0 | 170537 | 111 | 0.004744586057 | 0.01592980321 | 0.0006464428384 | 2/52/109 | 54 | 0.03703703704 | 0.01801801802 | 0.02424242424 | 0.0003051177637 |
| diagnosis | reduce | restart | all | v137 | budget_matched | supported | 0 | 170537 | 111 | 0.004744586057 | 0.01592980321 | 0.0006464428384 | 0/9/111 | 9 | 0 | 0 | 0 | 5.280884372e-05 |
| diagnosis | reduce | restart | below | v127 | original | supported | 0 | 18073 | 13 | 0.005117369155 | 0.08294730635 | 0.0007085612338 | 1/3/12 | 4 | 0.25 | 0.07692307692 | 0.1176470588 | 0.0001661129568 |
| diagnosis | reduce | restart | below | v127 | budget_matched | supported | 0 | 18073 | 13 | 0.005117369155 | 0.08294730635 | 0.0007085612338 | 1/3/12 | 4 | 0.25 | 0.07692307692 | 0.1176470588 | 0.0001661129568 |
| diagnosis | reduce | restart | below | v137 | original | supported | 0 | 18073 | 13 | 0.005275342539 | 0.02123148456 | 0.0007187673467 | 1/19/12 | 20 | 0.05 | 0.07692307692 | 0.06060606061 | 0.001052048726 |
| diagnosis | reduce | restart | below | v137 | budget_matched | supported | 0 | 18073 | 13 | 0.005275342539 | 0.02123148456 | 0.0007187673467 | 0/4/13 | 4 | 0 | 0 | 0 | 0.0002214839424 |
| diagnosis | reduce | restart | capped | v127 | original | supported | 0 | 152464 | 98 | 0.004606087111 | 0.01646184469 | 0.0006375271696 | 0/2/98 | 2 | 0 | 0 | 0 | 1.312628802e-05 |
| diagnosis | reduce | restart | capped | v127 | budget_matched | supported | 0 | 152464 | 98 | 0.004606087111 | 0.01646184469 | 0.0006375271696 | 0/2/98 | 2 | 0 | 0 | 0 | 1.312628802e-05 |
| diagnosis | reduce | restart | capped | v137 | original | supported | 0 | 152464 | 98 | 0.004681670471 | 0.01878435699 | 0.0006378695303 | 1/33/97 | 34 | 0.02941176471 | 0.01020408163 | 0.01515151515 | 0.0002165837523 |
| diagnosis | reduce | restart | capped | v137 | budget_matched | supported | 0 | 152464 | 98 | 0.004681670471 | 0.01878435699 | 0.0006378695303 | 1/5/97 | 6 | 0.1666666667 | 0.01020408163 | 0.01923076923 | 3.281572004e-05 |
| diagnosis | increase | quiet | all | v127 | original | supported | 0 | 249378 | 146 | 0.004049608028 | 0.02632506396 | 0.0005788047496 | 1/11/145 | 12 | 0.08333333333 | 0.006849315068 | 0.02577319588 | 4.413558452e-05 |
| diagnosis | increase | quiet | all | v127 | budget_matched | supported | 0 | 249378 | 146 | 0.004049608028 | 0.02632506396 | 0.0005788047496 | 2/20/144 | 22 | 0.09090909091 | 0.01369863014 | 0.04273504274 | 8.02465173e-05 |
| diagnosis | increase | quiet | all | v137 | original | supported | 0 | 249378 | 146 | 0.004122461558 | 0.02180921083 | 0.0005812405422 | 3/67/143 | 70 | 0.04285714286 | 0.02054794521 | 0.03521126761 | 0.000268825833 |
| diagnosis | increase | quiet | all | v137 | budget_matched | supported | 0 | 249378 | 146 | 0.004122461558 | 0.02180921083 | 0.0005812405422 | 1/22/145 | 23 | 0.04347826087 | 0.006849315068 | 0.02100840336 | 8.827116903e-05 |
| diagnosis | increase | quiet | below | v127 | original | supported | 0 | 70097 | 58 | 0.005115957922 | 0.05146623088 | 0.0008094506558 | 1/11/57 | 12 | 0.08333333333 | 0.01724137931 | 0.04716981132 | 0.0001570553549 |
| diagnosis | increase | quiet | below | v127 | budget_matched | supported | 0 | 70097 | 58 | 0.005115957922 | 0.05146623088 | 0.0008094506558 | 1/17/57 | 18 | 0.05555555556 | 0.01724137931 | 0.03846153846 | 0.0002427219121 |
| diagnosis | increase | quiet | below | v137 | original | supported | 0 | 70097 | 58 | 0.005264500231 | 0.04543681507 | 0.0008167161286 | 2/29/56 | 31 | 0.06451612903 | 0.03448275862 | 0.05494505495 | 0.0004140550265 |
| diagnosis | increase | quiet | below | v137 | budget_matched | supported | 0 | 70097 | 58 | 0.005264500231 | 0.04543681507 | 0.0008167161286 | 1/18/57 | 19 | 0.05263157895 | 0.01724137931 | 0.03731343284 | 0.0002569996716 |
| diagnosis | increase | quiet | capped | v127 | original | supported | 0 | 179281 | 88 | 0.003632676348 | 0.009659008533 | 0.0004886246073 | 0/0/88 | 0 | 0 | 0 | 0 | 0 |
| diagnosis | increase | quiet | capped | v127 | budget_matched | supported | 0 | 179281 | 88 | 0.003632676348 | 0.009659008533 | 0.0004886246073 | 0/0/88 | 0 | 0 | 0 | 0 | 0 |
| diagnosis | increase | quiet | capped | v137 | original | supported | 0 | 179281 | 88 | 0.003675936355 | 0.007585821795 | 0.0004891720453 | 1/38/87 | 39 | 0.02564102564 | 0.01136363636 | 0.02049180328 | 0.0002120618551 |
| diagnosis | increase | quiet | capped | v137 | budget_matched | supported | 0 | 179281 | 88 | 0.003675936355 | 0.007585821795 | 0.0004891720453 | 0/4/88 | 4 | 0 | 0 | 0 | 2.232230054e-05 |
| diagnosis | increase | first | all | v127 | original | supported | 0 | 22135 | 23 | 0.005800352807 | 0.1021830279 | 0.0009951415312 | 1/10/22 | 11 | 0.09090909091 | 0.04347826087 | 0.07462686567 | 0.0004522431259 |
| diagnosis | increase | first | all | v127 | budget_matched | supported | 0 | 22135 | 23 | 0.005800352807 | 0.1021830279 | 0.0009951415312 | 1/10/22 | 11 | 0.09090909091 | 0.04347826087 | 0.07462686567 | 0.0004522431259 |
| diagnosis | increase | first | all | v137 | original | supported | 0 | 22135 | 23 | 0.005729787532 | 0.09325041509 | 0.001015648827 | 1/11/22 | 12 | 0.08333333333 | 0.04347826087 | 0.07042253521 | 0.0004974674385 |
| diagnosis | increase | first | all | v137 | budget_matched | supported | 0 | 22135 | 23 | 0.005729787532 | 0.09325041509 | 0.001015648827 | 1/14/22 | 15 | 0.06666666667 | 0.04347826087 | 0.06024096386 | 0.0006331403763 |
| diagnosis | increase | first | below | v127 | original | supported | 0 | 22135 | 23 | 0.005800352807 | 0.1021830279 | 0.0009951415312 | 1/10/22 | 11 | 0.09090909091 | 0.04347826087 | 0.07462686567 | 0.0004522431259 |
| diagnosis | increase | first | below | v127 | budget_matched | supported | 0 | 22135 | 23 | 0.005800352807 | 0.1021830279 | 0.0009951415312 | 1/10/22 | 11 | 0.09090909091 | 0.04347826087 | 0.07462686567 | 0.0004522431259 |
| diagnosis | increase | first | below | v137 | original | supported | 0 | 22135 | 23 | 0.005729787532 | 0.09325041509 | 0.001015648827 | 1/11/22 | 12 | 0.08333333333 | 0.04347826087 | 0.07042253521 | 0.0004974674385 |
| diagnosis | increase | first | below | v137 | budget_matched | supported | 0 | 22135 | 23 | 0.005729787532 | 0.09325041509 | 0.001015648827 | 1/14/22 | 15 | 0.06666666667 | 0.04347826087 | 0.06024096386 | 0.0006331403763 |
| diagnosis | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis | increase | restart | all | v127 | original | supported | 0 | 227243 | 123 | 0.003879073685 | 0.01131112366 | 0.0005382507406 | 0/1/123 | 1 | 0 | 0 | 0 | 4.402958788e-06 |
| diagnosis | increase | restart | all | v127 | budget_matched | supported | 0 | 227243 | 123 | 0.003879073685 | 0.01131112366 | 0.0005382507406 | 0/1/123 | 1 | 0 | 0 | 0 | 4.402958788e-06 |
| diagnosis | increase | restart | all | v137 | original | supported | 0 | 227243 | 123 | 0.003965897174 | 0.008788697056 | 0.000538926247 | 2/56/121 | 58 | 0.03448275862 | 0.0162601626 | 0.02816901408 | 0.0002465656921 |
| diagnosis | increase | restart | all | v137 | budget_matched | supported | 0 | 227243 | 123 | 0.003965897174 | 0.008788697056 | 0.000538926247 | 0/5/123 | 5 | 0 | 0 | 0 | 2.201479394e-05 |
| diagnosis | increase | restart | below | v127 | original | supported | 0 | 47962 | 35 | 0.00480010202 | 0.01808798777 | 0.0007237522377 | 0/1/35 | 1 | 0 | 0 | 0 | 2.086506562e-05 |
| diagnosis | increase | restart | below | v127 | budget_matched | supported | 0 | 47962 | 35 | 0.00480010202 | 0.01808798777 | 0.0007237522377 | 0/1/35 | 1 | 0 | 0 | 0 | 2.086506562e-05 |
| diagnosis | increase | restart | below | v137 | original | supported | 0 | 47962 | 35 | 0.005049764932 | 0.01497723608 | 0.0007249064612 | 1/18/34 | 19 | 0.05263157895 | 0.02857142857 | 0.04504504505 | 0.0003755711812 |
| diagnosis | increase | restart | below | v137 | budget_matched | supported | 0 | 47962 | 35 | 0.005049764932 | 0.01497723608 | 0.0007249064612 | 0/1/35 | 1 | 0 | 0 | 0 | 2.086506562e-05 |
| diagnosis | increase | restart | capped | v127 | original | supported | 0 | 179281 | 88 | 0.003632676348 | 0.009659008533 | 0.0004886246073 | 0/0/88 | 0 | 0 | 0 | 0 | 0 |
| diagnosis | increase | restart | capped | v127 | budget_matched | supported | 0 | 179281 | 88 | 0.003632676348 | 0.009659008533 | 0.0004886246073 | 0/0/88 | 0 | 0 | 0 | 0 | 0 |
| diagnosis | increase | restart | capped | v137 | original | supported | 0 | 179281 | 88 | 0.003675936355 | 0.007585821795 | 0.0004891720453 | 1/38/87 | 39 | 0.02564102564 | 0.01136363636 | 0.02049180328 | 0.0002120618551 |
| diagnosis | increase | restart | capped | v137 | budget_matched | supported | 0 | 179281 | 88 | 0.003675936355 | 0.007585821795 | 0.0004891720453 | 0/4/88 | 4 | 0 | 0 | 0 | 2.232230054e-05 |
| training/2020-01 | reduce | quiet | all | v127 | original | supported | 0 | 41634 | 76 | 0.01108422042 | 0.05999439649 | 0.00178964801 | 1/5/75 | 6 | 0.1666666667 | 0.01315789474 | 0.0243902439 | 0.0001203137783 |
| training/2020-01 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 41634 | 76 | 0.01108422042 | 0.05999439649 | 0.00178964801 | 4/9/72 | 13 | 0.3076923077 | 0.05263157895 | 0.08988764045 | 0.000216564801 |
| training/2020-01 | reduce | quiet | all | v137 | original | supported | 0 | 41634 | 76 | 0.01116282941 | 0.06028370188 | 0.001792312823 | 3/23/73 | 26 | 0.1153846154 | 0.03947368421 | 0.05882352941 | 0.0005534433803 |
| training/2020-01 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 41634 | 76 | 0.01116282941 | 0.06028370188 | 0.001792312823 | 2/10/74 | 12 | 0.1666666667 | 0.02631578947 | 0.04545454545 | 0.0002406275567 |
| training/2020-01 | reduce | quiet | below | v127 | original | supported | 0 | 19488 | 29 | 0.009181762866 | 0.03523949851 | 0.00147605772 | 0/1/29 | 1 | 0 | 0 | 0 | 5.139010227e-05 |
| training/2020-01 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 19488 | 29 | 0.009181762866 | 0.03523949851 | 0.00147605772 | 0/6/29 | 6 | 0 | 0 | 0 | 0.0003083406136 |
| training/2020-01 | reduce | quiet | below | v137 | original | supported | 0 | 19488 | 29 | 0.009167843907 | 0.05977119384 | 0.00147387534 | 0/4/29 | 4 | 0 | 0 | 0 | 0.0002055604091 |
| training/2020-01 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 19488 | 29 | 0.009167843907 | 0.05977119384 | 0.00147387534 | 0/0/29 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | quiet | capped | v127 | original | supported | 0 | 22146 | 47 | 0.01275834184 | 0.08066337472 | 0.002065600669 | 1/4/46 | 5 | 0.2 | 0.02127659574 | 0.03846153846 | 0.0001810036653 |
| training/2020-01 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 22146 | 47 | 0.01275834184 | 0.08066337472 | 0.002065600669 | 4/6/43 | 10 | 0.4 | 0.08510638298 | 0.1403508772 | 0.000271505498 |
| training/2020-01 | reduce | quiet | capped | v137 | original | supported | 0 | 22146 | 47 | 0.01291837342 | 0.07218139146 | 0.002072530906 | 3/19/44 | 22 | 0.1363636364 | 0.06382978723 | 0.08695652174 | 0.0008597674103 |
| training/2020-01 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 22146 | 47 | 0.01291837342 | 0.07218139146 | 0.002072530906 | 2/8/45 | 10 | 0.2 | 0.04255319149 | 0.0701754386 | 0.0003620073306 |
| training/2020-01 | reduce | first | all | v127 | original | supported | 0 | 6566 | 24 | 0.02236701484 | 0.07568888099 | 0.003625193441 | 0/2/24 | 2 | 0 | 0 | 0 | 0.0003057169061 |
| training/2020-01 | reduce | first | all | v127 | budget_matched | supported | 0 | 6566 | 24 | 0.02236701484 | 0.07568888099 | 0.003625193441 | 0/2/24 | 2 | 0 | 0 | 0 | 0.0003057169061 |
| training/2020-01 | reduce | first | all | v137 | original | supported | 0 | 6566 | 24 | 0.02288631947 | 0.0615831241 | 0.003631019088 | 0/7/24 | 7 | 0 | 0 | 0 | 0.001070009172 |
| training/2020-01 | reduce | first | all | v137 | budget_matched | supported | 0 | 6566 | 24 | 0.02288631947 | 0.0615831241 | 0.003631019088 | 0/0/24 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | first | below | v127 | original | supported | 0 | 6157 | 19 | 0.01893805757 | 0.1016069786 | 0.003058339376 | 0/1/19 | 1 | 0 | 0 | 0 | 0.0001629195178 |
| training/2020-01 | reduce | first | below | v127 | budget_matched | supported | 0 | 6157 | 19 | 0.01893805757 | 0.1016069786 | 0.003058339376 | 0/1/19 | 1 | 0 | 0 | 0 | 0.0001629195178 |
| training/2020-01 | reduce | first | below | v137 | original | supported | 0 | 6157 | 19 | 0.01924685078 | 0.08245281637 | 0.003063240584 | 0/4/19 | 4 | 0 | 0 | 0 | 0.000651678071 |
| training/2020-01 | reduce | first | below | v137 | budget_matched | supported | 0 | 6157 | 19 | 0.01924685078 | 0.08245281637 | 0.003063240584 | 0/0/19 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | first | capped | v127 | original | supported | 0 | 409 | 5 | 0.07398581655 | 0.05719820407 | 0.01215849535 | 0/1/5 | 1 | 0 | 0 | 0 | 0.002475247525 |
| training/2020-01 | reduce | first | capped | v127 | budget_matched | supported | 0 | 409 | 5 | 0.07398581655 | 0.05719820407 | 0.01215849535 | 0/1/5 | 1 | 0 | 0 | 0 | 0.002475247525 |
| training/2020-01 | reduce | first | capped | v137 | original | supported | 0 | 409 | 5 | 0.0776741158 | 0.03463232452 | 0.0121782373 | 0/3/5 | 3 | 0 | 0 | 0 | 0.007425742574 |
| training/2020-01 | reduce | first | capped | v137 | budget_matched | supported | 0 | 409 | 5 | 0.0776741158 | 0.03463232452 | 0.0121782373 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | all | v127 | original | supported | 0 | 35068 | 52 | 0.008971672568 | 0.07153807942 | 0.00144596741 | 1/3/51 | 4 | 0.25 | 0.01923076923 | 0.03571428571 | 8.567511995e-05 |
| training/2020-01 | reduce | restart | all | v127 | budget_matched | supported | 0 | 35068 | 52 | 0.008971672568 | 0.07153807942 | 0.00144596741 | 1/3/51 | 4 | 0.25 | 0.01923076923 | 0.03571428571 | 8.567511995e-05 |
| training/2020-01 | reduce | restart | all | v137 | original | supported | 0 | 35068 | 52 | 0.00896776737 | 0.07351273323 | 0.0014480404 | 3/16/49 | 19 | 0.1578947368 | 0.05769230769 | 0.08450704225 | 0.000456933973 |
| training/2020-01 | reduce | restart | all | v137 | budget_matched | supported | 0 | 35068 | 52 | 0.00896776737 | 0.07351273323 | 0.0014480404 | 2/4/50 | 6 | 0.3333333333 | 0.03846153846 | 0.06896551724 | 0.0001142334933 |
| training/2020-01 | reduce | restart | below | v127 | original | supported | 0 | 13331 | 10 | 0.00467576133 | 0.02399808806 | 0.0007452717206 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | below | v127 | budget_matched | supported | 0 | 13331 | 10 | 0.00467576133 | 0.02399808806 | 0.0007452717206 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | below | v137 | original | supported | 0 | 13331 | 10 | 0.004512795874 | 0.07072143291 | 0.0007398177452 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | below | v137 | budget_matched | supported | 0 | 13331 | 10 | 0.004512795874 | 0.07072143291 | 0.0007398177452 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | capped | v127 | original | supported | 0 | 21737 | 42 | 0.01160629523 | 0.08756291247 | 0.001875694338 | 1/3/41 | 4 | 0.25 | 0.02380952381 | 0.04347826087 | 0.0001382807098 |
| training/2020-01 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 21737 | 42 | 0.01160629523 | 0.08756291247 | 0.001875694338 | 1/3/41 | 4 | 0.25 | 0.02380952381 | 0.04347826087 | 0.0001382807098 |
| training/2020-01 | reduce | restart | capped | v137 | original | supported | 0 | 21737 | 42 | 0.01169993947 | 0.07938254603 | 0.001882383512 | 3/16/39 | 19 | 0.1578947368 | 0.07142857143 | 0.09836065574 | 0.0007374971192 |
| training/2020-01 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 21737 | 42 | 0.01169993947 | 0.07938254603 | 0.001882383512 | 2/4/40 | 6 | 0.3333333333 | 0.04761904762 | 0.08333333333 | 0.0001843742798 |
| training/2020-01 | increase | quiet | all | v127 | original | supported | 0 | 41197 | 100 | 0.01362850559 | 0.07125864552 | 0.002366252512 | 2/14/98 | 16 | 0.125 | 0.02 | 0.06097560976 | 0.0003406574689 |
| training/2020-01 | increase | quiet | all | v127 | budget_matched | supported | 0 | 41197 | 100 | 0.01362850559 | 0.07125864552 | 0.002366252512 | 3/15/97 | 18 | 0.1666666667 | 0.03 | 0.08720930233 | 0.0003649901453 |
| training/2020-01 | increase | quiet | all | v137 | original | supported | 0 | 41197 | 100 | 0.01361397541 | 0.08081841408 | 0.002364571083 | 13/47/87 | 60 | 0.2166666667 | 0.13 | 0.1911764706 | 0.001143635789 |
| training/2020-01 | increase | quiet | all | v137 | budget_matched | supported | 0 | 41197 | 100 | 0.01361397541 | 0.08081841408 | 0.002364571083 | 2/16/98 | 18 | 0.1111111111 | 0.02 | 0.05813953488 | 0.0003893228216 |
| training/2020-01 | increase | quiet | below | v127 | original | supported | 0 | 19153 | 55 | 0.0150902199 | 0.0981861685 | 0.002775296296 | 1/8/54 | 9 | 0.1111111111 | 0.01818181818 | 0.05494505495 | 0.0004188920306 |
| training/2020-01 | increase | quiet | below | v127 | budget_matched | supported | 0 | 19153 | 55 | 0.0150902199 | 0.0981861685 | 0.002775296296 | 1/6/54 | 7 | 0.1428571429 | 0.01818181818 | 0.06024096386 | 0.0003141690229 |
| training/2020-01 | increase | quiet | below | v137 | original | supported | 0 | 19153 | 55 | 0.01542794586 | 0.1033146667 | 0.002781031729 | 9/15/46 | 24 | 0.375 | 0.1636363636 | 0.298013245 | 0.0007854225573 |
| training/2020-01 | increase | quiet | below | v137 | budget_matched | supported | 0 | 19153 | 55 | 0.01542794586 | 0.1033146667 | 0.002781031729 | 0/3/55 | 3 | 0 | 0 | 0 | 0.0001570845115 |
| training/2020-01 | increase | quiet | capped | v127 | original | supported | 0 | 22044 | 45 | 0.01235849043 | 0.04398035633 | 0.002010853509 | 1/6/44 | 7 | 0.1428571429 | 0.02222222222 | 0.06849315068 | 0.00027273967 |
| training/2020-01 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 22044 | 45 | 0.01235849043 | 0.04398035633 | 0.002010853509 | 1/6/44 | 7 | 0.1428571429 | 0.02222222222 | 0.06849315068 | 0.00027273967 |
| training/2020-01 | increase | quiet | capped | v137 | original | supported | 0 | 22044 | 45 | 0.01203790138 | 0.06301388168 | 0.002002727917 | 4/32/41 | 36 | 0.1111111111 | 0.08888888889 | 0.1058201058 | 0.001454611573 |
| training/2020-01 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 22044 | 45 | 0.01203790138 | 0.06301388168 | 0.002002727917 | 2/11/43 | 13 | 0.1538461538 | 0.04444444444 | 0.1030927835 | 0.0005000227283 |
| training/2020-01 | increase | first | all | v127 | original | supported | 0 | 2862 | 18 | 0.02817558334 | 0.1231868285 | 0.006000348067 | 0/4/18 | 4 | 0 | 0 | 0 | 0.001406469761 |
| training/2020-01 | increase | first | all | v127 | budget_matched | supported | 0 | 2862 | 18 | 0.02817558334 | 0.1231868285 | 0.006000348067 | 0/4/18 | 4 | 0 | 0 | 0 | 0.001406469761 |
| training/2020-01 | increase | first | all | v137 | original | supported | 0 | 2862 | 18 | 0.02804792331 | 0.1372598253 | 0.005961093043 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | first | all | v137 | budget_matched | supported | 0 | 2862 | 18 | 0.02804792331 | 0.1372598253 | 0.005961093043 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | first | below | v127 | original | supported | 0 | 2862 | 18 | 0.02817558334 | 0.1231868285 | 0.006000348067 | 0/4/18 | 4 | 0 | 0 | 0 | 0.001406469761 |
| training/2020-01 | increase | first | below | v127 | budget_matched | supported | 0 | 2862 | 18 | 0.02817558334 | 0.1231868285 | 0.006000348067 | 0/4/18 | 4 | 0 | 0 | 0 | 0.001406469761 |
| training/2020-01 | increase | first | below | v137 | original | supported | 0 | 2862 | 18 | 0.02804792331 | 0.1372598253 | 0.005961093043 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | first | below | v137 | budget_matched | supported | 0 | 2862 | 18 | 0.02804792331 | 0.1372598253 | 0.005961093043 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-01 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-01 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-01 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-01 | increase | restart | all | v127 | original | supported | 0 | 38335 | 82 | 0.01254245534 | 0.06386651462 | 0.002094939574 | 2/10/80 | 12 | 0.1666666667 | 0.0243902439 | 0.07692307692 | 0.0002614174052 |
| training/2020-01 | increase | restart | all | v127 | budget_matched | supported | 0 | 38335 | 82 | 0.01254245534 | 0.06386651462 | 0.002094939574 | 2/10/80 | 12 | 0.1666666667 | 0.0243902439 | 0.07692307692 | 0.0002614174052 |
| training/2020-01 | increase | restart | all | v137 | original | supported | 0 | 38335 | 82 | 0.01253637116 | 0.07481318239 | 0.0020960633 | 13/47/69 | 60 | 0.2166666667 | 0.1585365854 | 0.201863354 | 0.001228661804 |
| training/2020-01 | increase | restart | all | v137 | budget_matched | supported | 0 | 38335 | 82 | 0.01253637116 | 0.07481318239 | 0.0020960633 | 2/12/80 | 14 | 0.1428571429 | 0.0243902439 | 0.07246376812 | 0.0003137008862 |
| training/2020-01 | increase | restart | below | v127 | original | supported | 0 | 16291 | 37 | 0.01279138557 | 0.1085426939 | 0.002208719771 | 1/4/36 | 5 | 0.2 | 0.02702702703 | 0.08771929825 | 0.0002460932693 |
| training/2020-01 | increase | restart | below | v127 | budget_matched | supported | 0 | 16291 | 37 | 0.01279138557 | 0.1085426939 | 0.002208719771 | 1/4/36 | 5 | 0.2 | 0.02702702703 | 0.08771929825 | 0.0002460932693 |
| training/2020-01 | increase | restart | below | v137 | original | supported | 0 | 16291 | 37 | 0.01321087046 | 0.111217632 | 0.002222359119 | 9/15/28 | 24 | 0.375 | 0.2432432432 | 0.3383458647 | 0.0009228497601 |
| training/2020-01 | increase | restart | below | v137 | budget_matched | supported | 0 | 16291 | 37 | 0.01321087046 | 0.111217632 | 0.002222359119 | 0/1/37 | 1 | 0 | 0 | 0 | 6.152331734e-05 |
| training/2020-01 | increase | restart | capped | v127 | original | supported | 0 | 22044 | 45 | 0.01235849043 | 0.04398035633 | 0.002010853509 | 1/6/44 | 7 | 0.1428571429 | 0.02222222222 | 0.06849315068 | 0.00027273967 |
| training/2020-01 | increase | restart | capped | v127 | budget_matched | supported | 0 | 22044 | 45 | 0.01235849043 | 0.04398035633 | 0.002010853509 | 1/6/44 | 7 | 0.1428571429 | 0.02222222222 | 0.06849315068 | 0.00027273967 |
| training/2020-01 | increase | restart | capped | v137 | original | supported | 0 | 22044 | 45 | 0.01203790138 | 0.06301388168 | 0.002002727917 | 4/32/41 | 36 | 0.1111111111 | 0.08888888889 | 0.1058201058 | 0.001454611573 |
| training/2020-01 | increase | restart | capped | v137 | budget_matched | supported | 0 | 22044 | 45 | 0.01203790138 | 0.06301388168 | 0.002002727917 | 2/11/43 | 13 | 0.1538461538 | 0.04444444444 | 0.1030927835 | 0.0005000227283 |
| training/2020-02 | reduce | quiet | all | v127 | original | supported | 0 | 40200 | 88 | 0.01365533478 | 0.02633501111 | 0.002171096585 | 1/13/87 | 14 | 0.07142857143 | 0.01136363636 | 0.01960784314 | 0.0003240925409 |
| training/2020-02 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 40200 | 88 | 0.01365533478 | 0.02633501111 | 0.002171096585 | 2/23/86 | 25 | 0.08 | 0.02272727273 | 0.03539823009 | 0.0005733944954 |
| training/2020-02 | reduce | quiet | all | v137 | original | supported | 0 | 40200 | 88 | 0.01357597443 | 0.03370578025 | 0.002168879368 | 4/34/84 | 38 | 0.1052631579 | 0.04545454545 | 0.06349206349 | 0.0008476266454 |
| training/2020-02 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 40200 | 88 | 0.01357597443 | 0.03370578025 | 0.002168879368 | 2/18/86 | 20 | 0.1 | 0.02272727273 | 0.03703703704 | 0.0004487435181 |
| training/2020-02 | reduce | quiet | below | v127 | original | supported | 0 | 2767 | 10 | 0.02153158244 | 0.03475830018 | 0.003584022407 | 0/3/10 | 3 | 0 | 0 | 0 | 0.001088139282 |
| training/2020-02 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 2767 | 10 | 0.02153158244 | 0.03475830018 | 0.003584022407 | 0/1/10 | 1 | 0 | 0 | 0 | 0.0003627130939 |
| training/2020-02 | reduce | quiet | below | v137 | original | supported | 0 | 2767 | 10 | 0.02170942939 | 0.03988371351 | 0.003592147981 | 0/6/10 | 6 | 0 | 0 | 0 | 0.002176278564 |
| training/2020-02 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 2767 | 10 | 0.02170942939 | 0.03988371351 | 0.003592147981 | 0/4/10 | 4 | 0 | 0 | 0 | 0.001450852376 |
| training/2020-02 | reduce | quiet | capped | v127 | original | supported | 0 | 37433 | 78 | 0.01307313252 | 0.02674405113 | 0.002066654896 | 1/10/77 | 11 | 0.09090909091 | 0.01282051282 | 0.02247191011 | 0.0002677017802 |
| training/2020-02 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 37433 | 78 | 0.01307313252 | 0.02674405113 | 0.002066654896 | 2/17/76 | 19 | 0.1052631579 | 0.02564102564 | 0.0412371134 | 0.0004550930264 |
| training/2020-02 | reduce | quiet | capped | v137 | original | supported | 0 | 37433 | 78 | 0.01297475973 | 0.03567825737 | 0.002063673153 | 4/28/74 | 32 | 0.125 | 0.05128205128 | 0.07272727273 | 0.0007495649846 |
| training/2020-02 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 37433 | 78 | 0.01297475973 | 0.03567825737 | 0.002063673153 | 1/12/77 | 13 | 0.07692307692 | 0.01282051282 | 0.02197802198 | 0.0003212421363 |
| training/2020-02 | reduce | first | all | v127 | original | supported | 0 | 766 | 8 | 0.05833267115 | 0.1360437311 | 0.01031578025 | 0/3/8 | 3 | 0 | 0 | 0 | 0.003957783641 |
| training/2020-02 | reduce | first | all | v127 | budget_matched | supported | 0 | 766 | 8 | 0.05833267115 | 0.1360437311 | 0.01031578025 | 0/3/8 | 3 | 0 | 0 | 0 | 0.003957783641 |
| training/2020-02 | reduce | first | all | v137 | original | supported | 0 | 766 | 8 | 0.06038740176 | 0.1546089704 | 0.01034016469 | 0/2/8 | 2 | 0 | 0 | 0 | 0.002638522427 |
| training/2020-02 | reduce | first | all | v137 | budget_matched | supported | 0 | 766 | 8 | 0.06038740176 | 0.1546089704 | 0.01034016469 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | first | below | v127 | original | supported | 0 | 766 | 8 | 0.05833267115 | 0.1360437311 | 0.01031578025 | 0/3/8 | 3 | 0 | 0 | 0 | 0.003957783641 |
| training/2020-02 | reduce | first | below | v127 | budget_matched | supported | 0 | 766 | 8 | 0.05833267115 | 0.1360437311 | 0.01031578025 | 0/3/8 | 3 | 0 | 0 | 0 | 0.003957783641 |
| training/2020-02 | reduce | first | below | v137 | original | supported | 0 | 766 | 8 | 0.06038740176 | 0.1546089704 | 0.01034016469 | 0/2/8 | 2 | 0 | 0 | 0 | 0.002638522427 |
| training/2020-02 | reduce | first | below | v137 | budget_matched | supported | 0 | 766 | 8 | 0.06038740176 | 0.1546089704 | 0.01034016469 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | first | capped | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | reduce | first | capped | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | reduce | first | capped | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | reduce | first | capped | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | reduce | restart | all | v127 | original | supported | 0 | 39434 | 80 | 0.0127874837 | 0.0244130226 | 0.00201288723 | 1/10/79 | 11 | 0.09090909091 | 0.0125 | 0.02197802198 | 0.000254103776 |
| training/2020-02 | reduce | restart | all | v127 | budget_matched | supported | 0 | 39434 | 80 | 0.0127874837 | 0.0244130226 | 0.00201288723 | 1/10/79 | 11 | 0.09090909091 | 0.0125 | 0.02197802198 | 0.000254103776 |
| training/2020-02 | reduce | restart | all | v137 | original | supported | 0 | 39434 | 80 | 0.01266666892 | 0.03280213919 | 0.00201015328 | 4/32/76 | 36 | 0.1111111111 | 0.05 | 0.06896551724 | 0.0008131320831 |
| training/2020-02 | reduce | restart | all | v137 | budget_matched | supported | 0 | 39434 | 80 | 0.01266666892 | 0.03280213919 | 0.00201015328 | 0/10/80 | 10 | 0 | 0 | 0 | 0.000254103776 |
| training/2020-02 | reduce | restart | below | v127 | original | supported | 0 | 2001 | 2 | 0.007443809354 | 0.005618148137 | 0.00100704764 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | restart | below | v127 | budget_matched | supported | 0 | 2001 | 2 | 0.007443809354 | 0.005618148137 | 0.00100704764 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | restart | below | v137 | original | supported | 0 | 2001 | 2 | 0.006903169095 | 0.0183950831 | 0.001008949181 | 0/4/2 | 4 | 0 | 0 | 0 | 0.0020010005 |
| training/2020-02 | reduce | restart | below | v137 | budget_matched | supported | 0 | 2001 | 2 | 0.006903169095 | 0.0183950831 | 0.001008949181 | 0/1/2 | 1 | 0 | 0 | 0 | 0.0005002501251 |
| training/2020-02 | reduce | restart | capped | v127 | original | supported | 0 | 37433 | 78 | 0.01307313252 | 0.02674405113 | 0.002066654896 | 1/10/77 | 11 | 0.09090909091 | 0.01282051282 | 0.02247191011 | 0.0002677017802 |
| training/2020-02 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 37433 | 78 | 0.01307313252 | 0.02674405113 | 0.002066654896 | 1/10/77 | 11 | 0.09090909091 | 0.01282051282 | 0.02247191011 | 0.0002677017802 |
| training/2020-02 | reduce | restart | capped | v137 | original | supported | 0 | 37433 | 78 | 0.01297475973 | 0.03567825737 | 0.002063673153 | 4/28/74 | 32 | 0.125 | 0.05128205128 | 0.07272727273 | 0.0007495649846 |
| training/2020-02 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 37433 | 78 | 0.01297475973 | 0.03567825737 | 0.002063673153 | 0/10/78 | 10 | 0 | 0 | 0 | 0.0002677017802 |
| training/2020-02 | increase | quiet | all | v127 | original | supported | 0 | 39784 | 113 | 0.0176693335 | 0.02621767656 | 0.00281285202 | 0/4/113 | 4 | 0 | 0 | 0 | 0.0001008293212 |
| training/2020-02 | increase | quiet | all | v127 | budget_matched | supported | 0 | 39784 | 113 | 0.0176693335 | 0.02621767656 | 0.00281285202 | 0/5/113 | 5 | 0 | 0 | 0 | 0.0001260366515 |
| training/2020-02 | increase | quiet | all | v137 | original | supported | 0 | 39784 | 113 | 0.01773304932 | 0.02681505261 | 0.002813252222 | 6/47/107 | 53 | 0.1132075472 | 0.05309734513 | 0.09230769231 | 0.001184744524 |
| training/2020-02 | increase | quiet | all | v137 | budget_matched | supported | 0 | 39784 | 113 | 0.01773304932 | 0.02681505261 | 0.002813252222 | 0/9/113 | 9 | 0 | 0 | 0 | 0.0002268659726 |
| training/2020-02 | increase | quiet | below | v127 | original | supported | 0 | 2638 | 18 | 0.03745305817 | 0.04627225182 | 0.00672479265 | 0/3/18 | 3 | 0 | 0 | 0 | 0.001145038168 |
| training/2020-02 | increase | quiet | below | v127 | budget_matched | supported | 0 | 2638 | 18 | 0.03745305817 | 0.04627225182 | 0.00672479265 | 0/4/18 | 4 | 0 | 0 | 0 | 0.001526717557 |
| training/2020-02 | increase | quiet | below | v137 | original | supported | 0 | 2638 | 18 | 0.03733779994 | 0.06169153808 | 0.006699851504 | 1/3/17 | 4 | 0.25 | 0.05555555556 | 0.1470588235 | 0.001145038168 |
| training/2020-02 | increase | quiet | below | v137 | budget_matched | supported | 0 | 2638 | 18 | 0.03733779994 | 0.06169153808 | 0.006699851504 | 0/4/18 | 4 | 0 | 0 | 0 | 0.001526717557 |
| training/2020-02 | increase | quiet | capped | v127 | original | supported | 0 | 37146 | 95 | 0.01626435138 | 0.02524778184 | 0.002535037467 | 0/1/95 | 1 | 0 | 0 | 0 | 2.698982484e-05 |
| training/2020-02 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 37146 | 95 | 0.01626435138 | 0.02524778184 | 0.002535037467 | 0/1/95 | 1 | 0 | 0 | 0 | 2.698982484e-05 |
| training/2020-02 | increase | quiet | capped | v137 | original | supported | 0 | 37146 | 95 | 0.01634077742 | 0.02214855843 | 0.002537237337 | 5/44/90 | 49 | 0.1020408163 | 0.05263157895 | 0.08591065292 | 0.001187552293 |
| training/2020-02 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 37146 | 95 | 0.01634077742 | 0.02214855843 | 0.002537237337 | 0/14/95 | 14 | 0 | 0 | 0 | 0.0003778575477 |
| training/2020-02 | increase | first | all | v127 | original | supported | 0 | 390 | 7 | 0.09447302237 | 0.0623717393 | 0.01767722049 | 0/2/7 | 2 | 0 | 0 | 0 | 0.005221932115 |
| training/2020-02 | increase | first | all | v127 | budget_matched | supported | 0 | 390 | 7 | 0.09447302237 | 0.0623717393 | 0.01767722049 | 0/2/7 | 2 | 0 | 0 | 0 | 0.005221932115 |
| training/2020-02 | increase | first | all | v137 | original | supported | 0 | 390 | 7 | 0.08747524466 | 0.0839147464 | 0.01750019567 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | first | all | v137 | budget_matched | supported | 0 | 390 | 7 | 0.08747524466 | 0.0839147464 | 0.01750019567 | 0/2/7 | 2 | 0 | 0 | 0 | 0.005221932115 |
| training/2020-02 | increase | first | below | v127 | original | supported | 0 | 390 | 7 | 0.09447302237 | 0.0623717393 | 0.01767722049 | 0/2/7 | 2 | 0 | 0 | 0 | 0.005221932115 |
| training/2020-02 | increase | first | below | v127 | budget_matched | supported | 0 | 390 | 7 | 0.09447302237 | 0.0623717393 | 0.01767722049 | 0/2/7 | 2 | 0 | 0 | 0 | 0.005221932115 |
| training/2020-02 | increase | first | below | v137 | original | supported | 0 | 390 | 7 | 0.08747524466 | 0.0839147464 | 0.01750019567 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | first | below | v137 | budget_matched | supported | 0 | 390 | 7 | 0.08747524466 | 0.0839147464 | 0.01750019567 | 0/2/7 | 2 | 0 | 0 | 0 | 0.005221932115 |
| training/2020-02 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | increase | restart | all | v127 | original | supported | 0 | 39394 | 106 | 0.01690897815 | 0.02579376032 | 0.002665694999 | 0/2/106 | 2 | 0 | 0 | 0 | 5.09061291e-05 |
| training/2020-02 | increase | restart | all | v127 | budget_matched | supported | 0 | 39394 | 106 | 0.01690897815 | 0.02579376032 | 0.002665694999 | 0/2/106 | 2 | 0 | 0 | 0 | 5.09061291e-05 |
| training/2020-02 | increase | restart | all | v137 | original | supported | 0 | 39394 | 106 | 0.01704260265 | 0.0239350111 | 0.002667851706 | 6/47/100 | 53 | 0.1132075472 | 0.05660377358 | 0.09433962264 | 0.001196294034 |
| training/2020-02 | increase | restart | all | v137 | budget_matched | supported | 0 | 39394 | 106 | 0.01704260265 | 0.0239350111 | 0.002667851706 | 0/16/106 | 16 | 0 | 0 | 0 | 0.0004072490328 |
| training/2020-02 | increase | restart | below | v127 | original | supported | 0 | 2248 | 11 | 0.02756080459 | 0.04954015323 | 0.004824682838 | 0/1/11 | 1 | 0 | 0 | 0 | 0.0004470272687 |
| training/2020-02 | increase | restart | below | v127 | budget_matched | supported | 0 | 2248 | 11 | 0.02756080459 | 0.04954015323 | 0.004824682838 | 0/1/11 | 1 | 0 | 0 | 0 | 0.0004470272687 |
| training/2020-02 | increase | restart | below | v137 | original | supported | 0 | 2248 | 11 | 0.02863957777 | 0.06489099197 | 0.004826126316 | 1/3/10 | 4 | 0.25 | 0.09090909091 | 0.1851851852 | 0.001341081806 |
| training/2020-02 | increase | restart | below | v137 | budget_matched | supported | 0 | 2248 | 11 | 0.02863957777 | 0.06489099197 | 0.004826126316 | 0/2/11 | 2 | 0 | 0 | 0 | 0.0008940545373 |
| training/2020-02 | increase | restart | capped | v127 | original | supported | 0 | 37146 | 95 | 0.01626435138 | 0.02524778184 | 0.002535037467 | 0/1/95 | 1 | 0 | 0 | 0 | 2.698982484e-05 |
| training/2020-02 | increase | restart | capped | v127 | budget_matched | supported | 0 | 37146 | 95 | 0.01626435138 | 0.02524778184 | 0.002535037467 | 0/1/95 | 1 | 0 | 0 | 0 | 2.698982484e-05 |
| training/2020-02 | increase | restart | capped | v137 | original | supported | 0 | 37146 | 95 | 0.01634077742 | 0.02214855843 | 0.002537237337 | 5/44/90 | 49 | 0.1020408163 | 0.05263157895 | 0.08591065292 | 0.001187552293 |
| training/2020-02 | increase | restart | capped | v137 | budget_matched | supported | 0 | 37146 | 95 | 0.01634077742 | 0.02214855843 | 0.002537237337 | 0/14/95 | 14 | 0 | 0 | 0 | 0.0003778575477 |
| training/2020-03 | reduce | quiet | all | v127 | original | supported | 0 | 42075 | 136 | 0.01688227949 | 0.09210534871 | 0.003111675874 | 17/51/119 | 68 | 0.25 | 0.125 | 0.1666666667 | 0.001216051885 |
| training/2020-03 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 42075 | 136 | 0.01688227949 | 0.09210534871 | 0.003111675874 | 20/64/116 | 84 | 0.2380952381 | 0.1470588235 | 0.1818181818 | 0.001526025895 |
| training/2020-03 | reduce | quiet | all | v137 | original | supported | 0 | 42075 | 136 | 0.01691299209 | 0.09688750822 | 0.003117271786 | 31/155/105 | 186 | 0.1666666667 | 0.2279411765 | 0.1925465839 | 0.003695843964 |
| training/2020-03 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 42075 | 136 | 0.01691299209 | 0.09688750822 | 0.003117271786 | 17/49/119 | 66 | 0.2575757576 | 0.125 | 0.1683168317 | 0.001168363576 |
| training/2020-03 | reduce | quiet | below | v127 | original | supported | 0 | 6518 | 43 | 0.03222864192 | 0.120798639 | 0.006392866568 | 4/15/39 | 19 | 0.2105263158 | 0.09302325581 | 0.1290322581 | 0.002316602317 |
| training/2020-03 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 6518 | 43 | 0.03222864192 | 0.120798639 | 0.006392866568 | 5/20/38 | 25 | 0.2 | 0.1162790698 | 0.1470588235 | 0.003088803089 |
| training/2020-03 | reduce | quiet | below | v137 | original | supported | 0 | 6518 | 43 | 0.03245773372 | 0.1380105154 | 0.006409974534 | 10/36/33 | 46 | 0.2173913043 | 0.2325581395 | 0.2247191011 | 0.00555984556 |
| training/2020-03 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 6518 | 43 | 0.03245773372 | 0.1380105154 | 0.006409974534 | 4/5/39 | 9 | 0.4444444444 | 0.09302325581 | 0.1538461538 | 0.0007722007722 |
| training/2020-03 | reduce | quiet | capped | v127 | original | supported | 0 | 35557 | 93 | 0.01406911779 | 0.09777241994 | 0.002510196533 | 13/36/80 | 49 | 0.2653061224 | 0.1397849462 | 0.1830985915 | 0.001015113918 |
| training/2020-03 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 35557 | 93 | 0.01406911779 | 0.09777241994 | 0.002510196533 | 15/47/78 | 62 | 0.2419354839 | 0.1612903226 | 0.1935483871 | 0.001325287616 |
| training/2020-03 | reduce | quiet | capped | v137 | original | supported | 0 | 35557 | 93 | 0.01406346525 | 0.1043889462 | 0.002513682155 | 21/119/72 | 140 | 0.15 | 0.2258064516 | 0.1802575107 | 0.003355515452 |
| training/2020-03 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 35557 | 93 | 0.01406346525 | 0.1043889462 | 0.002513682155 | 10/41/83 | 51 | 0.1960784314 | 0.1075268817 | 0.1388888889 | 0.001156101963 |
| training/2020-03 | reduce | first | all | v127 | original | supported | 0 | 3630 | 37 | 0.04840451874 | 0.1341649742 | 0.009853660209 | 6/26/31 | 32 | 0.1875 | 0.1621621622 | 0.1739130435 | 0.007236292792 |
| training/2020-03 | reduce | first | all | v127 | budget_matched | supported | 0 | 3630 | 37 | 0.04840451874 | 0.1341649742 | 0.009853660209 | 6/26/31 | 32 | 0.1875 | 0.1621621622 | 0.1739130435 | 0.007236292792 |
| training/2020-03 | reduce | first | all | v137 | original | supported | 0 | 3630 | 37 | 0.04930355906 | 0.1745898483 | 0.009877732542 | 12/58/25 | 70 | 0.1714285714 | 0.3243243243 | 0.2242990654 | 0.0161424993 |
| training/2020-03 | reduce | first | all | v137 | budget_matched | supported | 0 | 3630 | 37 | 0.04930355906 | 0.1745898483 | 0.009877732542 | 4/19/33 | 23 | 0.1739130435 | 0.1081081081 | 0.1333333333 | 0.005288060117 |
| training/2020-03 | reduce | first | below | v127 | original | supported | 0 | 3431 | 30 | 0.04219717844 | 0.1497643156 | 0.008496747233 | 3/11/27 | 14 | 0.2142857143 | 0.1 | 0.1363636364 | 0.00323434284 |
| training/2020-03 | reduce | first | below | v127 | budget_matched | supported | 0 | 3431 | 30 | 0.04219717844 | 0.1497643156 | 0.008496747233 | 3/11/27 | 14 | 0.2142857143 | 0.1 | 0.1363636364 | 0.00323434284 |
| training/2020-03 | reduce | first | below | v137 | original | supported | 0 | 3431 | 30 | 0.04286028599 | 0.1972890545 | 0.00850378829 | 8/33/22 | 41 | 0.1951219512 | 0.2666666667 | 0.2253521127 | 0.009703028521 |
| training/2020-03 | reduce | first | below | v137 | budget_matched | supported | 0 | 3431 | 30 | 0.04286028599 | 0.1972890545 | 0.00850378829 | 3/6/27 | 9 | 0.3333333333 | 0.1 | 0.1538461538 | 0.001764187004 |
| training/2020-03 | reduce | first | capped | v127 | original | supported | 0 | 199 | 7 | 0.1554265516 | 0.1905664914 | 0.0332484764 | 3/15/4 | 18 | 0.1666666667 | 0.4285714286 | 0.24 | 0.078125 |
| training/2020-03 | reduce | first | capped | v127 | budget_matched | supported | 0 | 199 | 7 | 0.1554265516 | 0.1905664914 | 0.0332484764 | 3/15/4 | 18 | 0.1666666667 | 0.4285714286 | 0.24 | 0.078125 |
| training/2020-03 | reduce | first | capped | v137 | original | supported | 0 | 199 | 7 | 0.1603933575 | 0.2310556643 | 0.03356618845 | 4/25/3 | 29 | 0.1379310345 | 0.5714285714 | 0.2222222222 | 0.1302083333 |
| training/2020-03 | reduce | first | capped | v137 | budget_matched | supported | 0 | 199 | 7 | 0.1603933575 | 0.2310556643 | 0.03356618845 | 1/13/6 | 14 | 0.07142857143 | 0.1428571429 | 0.09523809524 | 0.06770833333 |
| training/2020-03 | reduce | restart | all | v127 | original | supported | 0 | 38445 | 99 | 0.01390593072 | 0.09328954021 | 0.002475093662 | 11/25/88 | 36 | 0.3055555556 | 0.1111111111 | 0.162962963 | 0.0006519584833 |
| training/2020-03 | reduce | restart | all | v127 | budget_matched | supported | 0 | 38445 | 99 | 0.01390593072 | 0.09328954021 | 0.002475093662 | 11/25/88 | 36 | 0.3055555556 | 0.1111111111 | 0.162962963 | 0.0006519584833 |
| training/2020-03 | reduce | restart | all | v137 | original | supported | 0 | 38445 | 99 | 0.0138546553 | 0.09607981888 | 0.002478945019 | 19/97/80 | 116 | 0.1637931034 | 0.1919191919 | 0.176744186 | 0.002529598915 |
| training/2020-03 | reduce | restart | all | v137 | budget_matched | supported | 0 | 38445 | 99 | 0.0138546553 | 0.09607981888 | 0.002478945019 | 7/22/92 | 29 | 0.2413793103 | 0.07070707071 | 0.109375 | 0.0005737234653 |
| training/2020-03 | reduce | restart | below | v127 | original | supported | 0 | 3087 | 13 | 0.02114926102 | 0.1511537287 | 0.004054539856 | 1/4/12 | 5 | 0.2 | 0.07692307692 | 0.1111111111 | 0.001301236174 |
| training/2020-03 | reduce | restart | below | v127 | budget_matched | supported | 0 | 3087 | 13 | 0.02114926102 | 0.1511537287 | 0.004054539856 | 1/4/12 | 5 | 0.2 | 0.07692307692 | 0.1111111111 | 0.001301236174 |
| training/2020-03 | reduce | restart | below | v137 | original | supported | 0 | 3087 | 13 | 0.02089597251 | 0.1170855811 | 0.004082836537 | 2/3/11 | 5 | 0.4 | 0.1538461538 | 0.2222222222 | 0.0009759271308 |
| training/2020-03 | reduce | restart | below | v137 | budget_matched | supported | 0 | 3087 | 13 | 0.02089597251 | 0.1170855811 | 0.004082836537 | 0/1/13 | 1 | 0 | 0 | 0 | 0.0003253090436 |
| training/2020-03 | reduce | restart | capped | v127 | original | supported | 0 | 35358 | 86 | 0.01327353746 | 0.09506610982 | 0.002337196994 | 10/21/76 | 31 | 0.3225806452 | 0.1162790698 | 0.1709401709 | 0.0005953731005 |
| training/2020-03 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 35358 | 86 | 0.01327353746 | 0.09506610982 | 0.002337196994 | 10/21/76 | 31 | 0.3225806452 | 0.1162790698 | 0.1709401709 | 0.0005953731005 |
| training/2020-03 | reduce | restart | capped | v137 | original | supported | 0 | 35358 | 86 | 0.0132398992 | 0.1007242942 | 0.002338914104 | 17/94/69 | 111 | 0.1531531532 | 0.1976744186 | 0.1725888325 | 0.002665003402 |
| training/2020-03 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 35358 | 86 | 0.0132398992 | 0.1007242942 | 0.002338914104 | 7/23/79 | 30 | 0.2333333333 | 0.08139534884 | 0.1206896552 | 0.0006520753005 |
| training/2020-03 | increase | quiet | all | v127 | original | supported | 0 | 41971 | 122 | 0.01556173976 | 0.08312399333 | 0.002792905797 | 8/23/114 | 31 | 0.2580645161 | 0.06557377049 | 0.162601626 | 0.0005495949724 |
| training/2020-03 | increase | quiet | all | v127 | budget_matched | supported | 0 | 41971 | 122 | 0.01556173976 | 0.08312399333 | 0.002792905797 | 11/27/111 | 38 | 0.2894736842 | 0.09016393443 | 0.200729927 | 0.0006451767067 |
| training/2020-03 | increase | quiet | all | v137 | original | supported | 0 | 41971 | 122 | 0.015518102 | 0.1005726025 | 0.002785519807 | 22/217/100 | 239 | 0.09205020921 | 0.1803278689 | 0.1020408163 | 0.005185309087 |
| training/2020-03 | increase | quiet | all | v137 | budget_matched | supported | 0 | 41971 | 122 | 0.015518102 | 0.1005726025 | 0.002785519807 | 11/25/111 | 36 | 0.3055555556 | 0.09016393443 | 0.2067669173 | 0.0005973858396 |
| training/2020-03 | increase | quiet | below | v127 | original | supported | 0 | 6348 | 47 | 0.03259913901 | 0.1545686015 | 0.006856341425 | 7/21/40 | 28 | 0.25 | 0.1489361702 | 0.2201257862 | 0.003332804317 |
| training/2020-03 | increase | quiet | below | v127 | budget_matched | supported | 0 | 6348 | 47 | 0.03259913901 | 0.1545686015 | 0.006856341425 | 7/20/40 | 27 | 0.2592592593 | 0.1489361702 | 0.2258064516 | 0.003174099349 |
| training/2020-03 | increase | quiet | below | v137 | original | supported | 0 | 6348 | 47 | 0.03229774448 | 0.1899755726 | 0.006816365126 | 13/69/34 | 82 | 0.1585365854 | 0.2765957447 | 0.1733333333 | 0.01095064276 |
| training/2020-03 | increase | quiet | below | v137 | budget_matched | supported | 0 | 6348 | 47 | 0.03229774448 | 0.1899755726 | 0.006816365126 | 9/20/38 | 29 | 0.3103448276 | 0.1914893617 | 0.2760736196 | 0.003174099349 |
| training/2020-03 | increase | quiet | capped | v127 | original | supported | 0 | 35623 | 75 | 0.01252568411 | 0.04105426861 | 0.00206880369 | 1/2/74 | 3 | 0.3333333333 | 0.01333333333 | 0.05747126437 | 5.626195567e-05 |
| training/2020-03 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 35623 | 75 | 0.01252568411 | 0.04105426861 | 0.00206880369 | 1/3/74 | 4 | 0.25 | 0.01333333333 | 0.05494505495 | 8.43929335e-05 |
| training/2020-03 | increase | quiet | capped | v137 | original | supported | 0 | 35623 | 75 | 0.01252797847 | 0.05750310835 | 0.002067225276 | 9/148/66 | 157 | 0.05732484076 | 0.12 | 0.0640113798 | 0.004163384719 |
| training/2020-03 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 35623 | 75 | 0.01252797847 | 0.05750310835 | 0.002067225276 | 4/9/71 | 13 | 0.3076923077 | 0.05333333333 | 0.157480315 | 0.0002531788005 |
| training/2020-03 | increase | first | all | v127 | original | supported | 0 | 1611 | 25 | 0.05513310565 | 0.2328083378 | 0.01381734549 | 5/15/20 | 20 | 0.25 | 0.2 | 0.2380952381 | 0.009457755359 |
| training/2020-03 | increase | first | all | v127 | budget_matched | supported | 0 | 1611 | 25 | 0.05513310565 | 0.2328083378 | 0.01381734549 | 5/15/20 | 20 | 0.25 | 0.2 | 0.2380952381 | 0.009457755359 |
| training/2020-03 | increase | first | all | v137 | original | supported | 0 | 1611 | 25 | 0.05418732593 | 0.2906708355 | 0.01359261143 | 7/7/18 | 14 | 0.5 | 0.28 | 0.4320987654 | 0.004413619168 |
| training/2020-03 | increase | first | all | v137 | budget_matched | supported | 0 | 1611 | 25 | 0.05418732593 | 0.2906708355 | 0.01359261143 | 7/11/18 | 18 | 0.3888888889 | 0.28 | 0.3608247423 | 0.006935687264 |
| training/2020-03 | increase | first | below | v127 | original | supported | 0 | 1611 | 25 | 0.05513310565 | 0.2328083378 | 0.01381734549 | 5/15/20 | 20 | 0.25 | 0.2 | 0.2380952381 | 0.009457755359 |
| training/2020-03 | increase | first | below | v127 | budget_matched | supported | 0 | 1611 | 25 | 0.05513310565 | 0.2328083378 | 0.01381734549 | 5/15/20 | 20 | 0.25 | 0.2 | 0.2380952381 | 0.009457755359 |
| training/2020-03 | increase | first | below | v137 | original | supported | 0 | 1611 | 25 | 0.05418732593 | 0.2906708355 | 0.01359261143 | 7/7/18 | 14 | 0.5 | 0.28 | 0.4320987654 | 0.004413619168 |
| training/2020-03 | increase | first | below | v137 | budget_matched | supported | 0 | 1611 | 25 | 0.05418732593 | 0.2906708355 | 0.01359261143 | 7/11/18 | 18 | 0.3888888889 | 0.28 | 0.3608247423 | 0.006935687264 |
| training/2020-03 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-03 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-03 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-03 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-03 | increase | restart | all | v127 | original | supported | 0 | 40360 | 97 | 0.01398221869 | 0.05300238965 | 0.002352856928 | 3/8/94 | 11 | 0.2727272727 | 0.03092783505 | 0.1063829787 | 0.0001986935896 |
| training/2020-03 | increase | restart | all | v127 | budget_matched | supported | 0 | 40360 | 97 | 0.01398221869 | 0.05300238965 | 0.002352856928 | 3/10/94 | 13 | 0.2307692308 | 0.03092783505 | 0.1006711409 | 0.0002483669871 |
| training/2020-03 | increase | restart | all | v137 | original | supported | 0 | 40360 | 97 | 0.01397459061 | 0.05487271125 | 0.002354146551 | 15/210/82 | 225 | 0.06666666667 | 0.1546391753 | 0.07522567703 | 0.005215706728 |
| training/2020-03 | increase | restart | all | v137 | budget_matched | supported | 0 | 40360 | 97 | 0.01397459061 | 0.05487271125 | 0.002354146551 | 6/14/91 | 20 | 0.3 | 0.0618556701 | 0.1694915254 | 0.0003477137819 |
| training/2020-03 | increase | restart | below | v127 | original | supported | 0 | 4737 | 22 | 0.02493559241 | 0.08398084729 | 0.004488982856 | 2/6/20 | 8 | 0.25 | 0.09090909091 | 0.1851851852 | 0.001272534464 |
| training/2020-03 | increase | restart | below | v127 | budget_matched | supported | 0 | 4737 | 22 | 0.02493559241 | 0.08398084729 | 0.004488982856 | 2/7/20 | 9 | 0.2222222222 | 0.09090909091 | 0.1724137931 | 0.001484623542 |
| training/2020-03 | increase | restart | below | v137 | original | supported | 0 | 4737 | 22 | 0.02485334597 | 0.07823586902 | 0.004511840574 | 6/62/16 | 68 | 0.08823529412 | 0.2727272727 | 0.1020408163 | 0.0131495228 |
| training/2020-03 | increase | restart | below | v137 | budget_matched | supported | 0 | 4737 | 22 | 0.02485334597 | 0.07823586902 | 0.004511840574 | 2/5/20 | 7 | 0.2857142857 | 0.09090909091 | 0.2 | 0.001060445387 |
| training/2020-03 | increase | restart | capped | v127 | original | supported | 0 | 35623 | 75 | 0.01252568411 | 0.04105426861 | 0.00206880369 | 1/2/74 | 3 | 0.3333333333 | 0.01333333333 | 0.05747126437 | 5.626195567e-05 |
| training/2020-03 | increase | restart | capped | v127 | budget_matched | supported | 0 | 35623 | 75 | 0.01252568411 | 0.04105426861 | 0.00206880369 | 1/3/74 | 4 | 0.25 | 0.01333333333 | 0.05494505495 | 8.43929335e-05 |
| training/2020-03 | increase | restart | capped | v137 | original | supported | 0 | 35623 | 75 | 0.01252797847 | 0.05750310835 | 0.002067225276 | 9/148/66 | 157 | 0.05732484076 | 0.12 | 0.0640113798 | 0.004163384719 |
| training/2020-03 | increase | restart | capped | v137 | budget_matched | supported | 0 | 35623 | 75 | 0.01252797847 | 0.05750310835 | 0.002067225276 | 4/9/71 | 13 | 0.3076923077 | 0.05333333333 | 0.157480315 | 0.0002531788005 |
| training/2020-04 | reduce | quiet | all | v127 | original | supported | 0 | 42463 | 41 | 0.006284123346 | 0.04676327419 | 0.0009519919019 | 1/6/40 | 7 | 0.1428571429 | 0.0243902439 | 0.04166666667 | 0.0001414360473 |
| training/2020-04 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 42463 | 41 | 0.006284123346 | 0.04676327419 | 0.0009519919019 | 1/9/40 | 10 | 0.1 | 0.0243902439 | 0.03921568627 | 0.000212154071 |
| training/2020-04 | reduce | quiet | all | v137 | original | supported | 0 | 42463 | 41 | 0.006432324605 | 0.03502310985 | 0.0009544867447 | 2/20/39 | 22 | 0.09090909091 | 0.0487804878 | 0.06349206349 | 0.0004714534911 |
| training/2020-04 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 42463 | 41 | 0.006432324605 | 0.03502310985 | 0.0009544867447 | 1/8/40 | 9 | 0.1111111111 | 0.0243902439 | 0.04 | 0.0001885813964 |
| training/2020-04 | reduce | quiet | below | v127 | original | supported | 0 | 17075 | 11 | 0.004221134476 | 0.01774217714 | 0.00064187222 | 0/6/11 | 6 | 0 | 0 | 0 | 0.0003516174402 |
| training/2020-04 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 17075 | 11 | 0.004221134476 | 0.01774217714 | 0.00064187222 | 0/9/11 | 9 | 0 | 0 | 0 | 0.0005274261603 |
| training/2020-04 | reduce | quiet | below | v137 | original | supported | 0 | 17075 | 11 | 0.004417231011 | 0.02926084805 | 0.0006397808016 | 1/8/10 | 9 | 0.1111111111 | 0.09090909091 | 0.1 | 0.0004688232536 |
| training/2020-04 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 17075 | 11 | 0.004417231011 | 0.02926084805 | 0.0006397808016 | 0/7/11 | 7 | 0 | 0 | 0 | 0.0004102203469 |
| training/2020-04 | reduce | quiet | capped | v127 | original | supported | 0 | 25388 | 30 | 0.007671610937 | 0.06412527585 | 0.001160566566 | 1/0/29 | 1 | 1 | 0.03333333333 | 0.06451612903 | 0 |
| training/2020-04 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 25388 | 30 | 0.007671610937 | 0.06412527585 | 0.001160566566 | 1/1/29 | 2 | 0.5 | 0.03333333333 | 0.0625 | 3.943528669e-05 |
| training/2020-04 | reduce | quiet | capped | v137 | original | supported | 0 | 25388 | 30 | 0.00778759966 | 0.0577601669 | 0.001166145953 | 1/12/29 | 13 | 0.07692307692 | 0.03333333333 | 0.04651162791 | 0.0004732234403 |
| training/2020-04 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 25388 | 30 | 0.00778759966 | 0.0577601669 | 0.001166145953 | 1/2/29 | 3 | 0.3333333333 | 0.03333333333 | 0.06060606061 | 7.887057339e-05 |
| training/2020-04 | reduce | first | all | v127 | original | supported | 0 | 3748 | 11 | 0.01800334039 | 0.03753321445 | 0.002910549916 | 0/4/11 | 4 | 0 | 0 | 0 | 0.001070377308 |
| training/2020-04 | reduce | first | all | v127 | budget_matched | supported | 0 | 3748 | 11 | 0.01800334039 | 0.03753321445 | 0.002910549916 | 0/4/11 | 4 | 0 | 0 | 0 | 0.001070377308 |
| training/2020-04 | reduce | first | all | v137 | original | supported | 0 | 3748 | 11 | 0.01884169975 | 0.1311138778 | 0.002908902962 | 1/2/10 | 3 | 0.3333333333 | 0.09090909091 | 0.1428571429 | 0.000535188654 |
| training/2020-04 | reduce | first | all | v137 | budget_matched | supported | 0 | 3748 | 11 | 0.01884169975 | 0.1311138778 | 0.002908902962 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | first | below | v127 | original | supported | 0 | 3302 | 8 | 0.01565048084 | 0.02527870047 | 0.002414099235 | 0/4/8 | 4 | 0 | 0 | 0 | 0.001214329083 |
| training/2020-04 | reduce | first | below | v127 | budget_matched | supported | 0 | 3302 | 8 | 0.01565048084 | 0.02527870047 | 0.002414099235 | 0/4/8 | 4 | 0 | 0 | 0 | 0.001214329083 |
| training/2020-04 | reduce | first | below | v137 | original | supported | 0 | 3302 | 8 | 0.01626991774 | 0.1411587527 | 0.002405643411 | 1/1/7 | 2 | 0.5 | 0.125 | 0.2 | 0.0003035822708 |
| training/2020-04 | reduce | first | below | v137 | budget_matched | supported | 0 | 3302 | 8 | 0.01626991774 | 0.1411587527 | 0.002405643411 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | first | capped | v127 | original | supported | 0 | 446 | 3 | 0.0354229418 | 0.3909848114 | 0.006586065946 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | first | capped | v127 | budget_matched | supported | 0 | 446 | 3 | 0.0354229418 | 0.3909848114 | 0.006586065946 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | first | capped | v137 | original | supported | 0 | 446 | 3 | 0.03788211271 | 0.1545454545 | 0.006634829058 | 0/1/3 | 1 | 0 | 0 | 0 | 0.002257336343 |
| training/2020-04 | reduce | first | capped | v137 | budget_matched | supported | 0 | 446 | 3 | 0.03788211271 | 0.1545454545 | 0.006634829058 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | restart | all | v127 | original | supported | 0 | 38715 | 30 | 0.005149585687 | 0.05638642615 | 0.0007623838576 | 1/2/29 | 3 | 0.3333333333 | 0.03333333333 | 0.06060606061 | 5.169962518e-05 |
| training/2020-04 | reduce | restart | all | v127 | budget_matched | supported | 0 | 38715 | 30 | 0.005149585687 | 0.05638642615 | 0.0007623838576 | 1/2/29 | 3 | 0.3333333333 | 0.03333333333 | 0.06060606061 | 5.169962518e-05 |
| training/2020-04 | reduce | restart | all | v137 | original | supported | 0 | 38715 | 30 | 0.005230972725 | 0.03792760602 | 0.0007652796677 | 1/18/29 | 19 | 0.05263157895 | 0.03333333333 | 0.04081632653 | 0.0004652966266 |
| training/2020-04 | reduce | restart | all | v137 | budget_matched | supported | 0 | 38715 | 30 | 0.005230972725 | 0.03792760602 | 0.0007652796677 | 0/1/30 | 1 | 0 | 0 | 0 | 2.584981259e-05 |
| training/2020-04 | reduce | restart | below | v127 | original | supported | 0 | 13773 | 3 | 0.001481012374 | 0.03113729059 | 0.0002169906689 | 0/2/3 | 2 | 0 | 0 | 0 | 0.0001452432825 |
| training/2020-04 | reduce | restart | below | v127 | budget_matched | supported | 0 | 13773 | 3 | 0.001481012374 | 0.03113729059 | 0.0002169906689 | 0/2/3 | 2 | 0 | 0 | 0 | 0.0001452432825 |
| training/2020-04 | reduce | restart | below | v137 | original | supported | 0 | 13773 | 3 | 0.001575615417 | 0.03523322006 | 0.0002164250813 | 0/7/3 | 7 | 0 | 0 | 0 | 0.0005083514887 |
| training/2020-04 | reduce | restart | below | v137 | budget_matched | supported | 0 | 13773 | 3 | 0.001575615417 | 0.03523322006 | 0.0002164250813 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | restart | capped | v127 | original | supported | 0 | 24942 | 27 | 0.007175375929 | 0.06465091838 | 0.00106355058 | 1/0/26 | 1 | 1 | 0.03703703704 | 0.07142857143 | 0 |
| training/2020-04 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 24942 | 27 | 0.007175375929 | 0.06465091838 | 0.00106355058 | 1/0/26 | 1 | 1 | 0.03703703704 | 0.07142857143 | 0 |
| training/2020-04 | reduce | restart | capped | v137 | original | supported | 0 | 24942 | 27 | 0.007249465075 | 0.05945328072 | 0.001068357778 | 1/11/26 | 12 | 0.08333333333 | 0.03703703704 | 0.05128205128 | 0.0004415011038 |
| training/2020-04 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 24942 | 27 | 0.007249465075 | 0.05945328072 | 0.001068357778 | 1/0/26 | 1 | 1 | 0.03703703704 | 0.07142857143 | 0 |
| training/2020-04 | increase | quiet | all | v127 | original | supported | 0 | 42320 | 46 | 0.006931504668 | 0.03612658433 | 0.001073526114 | 0/2/46 | 2 | 0 | 0 | 0 | 4.731040356e-05 |
| training/2020-04 | increase | quiet | all | v127 | budget_matched | supported | 0 | 42320 | 46 | 0.006931504668 | 0.03612658433 | 0.001073526114 | 0/1/46 | 1 | 0 | 0 | 0 | 2.365520178e-05 |
| training/2020-04 | increase | quiet | all | v137 | original | supported | 0 | 42320 | 46 | 0.006855581388 | 0.05315781708 | 0.001068629752 | 2/26/44 | 28 | 0.07142857143 | 0.04347826087 | 0.06329113924 | 0.0006150352463 |
| training/2020-04 | increase | quiet | all | v137 | budget_matched | supported | 0 | 42320 | 46 | 0.006855581388 | 0.05315781708 | 0.001068629752 | 2/3/44 | 5 | 0.4 | 0.04347826087 | 0.1515151515 | 7.096560534e-05 |
| training/2020-04 | increase | quiet | below | v127 | original | supported | 0 | 16933 | 19 | 0.006939699464 | 0.05507839402 | 0.001103914458 | 0/2/19 | 2 | 0 | 0 | 0 | 0.0001182452406 |
| training/2020-04 | increase | quiet | below | v127 | budget_matched | supported | 0 | 16933 | 19 | 0.006939699464 | 0.05507839402 | 0.001103914458 | 0/0/19 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | quiet | below | v137 | original | supported | 0 | 16933 | 19 | 0.006893563329 | 0.08013474296 | 0.001096566164 | 0/14/19 | 14 | 0 | 0 | 0 | 0.0008277166844 |
| training/2020-04 | increase | quiet | below | v137 | budget_matched | supported | 0 | 16933 | 19 | 0.006893563329 | 0.08013474296 | 0.001096566164 | 1/1/18 | 2 | 0.5 | 0.05263157895 | 0.1851851852 | 5.912262031e-05 |
| training/2020-04 | increase | quiet | capped | v127 | original | supported | 0 | 25387 | 27 | 0.00692603878 | 0.02918249468 | 0.001053257243 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 25387 | 27 | 0.00692603878 | 0.02918249468 | 0.001053257243 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | quiet | capped | v137 | original | supported | 0 | 25387 | 27 | 0.006830247627 | 0.07169737576 | 0.001049996307 | 2/12/25 | 14 | 0.1428571429 | 0.07407407407 | 0.1204819277 | 0.0004731861199 |
| training/2020-04 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 25387 | 27 | 0.006830247627 | 0.07169737576 | 0.001049996307 | 1/0/26 | 1 | 1 | 0.03703703704 | 0.1612903226 | 0 |
| training/2020-04 | increase | first | all | v127 | original | supported | 0 | 3558 | 11 | 0.01720488388 | 0.1144057117 | 0.002998060324 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | all | v127 | budget_matched | supported | 0 | 3558 | 11 | 0.01720488388 | 0.1144057117 | 0.002998060324 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | all | v137 | original | supported | 0 | 3558 | 11 | 0.01682608538 | 0.14548283 | 0.002975610305 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | all | v137 | budget_matched | supported | 0 | 3558 | 11 | 0.01682608538 | 0.14548283 | 0.002975610305 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | below | v127 | original | supported | 0 | 3558 | 11 | 0.01720488388 | 0.1144057117 | 0.002998060324 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | below | v127 | budget_matched | supported | 0 | 3558 | 11 | 0.01720488388 | 0.1144057117 | 0.002998060324 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | below | v137 | original | supported | 0 | 3558 | 11 | 0.01682608538 | 0.14548283 | 0.002975610305 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | below | v137 | budget_matched | supported | 0 | 3558 | 11 | 0.01682608538 | 0.14548283 | 0.002975610305 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-04 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-04 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-04 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-04 | increase | restart | all | v127 | original | supported | 0 | 38762 | 35 | 0.005988501643 | 0.01725816619 | 0.0008968713307 | 0/2/35 | 2 | 0 | 0 | 0 | 5.164355618e-05 |
| training/2020-04 | increase | restart | all | v127 | budget_matched | supported | 0 | 38762 | 35 | 0.005988501643 | 0.01725816619 | 0.0008968713307 | 0/2/35 | 2 | 0 | 0 | 0 | 5.164355618e-05 |
| training/2020-04 | increase | restart | all | v137 | original | supported | 0 | 38762 | 35 | 0.005940379561 | 0.0483024635 | 0.0008935862348 | 2/26/33 | 28 | 0.07142857143 | 0.05714285714 | 0.06802721088 | 0.0006713662303 |
| training/2020-04 | increase | restart | all | v137 | budget_matched | supported | 0 | 38762 | 35 | 0.005940379561 | 0.0483024635 | 0.0008935862348 | 1/0/34 | 1 | 1 | 0.02857142857 | 0.1282051282 | 0 |
| training/2020-04 | increase | restart | below | v127 | original | supported | 0 | 13375 | 8 | 0.004208968537 | 0.009319424836 | 0.0006000362528 | 0/2/8 | 2 | 0 | 0 | 0 | 0.0001496222039 |
| training/2020-04 | increase | restart | below | v127 | budget_matched | supported | 0 | 13375 | 8 | 0.004208968537 | 0.009319424836 | 0.0006000362528 | 0/2/8 | 2 | 0 | 0 | 0 | 0.0001496222039 |
| training/2020-04 | increase | restart | below | v137 | original | supported | 0 | 13375 | 8 | 0.004251326808 | 0.01126429602 | 0.0005967053006 | 0/14/8 | 14 | 0 | 0 | 0 | 0.001047355428 |
| training/2020-04 | increase | restart | below | v137 | budget_matched | supported | 0 | 13375 | 8 | 0.004251326808 | 0.01126429602 | 0.0005967053006 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | restart | capped | v127 | original | supported | 0 | 25387 | 27 | 0.00692603878 | 0.02918249468 | 0.001053257243 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | restart | capped | v127 | budget_matched | supported | 0 | 25387 | 27 | 0.00692603878 | 0.02918249468 | 0.001053257243 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | restart | capped | v137 | original | supported | 0 | 25387 | 27 | 0.006830247627 | 0.07169737576 | 0.001049996307 | 2/12/25 | 14 | 0.1428571429 | 0.07407407407 | 0.1204819277 | 0.0004731861199 |
| training/2020-04 | increase | restart | capped | v137 | budget_matched | supported | 0 | 25387 | 27 | 0.006830247627 | 0.07169737576 | 0.001049996307 | 1/0/26 | 1 | 1 | 0.03703703704 | 0.1612903226 | 0 |
| training/2020-05 | reduce | quiet | all | v127 | original | supported | 0 | 43723 | 53 | 0.006768713272 | 0.1148923857 | 0.001163394848 | 3/20/50 | 23 | 0.1304347826 | 0.05660377358 | 0.07894736842 | 0.0004579803068 |
| training/2020-05 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 43723 | 53 | 0.006768713272 | 0.1148923857 | 0.001163394848 | 5/20/48 | 25 | 0.2 | 0.09433962264 | 0.1282051282 | 0.0004579803068 |
| training/2020-05 | reduce | quiet | all | v137 | original | supported | 0 | 43723 | 53 | 0.006733994008 | 0.1443693037 | 0.001156873804 | 8/50/45 | 58 | 0.1379310345 | 0.1509433962 | 0.1441441441 | 0.001144950767 |
| training/2020-05 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 43723 | 53 | 0.006733994008 | 0.1443693037 | 0.001156873804 | 6/11/47 | 17 | 0.3529411765 | 0.1132075472 | 0.1714285714 | 0.0002518891688 |
| training/2020-05 | reduce | quiet | below | v127 | original | supported | 0 | 12331 | 19 | 0.008303045368 | 0.1731872285 | 0.001465107117 | 2/17/17 | 19 | 0.1052631579 | 0.1052631579 | 0.1052631579 | 0.001380766732 |
| training/2020-05 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 12331 | 19 | 0.008303045368 | 0.1731872285 | 0.001465107117 | 4/22/15 | 26 | 0.1538461538 | 0.2105263158 | 0.1777777778 | 0.001786874594 |
| training/2020-05 | reduce | quiet | below | v137 | original | supported | 0 | 12331 | 19 | 0.00788048678 | 0.226681998 | 0.00145024525 | 5/34/14 | 39 | 0.1282051282 | 0.2631578947 | 0.1724137931 | 0.002761533463 |
| training/2020-05 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 12331 | 19 | 0.00788048678 | 0.226681998 | 0.00145024525 | 4/5/15 | 9 | 0.4444444444 | 0.2105263158 | 0.2857142857 | 0.0004061078622 |
| training/2020-05 | reduce | quiet | capped | v127 | original | supported | 0 | 31392 | 34 | 0.006166016754 | 0.1038189475 | 0.001044880131 | 1/3/33 | 4 | 0.25 | 0.02941176471 | 0.05263157895 | 9.566936667e-05 |
| training/2020-05 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 31392 | 34 | 0.006166016754 | 0.1038189475 | 0.001044880131 | 2/10/32 | 12 | 0.1666666667 | 0.05882352941 | 0.08695652174 | 0.0003188978889 |
| training/2020-05 | reduce | quiet | capped | v137 | original | supported | 0 | 31392 | 34 | 0.006283643525 | 0.1174480503 | 0.001041635421 | 3/16/31 | 19 | 0.1578947368 | 0.08823529412 | 0.1132075472 | 0.0005102366222 |
| training/2020-05 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 31392 | 34 | 0.006283643525 | 0.1174480503 | 0.001041635421 | 2/5/32 | 7 | 0.2857142857 | 0.05882352941 | 0.09756097561 | 0.0001594489444 |
| training/2020-05 | reduce | first | all | v127 | original | supported | 0 | 7117 | 11 | 0.009719548747 | 0.02804195473 | 0.001537109402 | 0/17/11 | 17 | 0 | 0 | 0 | 0.002392344498 |
| training/2020-05 | reduce | first | all | v127 | budget_matched | supported | 0 | 7117 | 11 | 0.009719548747 | 0.02804195473 | 0.001537109402 | 0/17/11 | 17 | 0 | 0 | 0 | 0.002392344498 |
| training/2020-05 | reduce | first | all | v137 | original | supported | 0 | 7117 | 11 | 0.009362335163 | 0.08225807374 | 0.001522057939 | 2/31/9 | 33 | 0.06060606061 | 0.1818181818 | 0.09090909091 | 0.004362510554 |
| training/2020-05 | reduce | first | all | v137 | budget_matched | supported | 0 | 7117 | 11 | 0.009362335163 | 0.08225807374 | 0.001522057939 | 1/1/10 | 2 | 0.5 | 0.09090909091 | 0.1538461538 | 0.0001407261469 |
| training/2020-05 | reduce | first | below | v127 | original | supported | 0 | 6107 | 9 | 0.00959323447 | 0.02135075382 | 0.001474774112 | 0/17/9 | 17 | 0 | 0 | 0 | 0.002787799278 |
| training/2020-05 | reduce | first | below | v127 | budget_matched | supported | 0 | 6107 | 9 | 0.00959323447 | 0.02135075382 | 0.001474774112 | 0/17/9 | 17 | 0 | 0 | 0 | 0.002787799278 |
| training/2020-05 | reduce | first | below | v137 | original | supported | 0 | 6107 | 9 | 0.009192523344 | 0.03217586155 | 0.001458964428 | 1/29/8 | 30 | 0.03333333333 | 0.1111111111 | 0.05128205128 | 0.004755657593 |
| training/2020-05 | reduce | first | below | v137 | budget_matched | supported | 0 | 6107 | 9 | 0.009192523344 | 0.03217586155 | 0.001458964428 | 0/2/9 | 2 | 0 | 0 | 0 | 0.0003279763857 |
| training/2020-05 | reduce | first | capped | v127 | original | supported | 0 | 1010 | 2 | 0.0104833124 | 0.511627907 | 0.001914021887 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | first | capped | v127 | budget_matched | supported | 0 | 1010 | 2 | 0.0104833124 | 0.511627907 | 0.001914021887 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | first | capped | v137 | original | supported | 0 | 1010 | 2 | 0.01038910822 | 0.5112359551 | 0.001903555037 | 1/2/1 | 3 | 0.3333333333 | 0.5 | 0.4 | 0.001984126984 |
| training/2020-05 | reduce | first | capped | v137 | budget_matched | supported | 0 | 1010 | 2 | 0.01038910822 | 0.5112359551 | 0.001903555037 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | restart | all | v127 | original | supported | 0 | 36606 | 42 | 0.006195006883 | 0.1544589984 | 0.001090736637 | 3/3/39 | 6 | 0.5 | 0.07142857143 | 0.125 | 8.204791598e-05 |
| training/2020-05 | reduce | restart | all | v127 | budget_matched | supported | 0 | 36606 | 42 | 0.006195006883 | 0.1544589984 | 0.001090736637 | 3/3/39 | 6 | 0.5 | 0.07142857143 | 0.125 | 8.204791598e-05 |
| training/2020-05 | reduce | restart | all | v137 | original | supported | 0 | 36606 | 42 | 0.006222987506 | 0.1691326424 | 0.001085874091 | 6/19/36 | 25 | 0.24 | 0.1428571429 | 0.1791044776 | 0.0005196368012 |
| training/2020-05 | reduce | restart | all | v137 | budget_matched | supported | 0 | 36606 | 42 | 0.006222987506 | 0.1691326424 | 0.001085874091 | 5/7/37 | 12 | 0.4166666667 | 0.119047619 | 0.1851851852 | 0.0001914451373 |
| training/2020-05 | reduce | restart | below | v127 | original | supported | 0 | 6224 | 10 | 0.0070371095 | 0.4140184402 | 0.001455621843 | 2/0/8 | 2 | 1 | 0.2 | 0.3333333333 | 0 |
| training/2020-05 | reduce | restart | below | v127 | budget_matched | supported | 0 | 6224 | 10 | 0.0070371095 | 0.4140184402 | 0.001455621843 | 2/0/8 | 2 | 1 | 0.2 | 0.3333333333 | 0 |
| training/2020-05 | reduce | restart | below | v137 | original | supported | 0 | 6224 | 10 | 0.006593114142 | 0.4096331313 | 0.001441689976 | 4/5/6 | 9 | 0.4444444444 | 0.4 | 0.4210526316 | 0.0008046346958 |
| training/2020-05 | reduce | restart | below | v137 | budget_matched | supported | 0 | 6224 | 10 | 0.006593114142 | 0.4096331313 | 0.001441689976 | 2/0/8 | 2 | 1 | 0.2 | 0.3333333333 | 0 |
| training/2020-05 | reduce | restart | capped | v127 | original | supported | 0 | 30382 | 32 | 0.006022495307 | 0.1010912828 | 0.001015986932 | 1/3/31 | 4 | 0.25 | 0.03125 | 0.05555555556 | 9.884678748e-05 |
| training/2020-05 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 30382 | 32 | 0.006022495307 | 0.1010912828 | 0.001015986932 | 1/3/31 | 4 | 0.25 | 0.03125 | 0.05555555556 | 9.884678748e-05 |
| training/2020-05 | reduce | restart | capped | v137 | original | supported | 0 | 30382 | 32 | 0.006147164052 | 0.1142327474 | 0.001012982311 | 2/14/30 | 16 | 0.125 | 0.0625 | 0.08333333333 | 0.0004612850082 |
| training/2020-05 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 30382 | 32 | 0.006147164052 | 0.1142327474 | 0.001012982311 | 2/4/30 | 6 | 0.3333333333 | 0.0625 | 0.1052631579 | 0.0001317957166 |
| training/2020-05 | increase | quiet | all | v127 | original | supported | 0 | 43714 | 48 | 0.006965527263 | 0.04545037682 | 0.001075497626 | 1/9/47 | 10 | 0.1 | 0.02083333333 | 0.05681818182 | 0.0002061100169 |
| training/2020-05 | increase | quiet | all | v127 | budget_matched | supported | 0 | 43714 | 48 | 0.006965527263 | 0.04545037682 | 0.001075497626 | 2/7/46 | 9 | 0.2222222222 | 0.04166666667 | 0.119047619 | 0.000160307791 |
| training/2020-05 | increase | quiet | all | v137 | original | supported | 0 | 43714 | 48 | 0.006929183392 | 0.06595038119 | 0.001070018841 | 5/48/43 | 53 | 0.09433962264 | 0.1041666667 | 0.09615384615 | 0.001099253424 |
| training/2020-05 | increase | quiet | all | v137 | budget_matched | supported | 0 | 43714 | 48 | 0.006929183392 | 0.06595038119 | 0.001070018841 | 2/4/46 | 6 | 0.3333333333 | 0.04166666667 | 0.1388888889 | 9.160445198e-05 |
| training/2020-05 | increase | quiet | below | v127 | original | supported | 0 | 12321 | 21 | 0.009344179788 | 0.08357281225 | 0.001639543349 | 1/9/20 | 10 | 0.1 | 0.04761904762 | 0.08196721311 | 0.0007317073171 |
| training/2020-05 | increase | quiet | below | v127 | budget_matched | supported | 0 | 12321 | 21 | 0.009344179788 | 0.08357281225 | 0.001639543349 | 2/6/19 | 8 | 0.25 | 0.09523809524 | 0.1886792453 | 0.000487804878 |
| training/2020-05 | increase | quiet | below | v137 | original | supported | 0 | 12321 | 21 | 0.009231983467 | 0.1334331372 | 0.001621259359 | 3/26/18 | 29 | 0.1034482759 | 0.1428571429 | 0.1094890511 | 0.002113821138 |
| training/2020-05 | increase | quiet | below | v137 | budget_matched | supported | 0 | 12321 | 21 | 0.009231983467 | 0.1334331372 | 0.001621259359 | 2/3/19 | 5 | 0.4 | 0.09523809524 | 0.243902439 | 0.000243902439 |
| training/2020-05 | increase | quiet | capped | v127 | original | supported | 0 | 31393 | 27 | 0.006031963164 | 0.02297378044 | 0.0008541232001 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 31393 | 27 | 0.006031963164 | 0.02297378044 | 0.0008541232001 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | quiet | capped | v137 | original | supported | 0 | 31393 | 27 | 0.006025389561 | 0.01715384752 | 0.0008536701515 | 2/22/25 | 24 | 0.08333333333 | 0.07407407407 | 0.08130081301 | 0.0007013964165 |
| training/2020-05 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 31393 | 27 | 0.006025389561 | 0.01715384752 | 0.0008536701515 | 0/4/27 | 4 | 0 | 0 | 0 | 0.0001275266212 |
| training/2020-05 | increase | first | all | v127 | original | supported | 0 | 7381 | 7 | 0.004058423877 | 0.2240150501 | 0.000858027513 | 1/5/6 | 6 | 0.1666666667 | 0.1428571429 | 0.1612903226 | 0.0006780580418 |
| training/2020-05 | increase | first | all | v127 | budget_matched | supported | 0 | 7381 | 7 | 0.004058423877 | 0.2240150501 | 0.000858027513 | 1/5/6 | 6 | 0.1666666667 | 0.1428571429 | 0.1612903226 | 0.0006780580418 |
| training/2020-05 | increase | first | all | v137 | original | supported | 0 | 7381 | 7 | 0.003615042964 | 0.3397266473 | 0.0008262031284 | 2/2/5 | 4 | 0.5 | 0.2857142857 | 0.4347826087 | 0.0002712232167 |
| training/2020-05 | increase | first | all | v137 | budget_matched | supported | 0 | 7381 | 7 | 0.003615042964 | 0.3397266473 | 0.0008262031284 | 2/2/5 | 4 | 0.5 | 0.2857142857 | 0.4347826087 | 0.0002712232167 |
| training/2020-05 | increase | first | below | v127 | original | supported | 0 | 7381 | 7 | 0.004058423877 | 0.2240150501 | 0.000858027513 | 1/5/6 | 6 | 0.1666666667 | 0.1428571429 | 0.1612903226 | 0.0006780580418 |
| training/2020-05 | increase | first | below | v127 | budget_matched | supported | 0 | 7381 | 7 | 0.004058423877 | 0.2240150501 | 0.000858027513 | 1/5/6 | 6 | 0.1666666667 | 0.1428571429 | 0.1612903226 | 0.0006780580418 |
| training/2020-05 | increase | first | below | v137 | original | supported | 0 | 7381 | 7 | 0.003615042964 | 0.3397266473 | 0.0008262031284 | 2/2/5 | 4 | 0.5 | 0.2857142857 | 0.4347826087 | 0.0002712232167 |
| training/2020-05 | increase | first | below | v137 | budget_matched | supported | 0 | 7381 | 7 | 0.003615042964 | 0.3397266473 | 0.0008262031284 | 2/2/5 | 4 | 0.5 | 0.2857142857 | 0.4347826087 | 0.0002712232167 |
| training/2020-05 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-05 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-05 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-05 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-05 | increase | restart | all | v127 | original | supported | 0 | 36333 | 41 | 0.007556101399 | 0.01689953379 | 0.001119676387 | 0/4/41 | 4 | 0 | 0 | 0 | 0.0001102171277 |
| training/2020-05 | increase | restart | all | v127 | budget_matched | supported | 0 | 36333 | 41 | 0.007556101399 | 0.01689953379 | 0.001119676387 | 0/4/41 | 4 | 0 | 0 | 0 | 0.0001102171277 |
| training/2020-05 | increase | restart | all | v137 | original | supported | 0 | 36333 | 41 | 0.007602446555 | 0.01708715429 | 0.001119549675 | 3/46/38 | 49 | 0.0612244898 | 0.07317073171 | 0.06329113924 | 0.001267496969 |
| training/2020-05 | increase | restart | all | v137 | budget_matched | supported | 0 | 36333 | 41 | 0.007602446555 | 0.01708715429 | 0.001119549675 | 0/6/41 | 6 | 0 | 0 | 0 | 0.0001653256916 |
| training/2020-05 | increase | restart | below | v127 | original | supported | 0 | 4940 | 14 | 0.01724178391 | 0.02341813034 | 0.002807229257 | 0/4/14 | 4 | 0 | 0 | 0 | 0.0008120178644 |
| training/2020-05 | increase | restart | below | v127 | budget_matched | supported | 0 | 4940 | 14 | 0.01724178391 | 0.02341813034 | 0.002807229257 | 0/4/14 | 4 | 0 | 0 | 0 | 0.0008120178644 |
| training/2020-05 | increase | restart | below | v137 | original | supported | 0 | 4940 | 14 | 0.01762442028 | 0.02438660486 | 0.002809176372 | 1/24/13 | 25 | 0.04 | 0.07142857143 | 0.04385964912 | 0.004872107186 |
| training/2020-05 | increase | restart | below | v137 | budget_matched | supported | 0 | 4940 | 14 | 0.01762442028 | 0.02438660486 | 0.002809176372 | 0/1/14 | 1 | 0 | 0 | 0 | 0.0002030044661 |
| training/2020-05 | increase | restart | capped | v127 | original | supported | 0 | 31393 | 27 | 0.006031963164 | 0.02297378044 | 0.0008541232001 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | restart | capped | v127 | budget_matched | supported | 0 | 31393 | 27 | 0.006031963164 | 0.02297378044 | 0.0008541232001 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | restart | capped | v137 | original | supported | 0 | 31393 | 27 | 0.006025389561 | 0.01715384752 | 0.0008536701515 | 2/22/25 | 24 | 0.08333333333 | 0.07407407407 | 0.08130081301 | 0.0007013964165 |
| training/2020-05 | increase | restart | capped | v137 | budget_matched | supported | 0 | 31393 | 27 | 0.006025389561 | 0.01715384752 | 0.0008536701515 | 0/4/27 | 4 | 0 | 0 | 0 | 0.0001275266212 |
| training/2020-06 | reduce | quiet | all | v127 | original | supported | 0 | 42630 | 36 | 0.005917944699 | 0.01708809225 | 0.0008394224059 | 0/1/36 | 1 | 0 | 0 | 0 | 2.347748509e-05 |
| training/2020-06 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 42630 | 36 | 0.005917944699 | 0.01708809225 | 0.0008394224059 | 0/2/36 | 2 | 0 | 0 | 0 | 4.695497018e-05 |
| training/2020-06 | reduce | quiet | all | v137 | original | supported | 0 | 42630 | 36 | 0.005987296237 | 0.02358183445 | 0.0008394518744 | 1/7/35 | 8 | 0.125 | 0.02777777778 | 0.04545454545 | 0.0001643423956 |
| training/2020-06 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 42630 | 36 | 0.005987296237 | 0.02358183445 | 0.0008394518744 | 0/1/36 | 1 | 0 | 0 | 0 | 2.347748509e-05 |
| training/2020-06 | reduce | quiet | below | v127 | original | supported | 0 | 16645 | 18 | 0.007580606964 | 0.01579425973 | 0.001078864257 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 16645 | 18 | 0.007580606964 | 0.01579425973 | 0.001078864257 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | quiet | below | v137 | original | supported | 0 | 16645 | 18 | 0.007651075584 | 0.01690069891 | 0.001078881094 | 0/1/18 | 1 | 0 | 0 | 0 | 6.014314067e-05 |
| training/2020-06 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 16645 | 18 | 0.007651075584 | 0.01690069891 | 0.001078881094 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | quiet | capped | v127 | original | supported | 0 | 25985 | 18 | 0.004852906661 | 0.02716193072 | 0.0006860450874 | 0/1/18 | 1 | 0 | 0 | 0 | 3.851041707e-05 |
| training/2020-06 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 25985 | 18 | 0.004852906661 | 0.02716193072 | 0.0006860450874 | 0/2/18 | 2 | 0 | 0 | 0 | 7.702083414e-05 |
| training/2020-06 | reduce | quiet | capped | v137 | original | supported | 0 | 25985 | 18 | 0.00492154264 | 0.04110093417 | 0.0006860826471 | 1/6/17 | 7 | 0.1428571429 | 0.05555555556 | 0.08 | 0.0002310625024 |
| training/2020-06 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 25985 | 18 | 0.00492154264 | 0.04110093417 | 0.0006860826471 | 0/1/18 | 1 | 0 | 0 | 0 | 3.851041707e-05 |
| training/2020-06 | reduce | first | all | v127 | original | supported | 0 | 6120 | 11 | 0.0122609771 | 0.0257287209 | 0.001791636019 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | all | v127 | budget_matched | supported | 0 | 6120 | 11 | 0.0122609771 | 0.0257287209 | 0.001791636019 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | all | v137 | original | supported | 0 | 6120 | 11 | 0.01253747105 | 0.0313721586 | 0.001791865689 | 0/1/11 | 1 | 0 | 0 | 0 | 0.0001636929121 |
| training/2020-06 | reduce | first | all | v137 | budget_matched | supported | 0 | 6120 | 11 | 0.01253747105 | 0.0313721586 | 0.001791865689 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | below | v127 | original | supported | 0 | 6120 | 11 | 0.0122609771 | 0.0257287209 | 0.001791636019 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | below | v127 | budget_matched | supported | 0 | 6120 | 11 | 0.0122609771 | 0.0257287209 | 0.001791636019 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | below | v137 | original | supported | 0 | 6120 | 11 | 0.01253747105 | 0.0313721586 | 0.001791865689 | 0/1/11 | 1 | 0 | 0 | 0 | 0.0001636929121 |
| training/2020-06 | reduce | first | below | v137 | budget_matched | supported | 0 | 6120 | 11 | 0.01253747105 | 0.0313721586 | 0.001791865689 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | capped | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | reduce | first | capped | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | reduce | first | capped | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | reduce | first | capped | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | reduce | restart | all | v127 | original | supported | 0 | 36510 | 25 | 0.004854691938 | 0.0204408582 | 0.0006798073056 | 0/1/25 | 1 | 0 | 0 | 0 | 2.740852405e-05 |
| training/2020-06 | reduce | restart | all | v127 | budget_matched | supported | 0 | 36510 | 25 | 0.004854691938 | 0.0204408582 | 0.0006798073056 | 0/1/25 | 1 | 0 | 0 | 0 | 2.740852405e-05 |
| training/2020-06 | reduce | restart | all | v137 | original | supported | 0 | 36510 | 25 | 0.004889321165 | 0.03023699136 | 0.0006798032152 | 1/6/24 | 7 | 0.1428571429 | 0.04 | 0.0625 | 0.0001644511443 |
| training/2020-06 | reduce | restart | all | v137 | budget_matched | supported | 0 | 36510 | 25 | 0.004889321165 | 0.03023699136 | 0.0006798032152 | 0/0/25 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | below | v127 | original | supported | 0 | 10525 | 7 | 0.00485909958 | 0.009878824892 | 0.0006644069484 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | below | v127 | budget_matched | supported | 0 | 10525 | 7 | 0.00485909958 | 0.009878824892 | 0.0006644069484 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | below | v137 | original | supported | 0 | 10525 | 7 | 0.004809770095 | 0.01058907133 | 0.0006643000285 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | below | v137 | budget_matched | supported | 0 | 10525 | 7 | 0.004809770095 | 0.01058907133 | 0.0006643000285 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | capped | v127 | original | supported | 0 | 25985 | 18 | 0.004852906661 | 0.02716193072 | 0.0006860450874 | 0/1/18 | 1 | 0 | 0 | 0 | 3.851041707e-05 |
| training/2020-06 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 25985 | 18 | 0.004852906661 | 0.02716193072 | 0.0006860450874 | 0/1/18 | 1 | 0 | 0 | 0 | 3.851041707e-05 |
| training/2020-06 | reduce | restart | capped | v137 | original | supported | 0 | 25985 | 18 | 0.00492154264 | 0.04110093417 | 0.0006860826471 | 1/6/17 | 7 | 0.1428571429 | 0.05555555556 | 0.08 | 0.0002310625024 |
| training/2020-06 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 25985 | 18 | 0.00492154264 | 0.04110093417 | 0.0006860826471 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | quiet | all | v127 | original | supported | 0 | 42522 | 42 | 0.006670744844 | 0.0305426529 | 0.0009769801462 | 1/6/41 | 7 | 0.1428571429 | 0.02380952381 | 0.07142857143 | 0.0001412429379 |
| training/2020-06 | increase | quiet | all | v127 | budget_matched | supported | 0 | 42522 | 42 | 0.006670744844 | 0.0305426529 | 0.0009769801462 | 1/6/41 | 7 | 0.1428571429 | 0.02380952381 | 0.07142857143 | 0.0001412429379 |
| training/2020-06 | increase | quiet | all | v137 | original | supported | 0 | 42522 | 42 | 0.006700646652 | 0.02734500409 | 0.0009780983305 | 1/10/41 | 11 | 0.09090909091 | 0.02380952381 | 0.05813953488 | 0.0002354048964 |
| training/2020-06 | increase | quiet | all | v137 | budget_matched | supported | 0 | 42522 | 42 | 0.006700646652 | 0.02734500409 | 0.0009780983305 | 1/5/41 | 6 | 0.1666666667 | 0.02380952381 | 0.07575757576 | 0.0001177024482 |
| training/2020-06 | increase | quiet | below | v127 | original | supported | 0 | 16588 | 22 | 0.008168356873 | 0.05486943342 | 0.001301046308 | 1/6/21 | 7 | 0.1428571429 | 0.04545454545 | 0.1 | 0.0003621876132 |
| training/2020-06 | increase | quiet | below | v127 | budget_matched | supported | 0 | 16588 | 22 | 0.008168356873 | 0.05486943342 | 0.001301046308 | 1/5/21 | 6 | 0.1666666667 | 0.04545454545 | 0.1086956522 | 0.000301823011 |
| training/2020-06 | increase | quiet | below | v137 | original | supported | 0 | 16588 | 22 | 0.008253873867 | 0.04478986035 | 0.00130543458 | 1/8/21 | 9 | 0.1111111111 | 0.04545454545 | 0.08620689655 | 0.0004829168176 |
| training/2020-06 | increase | quiet | below | v137 | budget_matched | supported | 0 | 16588 | 22 | 0.008253873867 | 0.04478986035 | 0.00130543458 | 1/4/21 | 5 | 0.2 | 0.04545454545 | 0.119047619 | 0.0002414584088 |
| training/2020-06 | increase | quiet | capped | v127 | original | supported | 0 | 25934 | 20 | 0.005712836756 | 0.005732800295 | 0.0007696997617 | 0/0/20 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 25934 | 20 | 0.005712836756 | 0.005732800295 | 0.0007696997617 | 0/0/20 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | quiet | capped | v137 | original | supported | 0 | 25934 | 20 | 0.005707165776 | 0.01164859806 | 0.0007687263207 | 0/2/20 | 2 | 0 | 0 | 0 | 7.717835919e-05 |
| training/2020-06 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 25934 | 20 | 0.005707165776 | 0.01164859806 | 0.0007687263207 | 0/0/20 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | first | all | v127 | original | supported | 0 | 1174 | 9 | 0.03987369644 | 0.09656609737 | 0.007447858523 | 1/3/8 | 4 | 0.25 | 0.1111111111 | 0.2 | 0.002575107296 |
| training/2020-06 | increase | first | all | v127 | budget_matched | supported | 0 | 1174 | 9 | 0.03987369644 | 0.09656609737 | 0.007447858523 | 1/3/8 | 4 | 0.25 | 0.1111111111 | 0.2 | 0.002575107296 |
| training/2020-06 | increase | first | all | v137 | original | supported | 0 | 1174 | 9 | 0.03938944562 | 0.07043688983 | 0.007483718712 | 0/3/9 | 3 | 0 | 0 | 0 | 0.002575107296 |
| training/2020-06 | increase | first | all | v137 | budget_matched | supported | 0 | 1174 | 9 | 0.03938944562 | 0.07043688983 | 0.007483718712 | 0/3/9 | 3 | 0 | 0 | 0 | 0.002575107296 |
| training/2020-06 | increase | first | below | v127 | original | supported | 0 | 1174 | 9 | 0.03987369644 | 0.09656609737 | 0.007447858523 | 1/3/8 | 4 | 0.25 | 0.1111111111 | 0.2 | 0.002575107296 |
| training/2020-06 | increase | first | below | v127 | budget_matched | supported | 0 | 1174 | 9 | 0.03987369644 | 0.09656609737 | 0.007447858523 | 1/3/8 | 4 | 0.25 | 0.1111111111 | 0.2 | 0.002575107296 |
| training/2020-06 | increase | first | below | v137 | original | supported | 0 | 1174 | 9 | 0.03938944562 | 0.07043688983 | 0.007483718712 | 0/3/9 | 3 | 0 | 0 | 0 | 0.002575107296 |
| training/2020-06 | increase | first | below | v137 | budget_matched | supported | 0 | 1174 | 9 | 0.03938944562 | 0.07043688983 | 0.007483718712 | 0/3/9 | 3 | 0 | 0 | 0 | 0.002575107296 |
| training/2020-06 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | increase | restart | all | v127 | original | supported | 0 | 41348 | 33 | 0.005728008431 | 0.0169690912 | 0.0007932515206 | 0/3/33 | 3 | 0 | 0 | 0 | 7.261285247e-05 |
| training/2020-06 | increase | restart | all | v127 | budget_matched | supported | 0 | 41348 | 33 | 0.005728008431 | 0.0169690912 | 0.0007932515206 | 0/3/33 | 3 | 0 | 0 | 0 | 7.261285247e-05 |
| training/2020-06 | increase | restart | all | v137 | original | supported | 0 | 41348 | 33 | 0.005772508653 | 0.01939448122 | 0.0007933832698 | 1/7/32 | 8 | 0.125 | 0.0303030303 | 0.07692307692 | 0.0001694299891 |
| training/2020-06 | increase | restart | all | v137 | budget_matched | supported | 0 | 41348 | 33 | 0.005772508653 | 0.01939448122 | 0.0007933832698 | 0/1/33 | 1 | 0 | 0 | 0 | 2.420428416e-05 |
| training/2020-06 | increase | restart | below | v127 | original | supported | 0 | 15414 | 13 | 0.00575353472 | 0.03987382938 | 0.0008328772707 | 0/3/13 | 3 | 0 | 0 | 0 | 0.0001947925459 |
| training/2020-06 | increase | restart | below | v127 | budget_matched | supported | 0 | 15414 | 13 | 0.00575353472 | 0.03987382938 | 0.0008328772707 | 0/3/13 | 3 | 0 | 0 | 0 | 0.0001947925459 |
| training/2020-06 | increase | restart | below | v137 | original | supported | 0 | 15414 | 13 | 0.00588244781 | 0.03603578846 | 0.0008348684988 | 1/5/12 | 6 | 0.1666666667 | 0.07692307692 | 0.1351351351 | 0.0003246542432 |
| training/2020-06 | increase | restart | below | v137 | budget_matched | supported | 0 | 15414 | 13 | 0.00588244781 | 0.03603578846 | 0.0008348684988 | 0/1/13 | 1 | 0 | 0 | 0 | 6.493084865e-05 |
| training/2020-06 | increase | restart | capped | v127 | original | supported | 0 | 25934 | 20 | 0.005712836756 | 0.005732800295 | 0.0007696997617 | 0/0/20 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | restart | capped | v127 | budget_matched | supported | 0 | 25934 | 20 | 0.005712836756 | 0.005732800295 | 0.0007696997617 | 0/0/20 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | restart | capped | v137 | original | supported | 0 | 25934 | 20 | 0.005707165776 | 0.01164859806 | 0.0007687263207 | 0/2/20 | 2 | 0 | 0 | 0 | 7.717835919e-05 |
| training/2020-06 | increase | restart | capped | v137 | budget_matched | supported | 0 | 25934 | 20 | 0.005707165776 | 0.01164859806 | 0.0007687263207 | 0/0/20 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | quiet | all | v127 | original | supported | 0 | 44179 | 27 | 0.00379460393 | 0.09054796283 | 0.0005987479771 | 0/1/27 | 1 | 0 | 0 | 0 | 2.264903062e-05 |
| training/2020-07 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 44179 | 27 | 0.00379460393 | 0.09054796283 | 0.0005987479771 | 2/2/25 | 4 | 0.5 | 0.07407407407 | 0.1290322581 | 4.529806124e-05 |
| training/2020-07 | reduce | quiet | all | v137 | original | supported | 0 | 44179 | 27 | 0.003799511256 | 0.1389886515 | 0.0005978023527 | 2/3/25 | 5 | 0.4 | 0.07407407407 | 0.125 | 6.794709186e-05 |
| training/2020-07 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 44179 | 27 | 0.003799511256 | 0.1389886515 | 0.0005978023527 | 2/0/25 | 2 | 1 | 0.07407407407 | 0.1379310345 | 0 |
| training/2020-07 | reduce | quiet | below | v127 | original | supported | 0 | 25583 | 8 | 0.001924027601 | 0.1700604422 | 0.0003065614712 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 25583 | 8 | 0.001924027601 | 0.1700604422 | 0.0003065614712 | 1/0/7 | 1 | 1 | 0.125 | 0.2222222222 | 0 |
| training/2020-07 | reduce | quiet | below | v137 | original | supported | 0 | 25583 | 8 | 0.002016574538 | 0.06992542604 | 0.0003095310508 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 25583 | 8 | 0.002016574538 | 0.06992542604 | 0.0003095310508 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | quiet | capped | v127 | original | supported | 0 | 18596 | 19 | 0.006368004351 | 0.08532692345 | 0.001000716539 | 0/1/19 | 1 | 0 | 0 | 0 | 5.383000484e-05 |
| training/2020-07 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 18596 | 19 | 0.006368004351 | 0.08532692345 | 0.001000716539 | 2/2/17 | 4 | 0.5 | 0.1052631579 | 0.1739130435 | 0.0001076600097 |
| training/2020-07 | reduce | quiet | capped | v137 | original | supported | 0 | 18596 | 19 | 0.006252343589 | 0.1720151229 | 0.0009943846671 | 2/3/17 | 5 | 0.4 | 0.1052631579 | 0.1666666667 | 0.0001614900145 |
| training/2020-07 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 18596 | 19 | 0.006252343589 | 0.1720151229 | 0.0009943846671 | 2/0/17 | 2 | 1 | 0.1052631579 | 0.1904761905 | 0 |
| training/2020-07 | reduce | first | all | v127 | original | supported | 0 | 17551 | 7 | 0.002597572798 | 0.06388049778 | 0.0003954693732 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | all | v127 | budget_matched | supported | 0 | 17551 | 7 | 0.002597572798 | 0.06388049778 | 0.0003954693732 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | all | v137 | original | supported | 0 | 17551 | 7 | 0.00266393076 | 0.05714641921 | 0.0003966963438 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | all | v137 | budget_matched | supported | 0 | 17551 | 7 | 0.00266393076 | 0.05714641921 | 0.0003966963438 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | below | v127 | original | supported | 0 | 17337 | 5 | 0.00180082576 | 0.108994079 | 0.00028518027 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | below | v127 | budget_matched | supported | 0 | 17337 | 5 | 0.00180082576 | 0.108994079 | 0.00028518027 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | below | v137 | original | supported | 0 | 17337 | 5 | 0.001886862093 | 0.08211533686 | 0.0002864803589 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | below | v137 | budget_matched | supported | 0 | 17337 | 5 | 0.001886862093 | 0.08211533686 | 0.0002864803589 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | capped | v127 | original | supported | 0 | 214 | 2 | 0.0671452523 | 0.03225108225 | 0.009330432841 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | capped | v127 | budget_matched | supported | 0 | 214 | 2 | 0.0671452523 | 0.03225108225 | 0.009330432841 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | capped | v137 | original | supported | 0 | 214 | 2 | 0.06561738622 | 0.03921568627 | 0.009325736203 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | capped | v137 | budget_matched | supported | 0 | 214 | 2 | 0.06561738622 | 0.03921568627 | 0.009325736203 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | all | v127 | original | supported | 0 | 26628 | 20 | 0.00458358896 | 0.1079625407 | 0.000732732609 | 0/1/20 | 1 | 0 | 0 | 0 | 3.75826819e-05 |
| training/2020-07 | reduce | restart | all | v127 | budget_matched | supported | 0 | 26628 | 20 | 0.00458358896 | 0.1079625407 | 0.000732732609 | 0/1/20 | 1 | 0 | 0 | 0 | 3.75826819e-05 |
| training/2020-07 | reduce | restart | all | v137 | original | supported | 0 | 26628 | 20 | 0.004547993053 | 0.177289508 | 0.0007303549877 | 2/3/18 | 5 | 0.4 | 0.1 | 0.16 | 0.0001127480457 |
| training/2020-07 | reduce | restart | all | v137 | budget_matched | supported | 0 | 26628 | 20 | 0.004547993053 | 0.177289508 | 0.0007303549877 | 1/0/19 | 1 | 1 | 0.05 | 0.09523809524 | 0 |
| training/2020-07 | reduce | restart | below | v127 | original | supported | 0 | 8246 | 3 | 0.002183056261 | 0.3549692349 | 0.0003515148895 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | below | v127 | budget_matched | supported | 0 | 8246 | 3 | 0.002183056261 | 0.3549692349 | 0.0003515148895 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | below | v137 | original | supported | 0 | 8246 | 3 | 0.002289291571 | 0.1079010073 | 0.0003579945296 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | below | v137 | budget_matched | supported | 0 | 8246 | 3 | 0.002289291571 | 0.1079010073 | 0.0003579945296 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | capped | v127 | original | supported | 0 | 18382 | 17 | 0.005660446356 | 0.09473687647 | 0.0009037434519 | 0/1/17 | 1 | 0 | 0 | 0 | 5.445140212e-05 |
| training/2020-07 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 18382 | 17 | 0.005660446356 | 0.09473687647 | 0.0009037434519 | 0/1/17 | 1 | 0 | 0 | 0 | 5.445140212e-05 |
| training/2020-07 | reduce | restart | capped | v137 | original | supported | 0 | 18382 | 17 | 0.005561226239 | 0.1913409527 | 0.0008973925428 | 2/3/15 | 5 | 0.4 | 0.1176470588 | 0.1818181818 | 0.0001633542064 |
| training/2020-07 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 18382 | 17 | 0.005561226239 | 0.1913409527 | 0.0008973925428 | 1/0/16 | 1 | 1 | 0.05882352941 | 0.1111111111 | 0 |
| training/2020-07 | increase | quiet | all | v127 | original | supported | 0 | 44066 | 30 | 0.00475157971 | 0.01682369846 | 0.0006769510415 | 0/1/30 | 1 | 0 | 0 | 0 | 2.270869289e-05 |
| training/2020-07 | increase | quiet | all | v127 | budget_matched | supported | 0 | 44066 | 30 | 0.00475157971 | 0.01682369846 | 0.0006769510415 | 0/2/30 | 2 | 0 | 0 | 0 | 4.541738578e-05 |
| training/2020-07 | increase | quiet | all | v137 | original | supported | 0 | 44066 | 30 | 0.004766081191 | 0.02210737425 | 0.0006751399399 | 1/17/29 | 18 | 0.05555555556 | 0.03333333333 | 0.04901960784 | 0.0003860477791 |
| training/2020-07 | increase | quiet | all | v137 | budget_matched | supported | 0 | 44066 | 30 | 0.004766081191 | 0.02210737425 | 0.0006751399399 | 0/4/30 | 4 | 0 | 0 | 0 | 9.083477155e-05 |
| training/2020-07 | increase | quiet | below | v127 | original | supported | 0 | 25467 | 14 | 0.003819547828 | 0.01344133598 | 0.0005491153515 | 0/1/14 | 1 | 0 | 0 | 0 | 3.928809963e-05 |
| training/2020-07 | increase | quiet | below | v127 | budget_matched | supported | 0 | 25467 | 14 | 0.003819547828 | 0.01344133598 | 0.0005491153515 | 0/1/14 | 1 | 0 | 0 | 0 | 3.928809963e-05 |
| training/2020-07 | increase | quiet | below | v137 | original | supported | 0 | 25467 | 14 | 0.003887895001 | 0.01619404286 | 0.0005492348188 | 0/10/14 | 10 | 0 | 0 | 0 | 0.0003928809963 |
| training/2020-07 | increase | quiet | below | v137 | budget_matched | supported | 0 | 25467 | 14 | 0.003887895001 | 0.01619404286 | 0.0005492348188 | 0/2/14 | 2 | 0 | 0 | 0 | 7.857619927e-05 |
| training/2020-07 | increase | quiet | capped | v127 | original | supported | 0 | 18599 | 16 | 0.006027780363 | 0.04173890974 | 0.0008519922544 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 18599 | 16 | 0.006027780363 | 0.04173890974 | 0.0008519922544 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | quiet | capped | v137 | original | supported | 0 | 18599 | 16 | 0.005968552705 | 0.08601735404 | 0.0008475376881 | 1/7/15 | 8 | 0.125 | 0.0625 | 0.1041666667 | 0.0003766883711 |
| training/2020-07 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 18599 | 16 | 0.005968552705 | 0.08601735404 | 0.0008475376881 | 1/0/15 | 1 | 1 | 0.0625 | 0.25 | 0 |
| training/2020-07 | increase | first | all | v127 | original | supported | 0 | 11344 | 5 | 0.002997220118 | 0.02428286841 | 0.0004402817539 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | first | all | v127 | budget_matched | supported | 0 | 11344 | 5 | 0.002997220118 | 0.02428286841 | 0.0004402817539 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | first | all | v137 | original | supported | 0 | 11344 | 5 | 0.002956699368 | 0.03081527157 | 0.0004419939595 | 0/1/5 | 1 | 0 | 0 | 0 | 8.819119852e-05 |
| training/2020-07 | increase | first | all | v137 | budget_matched | supported | 0 | 11344 | 5 | 0.002956699368 | 0.03081527157 | 0.0004419939595 | 0/1/5 | 1 | 0 | 0 | 0 | 8.819119852e-05 |
| training/2020-07 | increase | first | below | v127 | original | supported | 0 | 11344 | 5 | 0.002997220118 | 0.02428286841 | 0.0004402817539 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | first | below | v127 | budget_matched | supported | 0 | 11344 | 5 | 0.002997220118 | 0.02428286841 | 0.0004402817539 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | first | below | v137 | original | supported | 0 | 11344 | 5 | 0.002956699368 | 0.03081527157 | 0.0004419939595 | 0/1/5 | 1 | 0 | 0 | 0 | 8.819119852e-05 |
| training/2020-07 | increase | first | below | v137 | budget_matched | supported | 0 | 11344 | 5 | 0.002956699368 | 0.03081527157 | 0.0004419939595 | 0/1/5 | 1 | 0 | 0 | 0 | 8.819119852e-05 |
| training/2020-07 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-07 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-07 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-07 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-07 | increase | restart | all | v127 | original | supported | 0 | 32722 | 25 | 0.005359777718 | 0.01882674329 | 0.0007589990948 | 0/1/25 | 1 | 0 | 0 | 0 | 3.058384561e-05 |
| training/2020-07 | increase | restart | all | v127 | budget_matched | supported | 0 | 32722 | 25 | 0.005359777718 | 0.01882674329 | 0.0007589990948 | 0/1/25 | 1 | 0 | 0 | 0 | 3.058384561e-05 |
| training/2020-07 | increase | restart | all | v137 | original | supported | 0 | 32722 | 25 | 0.0053933542 | 0.03705630748 | 0.0007559665398 | 1/16/24 | 17 | 0.05882352941 | 0.04 | 0.05376344086 | 0.0004893415298 |
| training/2020-07 | increase | restart | all | v137 | budget_matched | supported | 0 | 32722 | 25 | 0.0053933542 | 0.03705630748 | 0.0007559665398 | 1/2/24 | 3 | 0.3333333333 | 0.04 | 0.1351351351 | 6.116769123e-05 |
| training/2020-07 | increase | restart | below | v127 | original | supported | 0 | 14123 | 9 | 0.004480065108 | 0.01158011709 | 0.0006365336288 | 0/1/9 | 1 | 0 | 0 | 0 | 7.085163667e-05 |
| training/2020-07 | increase | restart | below | v127 | budget_matched | supported | 0 | 14123 | 9 | 0.004480065108 | 0.01158011709 | 0.0006365336288 | 0/1/9 | 1 | 0 | 0 | 0 | 7.085163667e-05 |
| training/2020-07 | increase | restart | below | v137 | original | supported | 0 | 14123 | 9 | 0.004635858129 | 0.01421793773 | 0.0006353737628 | 0/9/9 | 9 | 0 | 0 | 0 | 0.0006376647301 |
| training/2020-07 | increase | restart | below | v137 | budget_matched | supported | 0 | 14123 | 9 | 0.004635858129 | 0.01421793773 | 0.0006353737628 | 0/2/9 | 2 | 0 | 0 | 0 | 0.0001417032733 |
| training/2020-07 | increase | restart | capped | v127 | original | supported | 0 | 18599 | 16 | 0.006027780363 | 0.04173890974 | 0.0008519922544 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | restart | capped | v127 | budget_matched | supported | 0 | 18599 | 16 | 0.006027780363 | 0.04173890974 | 0.0008519922544 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | restart | capped | v137 | original | supported | 0 | 18599 | 16 | 0.005968552705 | 0.08601735404 | 0.0008475376881 | 1/7/15 | 8 | 0.125 | 0.0625 | 0.1041666667 | 0.0003766883711 |
| training/2020-07 | increase | restart | capped | v137 | budget_matched | supported | 0 | 18599 | 16 | 0.005968552705 | 0.08601735404 | 0.0008475376881 | 1/0/15 | 1 | 1 | 0.0625 | 0.25 | 0 |
| training/2020-08 | reduce | quiet | all | v127 | original | supported | 0 | 43683 | 58 | 0.008620639022 | 0.05584234999 | 0.001307627477 | 2/2/56 | 4 | 0.5 | 0.03448275862 | 0.06451612903 | 4.584527221e-05 |
| training/2020-08 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 43683 | 58 | 0.008620639022 | 0.05584234999 | 0.001307627477 | 2/6/56 | 8 | 0.25 | 0.03448275862 | 0.06060606061 | 0.0001375358166 |
| training/2020-08 | reduce | quiet | all | v137 | original | supported | 0 | 43683 | 58 | 0.008563473231 | 0.07300935958 | 0.001302330436 | 7/14/51 | 21 | 0.3333333333 | 0.1206896552 | 0.1772151899 | 0.0003209169054 |
| training/2020-08 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 43683 | 58 | 0.008563473231 | 0.07300935958 | 0.001302330436 | 2/8/56 | 10 | 0.2 | 0.03448275862 | 0.05882352941 | 0.0001833810888 |
| training/2020-08 | reduce | quiet | below | v127 | original | supported | 0 | 22761 | 29 | 0.008650351815 | 0.04858109793 | 0.001257614181 | 1/2/28 | 3 | 0.3333333333 | 0.03448275862 | 0.0625 | 8.798169981e-05 |
| training/2020-08 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 22761 | 29 | 0.008650351815 | 0.04858109793 | 0.001257614181 | 2/6/27 | 8 | 0.25 | 0.06896551724 | 0.1081081081 | 0.0002639450994 |
| training/2020-08 | reduce | quiet | below | v137 | original | supported | 0 | 22761 | 29 | 0.008476703849 | 0.078957358 | 0.001250345961 | 5/9/24 | 14 | 0.3571428571 | 0.1724137931 | 0.2325581395 | 0.0003959176491 |
| training/2020-08 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 22761 | 29 | 0.008476703849 | 0.078957358 | 0.001250345961 | 3/6/26 | 9 | 0.3333333333 | 0.1034482759 | 0.1578947368 | 0.0002639450994 |
| training/2020-08 | reduce | quiet | capped | v127 | original | supported | 0 | 20922 | 29 | 0.008588314537 | 0.09133122015 | 0.001362036836 | 1/0/28 | 1 | 1 | 0.03448275862 | 0.06666666667 | 0 |
| training/2020-08 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 20922 | 29 | 0.008588314537 | 0.09133122015 | 0.001362036836 | 2/1/27 | 3 | 0.6666666667 | 0.06896551724 | 0.125 | 4.78629206e-05 |
| training/2020-08 | reduce | quiet | capped | v137 | original | supported | 0 | 20922 | 29 | 0.00865786946 | 0.0725413829 | 0.001358884237 | 2/5/27 | 7 | 0.2857142857 | 0.06896551724 | 0.1111111111 | 0.000239314603 |
| training/2020-08 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 20922 | 29 | 0.00865786946 | 0.0725413829 | 0.001358884237 | 1/2/28 | 3 | 0.3333333333 | 0.03448275862 | 0.0625 | 9.572584119e-05 |
| training/2020-08 | reduce | first | all | v127 | original | supported | 0 | 15986 | 23 | 0.01022436451 | 0.06119409316 | 0.001429166887 | 1/1/22 | 2 | 0.5 | 0.04347826087 | 0.08 | 6.264486625e-05 |
| training/2020-08 | reduce | first | all | v127 | budget_matched | supported | 0 | 15986 | 23 | 0.01022436451 | 0.06119409316 | 0.001429166887 | 1/2/22 | 3 | 0.3333333333 | 0.04347826087 | 0.07692307692 | 0.0001252897325 |
| training/2020-08 | reduce | first | all | v137 | original | supported | 0 | 15986 | 23 | 0.01021776781 | 0.08998383779 | 0.001427532327 | 2/4/21 | 6 | 0.3333333333 | 0.08695652174 | 0.1379310345 | 0.000250579465 |
| training/2020-08 | reduce | first | all | v137 | budget_matched | supported | 0 | 15986 | 23 | 0.01021776781 | 0.08998383779 | 0.001427532327 | 0/0/23 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | first | below | v127 | original | supported | 0 | 15123 | 21 | 0.01002338188 | 0.06213882993 | 0.001379418213 | 1/1/20 | 2 | 0.5 | 0.04761904762 | 0.08695652174 | 6.621639518e-05 |
| training/2020-08 | reduce | first | below | v127 | budget_matched | supported | 0 | 15123 | 21 | 0.01002338188 | 0.06213882993 | 0.001379418213 | 1/2/20 | 3 | 0.3333333333 | 0.04761904762 | 0.08333333333 | 0.0001324327904 |
| training/2020-08 | reduce | first | below | v137 | original | supported | 0 | 15123 | 21 | 0.009922998802 | 0.1054515851 | 0.001377069019 | 2/3/19 | 5 | 0.4 | 0.09523809524 | 0.1538461538 | 0.0001986491855 |
| training/2020-08 | reduce | first | below | v137 | budget_matched | supported | 0 | 15123 | 21 | 0.009922998802 | 0.1054515851 | 0.001377069019 | 0/0/21 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | first | capped | v127 | original | supported | 0 | 863 | 2 | 0.01374633474 | 0.1166666667 | 0.002300950425 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | first | capped | v127 | budget_matched | supported | 0 | 863 | 2 | 0.01374633474 | 0.1166666667 | 0.002300950425 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | first | capped | v137 | original | supported | 0 | 863 | 2 | 0.0153832275 | 0.04483430799 | 0.002311838939 | 0/1/2 | 1 | 0 | 0 | 0 | 0.001161440186 |
| training/2020-08 | reduce | first | capped | v137 | budget_matched | supported | 0 | 863 | 2 | 0.0153832275 | 0.04483430799 | 0.002311838939 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | restart | all | v127 | original | supported | 0 | 27697 | 35 | 0.00769500969 | 0.07384553208 | 0.001237478038 | 1/1/34 | 2 | 0.5 | 0.02857142857 | 0.05405405405 | 3.615067602e-05 |
| training/2020-08 | reduce | restart | all | v127 | budget_matched | supported | 0 | 27697 | 35 | 0.00769500969 | 0.07384553208 | 0.001237478038 | 1/1/34 | 2 | 0.5 | 0.02857142857 | 0.05405405405 | 3.615067602e-05 |
| training/2020-08 | reduce | restart | all | v137 | original | supported | 0 | 27697 | 35 | 0.007608656712 | 0.09746266963 | 0.001230067107 | 5/10/30 | 15 | 0.3333333333 | 0.1428571429 | 0.2 | 0.0003615067602 |
| training/2020-08 | reduce | restart | all | v137 | budget_matched | supported | 0 | 27697 | 35 | 0.007608656712 | 0.09746266963 | 0.001230067107 | 1/1/34 | 2 | 0.5 | 0.02857142857 | 0.05405405405 | 3.615067602e-05 |
| training/2020-08 | reduce | restart | below | v127 | original | supported | 0 | 7638 | 8 | 0.005931795435 | 0.1079274371 | 0.001016446026 | 0/1/8 | 1 | 0 | 0 | 0 | 0.000131061599 |
| training/2020-08 | reduce | restart | below | v127 | budget_matched | supported | 0 | 7638 | 8 | 0.005931795435 | 0.1079274371 | 0.001016446026 | 0/1/8 | 1 | 0 | 0 | 0 | 0.000131061599 |
| training/2020-08 | reduce | restart | below | v137 | original | supported | 0 | 7638 | 8 | 0.005613085289 | 0.181659519 | 0.0009994382889 | 3/6/5 | 9 | 0.3333333333 | 0.375 | 0.3529411765 | 0.0007863695937 |
| training/2020-08 | reduce | restart | below | v137 | budget_matched | supported | 0 | 7638 | 8 | 0.005613085289 | 0.181659519 | 0.0009994382889 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | restart | capped | v127 | original | supported | 0 | 20059 | 27 | 0.008366400611 | 0.09563549619 | 0.00132164188 | 1/0/26 | 1 | 1 | 0.03703703704 | 0.07142857143 | 0 |
| training/2020-08 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 20059 | 27 | 0.008366400611 | 0.09563549619 | 0.00132164188 | 1/0/26 | 1 | 1 | 0.03703703704 | 0.07142857143 | 0 |
| training/2020-08 | reduce | restart | capped | v137 | original | supported | 0 | 20059 | 27 | 0.008368523831 | 0.07765578642 | 0.001317885189 | 2/4/25 | 6 | 0.3333333333 | 0.07407407407 | 0.1212121212 | 0.0001996805112 |
| training/2020-08 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 20059 | 27 | 0.008368523831 | 0.07765578642 | 0.001317885189 | 1/1/26 | 2 | 0.5 | 0.03703703704 | 0.06896551724 | 4.99201278e-05 |
| training/2020-08 | increase | quiet | all | v127 | original | supported | 0 | 43344 | 71 | 0.01020228485 | 0.03744688667 | 0.001610018417 | 0/4/71 | 4 | 0 | 0 | 0 | 9.243639221e-05 |
| training/2020-08 | increase | quiet | all | v127 | budget_matched | supported | 0 | 43344 | 71 | 0.01020228485 | 0.03744688667 | 0.001610018417 | 2/11/69 | 13 | 0.1538461538 | 0.02816901408 | 0.08130081301 | 0.0002542000786 |
| training/2020-08 | increase | quiet | all | v137 | original | supported | 0 | 43344 | 71 | 0.01024860959 | 0.04989797155 | 0.001607309612 | 2/36/69 | 38 | 0.05263157895 | 0.02816901408 | 0.04484304933 | 0.0008319275299 |
| training/2020-08 | increase | quiet | all | v137 | budget_matched | supported | 0 | 43344 | 71 | 0.01024860959 | 0.04989797155 | 0.001607309612 | 3/11/68 | 14 | 0.2142857143 | 0.04225352113 | 0.1181102362 | 0.0002542000786 |
| training/2020-08 | increase | quiet | below | v127 | original | supported | 0 | 22493 | 50 | 0.01312598108 | 0.05021694655 | 0.002172226695 | 0/4/50 | 4 | 0 | 0 | 0 | 0.000178229292 |
| training/2020-08 | increase | quiet | below | v127 | budget_matched | supported | 0 | 22493 | 50 | 0.01312598108 | 0.05021694655 | 0.002172226695 | 1/6/49 | 7 | 0.1428571429 | 0.02 | 0.0641025641 | 0.000267343938 |
| training/2020-08 | increase | quiet | below | v137 | original | supported | 0 | 22493 | 50 | 0.01325344766 | 0.06889366963 | 0.002166742424 | 2/17/48 | 19 | 0.1052631579 | 0.04 | 0.07936507937 | 0.0007574744909 |
| training/2020-08 | increase | quiet | below | v137 | budget_matched | supported | 0 | 22493 | 50 | 0.01325344766 | 0.06889366963 | 0.002166742424 | 2/10/48 | 12 | 0.1666666667 | 0.04 | 0.1020408163 | 0.00044557323 |
| training/2020-08 | increase | quiet | capped | v127 | original | supported | 0 | 20851 | 21 | 0.007048349823 | 0.009417914259 | 0.001003536676 | 0/0/21 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 20851 | 21 | 0.007048349823 | 0.009417914259 | 0.001003536676 | 0/0/21 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | quiet | capped | v137 | original | supported | 0 | 20851 | 21 | 0.007007142876 | 0.01087921493 | 0.001003821903 | 0/19/21 | 19 | 0 | 0 | 0 | 0.0009121459434 |
| training/2020-08 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 20851 | 21 | 0.007007142876 | 0.01087921493 | 0.001003821903 | 0/3/21 | 3 | 0 | 0 | 0 | 0.0001440230437 |
| training/2020-08 | increase | first | all | v127 | original | supported | 0 | 9382 | 19 | 0.01056218655 | 0.08246140397 | 0.001945633174 | 0/4/19 | 4 | 0 | 0 | 0 | 0.0004272134999 |
| training/2020-08 | increase | first | all | v127 | budget_matched | supported | 0 | 9382 | 19 | 0.01056218655 | 0.08246140397 | 0.001945633174 | 0/4/19 | 4 | 0 | 0 | 0 | 0.0004272134999 |
| training/2020-08 | increase | first | all | v137 | original | supported | 0 | 9382 | 19 | 0.01033505978 | 0.1354154146 | 0.001921637016 | 1/2/18 | 3 | 0.3333333333 | 0.05263157895 | 0.1612903226 | 0.00021360675 |
| training/2020-08 | increase | first | all | v137 | budget_matched | supported | 0 | 9382 | 19 | 0.01033505978 | 0.1354154146 | 0.001921637016 | 1/7/18 | 8 | 0.125 | 0.05263157895 | 0.09803921569 | 0.0007476236249 |
| training/2020-08 | increase | first | below | v127 | original | supported | 0 | 9382 | 19 | 0.01056218655 | 0.08246140397 | 0.001945633174 | 0/4/19 | 4 | 0 | 0 | 0 | 0.0004272134999 |
| training/2020-08 | increase | first | below | v127 | budget_matched | supported | 0 | 9382 | 19 | 0.01056218655 | 0.08246140397 | 0.001945633174 | 0/4/19 | 4 | 0 | 0 | 0 | 0.0004272134999 |
| training/2020-08 | increase | first | below | v137 | original | supported | 0 | 9382 | 19 | 0.01033505978 | 0.1354154146 | 0.001921637016 | 1/2/18 | 3 | 0.3333333333 | 0.05263157895 | 0.1612903226 | 0.00021360675 |
| training/2020-08 | increase | first | below | v137 | budget_matched | supported | 0 | 9382 | 19 | 0.01033505978 | 0.1354154146 | 0.001921637016 | 1/7/18 | 8 | 0.125 | 0.05263157895 | 0.09803921569 | 0.0007476236249 |
| training/2020-08 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-08 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-08 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-08 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-08 | increase | restart | all | v127 | original | supported | 0 | 33962 | 52 | 0.01010286203 | 0.02656856792 | 0.001517304865 | 0/0/52 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | all | v127 | budget_matched | supported | 0 | 33962 | 52 | 0.01010286203 | 0.02656856792 | 0.001517304865 | 0/0/52 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | all | v137 | original | supported | 0 | 33962 | 52 | 0.01022472774 | 0.01776775728 | 0.00152047669 | 1/34/51 | 35 | 0.02857142857 | 0.01923076923 | 0.02604166667 | 0.001002654084 |
| training/2020-08 | increase | restart | all | v137 | budget_matched | supported | 0 | 33962 | 52 | 0.01022472774 | 0.01776775728 | 0.00152047669 | 0/3/52 | 3 | 0 | 0 | 0 | 8.846947803e-05 |
| training/2020-08 | increase | restart | below | v127 | original | supported | 0 | 13111 | 31 | 0.01496058715 | 0.04107394044 | 0.002334373015 | 0/0/31 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | below | v127 | budget_matched | supported | 0 | 13111 | 31 | 0.01496058715 | 0.04107394044 | 0.002334373015 | 0/0/31 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | below | v137 | original | supported | 0 | 13111 | 31 | 0.01534179447 | 0.03415926751 | 0.002342135524 | 1/15/30 | 16 | 0.0625 | 0.03225806452 | 0.05263157895 | 0.001146788991 |
| training/2020-08 | increase | restart | below | v137 | budget_matched | supported | 0 | 13111 | 31 | 0.01534179447 | 0.03415926751 | 0.002342135524 | 0/0/31 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | capped | v127 | original | supported | 0 | 20851 | 21 | 0.007048349823 | 0.009417914259 | 0.001003536676 | 0/0/21 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | capped | v127 | budget_matched | supported | 0 | 20851 | 21 | 0.007048349823 | 0.009417914259 | 0.001003536676 | 0/0/21 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | capped | v137 | original | supported | 0 | 20851 | 21 | 0.007007142876 | 0.01087921493 | 0.001003821903 | 0/19/21 | 19 | 0 | 0 | 0 | 0.0009121459434 |
| training/2020-08 | increase | restart | capped | v137 | budget_matched | supported | 0 | 20851 | 21 | 0.007007142876 | 0.01087921493 | 0.001003821903 | 0/3/21 | 3 | 0 | 0 | 0 | 0.0001440230437 |
| training/2020-09 | reduce | quiet | all | v127 | original | supported | 0 | 42111 | 65 | 0.009403453565 | 0.05914481877 | 0.001510083554 | 3/11/62 | 14 | 0.2142857143 | 0.04615384615 | 0.07594936709 | 0.0002616182277 |
| training/2020-09 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 42111 | 65 | 0.009403453565 | 0.05914481877 | 0.001510083554 | 5/17/60 | 22 | 0.2272727273 | 0.07692307692 | 0.1149425287 | 0.0004043190791 |
| training/2020-09 | reduce | quiet | all | v137 | original | supported | 0 | 42111 | 65 | 0.009587430512 | 0.04977861762 | 0.001512822494 | 5/34/60 | 39 | 0.1282051282 | 0.07692307692 | 0.09615384615 | 0.0008086381582 |
| training/2020-09 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 42111 | 65 | 0.009587430512 | 0.04977861762 | 0.001512822494 | 4/25/61 | 29 | 0.1379310345 | 0.06153846154 | 0.08510638298 | 0.000594586881 |
| training/2020-09 | reduce | quiet | below | v127 | original | supported | 0 | 11717 | 18 | 0.009731855045 | 0.03618536958 | 0.001519348484 | 1/5/17 | 6 | 0.1666666667 | 0.05555555556 | 0.08333333333 | 0.0004273869562 |
| training/2020-09 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 11717 | 18 | 0.009731855045 | 0.03618536958 | 0.001519348484 | 1/11/17 | 12 | 0.08333333333 | 0.05555555556 | 0.06666666667 | 0.0009402513035 |
| training/2020-09 | reduce | quiet | below | v137 | original | supported | 0 | 11717 | 18 | 0.01005289116 | 0.02458898134 | 0.001529030295 | 0/14/18 | 14 | 0 | 0 | 0 | 0.001196683477 |
| training/2020-09 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 11717 | 18 | 0.01005289116 | 0.02458898134 | 0.001529030295 | 0/13/18 | 13 | 0 | 0 | 0 | 0.001111206086 |
| training/2020-09 | reduce | quiet | capped | v127 | original | supported | 0 | 30394 | 47 | 0.009276853574 | 0.07437876471 | 0.001506511889 | 2/6/45 | 8 | 0.25 | 0.04255319149 | 0.07272727273 | 0.0001977131183 |
| training/2020-09 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 30394 | 47 | 0.009276853574 | 0.07437876471 | 0.001506511889 | 4/8/43 | 12 | 0.3333333333 | 0.08510638298 | 0.1355932203 | 0.000263617491 |
| training/2020-09 | reduce | quiet | capped | v137 | original | supported | 0 | 30394 | 47 | 0.009407993702 | 0.06606675124 | 0.001506574327 | 5/20/42 | 25 | 0.2 | 0.1063829787 | 0.1388888889 | 0.0006590437276 |
| training/2020-09 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 30394 | 47 | 0.009407993702 | 0.06606675124 | 0.001506574327 | 3/14/44 | 17 | 0.1764705882 | 0.06382978723 | 0.09375 | 0.0004613306093 |
| training/2020-09 | reduce | first | all | v127 | original | supported | 0 | 10548 | 13 | 0.008511277732 | 0.09883501639 | 0.001221624592 | 1/0/12 | 1 | 1 | 0.07692307692 | 0.1428571429 | 0 |
| training/2020-09 | reduce | first | all | v127 | budget_matched | supported | 0 | 10548 | 13 | 0.008511277732 | 0.09883501639 | 0.001221624592 | 1/0/12 | 1 | 1 | 0.07692307692 | 0.1428571429 | 0 |
| training/2020-09 | reduce | first | all | v137 | original | supported | 0 | 10548 | 13 | 0.008570561935 | 0.107884063 | 0.001225849226 | 0/0/13 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | all | v137 | budget_matched | supported | 0 | 10548 | 13 | 0.008570561935 | 0.107884063 | 0.001225849226 | 0/0/13 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | below | v127 | original | supported | 0 | 10218 | 12 | 0.007965950135 | 0.1069547976 | 0.001163247687 | 1/0/11 | 1 | 1 | 0.08333333333 | 0.1538461538 | 0 |
| training/2020-09 | reduce | first | below | v127 | budget_matched | supported | 0 | 10218 | 12 | 0.007965950135 | 0.1069547976 | 0.001163247687 | 1/0/11 | 1 | 1 | 0.08333333333 | 0.1538461538 | 0 |
| training/2020-09 | reduce | first | below | v137 | original | supported | 0 | 10218 | 12 | 0.008023690624 | 0.1184441538 | 0.001167591269 | 0/0/12 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | below | v137 | budget_matched | supported | 0 | 10218 | 12 | 0.008023690624 | 0.1184441538 | 0.001167591269 | 0/0/12 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | capped | v127 | original | supported | 0 | 330 | 1 | 0.02539660313 | 0.01086956522 | 0.003029185844 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | capped | v127 | budget_matched | supported | 0 | 330 | 1 | 0.02539660313 | 0.01086956522 | 0.003029185844 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | capped | v137 | original | supported | 0 | 330 | 1 | 0.02550368634 | 0.008849557522 | 0.003029727402 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | capped | v137 | budget_matched | supported | 0 | 330 | 1 | 0.02550368634 | 0.008849557522 | 0.003029727402 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | restart | all | v127 | original | supported | 0 | 31563 | 52 | 0.009701608706 | 0.06684932843 | 0.001606483298 | 2/11/50 | 13 | 0.1538461538 | 0.03846153846 | 0.06153846154 | 0.0003490844467 |
| training/2020-09 | reduce | restart | all | v127 | budget_matched | supported | 0 | 31563 | 52 | 0.009701608706 | 0.06684932843 | 0.001606483298 | 2/11/50 | 13 | 0.1538461538 | 0.03846153846 | 0.06153846154 | 0.0003490844467 |
| training/2020-09 | reduce | restart | all | v137 | original | supported | 0 | 31563 | 52 | 0.009927256566 | 0.05752448155 | 0.001608725736 | 5/34/47 | 39 | 0.1282051282 | 0.09615384615 | 0.1098901099 | 0.00107898829 |
| training/2020-09 | reduce | restart | all | v137 | budget_matched | supported | 0 | 31563 | 52 | 0.009927256566 | 0.05752448155 | 0.001608725736 | 3/14/49 | 17 | 0.1764705882 | 0.05769230769 | 0.08695652174 | 0.0004442892958 |
| training/2020-09 | reduce | restart | below | v127 | original | supported | 0 | 1499 | 6 | 0.0217692242 | 0.05173536897 | 0.003946725361 | 0/5/6 | 5 | 0 | 0 | 0 | 0.003348961822 |
| training/2020-09 | reduce | restart | below | v127 | budget_matched | supported | 0 | 1499 | 6 | 0.0217692242 | 0.05173536897 | 0.003946725361 | 0/5/6 | 5 | 0 | 0 | 0 | 0.003348961822 |
| training/2020-09 | reduce | restart | below | v137 | original | supported | 0 | 1499 | 6 | 0.02388502661 | 0.02637144491 | 0.003992795444 | 0/14/6 | 14 | 0 | 0 | 0 | 0.009377093101 |
| training/2020-09 | reduce | restart | below | v137 | budget_matched | supported | 0 | 1499 | 6 | 0.02388502661 | 0.02637144491 | 0.003992795444 | 0/0/6 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | restart | capped | v127 | original | supported | 0 | 30064 | 46 | 0.0090999138 | 0.07596174514 | 0.001489798132 | 2/6/44 | 8 | 0.25 | 0.04347826087 | 0.07407407407 | 0.000199880072 |
| training/2020-09 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 30064 | 46 | 0.0090999138 | 0.07596174514 | 0.001489798132 | 2/6/44 | 8 | 0.25 | 0.04347826087 | 0.07407407407 | 0.000199880072 |
| training/2020-09 | reduce | restart | capped | v137 | original | supported | 0 | 30064 | 46 | 0.009231317991 | 0.06751642144 | 0.00148985531 | 5/20/41 | 25 | 0.2 | 0.1086956522 | 0.1408450704 | 0.0006662669065 |
| training/2020-09 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 30064 | 46 | 0.009231317991 | 0.06751642144 | 0.00148985531 | 3/10/43 | 13 | 0.2307692308 | 0.0652173913 | 0.1016949153 | 0.0003331334533 |
| training/2020-09 | increase | quiet | all | v127 | original | supported | 0 | 41959 | 70 | 0.0102212871 | 0.05365827941 | 0.001633598306 | 2/6/68 | 8 | 0.25 | 0.02857142857 | 0.09803921569 | 0.0001432356943 |
| training/2020-09 | increase | quiet | all | v127 | budget_matched | supported | 0 | 41959 | 70 | 0.0102212871 | 0.05365827941 | 0.001633598306 | 1/10/69 | 11 | 0.09090909091 | 0.01428571429 | 0.04385964912 | 0.0002387261572 |
| training/2020-09 | increase | quiet | all | v137 | original | supported | 0 | 41959 | 70 | 0.01026823091 | 0.05792668529 | 0.001631127747 | 4/35/66 | 39 | 0.1025641026 | 0.05714285714 | 0.08849557522 | 0.0008355415503 |
| training/2020-09 | increase | quiet | all | v137 | budget_matched | supported | 0 | 41959 | 70 | 0.01026823091 | 0.05792668529 | 0.001631127747 | 2/7/68 | 9 | 0.2222222222 | 0.02857142857 | 0.09433962264 | 0.0001671083101 |
| training/2020-09 | increase | quiet | below | v127 | original | supported | 0 | 11605 | 20 | 0.00879790905 | 0.1184539793 | 0.001649006437 | 1/4/19 | 5 | 0.2 | 0.05 | 0.125 | 0.0003452740613 |
| training/2020-09 | increase | quiet | below | v127 | budget_matched | supported | 0 | 11605 | 20 | 0.00879790905 | 0.1184539793 | 0.001649006437 | 1/7/19 | 8 | 0.125 | 0.05 | 0.09615384615 | 0.0006042296073 |
| training/2020-09 | increase | quiet | below | v137 | original | supported | 0 | 11605 | 20 | 0.008789155695 | 0.1234082563 | 0.001641624363 | 1/10/19 | 11 | 0.09090909091 | 0.05 | 0.078125 | 0.0008631851532 |
| training/2020-09 | increase | quiet | below | v137 | budget_matched | supported | 0 | 11605 | 20 | 0.008789155695 | 0.1234082563 | 0.001641624363 | 1/4/19 | 5 | 0.2 | 0.05 | 0.125 | 0.0003452740613 |
| training/2020-09 | increase | quiet | capped | v127 | original | supported | 0 | 30354 | 50 | 0.01076547575 | 0.03484235134 | 0.00162770744 | 1/2/49 | 3 | 0.3333333333 | 0.02 | 0.08064516129 | 6.599788807e-05 |
| training/2020-09 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 30354 | 50 | 0.01076547575 | 0.03484235134 | 0.00162770744 | 1/2/49 | 3 | 0.3333333333 | 0.02 | 0.08064516129 | 6.599788807e-05 |
| training/2020-09 | increase | quiet | capped | v137 | original | supported | 0 | 30354 | 50 | 0.01083371381 | 0.0451453047 | 0.001627114661 | 3/25/47 | 28 | 0.1071428571 | 0.06 | 0.09259259259 | 0.0008249736008 |
| training/2020-09 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 30354 | 50 | 0.01083371381 | 0.0451453047 | 0.001627114661 | 1/3/49 | 4 | 0.25 | 0.02 | 0.07575757576 | 9.89968321e-05 |
| training/2020-09 | increase | first | all | v127 | original | supported | 0 | 1140 | 11 | 0.04140489036 | 0.189347234 | 0.008993022677 | 1/4/10 | 5 | 0.2 | 0.09090909091 | 0.1612903226 | 0.00354295837 |
| training/2020-09 | increase | first | all | v127 | budget_matched | supported | 0 | 1140 | 11 | 0.04140489036 | 0.189347234 | 0.008993022677 | 1/4/10 | 5 | 0.2 | 0.09090909091 | 0.1612903226 | 0.00354295837 |
| training/2020-09 | increase | first | all | v137 | original | supported | 0 | 1140 | 11 | 0.04018073538 | 0.2032225848 | 0.008887351829 | 1/3/10 | 4 | 0.25 | 0.09090909091 | 0.1851851852 | 0.002657218778 |
| training/2020-09 | increase | first | all | v137 | budget_matched | supported | 0 | 1140 | 11 | 0.04018073538 | 0.2032225848 | 0.008887351829 | 1/4/10 | 5 | 0.2 | 0.09090909091 | 0.1612903226 | 0.00354295837 |
| training/2020-09 | increase | first | below | v127 | original | supported | 0 | 1140 | 11 | 0.04140489036 | 0.189347234 | 0.008993022677 | 1/4/10 | 5 | 0.2 | 0.09090909091 | 0.1612903226 | 0.00354295837 |
| training/2020-09 | increase | first | below | v127 | budget_matched | supported | 0 | 1140 | 11 | 0.04140489036 | 0.189347234 | 0.008993022677 | 1/4/10 | 5 | 0.2 | 0.09090909091 | 0.1612903226 | 0.00354295837 |
| training/2020-09 | increase | first | below | v137 | original | supported | 0 | 1140 | 11 | 0.04018073538 | 0.2032225848 | 0.008887351829 | 1/3/10 | 4 | 0.25 | 0.09090909091 | 0.1851851852 | 0.002657218778 |
| training/2020-09 | increase | first | below | v137 | budget_matched | supported | 0 | 1140 | 11 | 0.04018073538 | 0.2032225848 | 0.008887351829 | 1/4/10 | 5 | 0.2 | 0.09090909091 | 0.1612903226 | 0.00354295837 |
| training/2020-09 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-09 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-09 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-09 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-09 | increase | restart | all | v127 | original | supported | 0 | 40819 | 59 | 0.009350386106 | 0.03175958042 | 0.001428063046 | 1/2/58 | 3 | 0.3333333333 | 0.01694915254 | 0.07042253521 | 4.906771344e-05 |
| training/2020-09 | increase | restart | all | v127 | budget_matched | supported | 0 | 40819 | 59 | 0.009350386106 | 0.03175958042 | 0.001428063046 | 1/2/58 | 3 | 0.3333333333 | 0.01694915254 | 0.07042253521 | 4.906771344e-05 |
| training/2020-09 | increase | restart | all | v137 | original | supported | 0 | 40819 | 59 | 0.00943282938 | 0.04150730303 | 0.001428474682 | 3/32/56 | 35 | 0.08571428571 | 0.05084745763 | 0.07537688442 | 0.0007850834151 |
| training/2020-09 | increase | restart | all | v137 | budget_matched | supported | 0 | 40819 | 59 | 0.00943282938 | 0.04150730303 | 0.001428474682 | 1/2/58 | 3 | 0.3333333333 | 0.01694915254 | 0.07042253521 | 4.906771344e-05 |
| training/2020-09 | increase | restart | below | v127 | original | supported | 0 | 10465 | 9 | 0.005245882419 | 0.038470383 | 0.0008489893784 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | restart | below | v127 | budget_matched | supported | 0 | 10465 | 9 | 0.005245882419 | 0.038470383 | 0.0008489893784 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | restart | below | v137 | original | supported | 0 | 10465 | 9 | 0.005369528286 | 0.0289258856 | 0.0008523143478 | 0/7/9 | 7 | 0 | 0 | 0 | 0.0006694720735 |
| training/2020-09 | increase | restart | below | v137 | budget_matched | supported | 0 | 10465 | 9 | 0.005369528286 | 0.0289258856 | 0.0008523143478 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | restart | capped | v127 | original | supported | 0 | 30354 | 50 | 0.01076547575 | 0.03484235134 | 0.00162770744 | 1/2/49 | 3 | 0.3333333333 | 0.02 | 0.08064516129 | 6.599788807e-05 |
| training/2020-09 | increase | restart | capped | v127 | budget_matched | supported | 0 | 30354 | 50 | 0.01076547575 | 0.03484235134 | 0.00162770744 | 1/2/49 | 3 | 0.3333333333 | 0.02 | 0.08064516129 | 6.599788807e-05 |
| training/2020-09 | increase | restart | capped | v137 | original | supported | 0 | 30354 | 50 | 0.01083371381 | 0.0451453047 | 0.001627114661 | 3/25/47 | 28 | 0.1071428571 | 0.06 | 0.09259259259 | 0.0008249736008 |
| training/2020-09 | increase | restart | capped | v137 | budget_matched | supported | 0 | 30354 | 50 | 0.01083371381 | 0.0451453047 | 0.001627114661 | 1/3/49 | 4 | 0.25 | 0.02 | 0.07575757576 | 9.89968321e-05 |
| training/2020-10 | reduce | quiet | all | v127 | original | supported | 0 | 44016 | 34 | 0.005110380361 | 0.02945835132 | 0.0007645263675 | 1/4/33 | 5 | 0.2 | 0.02941176471 | 0.05128205128 | 9.094629621e-05 |
| training/2020-10 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 44016 | 34 | 0.005110380361 | 0.02945835132 | 0.0007645263675 | 1/4/33 | 5 | 0.2 | 0.02941176471 | 0.05128205128 | 9.094629621e-05 |
| training/2020-10 | reduce | quiet | all | v137 | original | supported | 0 | 44016 | 34 | 0.005146736627 | 0.01793040632 | 0.0007677904 | 1/23/33 | 24 | 0.04166666667 | 0.02941176471 | 0.03448275862 | 0.0005229412032 |
| training/2020-10 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 44016 | 34 | 0.005146736627 | 0.01793040632 | 0.0007677904 | 0/3/34 | 3 | 0 | 0 | 0 | 6.820972216e-05 |
| training/2020-10 | reduce | quiet | below | v127 | original | supported | 0 | 26913 | 12 | 0.002914928057 | 0.04901345413 | 0.0004382573791 | 1/3/11 | 4 | 0.25 | 0.08333333333 | 0.125 | 0.0001115200178 |
| training/2020-10 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 26913 | 12 | 0.002914928057 | 0.04901345413 | 0.0004382573791 | 1/7/11 | 8 | 0.125 | 0.08333333333 | 0.1 | 0.000260213375 |
| training/2020-10 | reduce | quiet | below | v137 | original | supported | 0 | 26913 | 12 | 0.002936002575 | 0.04795885732 | 0.0004415858758 | 1/6/11 | 7 | 0.1428571429 | 0.08333333333 | 0.1052631579 | 0.0002230400357 |
| training/2020-10 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 26913 | 12 | 0.002936002575 | 0.04795885732 | 0.0004415858758 | 0/1/12 | 1 | 0 | 0 | 0 | 3.717333928e-05 |
| training/2020-10 | reduce | quiet | capped | v127 | original | supported | 0 | 17103 | 22 | 0.008565108061 | 0.02494502176 | 0.001277937891 | 0/1/22 | 1 | 0 | 0 | 0 | 5.85445817e-05 |
| training/2020-10 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 17103 | 22 | 0.008565108061 | 0.02494502176 | 0.001277937891 | 0/2/22 | 2 | 0 | 0 | 0 | 0.0001170891634 |
| training/2020-10 | reduce | quiet | capped | v137 | original | supported | 0 | 17103 | 22 | 0.008625511434 | 0.015698863 | 0.001281100484 | 0/17/22 | 17 | 0 | 0 | 0 | 0.0009952578889 |
| training/2020-10 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 17103 | 22 | 0.008625511434 | 0.015698863 | 0.001281100484 | 0/2/22 | 2 | 0 | 0 | 0 | 0.0001170891634 |
| training/2020-10 | reduce | first | all | v127 | original | supported | 0 | 26408 | 13 | 0.003391760773 | 0.05204236344 | 0.0004850621071 | 1/3/12 | 4 | 0.25 | 0.07692307692 | 0.1176470588 | 0.0001136578898 |
| training/2020-10 | reduce | first | all | v127 | budget_matched | supported | 0 | 26408 | 13 | 0.003391760773 | 0.05204236344 | 0.0004850621071 | 1/3/12 | 4 | 0.25 | 0.07692307692 | 0.1176470588 | 0.0001136578898 |
| training/2020-10 | reduce | first | all | v137 | original | supported | 0 | 26408 | 13 | 0.003446387902 | 0.06258778668 | 0.0004880095829 | 1/10/12 | 11 | 0.09090909091 | 0.07692307692 | 0.08333333333 | 0.0003788596325 |
| training/2020-10 | reduce | first | all | v137 | budget_matched | supported | 0 | 26408 | 13 | 0.003446387902 | 0.06258778668 | 0.0004880095829 | 0/1/13 | 1 | 0 | 0 | 0 | 3.788596325e-05 |
| training/2020-10 | reduce | first | below | v127 | original | supported | 0 | 18845 | 9 | 0.003164711101 | 0.06748063376 | 0.0004682497355 | 1/2/8 | 3 | 0.3333333333 | 0.1111111111 | 0.1666666667 | 0.000106179656 |
| training/2020-10 | reduce | first | below | v127 | budget_matched | supported | 0 | 18845 | 9 | 0.003164711101 | 0.06748063376 | 0.0004682497355 | 1/2/8 | 3 | 0.3333333333 | 0.1111111111 | 0.1666666667 | 0.000106179656 |
| training/2020-10 | reduce | first | below | v137 | original | supported | 0 | 18845 | 9 | 0.003216697926 | 0.1298949743 | 0.0004726379442 | 1/4/8 | 5 | 0.2 | 0.1111111111 | 0.1428571429 | 0.000212359312 |
| training/2020-10 | reduce | first | below | v137 | budget_matched | supported | 0 | 18845 | 9 | 0.003216697926 | 0.1298949743 | 0.0004726379442 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | first | capped | v127 | original | supported | 0 | 7563 | 4 | 0.003957508633 | 0.02539876684 | 0.0005269541001 | 0/1/4 | 1 | 0 | 0 | 0 | 0.0001322926313 |
| training/2020-10 | reduce | first | capped | v127 | budget_matched | supported | 0 | 7563 | 4 | 0.003957508633 | 0.02539876684 | 0.0005269541001 | 0/1/4 | 1 | 0 | 0 | 0 | 0.0001322926313 |
| training/2020-10 | reduce | first | capped | v137 | original | supported | 0 | 7563 | 4 | 0.004018714702 | 0.03483762249 | 0.0005263116496 | 0/6/4 | 6 | 0 | 0 | 0 | 0.0007937557878 |
| training/2020-10 | reduce | first | capped | v137 | budget_matched | supported | 0 | 7563 | 4 | 0.004018714702 | 0.03483762249 | 0.0005263116496 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | restart | all | v127 | original | supported | 0 | 17608 | 21 | 0.007687919324 | 0.02205811569 | 0.001183659272 | 0/1/21 | 1 | 0 | 0 | 0 | 5.686018082e-05 |
| training/2020-10 | reduce | restart | all | v127 | budget_matched | supported | 0 | 17608 | 21 | 0.007687919324 | 0.02205811569 | 0.001183659272 | 0/1/21 | 1 | 0 | 0 | 0 | 5.686018082e-05 |
| training/2020-10 | reduce | restart | all | v137 | original | supported | 0 | 17608 | 21 | 0.007696873447 | 0.01599637598 | 0.001187398068 | 0/13/21 | 13 | 0 | 0 | 0 | 0.0007391823506 |
| training/2020-10 | reduce | restart | all | v137 | budget_matched | supported | 0 | 17608 | 21 | 0.007696873447 | 0.01599637598 | 0.001187398068 | 0/0/21 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | restart | below | v127 | original | supported | 0 | 8068 | 3 | 0.002331492078 | 0.04899380805 | 0.0003682021044 | 0/1/3 | 1 | 0 | 0 | 0 | 0.0001239925604 |
| training/2020-10 | reduce | restart | below | v127 | budget_matched | supported | 0 | 8068 | 3 | 0.002331492078 | 0.04899380805 | 0.0003682021044 | 0/1/3 | 1 | 0 | 0 | 0 | 0.0001239925604 |
| training/2020-10 | reduce | restart | below | v137 | original | supported | 0 | 8068 | 3 | 0.002280362528 | 0.03114068985 | 0.0003690553566 | 0/2/3 | 2 | 0 | 0 | 0 | 0.0002479851209 |
| training/2020-10 | reduce | restart | below | v137 | budget_matched | supported | 0 | 8068 | 3 | 0.002280362528 | 0.03114068985 | 0.0003690553566 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | restart | capped | v127 | original | supported | 0 | 9540 | 18 | 0.0122178622 | 0.02719297727 | 0.00187329328 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 9540 | 18 | 0.0122178622 | 0.02719297727 | 0.00187329328 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | restart | capped | v137 | original | supported | 0 | 9540 | 18 | 0.01227762922 | 0.01584852076 | 0.001879472386 | 0/11/18 | 11 | 0 | 0 | 0 | 0.001155219492 |
| training/2020-10 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 9540 | 18 | 0.01227762922 | 0.01584852076 | 0.001879472386 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | quiet | all | v127 | original | supported | 0 | 43781 | 39 | 0.005624644568 | 0.08777044288 | 0.0008611917825 | 2/0/37 | 2 | 1 | 0.05128205128 | 0.2127659574 | 0 |
| training/2020-10 | increase | quiet | all | v127 | budget_matched | supported | 0 | 43781 | 39 | 0.005624644568 | 0.08777044288 | 0.0008611917825 | 2/1/37 | 3 | 0.6666666667 | 0.05128205128 | 0.1960784314 | 2.286132321e-05 |
| training/2020-10 | increase | quiet | all | v137 | original | supported | 0 | 43781 | 39 | 0.005644560175 | 0.06790693614 | 0.0008707615092 | 2/8/37 | 10 | 0.2 | 0.05128205128 | 0.1265822785 | 0.0001828905857 |
| training/2020-10 | increase | quiet | all | v137 | budget_matched | supported | 0 | 43781 | 39 | 0.005644560175 | 0.06790693614 | 0.0008707615092 | 1/1/38 | 2 | 0.5 | 0.02564102564 | 0.1063829787 | 2.286132321e-05 |
| training/2020-10 | increase | quiet | below | v127 | original | supported | 0 | 26805 | 21 | 0.004786463375 | 0.09683000885 | 0.0007545344825 | 1/0/20 | 1 | 1 | 0.04761904762 | 0.2 | 0 |
| training/2020-10 | increase | quiet | below | v127 | budget_matched | supported | 0 | 26805 | 21 | 0.004786463375 | 0.09683000885 | 0.0007545344825 | 1/1/20 | 2 | 0.5 | 0.04761904762 | 0.1724137931 | 3.733572282e-05 |
| training/2020-10 | increase | quiet | below | v137 | original | supported | 0 | 26805 | 21 | 0.004836904473 | 0.09562738237 | 0.0007580690168 | 1/5/20 | 6 | 0.1666666667 | 0.04761904762 | 0.1111111111 | 0.0001866786141 |
| training/2020-10 | increase | quiet | below | v137 | budget_matched | supported | 0 | 26805 | 21 | 0.004836904473 | 0.09562738237 | 0.0007580690168 | 1/1/20 | 2 | 0.5 | 0.04761904762 | 0.1724137931 | 3.733572282e-05 |
| training/2020-10 | increase | quiet | capped | v127 | original | supported | 0 | 16976 | 18 | 0.006948127537 | 0.07442257932 | 0.001029603006 | 1/0/17 | 1 | 1 | 0.05555555556 | 0.2272727273 | 0 |
| training/2020-10 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 16976 | 18 | 0.006948127537 | 0.07442257932 | 0.001029603006 | 1/0/17 | 1 | 1 | 0.05555555556 | 0.2272727273 | 0 |
| training/2020-10 | increase | quiet | capped | v137 | original | supported | 0 | 16976 | 18 | 0.006919843582 | 0.0782985327 | 0.001048702264 | 1/3/17 | 4 | 0.25 | 0.05555555556 | 0.1470588235 | 0.0001769076542 |
| training/2020-10 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 16976 | 18 | 0.006919843582 | 0.0782985327 | 0.001048702264 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | first | all | v127 | original | supported | 0 | 7195 | 10 | 0.006738418233 | 0.1873638765 | 0.001294904677 | 1/0/9 | 1 | 1 | 0.1 | 0.3571428571 | 0 |
| training/2020-10 | increase | first | all | v127 | budget_matched | supported | 0 | 7195 | 10 | 0.006738418233 | 0.1873638765 | 0.001294904677 | 1/0/9 | 1 | 1 | 0.1 | 0.3571428571 | 0 |
| training/2020-10 | increase | first | all | v137 | original | supported | 0 | 7195 | 10 | 0.007004038551 | 0.1864336131 | 0.001302766029 | 1/1/9 | 2 | 0.5 | 0.1 | 0.2777777778 | 0.0001391788448 |
| training/2020-10 | increase | first | all | v137 | budget_matched | supported | 0 | 7195 | 10 | 0.007004038551 | 0.1864336131 | 0.001302766029 | 1/1/9 | 2 | 0.5 | 0.1 | 0.2777777778 | 0.0001391788448 |
| training/2020-10 | increase | first | below | v127 | original | supported | 0 | 7195 | 10 | 0.006738418233 | 0.1873638765 | 0.001294904677 | 1/0/9 | 1 | 1 | 0.1 | 0.3571428571 | 0 |
| training/2020-10 | increase | first | below | v127 | budget_matched | supported | 0 | 7195 | 10 | 0.006738418233 | 0.1873638765 | 0.001294904677 | 1/0/9 | 1 | 1 | 0.1 | 0.3571428571 | 0 |
| training/2020-10 | increase | first | below | v137 | original | supported | 0 | 7195 | 10 | 0.007004038551 | 0.1864336131 | 0.001302766029 | 1/1/9 | 2 | 0.5 | 0.1 | 0.2777777778 | 0.0001391788448 |
| training/2020-10 | increase | first | below | v137 | budget_matched | supported | 0 | 7195 | 10 | 0.007004038551 | 0.1864336131 | 0.001302766029 | 1/1/9 | 2 | 0.5 | 0.1 | 0.2777777778 | 0.0001391788448 |
| training/2020-10 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-10 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-10 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-10 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-10 | increase | restart | all | v127 | original | supported | 0 | 36586 | 29 | 0.005405609923 | 0.05884516378 | 0.0007758978373 | 1/0/28 | 1 | 1 | 0.03448275862 | 0.1515151515 | 0 |
| training/2020-10 | increase | restart | all | v127 | budget_matched | supported | 0 | 36586 | 29 | 0.005405609923 | 0.05884516378 | 0.0007758978373 | 1/0/28 | 1 | 1 | 0.03448275862 | 0.1515151515 | 0 |
| training/2020-10 | increase | restart | all | v137 | original | supported | 0 | 36586 | 29 | 0.00537720526 | 0.0523940153 | 0.0007858035329 | 1/7/28 | 8 | 0.125 | 0.03448275862 | 0.08196721311 | 0.0001914817956 |
| training/2020-10 | increase | restart | all | v137 | budget_matched | supported | 0 | 36586 | 29 | 0.00537720526 | 0.0523940153 | 0.0007858035329 | 0/0/29 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | restart | below | v127 | original | supported | 0 | 19610 | 11 | 0.004070282079 | 0.09716068141 | 0.0005562701504 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | restart | below | v127 | budget_matched | supported | 0 | 19610 | 11 | 0.004070282079 | 0.09716068141 | 0.0005562701504 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | restart | below | v137 | original | supported | 0 | 19610 | 11 | 0.004041772922 | 0.01575492286 | 0.0005582171553 | 0/4/11 | 4 | 0 | 0 | 0 | 0.0002040920455 |
| training/2020-10 | increase | restart | below | v137 | budget_matched | supported | 0 | 19610 | 11 | 0.004041772922 | 0.01575492286 | 0.0005582171553 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | restart | capped | v127 | original | supported | 0 | 16976 | 18 | 0.006948127537 | 0.07442257932 | 0.001029603006 | 1/0/17 | 1 | 1 | 0.05555555556 | 0.2272727273 | 0 |
| training/2020-10 | increase | restart | capped | v127 | budget_matched | supported | 0 | 16976 | 18 | 0.006948127537 | 0.07442257932 | 0.001029603006 | 1/0/17 | 1 | 1 | 0.05555555556 | 0.2272727273 | 0 |
| training/2020-10 | increase | restart | capped | v137 | original | supported | 0 | 16976 | 18 | 0.006919843582 | 0.0782985327 | 0.001048702264 | 1/3/17 | 4 | 0.25 | 0.05555555556 | 0.1470588235 | 0.0001769076542 |
| training/2020-10 | increase | restart | capped | v137 | budget_matched | supported | 0 | 16976 | 18 | 0.006919843582 | 0.0782985327 | 0.001048702264 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | quiet | all | v127 | original | supported | 0 | 31161 | 82 | 0.01536403643 | 0.0448245266 | 0.002582508819 | 7/13/75 | 20 | 0.35 | 0.08536585366 | 0.137254902 | 0.0004182888767 |
| training/2020-11 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 31161 | 82 | 0.01536403643 | 0.0448245266 | 0.002582508819 | 3/17/79 | 20 | 0.15 | 0.03658536585 | 0.05882352941 | 0.0005469931465 |
| training/2020-11 | reduce | quiet | all | v137 | original | supported | 0 | 31161 | 82 | 0.01547721267 | 0.05124726004 | 0.002584563722 | 8/37/74 | 45 | 0.1777777778 | 0.09756097561 | 0.125984252 | 0.001190514495 |
| training/2020-11 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 31161 | 82 | 0.01547721267 | 0.05124726004 | 0.002584563722 | 2/15/80 | 17 | 0.1176470588 | 0.0243902439 | 0.0404040404 | 0.0004826410116 |
| training/2020-11 | reduce | quiet | below | v127 | original | supported | 0 | 18421 | 35 | 0.01129956807 | 0.05744857599 | 0.001864289181 | 4/7/31 | 11 | 0.3636363636 | 0.1142857143 | 0.1739130435 | 0.0003807244643 |
| training/2020-11 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 18421 | 35 | 0.01129956807 | 0.05744857599 | 0.001864289181 | 3/15/32 | 18 | 0.1666666667 | 0.08571428571 | 0.1132075472 | 0.0008158381377 |
| training/2020-11 | reduce | quiet | below | v137 | original | supported | 0 | 18421 | 35 | 0.01141610924 | 0.1015175046 | 0.001865304837 | 4/14/31 | 18 | 0.2222222222 | 0.1142857143 | 0.1509433962 | 0.0007614489285 |
| training/2020-11 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 18421 | 35 | 0.01141610924 | 0.1015175046 | 0.001865304837 | 3/1/32 | 4 | 0.75 | 0.08571428571 | 0.1538461538 | 5.438920918e-05 |
| training/2020-11 | reduce | quiet | capped | v127 | original | supported | 0 | 12740 | 47 | 0.02124092588 | 0.04392100799 | 0.003620995786 | 3/6/44 | 9 | 0.3333333333 | 0.06382978723 | 0.1071428571 | 0.000472701489 |
| training/2020-11 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 12740 | 47 | 0.02124092588 | 0.04392100799 | 0.003620995786 | 3/11/44 | 14 | 0.2142857143 | 0.06382978723 | 0.09836065574 | 0.0008666193965 |
| training/2020-11 | reduce | quiet | capped | v137 | original | supported | 0 | 12740 | 47 | 0.02134923672 | 0.04641162717 | 0.003624553355 | 4/23/43 | 27 | 0.1481481481 | 0.08510638298 | 0.1081081081 | 0.001812022375 |
| training/2020-11 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 12740 | 47 | 0.02134923672 | 0.04641162717 | 0.003624553355 | 1/9/46 | 10 | 0.1 | 0.02127659574 | 0.0350877193 | 0.0007090522335 |
| training/2020-11 | reduce | first | all | v127 | original | supported | 0 | 12976 | 24 | 0.01104248704 | 0.1134939544 | 0.00181262242 | 4/6/20 | 10 | 0.4 | 0.1666666667 | 0.2352941176 | 0.0004632489191 |
| training/2020-11 | reduce | first | all | v127 | budget_matched | supported | 0 | 12976 | 24 | 0.01104248704 | 0.1134939544 | 0.00181262242 | 4/6/20 | 10 | 0.4 | 0.1666666667 | 0.2352941176 | 0.0004632489191 |
| training/2020-11 | reduce | first | all | v137 | original | supported | 0 | 12976 | 24 | 0.01149789282 | 0.1217848875 | 0.001827220475 | 2/12/22 | 14 | 0.1428571429 | 0.08333333333 | 0.1052631579 | 0.0009264978382 |
| training/2020-11 | reduce | first | all | v137 | budget_matched | supported | 0 | 12976 | 24 | 0.01149789282 | 0.1217848875 | 0.001827220475 | 1/0/23 | 1 | 1 | 0.04166666667 | 0.08 | 0 |
| training/2020-11 | reduce | first | below | v127 | original | supported | 0 | 12426 | 21 | 0.01011364972 | 0.1224563612 | 0.001653031122 | 4/6/17 | 10 | 0.4 | 0.1904761905 | 0.2580645161 | 0.0004836759371 |
| training/2020-11 | reduce | first | below | v127 | budget_matched | supported | 0 | 12426 | 21 | 0.01011364972 | 0.1224563612 | 0.001653031122 | 4/6/17 | 10 | 0.4 | 0.1904761905 | 0.2580645161 | 0.0004836759371 |
| training/2020-11 | reduce | first | below | v137 | original | supported | 0 | 12426 | 21 | 0.01051708948 | 0.1336678883 | 0.001667887836 | 2/11/19 | 13 | 0.1538461538 | 0.09523809524 | 0.1176470588 | 0.0008867392181 |
| training/2020-11 | reduce | first | below | v137 | budget_matched | supported | 0 | 12426 | 21 | 0.01051708948 | 0.1336678883 | 0.001667887836 | 1/0/20 | 1 | 1 | 0.04761904762 | 0.09090909091 | 0 |
| training/2020-11 | reduce | first | capped | v127 | original | supported | 0 | 550 | 3 | 0.03202745511 | 0.1017998753 | 0.005418225091 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | first | capped | v127 | budget_matched | supported | 0 | 550 | 3 | 0.03202745511 | 0.1017998753 | 0.005418225091 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | first | capped | v137 | original | supported | 0 | 550 | 3 | 0.0336569153 | 0.07657754011 | 0.005426979334 | 0/1/3 | 1 | 0 | 0 | 0 | 0.001828153565 |
| training/2020-11 | reduce | first | capped | v137 | budget_matched | supported | 0 | 550 | 3 | 0.0336569153 | 0.07657754011 | 0.005426979334 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | restart | all | v127 | original | supported | 0 | 18185 | 58 | 0.01844770015 | 0.04045183292 | 0.003131865207 | 3/7/55 | 10 | 0.3 | 0.05172413793 | 0.08823529412 | 0.0003861642853 |
| training/2020-11 | reduce | restart | all | v127 | budget_matched | supported | 0 | 18185 | 58 | 0.01844770015 | 0.04045183292 | 0.003131865207 | 3/8/55 | 11 | 0.2727272727 | 0.05172413793 | 0.08695652174 | 0.0004413306118 |
| training/2020-11 | reduce | restart | all | v137 | original | supported | 0 | 18185 | 58 | 0.01831667676 | 0.05427650959 | 0.00312496988 | 6/25/52 | 31 | 0.1935483871 | 0.1034482759 | 0.1348314607 | 0.001379158162 |
| training/2020-11 | reduce | restart | all | v137 | budget_matched | supported | 0 | 18185 | 58 | 0.01831667676 | 0.05427650959 | 0.00312496988 | 1/6/57 | 7 | 0.1428571429 | 0.01724137931 | 0.03076923077 | 0.0003309979588 |
| training/2020-11 | reduce | restart | below | v127 | original | supported | 0 | 5995 | 14 | 0.01375765337 | 0.03469202935 | 0.002302169521 | 0/1/14 | 1 | 0 | 0 | 0 | 0.000167196121 |
| training/2020-11 | reduce | restart | below | v127 | budget_matched | supported | 0 | 5995 | 14 | 0.01375765337 | 0.03469202935 | 0.002302169521 | 0/1/14 | 1 | 0 | 0 | 0 | 0.000167196121 |
| training/2020-11 | reduce | restart | below | v137 | original | supported | 0 | 5995 | 14 | 0.01327953202 | 0.1280307441 | 0.002274496439 | 2/3/12 | 5 | 0.4 | 0.1428571429 | 0.2105263158 | 0.0005015883631 |
| training/2020-11 | reduce | restart | below | v137 | budget_matched | supported | 0 | 5995 | 14 | 0.01327953202 | 0.1280307441 | 0.002274496439 | 0/0/14 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | restart | capped | v127 | original | supported | 0 | 12190 | 44 | 0.020754249 | 0.04543577049 | 0.003539906687 | 3/6/41 | 9 | 0.3333333333 | 0.06818181818 | 0.1132075472 | 0.0004939897909 |
| training/2020-11 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 12190 | 44 | 0.020754249 | 0.04543577049 | 0.003539906687 | 3/7/41 | 10 | 0.3 | 0.06818181818 | 0.1111111111 | 0.0005763214227 |
| training/2020-11 | reduce | restart | capped | v137 | original | supported | 0 | 12190 | 44 | 0.02079392718 | 0.04842207118 | 0.003543229788 | 4/22/40 | 26 | 0.1538461538 | 0.09090909091 | 0.1142857143 | 0.0018112959 |
| training/2020-11 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 12190 | 44 | 0.02079392718 | 0.04842207118 | 0.003543229788 | 1/6/43 | 7 | 0.1428571429 | 0.02272727273 | 0.03921568627 | 0.0004939897909 |
| training/2020-11 | increase | quiet | all | v127 | original | supported | 0 | 30705 | 106 | 0.0184540905 | 0.08544448641 | 0.003326678264 | 10/22/96 | 32 | 0.3125 | 0.09433962264 | 0.2136752137 | 0.0007189777444 |
| training/2020-11 | increase | quiet | all | v127 | budget_matched | supported | 0 | 30705 | 106 | 0.0184540905 | 0.08544448641 | 0.003326678264 | 9/20/97 | 29 | 0.3103448276 | 0.08490566038 | 0.2027027027 | 0.0006536161312 |
| training/2020-11 | increase | quiet | all | v137 | original | supported | 0 | 30705 | 106 | 0.0183066071 | 0.120997462 | 0.003301452326 | 17/78/89 | 95 | 0.1789473684 | 0.1603773585 | 0.1748971193 | 0.002549102912 |
| training/2020-11 | increase | quiet | all | v137 | budget_matched | supported | 0 | 30705 | 106 | 0.0183066071 | 0.120997462 | 0.003301452326 | 11/15/95 | 26 | 0.4230769231 | 0.1037735849 | 0.2619047619 | 0.0004902120984 |
| training/2020-11 | increase | quiet | below | v127 | original | supported | 0 | 18132 | 55 | 0.01537896118 | 0.1197088403 | 0.002867000822 | 9/21/46 | 30 | 0.3 | 0.1636363636 | 0.2571428571 | 0.001161697184 |
| training/2020-11 | increase | quiet | below | v127 | budget_matched | supported | 0 | 18132 | 55 | 0.01537896118 | 0.1197088403 | 0.002867000822 | 8/17/47 | 25 | 0.32 | 0.1454545455 | 0.2580645161 | 0.0009404215301 |
| training/2020-11 | increase | quiet | below | v137 | original | supported | 0 | 18132 | 55 | 0.01524349124 | 0.1734861544 | 0.002833414156 | 13/36/42 | 49 | 0.2653061224 | 0.2363636364 | 0.2589641434 | 0.001991480887 |
| training/2020-11 | increase | quiet | below | v137 | budget_matched | supported | 0 | 18132 | 55 | 0.01524349124 | 0.1734861544 | 0.002833414156 | 8/12/47 | 20 | 0.4 | 0.1454545455 | 0.2962962963 | 0.0006638269624 |
| training/2020-11 | increase | quiet | capped | v127 | original | supported | 0 | 12573 | 51 | 0.02288885109 | 0.06021515234 | 0.003989596532 | 1/1/50 | 2 | 0.5 | 0.01960784314 | 0.08474576271 | 7.985944737e-05 |
| training/2020-11 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 12573 | 51 | 0.02288885109 | 0.06021515234 | 0.003989596532 | 1/1/50 | 2 | 0.5 | 0.01960784314 | 0.08474576271 | 7.985944737e-05 |
| training/2020-11 | increase | quiet | capped | v137 | original | supported | 0 | 12573 | 51 | 0.02272404264 | 0.06924262669 | 0.003976427917 | 4/42/47 | 46 | 0.08695652174 | 0.07843137255 | 0.08510638298 | 0.00335409679 |
| training/2020-11 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 12573 | 51 | 0.02272404264 | 0.06924262669 | 0.003976427917 | 2/12/49 | 14 | 0.1428571429 | 0.03921568627 | 0.09345794393 | 0.0009583133685 |
| training/2020-11 | increase | first | all | v127 | original | supported | 0 | 5535 | 23 | 0.01779071231 | 0.1702381296 | 0.00381471519 | 5/13/18 | 18 | 0.2777777778 | 0.2173913043 | 0.2631578947 | 0.002358490566 |
| training/2020-11 | increase | first | all | v127 | budget_matched | supported | 0 | 5535 | 23 | 0.01779071231 | 0.1702381296 | 0.00381471519 | 5/13/18 | 18 | 0.2777777778 | 0.2173913043 | 0.2631578947 | 0.002358490566 |
| training/2020-11 | increase | first | all | v137 | original | supported | 0 | 5535 | 23 | 0.0171724253 | 0.2698404606 | 0.00368035722 | 6/8/17 | 14 | 0.4285714286 | 0.2608695652 | 0.3797468354 | 0.00145137881 |
| training/2020-11 | increase | first | all | v137 | budget_matched | supported | 0 | 5535 | 23 | 0.0171724253 | 0.2698404606 | 0.00368035722 | 6/8/17 | 14 | 0.4285714286 | 0.2608695652 | 0.3797468354 | 0.00145137881 |
| training/2020-11 | increase | first | below | v127 | original | supported | 0 | 5535 | 23 | 0.01779071231 | 0.1702381296 | 0.00381471519 | 5/13/18 | 18 | 0.2777777778 | 0.2173913043 | 0.2631578947 | 0.002358490566 |
| training/2020-11 | increase | first | below | v127 | budget_matched | supported | 0 | 5535 | 23 | 0.01779071231 | 0.1702381296 | 0.00381471519 | 5/13/18 | 18 | 0.2777777778 | 0.2173913043 | 0.2631578947 | 0.002358490566 |
| training/2020-11 | increase | first | below | v137 | original | supported | 0 | 5535 | 23 | 0.0171724253 | 0.2698404606 | 0.00368035722 | 6/8/17 | 14 | 0.4285714286 | 0.2608695652 | 0.3797468354 | 0.00145137881 |
| training/2020-11 | increase | first | below | v137 | budget_matched | supported | 0 | 5535 | 23 | 0.0171724253 | 0.2698404606 | 0.00368035722 | 6/8/17 | 14 | 0.4285714286 | 0.2608695652 | 0.3797468354 | 0.00145137881 |
| training/2020-11 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-11 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-11 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-11 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-11 | increase | restart | all | v127 | original | supported | 0 | 25170 | 83 | 0.01859997045 | 0.06405267224 | 0.003219356675 | 5/9/78 | 14 | 0.3571428571 | 0.06024096386 | 0.1798561151 | 0.0003587515446 |
| training/2020-11 | increase | restart | all | v127 | budget_matched | supported | 0 | 25170 | 83 | 0.01859997045 | 0.06405267224 | 0.003219356675 | 5/9/78 | 14 | 0.3571428571 | 0.06024096386 | 0.1798561151 | 0.0003587515446 |
| training/2020-11 | increase | restart | all | v137 | original | supported | 0 | 25170 | 83 | 0.01855601895 | 0.09332514093 | 0.003218129379 | 11/70/72 | 81 | 0.1358024691 | 0.1325301205 | 0.1351351351 | 0.002790289792 |
| training/2020-11 | increase | restart | all | v137 | budget_matched | supported | 0 | 25170 | 83 | 0.01855601895 | 0.09332514093 | 0.003218129379 | 5/15/78 | 20 | 0.25 | 0.06024096386 | 0.1533742331 | 0.000597919241 |
| training/2020-11 | increase | restart | below | v127 | original | supported | 0 | 12597 | 32 | 0.01431926105 | 0.09012454314 | 0.002450584292 | 4/8/28 | 12 | 0.3333333333 | 0.125 | 0.25 | 0.0006366892161 |
| training/2020-11 | increase | restart | below | v127 | budget_matched | supported | 0 | 12597 | 32 | 0.01431926105 | 0.09012454314 | 0.002450584292 | 4/8/28 | 12 | 0.3333333333 | 0.125 | 0.25 | 0.0006366892161 |
| training/2020-11 | increase | restart | below | v137 | original | supported | 0 | 12597 | 32 | 0.01439593626 | 0.141249337 | 0.002461275562 | 7/28/25 | 35 | 0.2 | 0.21875 | 0.2034883721 | 0.002228412256 |
| training/2020-11 | increase | restart | below | v137 | budget_matched | supported | 0 | 12597 | 32 | 0.01439593626 | 0.141249337 | 0.002461275562 | 3/3/29 | 6 | 0.5 | 0.09375 | 0.2678571429 | 0.000238758456 |
| training/2020-11 | increase | restart | capped | v127 | original | supported | 0 | 12573 | 51 | 0.02288885109 | 0.06021515234 | 0.003989596532 | 1/1/50 | 2 | 0.5 | 0.01960784314 | 0.08474576271 | 7.985944737e-05 |
| training/2020-11 | increase | restart | capped | v127 | budget_matched | supported | 0 | 12573 | 51 | 0.02288885109 | 0.06021515234 | 0.003989596532 | 1/1/50 | 2 | 0.5 | 0.01960784314 | 0.08474576271 | 7.985944737e-05 |
| training/2020-11 | increase | restart | capped | v137 | original | supported | 0 | 12573 | 51 | 0.02272404264 | 0.06924262669 | 0.003976427917 | 4/42/47 | 46 | 0.08695652174 | 0.07843137255 | 0.08510638298 | 0.00335409679 |
| training/2020-11 | increase | restart | capped | v137 | budget_matched | supported | 0 | 12573 | 51 | 0.02272404264 | 0.06924262669 | 0.003976427917 | 2/12/49 | 14 | 0.1428571429 | 0.03921568627 | 0.09345794393 | 0.0009583133685 |
| training/2020-12 | reduce | quiet | all | v127 | original | supported | 0 | 39344 | 48 | 0.007158836861 | 0.0489857833 | 0.001194580282 | 4/20/44 | 24 | 0.1666666667 | 0.08333333333 | 0.1111111111 | 0.0005089576547 |
| training/2020-12 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 39344 | 48 | 0.007158836861 | 0.0489857833 | 0.001194580282 | 3/32/45 | 35 | 0.08571428571 | 0.0625 | 0.07228915663 | 0.0008143322476 |
| training/2020-12 | reduce | quiet | all | v137 | original | supported | 0 | 39344 | 48 | 0.007157962274 | 0.04459483386 | 0.00119735908 | 5/59/43 | 64 | 0.078125 | 0.1041666667 | 0.08928571429 | 0.001501425081 |
| training/2020-12 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 39344 | 48 | 0.007157962274 | 0.04459483386 | 0.00119735908 | 2/40/46 | 42 | 0.04761904762 | 0.04166666667 | 0.04444444444 | 0.001017915309 |
| training/2020-12 | reduce | quiet | below | v127 | original | supported | 0 | 23175 | 12 | 0.003385818894 | 0.1462003755 | 0.0005062132803 | 2/1/10 | 3 | 0.6666666667 | 0.1666666667 | 0.2666666667 | 4.317230065e-05 |
| training/2020-12 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 23175 | 12 | 0.003385818894 | 0.1462003755 | 0.0005062132803 | 2/1/10 | 3 | 0.6666666667 | 0.1666666667 | 0.2666666667 | 4.317230065e-05 |
| training/2020-12 | reduce | quiet | below | v137 | original | supported | 0 | 23175 | 12 | 0.003450150803 | 0.06815164387 | 0.0005133563609 | 1/3/11 | 4 | 0.25 | 0.08333333333 | 0.125 | 0.000129516902 |
| training/2020-12 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 23175 | 12 | 0.003450150803 | 0.06815164387 | 0.0005133563609 | 0/0/12 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | quiet | capped | v127 | original | supported | 0 | 16169 | 36 | 0.01256669705 | 0.05233629523 | 0.002181215526 | 2/19/34 | 21 | 0.09523809524 | 0.05555555556 | 0.0701754386 | 0.001177710283 |
| training/2020-12 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 16169 | 36 | 0.01256669705 | 0.05233629523 | 0.002181215526 | 2/29/34 | 31 | 0.06451612903 | 0.05555555556 | 0.05970149254 | 0.001797557801 |
| training/2020-12 | reduce | quiet | capped | v137 | original | supported | 0 | 16169 | 36 | 0.0124723621 | 0.05079116693 | 0.002177739005 | 4/56/32 | 60 | 0.06666666667 | 0.1111111111 | 0.08333333333 | 0.003471146098 |
| training/2020-12 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 16169 | 36 | 0.0124723621 | 0.05079116693 | 0.002177739005 | 2/36/34 | 38 | 0.05263157895 | 0.05555555556 | 0.05405405405 | 0.002231451063 |
| training/2020-12 | reduce | first | all | v127 | original | supported | 0 | 24762 | 12 | 0.003045303234 | 0.2441311704 | 0.0004711874008 | 2/1/10 | 3 | 0.6666666667 | 0.1666666667 | 0.2666666667 | 4.04040404e-05 |
| training/2020-12 | reduce | first | all | v127 | budget_matched | supported | 0 | 24762 | 12 | 0.003045303234 | 0.2441311704 | 0.0004711874008 | 2/1/10 | 3 | 0.6666666667 | 0.1666666667 | 0.2666666667 | 4.04040404e-05 |
| training/2020-12 | reduce | first | all | v137 | original | supported | 0 | 24762 | 12 | 0.003093257617 | 0.22129006 | 0.0004765243525 | 2/3/10 | 5 | 0.4 | 0.1666666667 | 0.2352941176 | 0.0001212121212 |
| training/2020-12 | reduce | first | all | v137 | budget_matched | supported | 0 | 24762 | 12 | 0.003093257617 | 0.22129006 | 0.0004765243525 | 1/0/11 | 1 | 1 | 0.08333333333 | 0.1538461538 | 0 |
| training/2020-12 | reduce | first | below | v127 | original | supported | 0 | 21368 | 8 | 0.00240151263 | 0.2566783041 | 0.0003620345474 | 2/1/6 | 3 | 0.6666666667 | 0.25 | 0.3636363636 | 4.68164794e-05 |
| training/2020-12 | reduce | first | below | v127 | budget_matched | supported | 0 | 21368 | 8 | 0.00240151263 | 0.2566783041 | 0.0003620345474 | 2/1/6 | 3 | 0.6666666667 | 0.25 | 0.3636363636 | 4.68164794e-05 |
| training/2020-12 | reduce | first | below | v137 | original | supported | 0 | 21368 | 8 | 0.002468739639 | 0.1754683696 | 0.0003700864038 | 1/2/7 | 3 | 0.3333333333 | 0.125 | 0.1818181818 | 9.36329588e-05 |
| training/2020-12 | reduce | first | below | v137 | budget_matched | supported | 0 | 21368 | 8 | 0.002468739639 | 0.1754683696 | 0.0003700864038 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | first | capped | v127 | original | supported | 0 | 3394 | 4 | 0.007098490512 | 0.3544169326 | 0.001158393698 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | first | capped | v127 | budget_matched | supported | 0 | 3394 | 4 | 0.007098490512 | 0.3544169326 | 0.001158393698 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | first | capped | v137 | original | supported | 0 | 3394 | 4 | 0.007025107985 | 0.3245898815 | 0.001146638109 | 1/1/3 | 2 | 0.5 | 0.25 | 0.3333333333 | 0.0002949852507 |
| training/2020-12 | reduce | first | capped | v137 | budget_matched | supported | 0 | 3394 | 4 | 0.007025107985 | 0.3245898815 | 0.001146638109 | 1/0/3 | 1 | 1 | 0.25 | 0.4 | 0 |
| training/2020-12 | reduce | restart | all | v127 | original | supported | 0 | 14582 | 36 | 0.01414411458 | 0.04728130468 | 0.002422988904 | 2/19/34 | 21 | 0.09523809524 | 0.05555555556 | 0.0701754386 | 0.001306201017 |
| training/2020-12 | reduce | restart | all | v127 | budget_matched | supported | 0 | 14582 | 36 | 0.01414411458 | 0.04728130468 | 0.002422988904 | 2/19/34 | 21 | 0.09523809524 | 0.05555555556 | 0.0701754386 | 0.001306201017 |
| training/2020-12 | reduce | restart | all | v137 | original | supported | 0 | 14582 | 36 | 0.01406032249 | 0.04496009459 | 0.002421423647 | 3/56/33 | 59 | 0.05084745763 | 0.08333333333 | 0.06315789474 | 0.00384985563 |
| training/2020-12 | reduce | restart | all | v137 | budget_matched | supported | 0 | 14582 | 36 | 0.01406032249 | 0.04496009459 | 0.002421423647 | 2/20/34 | 22 | 0.09090909091 | 0.05555555556 | 0.06896551724 | 0.001374948439 |
| training/2020-12 | reduce | restart | below | v127 | original | supported | 0 | 1807 | 4 | 0.01502536303 | 0.0105315675 | 0.00221114475 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | restart | below | v127 | budget_matched | supported | 0 | 1807 | 4 | 0.01502536303 | 0.0105315675 | 0.00221114475 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | restart | below | v137 | original | supported | 0 | 1807 | 4 | 0.01505546002 | 0.01516132298 | 0.002207541444 | 0/1/4 | 1 | 0 | 0 | 0 | 0.0005546311703 |
| training/2020-12 | reduce | restart | below | v137 | budget_matched | supported | 0 | 1807 | 4 | 0.01505546002 | 0.01516132298 | 0.002207541444 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | restart | capped | v127 | original | supported | 0 | 12775 | 32 | 0.01401946362 | 0.05287298818 | 0.002452953866 | 2/19/30 | 21 | 0.09523809524 | 0.0625 | 0.07547169811 | 0.001491014675 |
| training/2020-12 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 12775 | 32 | 0.01401946362 | 0.05287298818 | 0.002452953866 | 2/19/30 | 21 | 0.09523809524 | 0.0625 | 0.07547169811 | 0.001491014675 |
| training/2020-12 | reduce | restart | capped | v137 | original | supported | 0 | 12775 | 32 | 0.01391956214 | 0.04955427057 | 0.002451676887 | 3/55/29 | 58 | 0.05172413793 | 0.09375 | 0.06666666667 | 0.004316095111 |
| training/2020-12 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 12775 | 32 | 0.01391956214 | 0.04955427057 | 0.002451676887 | 2/23/30 | 25 | 0.08 | 0.0625 | 0.0701754386 | 0.001804912501 |
| training/2020-12 | increase | quiet | all | v127 | original | supported | 0 | 39318 | 40 | 0.005784566278 | 0.1096076669 | 0.0009795416833 | 2/5/38 | 7 | 0.2857142857 | 0.05 | 0.1470588235 | 0.0001272977239 |
| training/2020-12 | increase | quiet | all | v127 | budget_matched | supported | 0 | 39318 | 40 | 0.005784566278 | 0.1096076669 | 0.0009795416833 | 4/8/36 | 12 | 0.3333333333 | 0.1 | 0.2272727273 | 0.0002036763583 |
| training/2020-12 | increase | quiet | all | v137 | original | supported | 0 | 39318 | 40 | 0.00573596427 | 0.1056865977 | 0.0009770835589 | 6/56/34 | 62 | 0.09677419355 | 0.15 | 0.1041666667 | 0.001425734508 |
| training/2020-12 | increase | quiet | all | v137 | budget_matched | supported | 0 | 39318 | 40 | 0.00573596427 | 0.1056865977 | 0.0009770835589 | 3/10/37 | 13 | 0.2307692308 | 0.075 | 0.1630434783 | 0.0002545954478 |
| training/2020-12 | increase | quiet | below | v127 | original | supported | 0 | 22929 | 24 | 0.005690583016 | 0.1524482734 | 0.0009956864301 | 2/5/22 | 7 | 0.2857142857 | 0.08333333333 | 0.1923076923 | 0.0002182929491 |
| training/2020-12 | increase | quiet | below | v127 | budget_matched | supported | 0 | 22929 | 24 | 0.005690583016 | 0.1524482734 | 0.0009956864301 | 3/6/21 | 9 | 0.3333333333 | 0.125 | 0.25 | 0.000261951539 |
| training/2020-12 | increase | quiet | below | v137 | original | supported | 0 | 22929 | 24 | 0.005594918848 | 0.1405909118 | 0.0009925944113 | 4/24/20 | 28 | 0.1428571429 | 0.1666666667 | 0.1470588235 | 0.001047806156 |
| training/2020-12 | increase | quiet | below | v137 | budget_matched | supported | 0 | 22929 | 24 | 0.005594918848 | 0.1405909118 | 0.0009925944113 | 3/6/21 | 9 | 0.3333333333 | 0.125 | 0.25 | 0.000261951539 |
| training/2020-12 | increase | quiet | capped | v127 | original | supported | 0 | 16389 | 16 | 0.005916053387 | 0.1011805274 | 0.0009569544053 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 16389 | 16 | 0.005916053387 | 0.1011805274 | 0.0009569544053 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | quiet | capped | v137 | original | supported | 0 | 16389 | 16 | 0.005933293607 | 0.09957229634 | 0.0009553831296 | 2/32/14 | 34 | 0.05882352941 | 0.125 | 0.06578947368 | 0.001954437183 |
| training/2020-12 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 16389 | 16 | 0.005933293607 | 0.09957229634 | 0.0009553831296 | 1/0/15 | 1 | 1 | 0.0625 | 0.25 | 0 |
| training/2020-12 | increase | first | all | v127 | original | supported | 0 | 13852 | 7 | 0.002530518838 | 0.4055768557 | 0.0004452842408 | 2/4/5 | 6 | 0.3333333333 | 0.2857142857 | 0.3225806452 | 0.000288912965 |
| training/2020-12 | increase | first | all | v127 | budget_matched | supported | 0 | 13852 | 7 | 0.002530518838 | 0.4055768557 | 0.0004452842408 | 2/4/5 | 6 | 0.3333333333 | 0.2857142857 | 0.3225806452 | 0.000288912965 |
| training/2020-12 | increase | first | all | v137 | original | supported | 0 | 13852 | 7 | 0.002512061965 | 0.3158021939 | 0.0004474499794 | 1/2/6 | 3 | 0.3333333333 | 0.1428571429 | 0.2631578947 | 0.0001444564825 |
| training/2020-12 | increase | first | all | v137 | budget_matched | supported | 0 | 13852 | 7 | 0.002512061965 | 0.3158021939 | 0.0004474499794 | 1/3/6 | 4 | 0.25 | 0.1428571429 | 0.2173913043 | 0.0002166847237 |
| training/2020-12 | increase | first | below | v127 | original | supported | 0 | 13852 | 7 | 0.002530518838 | 0.4055768557 | 0.0004452842408 | 2/4/5 | 6 | 0.3333333333 | 0.2857142857 | 0.3225806452 | 0.000288912965 |
| training/2020-12 | increase | first | below | v127 | budget_matched | supported | 0 | 13852 | 7 | 0.002530518838 | 0.4055768557 | 0.0004452842408 | 2/4/5 | 6 | 0.3333333333 | 0.2857142857 | 0.3225806452 | 0.000288912965 |
| training/2020-12 | increase | first | below | v137 | original | supported | 0 | 13852 | 7 | 0.002512061965 | 0.3158021939 | 0.0004474499794 | 1/2/6 | 3 | 0.3333333333 | 0.1428571429 | 0.2631578947 | 0.0001444564825 |
| training/2020-12 | increase | first | below | v137 | budget_matched | supported | 0 | 13852 | 7 | 0.002512061965 | 0.3158021939 | 0.0004474499794 | 1/3/6 | 4 | 0.25 | 0.1428571429 | 0.2173913043 | 0.0002166847237 |
| training/2020-12 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-12 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-12 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-12 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-12 | increase | restart | all | v127 | original | supported | 0 | 25466 | 33 | 0.007554575905 | 0.04780641403 | 0.001270146179 | 0/1/33 | 1 | 0 | 0 | 0 | 3.931899501e-05 |
| training/2020-12 | increase | restart | all | v127 | budget_matched | supported | 0 | 25466 | 33 | 0.007554575905 | 0.04780641403 | 0.001270146179 | 0/1/33 | 1 | 0 | 0 | 0 | 3.931899501e-05 |
| training/2020-12 | increase | restart | all | v137 | original | supported | 0 | 25466 | 33 | 0.007489576723 | 0.1066824999 | 0.001265172946 | 5/54/28 | 59 | 0.08474576271 | 0.1515151515 | 0.09293680297 | 0.00212322573 |
| training/2020-12 | increase | restart | all | v137 | budget_matched | supported | 0 | 25466 | 33 | 0.007489576723 | 0.1066824999 | 0.001265172946 | 2/2/31 | 4 | 0.5 | 0.06060606061 | 0.2040816327 | 7.863799001e-05 |
| training/2020-12 | increase | restart | below | v127 | original | supported | 0 | 9077 | 17 | 0.01051301433 | 0.04895244092 | 0.001835630368 | 0/1/17 | 1 | 0 | 0 | 0 | 0.0001103752759 |
| training/2020-12 | increase | restart | below | v127 | budget_matched | supported | 0 | 9077 | 17 | 0.01051301433 | 0.04895244092 | 0.001835630368 | 0/1/17 | 1 | 0 | 0 | 0 | 0.0001103752759 |
| training/2020-12 | increase | restart | below | v137 | original | supported | 0 | 9077 | 17 | 0.01029952759 | 0.1233235859 | 0.001824514723 | 3/22/14 | 25 | 0.12 | 0.1764705882 | 0.1282051282 | 0.002428256071 |
| training/2020-12 | increase | restart | below | v137 | budget_matched | supported | 0 | 9077 | 17 | 0.01029952759 | 0.1233235859 | 0.001824514723 | 1/2/16 | 3 | 0.3333333333 | 0.05882352941 | 0.1724137931 | 0.0002207505519 |
| training/2020-12 | increase | restart | capped | v127 | original | supported | 0 | 16389 | 16 | 0.005916053387 | 0.1011805274 | 0.0009569544053 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | restart | capped | v127 | budget_matched | supported | 0 | 16389 | 16 | 0.005916053387 | 0.1011805274 | 0.0009569544053 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | restart | capped | v137 | original | supported | 0 | 16389 | 16 | 0.005933293607 | 0.09957229634 | 0.0009553831296 | 2/32/14 | 34 | 0.05882352941 | 0.125 | 0.06578947368 | 0.001954437183 |
| training/2020-12 | increase | restart | capped | v137 | budget_matched | supported | 0 | 16389 | 16 | 0.005933293607 | 0.09957229634 | 0.0009553831296 | 1/0/15 | 1 | 1 | 0.0625 | 0.25 | 0 |
| calibration/2021-01 | reduce | quiet | all | v127 | original | supported | 0 | 40342 | 81 | 0.01182994363 | 0.04432063624 | 0.001961916949 | 10/78/71 | 88 | 0.1136363636 | 0.1234567901 | 0.1183431953 | 0.001937358734 |
| calibration/2021-01 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 40342 | 81 | 0.01182994363 | 0.04432063624 | 0.001961916949 | 11/86/70 | 97 | 0.1134020619 | 0.1358024691 | 0.1235955056 | 0.002136062194 |
| calibration/2021-01 | reduce | quiet | all | v137 | original | supported | 0 | 40342 | 81 | 0.01196619764 | 0.03390830409 | 0.001970525017 | 20/316/61 | 336 | 0.05952380952 | 0.2469135802 | 0.09592326139 | 0.007848786667 |
| calibration/2021-01 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 40342 | 81 | 0.01196619764 | 0.03390830409 | 0.001970525017 | 8/97/73 | 105 | 0.07619047619 | 0.0987654321 | 0.08602150538 | 0.002409279452 |
| calibration/2021-01 | reduce | quiet | below | v127 | original | supported | 0 | 17639 | 23 | 0.008343218209 | 0.01449790882 | 0.001299333847 | 0/16/23 | 16 | 0 | 0 | 0 | 0.0009082652134 |
| calibration/2021-01 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 17639 | 23 | 0.008343218209 | 0.01449790882 | 0.001299333847 | 0/29/23 | 29 | 0 | 0 | 0 | 0.001646230699 |
| calibration/2021-01 | reduce | quiet | below | v137 | original | supported | 0 | 17639 | 23 | 0.008564874484 | 0.01652905049 | 0.00129854312 | 5/96/18 | 101 | 0.0495049505 | 0.2173913043 | 0.08064516129 | 0.005449591281 |
| calibration/2021-01 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 17639 | 23 | 0.008564874484 | 0.01652905049 | 0.00129854312 | 1/31/22 | 32 | 0.03125 | 0.04347826087 | 0.03636363636 | 0.001759763851 |
| calibration/2021-01 | reduce | quiet | capped | v127 | original | supported | 0 | 22703 | 58 | 0.01453894022 | 0.05761877739 | 0.002476708094 | 10/62/48 | 72 | 0.1388888889 | 0.1724137931 | 0.1538461538 | 0.002737911239 |
| calibration/2021-01 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 22703 | 58 | 0.01453894022 | 0.05761877739 | 0.002476708094 | 10/59/48 | 69 | 0.1449275362 | 0.1724137931 | 0.157480315 | 0.002605431663 |
| calibration/2021-01 | reduce | quiet | capped | v137 | original | supported | 0 | 22703 | 58 | 0.0146088413 | 0.04407113515 | 0.002492618515 | 15/220/43 | 235 | 0.06382978723 | 0.2586206897 | 0.1023890785 | 0.009715168911 |
| calibration/2021-01 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 22703 | 58 | 0.0146088413 | 0.04407113515 | 0.002492618515 | 8/64/50 | 72 | 0.1111111111 | 0.1379310345 | 0.1230769231 | 0.002826230956 |
| calibration/2021-01 | reduce | first | all | v127 | original | supported | 0 | 15833 | 25 | 0.009953395684 | 0.01399877884 | 0.001570182965 | 1/37/24 | 38 | 0.02631578947 | 0.04 | 0.03174603175 | 0.002340587045 |
| calibration/2021-01 | reduce | first | all | v127 | budget_matched | supported | 0 | 15833 | 25 | 0.009953395684 | 0.01399877884 | 0.001570182965 | 1/37/24 | 38 | 0.02631578947 | 0.04 | 0.03174603175 | 0.002340587045 |
| calibration/2021-01 | reduce | first | all | v137 | original | supported | 0 | 15833 | 25 | 0.009995657289 | 0.01642425334 | 0.001566769124 | 5/183/20 | 188 | 0.02659574468 | 0.2 | 0.04694835681 | 0.011576417 |
| calibration/2021-01 | reduce | first | all | v137 | budget_matched | supported | 0 | 15833 | 25 | 0.009995657289 | 0.01642425334 | 0.001566769124 | 1/34/24 | 35 | 0.02857142857 | 0.04 | 0.03333333333 | 0.002150809717 |
| calibration/2021-01 | reduce | first | below | v127 | original | supported | 0 | 12404 | 18 | 0.008786701686 | 0.02175838294 | 0.001437947897 | 0/14/18 | 14 | 0 | 0 | 0 | 0.001130308413 |
| calibration/2021-01 | reduce | first | below | v127 | budget_matched | supported | 0 | 12404 | 18 | 0.008786701686 | 0.02175838294 | 0.001437947897 | 0/14/18 | 14 | 0 | 0 | 0 | 0.001130308413 |
| calibration/2021-01 | reduce | first | below | v137 | original | supported | 0 | 12404 | 18 | 0.008798081821 | 0.0447879405 | 0.001430146096 | 4/65/14 | 69 | 0.05797101449 | 0.2222222222 | 0.09195402299 | 0.005247860488 |
| calibration/2021-01 | reduce | first | below | v137 | budget_matched | supported | 0 | 12404 | 18 | 0.008798081821 | 0.0447879405 | 0.001430146096 | 1/4/17 | 5 | 0.2 | 0.05555555556 | 0.08695652174 | 0.0003229452608 |
| calibration/2021-01 | reduce | first | capped | v127 | original | supported | 0 | 3429 | 7 | 0.01417377258 | 0.01060305725 | 0.002048527608 | 1/23/6 | 24 | 0.04166666667 | 0.1428571429 | 0.06451612903 | 0.006721215663 |
| calibration/2021-01 | reduce | first | capped | v127 | budget_matched | supported | 0 | 3429 | 7 | 0.01417377258 | 0.01060305725 | 0.002048527608 | 1/23/6 | 24 | 0.04166666667 | 0.1428571429 | 0.06451612903 | 0.006721215663 |
| calibration/2021-01 | reduce | first | capped | v137 | original | supported | 0 | 3429 | 7 | 0.01432774422 | 0.007849075545 | 0.00206098669 | 1/118/6 | 119 | 0.008403361345 | 0.1428571429 | 0.01587301587 | 0.03448275862 |
| calibration/2021-01 | reduce | first | capped | v137 | budget_matched | supported | 0 | 3429 | 7 | 0.01432774422 | 0.007849075545 | 0.00206098669 | 0/29/7 | 29 | 0 | 0 | 0 | 0.008474576271 |
| calibration/2021-01 | reduce | restart | all | v127 | original | supported | 0 | 24509 | 56 | 0.01304220784 | 0.05786860446 | 0.002214980076 | 9/41/47 | 50 | 0.18 | 0.1607142857 | 0.1698113208 | 0.001676685887 |
| calibration/2021-01 | reduce | restart | all | v127 | budget_matched | supported | 0 | 24509 | 56 | 0.01304220784 | 0.05786860446 | 0.002214980076 | 9/41/47 | 50 | 0.18 | 0.1607142857 | 0.1698113208 | 0.001676685887 |
| calibration/2021-01 | reduce | restart | all | v137 | original | supported | 0 | 24509 | 56 | 0.01323918165 | 0.0419438002 | 0.002231354389 | 15/133/41 | 148 | 0.1013513514 | 0.2678571429 | 0.1470588235 | 0.005439005439 |
| calibration/2021-01 | reduce | restart | all | v137 | budget_matched | supported | 0 | 24509 | 56 | 0.01323918165 | 0.0419438002 | 0.002231354389 | 5/45/51 | 50 | 0.1 | 0.08928571429 | 0.09433962264 | 0.001840264998 |
| calibration/2021-01 | reduce | restart | below | v127 | original | supported | 0 | 5235 | 5 | 0.00729241228 | 0.01124907757 | 0.0009708966587 | 0/2/5 | 2 | 0 | 0 | 0 | 0.0003824091778 |
| calibration/2021-01 | reduce | restart | below | v127 | budget_matched | supported | 0 | 5235 | 5 | 0.00729241228 | 0.01124907757 | 0.0009708966587 | 0/2/5 | 2 | 0 | 0 | 0 | 0.0003824091778 |
| calibration/2021-01 | reduce | restart | below | v137 | original | supported | 0 | 5235 | 5 | 0.00801230451 | 0.01300157732 | 0.0009867182277 | 1/31/4 | 32 | 0.03125 | 0.2 | 0.05405405405 | 0.005927342256 |
| calibration/2021-01 | reduce | restart | below | v137 | budget_matched | supported | 0 | 5235 | 5 | 0.00801230451 | 0.01300157732 | 0.0009867182277 | 0/3/5 | 3 | 0 | 0 | 0 | 0.0005736137667 |
| calibration/2021-01 | reduce | restart | capped | v127 | original | supported | 0 | 19274 | 51 | 0.01460390649 | 0.06540703034 | 0.002552884854 | 9/39/42 | 48 | 0.1875 | 0.1764705882 | 0.1818181818 | 0.002028819643 |
| calibration/2021-01 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 19274 | 51 | 0.01460390649 | 0.06540703034 | 0.002552884854 | 9/39/42 | 48 | 0.1875 | 0.1764705882 | 0.1818181818 | 0.002028819643 |
| calibration/2021-01 | reduce | restart | capped | v137 | original | supported | 0 | 19274 | 51 | 0.01465885073 | 0.05237746865 | 0.002569409297 | 14/102/37 | 116 | 0.1206896552 | 0.2745098039 | 0.1676646707 | 0.005306143682 |
| calibration/2021-01 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 19274 | 51 | 0.01465885073 | 0.05237746865 | 0.002569409297 | 5/38/46 | 43 | 0.1162790698 | 0.09803921569 | 0.1063829787 | 0.001976798627 |
| calibration/2021-01 | increase | quiet | all | v127 | original | supported | 0 | 40191 | 75 | 0.01143749027 | 0.02857047076 | 0.001839441225 | 4/23/71 | 27 | 0.1481481481 | 0.05333333333 | 0.1092896175 | 0.0005733373218 |
| calibration/2021-01 | increase | quiet | all | v127 | budget_matched | supported | 0 | 40191 | 75 | 0.01143749027 | 0.02857047076 | 0.001839441225 | 2/31/73 | 33 | 0.06060606061 | 0.02666666667 | 0.04830917874 | 0.0007727589989 |
| calibration/2021-01 | increase | quiet | all | v137 | original | supported | 0 | 40191 | 75 | 0.01146050969 | 0.03084610703 | 0.001835192055 | 7/182/68 | 189 | 0.03703703704 | 0.09333333333 | 0.0421179302 | 0.004536843155 |
| calibration/2021-01 | increase | quiet | all | v137 | budget_matched | supported | 0 | 40191 | 75 | 0.01146050969 | 0.03084610703 | 0.001835192055 | 4/32/71 | 36 | 0.1111111111 | 0.05333333333 | 0.09132420091 | 0.0007976867085 |
| calibration/2021-01 | increase | quiet | below | v127 | original | supported | 0 | 17428 | 33 | 0.01051425848 | 0.05271453234 | 0.001838712 | 4/15/29 | 19 | 0.2105263158 | 0.1212121212 | 0.1834862385 | 0.0008623167577 |
| calibration/2021-01 | increase | quiet | below | v127 | budget_matched | supported | 0 | 17428 | 33 | 0.01051425848 | 0.05271453234 | 0.001838712 | 1/15/32 | 16 | 0.0625 | 0.0303030303 | 0.05154639175 | 0.0008623167577 |
| calibration/2021-01 | increase | quiet | below | v137 | original | supported | 0 | 17428 | 33 | 0.0105790332 | 0.05716376911 | 0.001833886663 | 5/78/28 | 83 | 0.06024096386 | 0.1515151515 | 0.06849315068 | 0.00448404714 |
| calibration/2021-01 | increase | quiet | below | v137 | budget_matched | supported | 0 | 17428 | 33 | 0.0105790332 | 0.05716376911 | 0.001833886663 | 3/21/30 | 24 | 0.125 | 0.09090909091 | 0.1162790698 | 0.001207243461 |
| calibration/2021-01 | increase | quiet | capped | v127 | original | supported | 0 | 22763 | 42 | 0.01214434278 | 0.0117505023 | 0.00183999954 | 0/8/42 | 8 | 0 | 0 | 0 | 0.0003520971788 |
| calibration/2021-01 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 22763 | 42 | 0.01214434278 | 0.0117505023 | 0.00183999954 | 0/8/42 | 8 | 0 | 0 | 0 | 0.0003520971788 |
| calibration/2021-01 | increase | quiet | capped | v137 | original | supported | 0 | 22763 | 42 | 0.01213539316 | 0.01201276848 | 0.0018361915 | 2/104/40 | 106 | 0.01886792453 | 0.04761904762 | 0.02145922747 | 0.004577263325 |
| calibration/2021-01 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 22763 | 42 | 0.01213539316 | 0.01201276848 | 0.0018361915 | 0/10/42 | 10 | 0 | 0 | 0 | 0.0004401214735 |
| calibration/2021-01 | increase | first | all | v127 | original | supported | 0 | 5617 | 19 | 0.01759073017 | 0.05702502977 | 0.003292774694 | 1/8/18 | 9 | 0.1111111111 | 0.05263157895 | 0.09090909091 | 0.001429081815 |
| calibration/2021-01 | increase | first | all | v127 | budget_matched | supported | 0 | 5617 | 19 | 0.01759073017 | 0.05702502977 | 0.003292774694 | 1/8/18 | 9 | 0.1111111111 | 0.05263157895 | 0.09090909091 | 0.001429081815 |
| calibration/2021-01 | increase | first | all | v137 | original | supported | 0 | 5617 | 19 | 0.01670141354 | 0.07918591381 | 0.003236801829 | 2/12/17 | 14 | 0.1428571429 | 0.1052631579 | 0.1333333333 | 0.002143622722 |
| calibration/2021-01 | increase | first | all | v137 | budget_matched | supported | 0 | 5617 | 19 | 0.01670141354 | 0.07918591381 | 0.003236801829 | 2/13/17 | 15 | 0.1333333333 | 0.1052631579 | 0.1265822785 | 0.002322257949 |
| calibration/2021-01 | increase | first | below | v127 | original | supported | 0 | 5617 | 19 | 0.01759073017 | 0.05702502977 | 0.003292774694 | 1/8/18 | 9 | 0.1111111111 | 0.05263157895 | 0.09090909091 | 0.001429081815 |
| calibration/2021-01 | increase | first | below | v127 | budget_matched | supported | 0 | 5617 | 19 | 0.01759073017 | 0.05702502977 | 0.003292774694 | 1/8/18 | 9 | 0.1111111111 | 0.05263157895 | 0.09090909091 | 0.001429081815 |
| calibration/2021-01 | increase | first | below | v137 | original | supported | 0 | 5617 | 19 | 0.01670141354 | 0.07918591381 | 0.003236801829 | 2/12/17 | 14 | 0.1428571429 | 0.1052631579 | 0.1333333333 | 0.002143622722 |
| calibration/2021-01 | increase | first | below | v137 | budget_matched | supported | 0 | 5617 | 19 | 0.01670141354 | 0.07918591381 | 0.003236801829 | 2/13/17 | 15 | 0.1333333333 | 0.1052631579 | 0.1265822785 | 0.002322257949 |
| calibration/2021-01 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-01 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-01 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-01 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-01 | increase | restart | all | v127 | original | supported | 0 | 34574 | 56 | 0.0104378157 | 0.02344770992 | 0.001603328131 | 3/15/53 | 18 | 0.1666666667 | 0.05357142857 | 0.1171875 | 0.0004345558839 |
| calibration/2021-01 | increase | restart | all | v127 | budget_matched | supported | 0 | 34574 | 56 | 0.0104378157 | 0.02344770992 | 0.001603328131 | 3/15/53 | 18 | 0.1666666667 | 0.05357142857 | 0.1171875 | 0.0004345558839 |
| calibration/2021-01 | increase | restart | all | v137 | original | supported | 0 | 34574 | 56 | 0.01060905609 | 0.01671598663 | 0.001607482154 | 5/170/51 | 175 | 0.02857142857 | 0.08928571429 | 0.03306878307 | 0.004924966684 |
| calibration/2021-01 | increase | restart | all | v137 | budget_matched | supported | 0 | 34574 | 56 | 0.01060905609 | 0.01671598663 | 0.001607482154 | 1/19/55 | 20 | 0.05 | 0.01785714286 | 0.03676470588 | 0.0005504374529 |
| calibration/2021-01 | increase | restart | below | v127 | original | supported | 0 | 11811 | 14 | 0.00714887523 | 0.07607864319 | 0.001147198144 | 3/7/11 | 10 | 0.3 | 0.2142857143 | 0.2777777778 | 0.0005933711961 |
| calibration/2021-01 | increase | restart | below | v127 | budget_matched | supported | 0 | 11811 | 14 | 0.00714887523 | 0.07607864319 | 0.001147198144 | 3/7/11 | 10 | 0.3 | 0.2142857143 | 0.2777777778 | 0.0005933711961 |
| calibration/2021-01 | increase | restart | below | v137 | original | supported | 0 | 11811 | 14 | 0.007667390627 | 0.09169932613 | 0.001166697221 | 3/66/11 | 69 | 0.04347826087 | 0.2142857143 | 0.05172413793 | 0.005594642706 |
| calibration/2021-01 | increase | restart | below | v137 | budget_matched | supported | 0 | 11811 | 14 | 0.007667390627 | 0.09169932613 | 0.001166697221 | 1/9/13 | 10 | 0.1 | 0.07142857143 | 0.09259259259 | 0.0007629058235 |
| calibration/2021-01 | increase | restart | capped | v127 | original | supported | 0 | 22763 | 42 | 0.01214434278 | 0.0117505023 | 0.00183999954 | 0/8/42 | 8 | 0 | 0 | 0 | 0.0003520971788 |
| calibration/2021-01 | increase | restart | capped | v127 | budget_matched | supported | 0 | 22763 | 42 | 0.01214434278 | 0.0117505023 | 0.00183999954 | 0/8/42 | 8 | 0 | 0 | 0 | 0.0003520971788 |
| calibration/2021-01 | increase | restart | capped | v137 | original | supported | 0 | 22763 | 42 | 0.01213539316 | 0.01201276848 | 0.0018361915 | 2/104/40 | 106 | 0.01886792453 | 0.04761904762 | 0.02145922747 | 0.004577263325 |
| calibration/2021-01 | increase | restart | capped | v137 | budget_matched | supported | 0 | 22763 | 42 | 0.01213539316 | 0.01201276848 | 0.0018361915 | 0/10/42 | 10 | 0 | 0 | 0 | 0.0004401214735 |
| calibration/2021-02 | reduce | quiet | all | v127 | original | supported | 0 | 39227 | 60 | 0.009891405993 | 0.02225985247 | 0.00151716812 | 4/25/56 | 29 | 0.1379310345 | 0.06666666667 | 0.08988764045 | 0.0006382924401 |
| calibration/2021-02 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 39227 | 60 | 0.009891405993 | 0.02225985247 | 0.00151716812 | 3/38/57 | 41 | 0.07317073171 | 0.05 | 0.05940594059 | 0.0009702045089 |
| calibration/2021-02 | reduce | quiet | all | v137 | original | supported | 0 | 39227 | 60 | 0.01004466844 | 0.02249820322 | 0.001514364106 | 4/75/56 | 79 | 0.05063291139 | 0.06666666667 | 0.05755395683 | 0.00191487732 |
| calibration/2021-02 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 39227 | 60 | 0.01004466844 | 0.02249820322 | 0.001514364106 | 2/22/58 | 24 | 0.08333333333 | 0.03333333333 | 0.04761904762 | 0.0005616973473 |
| calibration/2021-02 | reduce | quiet | below | v127 | original | supported | 0 | 14727 | 7 | 0.003741942449 | 0.01954212025 | 0.0004726924147 | 0/2/7 | 2 | 0 | 0 | 0 | 0.0001358695652 |
| calibration/2021-02 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 14727 | 7 | 0.003741942449 | 0.01954212025 | 0.0004726924147 | 0/1/7 | 1 | 0 | 0 | 0 | 6.793478261e-05 |
| calibration/2021-02 | reduce | quiet | below | v137 | original | supported | 0 | 14727 | 7 | 0.004064768455 | 0.005337992654 | 0.0004757917519 | 0/9/7 | 9 | 0 | 0 | 0 | 0.0006114130435 |
| calibration/2021-02 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 14727 | 7 | 0.004064768455 | 0.005337992654 | 0.0004757917519 | 0/1/7 | 1 | 0 | 0 | 0 | 6.793478261e-05 |
| calibration/2021-02 | reduce | quiet | capped | v127 | original | supported | 0 | 24500 | 53 | 0.01358786108 | 0.02397314663 | 0.002145004598 | 4/23/49 | 27 | 0.1481481481 | 0.07547169811 | 0.1 | 0.0009408107334 |
| calibration/2021-02 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 24500 | 53 | 0.01358786108 | 0.02397314663 | 0.002145004598 | 3/34/50 | 37 | 0.08108108108 | 0.05660377358 | 0.06666666667 | 0.001390763693 |
| calibration/2021-02 | reduce | quiet | capped | v137 | original | supported | 0 | 24500 | 53 | 0.01363919853 | 0.0260307408 | 0.002138652067 | 4/66/49 | 70 | 0.05714285714 | 0.07547169811 | 0.06504065041 | 0.002699717757 |
| calibration/2021-02 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 24500 | 53 | 0.01363919853 | 0.0260307408 | 0.002138652067 | 2/18/51 | 20 | 0.1 | 0.03773584906 | 0.05479452055 | 0.0007362866609 |
| calibration/2021-02 | reduce | first | all | v127 | original | supported | 0 | 11594 | 11 | 0.006602912262 | 0.02891809941 | 0.0009398543585 | 1/9/10 | 10 | 0.1 | 0.09090909091 | 0.09523809524 | 0.000777000777 |
| calibration/2021-02 | reduce | first | all | v127 | budget_matched | supported | 0 | 11594 | 11 | 0.006602912262 | 0.02891809941 | 0.0009398543585 | 1/9/10 | 10 | 0.1 | 0.09090909091 | 0.09523809524 | 0.000777000777 |
| calibration/2021-02 | reduce | first | all | v137 | original | supported | 0 | 11594 | 11 | 0.006679633277 | 0.02357602692 | 0.0009410008411 | 1/41/10 | 42 | 0.02380952381 | 0.09090909091 | 0.03773584906 | 0.003539670206 |
| calibration/2021-02 | reduce | first | all | v137 | budget_matched | supported | 0 | 11594 | 11 | 0.006679633277 | 0.02357602692 | 0.0009410008411 | 0/7/11 | 7 | 0 | 0 | 0 | 0.0006043339377 |
| calibration/2021-02 | reduce | first | below | v127 | original | supported | 0 | 8837 | 5 | 0.00427579604 | 0.03407885822 | 0.0005612281835 | 0/2/5 | 2 | 0 | 0 | 0 | 0.0002264492754 |
| calibration/2021-02 | reduce | first | below | v127 | budget_matched | supported | 0 | 8837 | 5 | 0.00427579604 | 0.03407885822 | 0.0005612281835 | 0/2/5 | 2 | 0 | 0 | 0 | 0.0002264492754 |
| calibration/2021-02 | reduce | first | below | v137 | original | supported | 0 | 8837 | 5 | 0.004391786456 | 0.02317782839 | 0.0005630017535 | 0/9/5 | 9 | 0 | 0 | 0 | 0.001019021739 |
| calibration/2021-02 | reduce | first | below | v137 | budget_matched | supported | 0 | 8837 | 5 | 0.004391786456 | 0.02317782839 | 0.0005630017535 | 0/2/5 | 2 | 0 | 0 | 0 | 0.0002264492754 |
| calibration/2021-02 | reduce | first | capped | v127 | original | supported | 0 | 2757 | 6 | 0.01406200767 | 0.03563620491 | 0.002153463175 | 1/7/5 | 8 | 0.125 | 0.1666666667 | 0.1428571429 | 0.002544529262 |
| calibration/2021-02 | reduce | first | capped | v127 | budget_matched | supported | 0 | 2757 | 6 | 0.01406200767 | 0.03563620491 | 0.002153463175 | 1/7/5 | 8 | 0.125 | 0.1666666667 | 0.1428571429 | 0.002544529262 |
| calibration/2021-02 | reduce | first | capped | v137 | original | supported | 0 | 2757 | 6 | 0.01401285865 | 0.03577232775 | 0.002152599658 | 1/32/5 | 33 | 0.0303030303 | 0.1666666667 | 0.05128205128 | 0.01163213377 |
| calibration/2021-02 | reduce | first | capped | v137 | budget_matched | supported | 0 | 2757 | 6 | 0.01401285865 | 0.03577232775 | 0.002152599658 | 0/5/6 | 5 | 0 | 0 | 0 | 0.001817520901 |
| calibration/2021-02 | reduce | restart | all | v127 | original | supported | 0 | 27633 | 49 | 0.01127116195 | 0.02320313107 | 0.001759392118 | 3/16/46 | 19 | 0.1578947368 | 0.0612244898 | 0.08823529412 | 0.0005800464037 |
| calibration/2021-02 | reduce | restart | all | v127 | budget_matched | supported | 0 | 27633 | 49 | 0.01127116195 | 0.02320313107 | 0.001759392118 | 3/16/46 | 19 | 0.1578947368 | 0.0612244898 | 0.08823529412 | 0.0005800464037 |
| calibration/2021-02 | reduce | restart | all | v137 | original | supported | 0 | 27633 | 49 | 0.01145653895 | 0.02418863002 | 0.001754930591 | 3/34/46 | 37 | 0.08108108108 | 0.0612244898 | 0.06976744186 | 0.001232598608 |
| calibration/2021-02 | reduce | restart | all | v137 | budget_matched | supported | 0 | 27633 | 49 | 0.01145653895 | 0.02418863002 | 0.001754930591 | 2/9/47 | 11 | 0.1818181818 | 0.04081632653 | 0.06666666667 | 0.0003262761021 |
| calibration/2021-02 | reduce | restart | below | v127 | original | supported | 0 | 5890 | 2 | 0.002940980788 | 0.01085648511 | 0.0003398586984 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | restart | below | v127 | budget_matched | supported | 0 | 5890 | 2 | 0.002940980788 | 0.01085648511 | 0.0003398586984 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | restart | below | v137 | original | supported | 0 | 5890 | 2 | 0.003574130411 | 0.00212916266 | 0.0003449471367 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | restart | below | v137 | budget_matched | supported | 0 | 5890 | 2 | 0.003574130411 | 0.00212916266 | 0.0003449471367 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | restart | capped | v127 | original | supported | 0 | 21743 | 47 | 0.01352773956 | 0.02451505164 | 0.002143932055 | 3/16/44 | 19 | 0.1578947368 | 0.06382978723 | 0.09090909091 | 0.0007374631268 |
| calibration/2021-02 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 21743 | 47 | 0.01352773956 | 0.02451505164 | 0.002143932055 | 3/16/44 | 19 | 0.1578947368 | 0.06382978723 | 0.09090909091 | 0.0007374631268 |
| calibration/2021-02 | reduce | restart | capped | v137 | original | supported | 0 | 21743 | 47 | 0.01359181864 | 0.02694766759 | 0.002136883521 | 3/34/44 | 37 | 0.08108108108 | 0.06382978723 | 0.07142857143 | 0.001567109145 |
| calibration/2021-02 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 21743 | 47 | 0.01359181864 | 0.02694766759 | 0.002136883521 | 2/10/45 | 12 | 0.1666666667 | 0.04255319149 | 0.06779661017 | 0.0004609144543 |
| calibration/2021-02 | increase | quiet | all | v127 | original | supported | 0 | 39150 | 60 | 0.01021059255 | 0.0154047971 | 0.001525340426 | 0/7/60 | 7 | 0 | 0 | 0 | 0.000179073932 |
| calibration/2021-02 | increase | quiet | all | v127 | budget_matched | supported | 0 | 39150 | 60 | 0.01021059255 | 0.0154047971 | 0.001525340426 | 0/5/60 | 5 | 0 | 0 | 0 | 0.0001279099514 |
| calibration/2021-02 | increase | quiet | all | v137 | original | supported | 0 | 39150 | 60 | 0.01022536995 | 0.01512879602 | 0.001522697118 | 1/54/59 | 55 | 0.01818181818 | 0.01666666667 | 0.01785714286 | 0.001381427475 |
| calibration/2021-02 | increase | quiet | all | v137 | budget_matched | supported | 0 | 39150 | 60 | 0.01022536995 | 0.01512879602 | 0.001522697118 | 0/11/60 | 11 | 0 | 0 | 0 | 0.0002814018931 |
| calibration/2021-02 | increase | quiet | below | v127 | original | supported | 0 | 14601 | 17 | 0.007433965582 | 0.03269207717 | 0.001157450009 | 0/5/17 | 5 | 0 | 0 | 0 | 0.0003428414701 |
| calibration/2021-02 | increase | quiet | below | v127 | budget_matched | supported | 0 | 14601 | 17 | 0.007433965582 | 0.03269207717 | 0.001157450009 | 0/5/17 | 5 | 0 | 0 | 0 | 0.0003428414701 |
| calibration/2021-02 | increase | quiet | below | v137 | original | supported | 0 | 14601 | 17 | 0.007401545993 | 0.02946303106 | 0.001151783065 | 0/4/17 | 4 | 0 | 0 | 0 | 0.0002742731761 |
| calibration/2021-02 | increase | quiet | below | v137 | budget_matched | supported | 0 | 14601 | 17 | 0.007401545993 | 0.02946303106 | 0.001151783065 | 0/3/17 | 3 | 0 | 0 | 0 | 0.0002057048821 |
| calibration/2021-02 | increase | quiet | capped | v127 | original | supported | 0 | 24549 | 43 | 0.01186204598 | 0.01013511232 | 0.001744150479 | 0/2/43 | 2 | 0 | 0 | 0 | 8.161266629e-05 |
| calibration/2021-02 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 24549 | 43 | 0.01186204598 | 0.01013511232 | 0.001744150479 | 0/2/43 | 2 | 0 | 0 | 0 | 8.161266629e-05 |
| calibration/2021-02 | increase | quiet | capped | v137 | original | supported | 0 | 24549 | 43 | 0.01190489472 | 0.01073698823 | 0.001743305538 | 1/50/42 | 51 | 0.01960784314 | 0.02325581395 | 0.02024291498 | 0.002040316657 |
| calibration/2021-02 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 24549 | 43 | 0.01190489472 | 0.01073698823 | 0.001743305538 | 0/5/43 | 5 | 0 | 0 | 0 | 0.0002040316657 |
| calibration/2021-02 | increase | first | all | v127 | original | supported | 0 | 7439 | 8 | 0.006095221373 | 0.04739705467 | 0.00107304135 | 0/5/8 | 5 | 0 | 0 | 0 | 0.0006728569506 |
| calibration/2021-02 | increase | first | all | v127 | budget_matched | supported | 0 | 7439 | 8 | 0.006095221373 | 0.04739705467 | 0.00107304135 | 0/5/8 | 5 | 0 | 0 | 0 | 0.0006728569506 |
| calibration/2021-02 | increase | first | all | v137 | original | supported | 0 | 7439 | 8 | 0.006166560231 | 0.04512281744 | 0.001059997769 | 0/1/8 | 1 | 0 | 0 | 0 | 0.0001345713901 |
| calibration/2021-02 | increase | first | all | v137 | budget_matched | supported | 0 | 7439 | 8 | 0.006166560231 | 0.04512281744 | 0.001059997769 | 0/1/8 | 1 | 0 | 0 | 0 | 0.0001345713901 |
| calibration/2021-02 | increase | first | below | v127 | original | supported | 0 | 7439 | 8 | 0.006095221373 | 0.04739705467 | 0.00107304135 | 0/5/8 | 5 | 0 | 0 | 0 | 0.0006728569506 |
| calibration/2021-02 | increase | first | below | v127 | budget_matched | supported | 0 | 7439 | 8 | 0.006095221373 | 0.04739705467 | 0.00107304135 | 0/5/8 | 5 | 0 | 0 | 0 | 0.0006728569506 |
| calibration/2021-02 | increase | first | below | v137 | original | supported | 0 | 7439 | 8 | 0.006166560231 | 0.04512281744 | 0.001059997769 | 0/1/8 | 1 | 0 | 0 | 0 | 0.0001345713901 |
| calibration/2021-02 | increase | first | below | v137 | budget_matched | supported | 0 | 7439 | 8 | 0.006166560231 | 0.04512281744 | 0.001059997769 | 0/1/8 | 1 | 0 | 0 | 0 | 0.0001345713901 |
| calibration/2021-02 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-02 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-02 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-02 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-02 | increase | restart | all | v127 | original | supported | 0 | 31711 | 52 | 0.01117600664 | 0.01126930527 | 0.001631444075 | 0/2/52 | 2 | 0 | 0 | 0 | 6.31731893e-05 |
| calibration/2021-02 | increase | restart | all | v127 | budget_matched | supported | 0 | 31711 | 52 | 0.01117600664 | 0.01126930527 | 0.001631444075 | 0/2/52 | 2 | 0 | 0 | 0 | 6.31731893e-05 |
| calibration/2021-02 | increase | restart | all | v137 | original | supported | 0 | 31711 | 52 | 0.01117751543 | 0.01117305413 | 0.00163124054 | 1/53/51 | 54 | 0.01851851852 | 0.01923076923 | 0.01865671642 | 0.001674089516 |
| calibration/2021-02 | increase | restart | all | v137 | budget_matched | supported | 0 | 31711 | 52 | 0.01117751543 | 0.01117305413 | 0.00163124054 | 0/5/52 | 5 | 0 | 0 | 0 | 0.0001579329732 |
| calibration/2021-02 | increase | restart | below | v127 | original | supported | 0 | 7162 | 9 | 0.008824487527 | 0.02869663859 | 0.001245123286 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | restart | below | v127 | budget_matched | supported | 0 | 7162 | 9 | 0.008824487527 | 0.02869663859 | 0.001245123286 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | restart | below | v137 | original | supported | 0 | 7162 | 9 | 0.008684296494 | 0.02880030509 | 0.001247118281 | 0/3/9 | 3 | 0 | 0 | 0 | 0.0004194044457 |
| calibration/2021-02 | increase | restart | below | v137 | budget_matched | supported | 0 | 7162 | 9 | 0.008684296494 | 0.02880030509 | 0.001247118281 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | restart | capped | v127 | original | supported | 0 | 24549 | 43 | 0.01186204598 | 0.01013511232 | 0.001744150479 | 0/2/43 | 2 | 0 | 0 | 0 | 8.161266629e-05 |
| calibration/2021-02 | increase | restart | capped | v127 | budget_matched | supported | 0 | 24549 | 43 | 0.01186204598 | 0.01013511232 | 0.001744150479 | 0/2/43 | 2 | 0 | 0 | 0 | 8.161266629e-05 |
| calibration/2021-02 | increase | restart | capped | v137 | original | supported | 0 | 24549 | 43 | 0.01190489472 | 0.01073698823 | 0.001743305538 | 1/50/42 | 51 | 0.01960784314 | 0.02325581395 | 0.02024291498 | 0.002040316657 |
| calibration/2021-02 | increase | restart | capped | v137 | budget_matched | supported | 0 | 24549 | 43 | 0.01190489472 | 0.01073698823 | 0.001743305538 | 0/5/43 | 5 | 0 | 0 | 0 | 0.0002040316657 |
| calibration/2021-03 | reduce | quiet | all | v127 | original | supported | 0 | 44177 | 26 | 0.004607983098 | 0.004771371167 | 0.0005877067506 | 0/0/26 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 44177 | 26 | 0.004607983098 | 0.004771371167 | 0.0005877067506 | 0/0/26 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | all | v137 | original | supported | 0 | 44177 | 26 | 0.004640769619 | 0.004667332739 | 0.0005877965655 | 0/1/26 | 1 | 0 | 0 | 0 | 2.264954361e-05 |
| calibration/2021-03 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 44177 | 26 | 0.004640769619 | 0.004667332739 | 0.0005877965655 | 0/0/26 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | below | v127 | original | supported | 0 | 3652 | 0 | 0.0004366146925 | 없음 | 9.04069239e-07 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 3652 | 0 | 0.0004366146925 | 없음 | 9.04069239e-07 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | below | v137 | original | supported | 0 | 3652 | 0 | 0.0005126393397 | 없음 | 2.424743095e-06 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 3652 | 0 | 0.0005126393397 | 없음 | 2.424743095e-06 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | capped | v127 | original | supported | 0 | 40525 | 26 | 0.004983895187 | 0.004902321773 | 0.0006405877721 | 0/0/26 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 40525 | 26 | 0.004983895187 | 0.004902321773 | 0.0006405877721 | 0/0/26 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | capped | v137 | original | supported | 0 | 40525 | 26 | 0.005012785209 | 0.005023529503 | 0.0006405486418 | 0/1/26 | 1 | 0 | 0 | 0 | 2.46919677e-05 |
| calibration/2021-03 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 40525 | 26 | 0.005012785209 | 0.005023529503 | 0.0006405486418 | 0/0/26 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | all | v127 | original | supported | 0 | 2892 | 1 | 0.002762486035 | 0.004291845494 | 0.0003456001605 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | all | v127 | budget_matched | supported | 0 | 2892 | 1 | 0.002762486035 | 0.004291845494 | 0.0003456001605 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | all | v137 | original | supported | 0 | 2892 | 1 | 0.002717008233 | 0.007936507937 | 0.0003454988973 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | all | v137 | budget_matched | supported | 0 | 2892 | 1 | 0.002717008233 | 0.007936507937 | 0.0003454988973 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | below | v127 | original | supported | 0 | 2867 | 0 | 0.0001905382301 | 없음 | 2.00669597e-07 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | below | v127 | budget_matched | supported | 0 | 2867 | 0 | 0.0001905382301 | 없음 | 2.00669597e-07 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | below | v137 | original | supported | 0 | 2867 | 0 | 0.0001401524033 | 없음 | 6.975768502e-08 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | below | v137 | budget_matched | supported | 0 | 2867 | 0 | 0.0001401524033 | 없음 | 6.975768502e-08 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | capped | v127 | original | supported | 0 | 25 | 1 | 0.2977134603 | 0.04 | 0.03995601378 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | capped | v127 | budget_matched | supported | 0 | 25 | 1 | 0.2977134603 | 0.04 | 0.03995601378 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | capped | v137 | original | supported | 0 | 25 | 1 | 0.2982308348 | 0.05263157895 | 0.03995931263 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | capped | v137 | budget_matched | supported | 0 | 25 | 1 | 0.2982308348 | 0.05263157895 | 0.03995931263 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | all | v127 | original | supported | 0 | 41285 | 25 | 0.00473725953 | 0.004931061888 | 0.0006046662337 | 0/0/25 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | all | v127 | budget_matched | supported | 0 | 41285 | 25 | 0.00473725953 | 0.004931061888 | 0.0006046662337 | 0/0/25 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | all | v137 | original | supported | 0 | 41285 | 25 | 0.00477552844 | 0.004818857393 | 0.0006047694335 | 0/1/25 | 1 | 0 | 0 | 0 | 2.423654872e-05 |
| calibration/2021-03 | reduce | restart | all | v137 | budget_matched | supported | 0 | 41285 | 25 | 0.00477552844 | 0.004818857393 | 0.0006047694335 | 0/0/25 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | below | v127 | original | supported | 0 | 785 | 0 | 0.001335342359 | 없음 | 3.473046021e-06 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | below | v127 | budget_matched | supported | 0 | 785 | 0 | 0.001335342359 | 없음 | 3.473046021e-06 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | below | v137 | original | supported | 0 | 785 | 0 | 0.001873047043 | 없음 | 1.102568981e-05 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | below | v137 | budget_matched | supported | 0 | 785 | 0 | 0.001873047043 | 없음 | 1.102568981e-05 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | capped | v127 | original | supported | 0 | 40500 | 25 | 0.004803197925 | 0.005046653655 | 0.0006163189906 | 0/0/25 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 40500 | 25 | 0.004803197925 | 0.005046653655 | 0.0006163189906 | 0/0/25 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | capped | v137 | original | supported | 0 | 40500 | 25 | 0.004831786413 | 0.00518567249 | 0.0006162777999 | 0/1/25 | 1 | 0 | 0 | 0 | 2.470660902e-05 |
| calibration/2021-03 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 40500 | 25 | 0.004831786413 | 0.00518567249 | 0.0006162777999 | 0/0/25 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | quiet | all | v127 | original | supported | 0 | 44162 | 23 | 0.00425363475 | 0.003536983 | 0.0005217332734 | 0/3/23 | 3 | 0 | 0 | 0 | 6.796710392e-05 |
| calibration/2021-03 | increase | quiet | all | v127 | budget_matched | supported | 0 | 44162 | 23 | 0.00425363475 | 0.003536983 | 0.0005217332734 | 0/3/23 | 3 | 0 | 0 | 0 | 6.796710392e-05 |
| calibration/2021-03 | increase | quiet | all | v137 | original | supported | 0 | 44162 | 23 | 0.004323271993 | 0.001945765854 | 0.0005221133627 | 0/10/23 | 10 | 0 | 0 | 0 | 0.0002265570131 |
| calibration/2021-03 | increase | quiet | all | v137 | budget_matched | supported | 0 | 44162 | 23 | 0.004323271993 | 0.001945765854 | 0.0005221133627 | 0/2/23 | 2 | 0 | 0 | 0 | 4.531140261e-05 |
| calibration/2021-03 | increase | quiet | below | v127 | original | supported | 0 | 3607 | 4 | 0.00718970312 | 0.01887747551 | 0.001110557132 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | quiet | below | v127 | budget_matched | supported | 0 | 3607 | 4 | 0.00718970312 | 0.01887747551 | 0.001110557132 | 0/1/4 | 1 | 0 | 0 | 0 | 0.000277546489 |
| calibration/2021-03 | increase | quiet | below | v137 | original | supported | 0 | 3607 | 4 | 0.007757975958 | 0.009738003857 | 0.001117231109 | 0/1/4 | 1 | 0 | 0 | 0 | 0.000277546489 |
| calibration/2021-03 | increase | quiet | below | v137 | budget_matched | supported | 0 | 3607 | 4 | 0.007757975958 | 0.009738003857 | 0.001117231109 | 0/1/4 | 1 | 0 | 0 | 0 | 0.000277546489 |
| calibration/2021-03 | increase | quiet | capped | v127 | original | supported | 0 | 40555 | 19 | 0.003992498056 | 0.001976210806 | 0.0004693627234 | 0/3/19 | 3 | 0 | 0 | 0 | 7.400828893e-05 |
| calibration/2021-03 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 40555 | 19 | 0.003992498056 | 0.001976210806 | 0.0004693627234 | 0/3/19 | 3 | 0 | 0 | 0 | 7.400828893e-05 |
| calibration/2021-03 | increase | quiet | capped | v137 | original | supported | 0 | 40555 | 19 | 0.004017786179 | 0.001499379055 | 0.0004691830283 | 0/9/19 | 9 | 0 | 0 | 0 | 0.0002220248668 |
| calibration/2021-03 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 40555 | 19 | 0.004017786179 | 0.001499379055 | 0.0004691830283 | 0/2/19 | 2 | 0 | 0 | 0 | 4.933885929e-05 |
| calibration/2021-03 | increase | first | all | v127 | original | supported | 0 | 2539 | 1 | 0.003803987359 | 0.003311258278 | 0.0004097094219 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | first | all | v127 | budget_matched | supported | 0 | 2539 | 1 | 0.003803987359 | 0.003311258278 | 0.0004097094219 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | first | all | v137 | original | supported | 0 | 2539 | 1 | 0.003697595987 | 0.002331002331 | 0.0004116151228 | 0/1/1 | 1 | 0 | 0 | 0 | 0.0003940110323 |
| calibration/2021-03 | increase | first | all | v137 | budget_matched | supported | 0 | 2539 | 1 | 0.003697595987 | 0.002331002331 | 0.0004116151228 | 0/1/1 | 1 | 0 | 0 | 0 | 0.0003940110323 |
| calibration/2021-03 | increase | first | below | v127 | original | supported | 0 | 2539 | 1 | 0.003803987359 | 0.003311258278 | 0.0004097094219 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | first | below | v127 | budget_matched | supported | 0 | 2539 | 1 | 0.003803987359 | 0.003311258278 | 0.0004097094219 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | first | below | v137 | original | supported | 0 | 2539 | 1 | 0.003697595987 | 0.002331002331 | 0.0004116151228 | 0/1/1 | 1 | 0 | 0 | 0 | 0.0003940110323 |
| calibration/2021-03 | increase | first | below | v137 | budget_matched | supported | 0 | 2539 | 1 | 0.003697595987 | 0.002331002331 | 0.0004116151228 | 0/1/1 | 1 | 0 | 0 | 0 | 0.0003940110323 |
| calibration/2021-03 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-03 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-03 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-03 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-03 | increase | restart | all | v127 | original | supported | 0 | 41623 | 22 | 0.004281063208 | 0.004163693276 | 0.0005285667203 | 0/3/22 | 3 | 0 | 0 | 0 | 7.211365111e-05 |
| calibration/2021-03 | increase | restart | all | v127 | budget_matched | supported | 0 | 41623 | 22 | 0.004281063208 | 0.004163693276 | 0.0005285667203 | 0/3/22 | 3 | 0 | 0 | 0 | 7.211365111e-05 |
| calibration/2021-03 | increase | restart | all | v137 | original | supported | 0 | 41623 | 22 | 0.004361438184 | 0.002171512462 | 0.0005288537474 | 0/9/22 | 9 | 0 | 0 | 0 | 0.0002163409533 |
| calibration/2021-03 | increase | restart | all | v137 | budget_matched | supported | 0 | 41623 | 22 | 0.004361438184 | 0.002171512462 | 0.0005288537474 | 0/2/22 | 2 | 0 | 0 | 0 | 4.807576741e-05 |
| calibration/2021-03 | increase | restart | below | v127 | original | supported | 0 | 1068 | 3 | 0.01523870342 | 0.05502755503 | 0.002776711004 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | below | v127 | budget_matched | supported | 0 | 1068 | 3 | 0.01523870342 | 0.05502755503 | 0.002776711004 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | below | v137 | original | supported | 0 | 1068 | 3 | 0.01741088302 | 0.04826653137 | 0.0027947208 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | below | v137 | budget_matched | supported | 0 | 1068 | 3 | 0.01741088302 | 0.04826653137 | 0.0027947208 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | capped | v127 | original | supported | 0 | 40555 | 19 | 0.003992498056 | 0.001976210806 | 0.0004693627234 | 0/3/19 | 3 | 0 | 0 | 0 | 7.400828893e-05 |
| calibration/2021-03 | increase | restart | capped | v127 | budget_matched | supported | 0 | 40555 | 19 | 0.003992498056 | 0.001976210806 | 0.0004693627234 | 0/3/19 | 3 | 0 | 0 | 0 | 7.400828893e-05 |
| calibration/2021-03 | increase | restart | capped | v137 | original | supported | 0 | 40555 | 19 | 0.004017786179 | 0.001499379055 | 0.0004691830283 | 0/9/19 | 9 | 0 | 0 | 0 | 0.0002220248668 |
| calibration/2021-03 | increase | restart | capped | v137 | budget_matched | supported | 0 | 40555 | 19 | 0.004017786179 | 0.001499379055 | 0.0004691830283 | 0/2/19 | 2 | 0 | 0 | 0 | 4.933885929e-05 |
| calibration/2021-04 | reduce | quiet | all | v127 | original | supported | 0 | 42516 | 36 | 0.005785570458 | 0.04990120278 | 0.0008351634561 | 1/2/35 | 3 | 0.3333333333 | 0.02777777778 | 0.05128205128 | 4.708097928e-05 |
| calibration/2021-04 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 42516 | 36 | 0.005785570458 | 0.04990120278 | 0.0008351634561 | 1/6/35 | 7 | 0.1428571429 | 0.02777777778 | 0.04651162791 | 0.0001412429379 |
| calibration/2021-04 | reduce | quiet | all | v137 | original | supported | 0 | 42516 | 36 | 0.005857152242 | 0.03490603777 | 0.0008380377493 | 2/7/34 | 9 | 0.2222222222 | 0.05555555556 | 0.08888888889 | 0.0001647834275 |
| calibration/2021-04 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 42516 | 36 | 0.005857152242 | 0.03490603777 | 0.0008380377493 | 2/4/34 | 6 | 0.3333333333 | 0.05555555556 | 0.09523809524 | 9.416195857e-05 |
| calibration/2021-04 | reduce | quiet | below | v127 | original | supported | 0 | 2228 | 2 | 0.006738427604 | 0.008309551788 | 0.0008966845586 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 2228 | 2 | 0.006738427604 | 0.008309551788 | 0.0008966845586 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | quiet | below | v137 | original | supported | 0 | 2228 | 2 | 0.006823008148 | 0.00558968059 | 0.0008968962016 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 2228 | 2 | 0.006823008148 | 0.00558968059 | 0.0008968962016 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | quiet | capped | v127 | original | supported | 0 | 40288 | 34 | 0.005732875718 | 0.05294331811 | 0.0008317612268 | 1/2/33 | 3 | 0.3333333333 | 0.02941176471 | 0.05405405405 | 4.96845034e-05 |
| calibration/2021-04 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 40288 | 34 | 0.005732875718 | 0.05294331811 | 0.0008317612268 | 1/4/33 | 5 | 0.2 | 0.02941176471 | 0.05128205128 | 9.936900681e-05 |
| calibration/2021-04 | reduce | quiet | capped | v137 | original | supported | 0 | 40288 | 34 | 0.005803738646 | 0.03720652072 | 0.0008347827693 | 2/7/32 | 9 | 0.2222222222 | 0.05882352941 | 0.09302325581 | 0.0001738957619 |
| calibration/2021-04 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 40288 | 34 | 0.005803738646 | 0.03720652072 | 0.0008347827693 | 1/3/33 | 4 | 0.25 | 0.02941176471 | 0.05263157895 | 7.452675511e-05 |
| calibration/2021-04 | reduce | first | all | v127 | original | supported | 0 | 3363 | 2 | 0.004889907607 | 0.00230539699 | 0.0005950695097 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | all | v127 | budget_matched | supported | 0 | 3363 | 2 | 0.004889907607 | 0.00230539699 | 0.0005950695097 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | all | v137 | original | supported | 0 | 3363 | 2 | 0.004717083814 | 0.003676000511 | 0.0005946278684 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | all | v137 | budget_matched | supported | 0 | 3363 | 2 | 0.004717083814 | 0.003676000511 | 0.0005946278684 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | below | v127 | original | supported | 0 | 2225 | 1 | 0.003970911105 | 0.001560062402 | 0.000450321145 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | below | v127 | budget_matched | supported | 0 | 2225 | 1 | 0.003970911105 | 0.001560062402 | 0.000450321145 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | below | v137 | original | supported | 0 | 2225 | 1 | 0.003930491054 | 0.001811594203 | 0.0004500793003 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | below | v137 | budget_matched | supported | 0 | 2225 | 1 | 0.003930491054 | 0.001811594203 | 0.0004500793003 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | capped | v127 | original | supported | 0 | 1138 | 1 | 0.006686715355 | 0.00641025641 | 0.0008780792737 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | capped | v127 | budget_matched | supported | 0 | 1138 | 1 | 0.006686715355 | 0.00641025641 | 0.0008780792737 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | capped | v137 | original | supported | 0 | 1138 | 1 | 0.006255017813 | 0.0125 | 0.000877246993 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | capped | v137 | budget_matched | supported | 0 | 1138 | 1 | 0.006255017813 | 0.0125 | 0.000877246993 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | all | v127 | original | supported | 0 | 39153 | 34 | 0.005862502345 | 0.05328473541 | 0.0008557860379 | 1/2/33 | 3 | 0.3333333333 | 0.02941176471 | 0.05405405405 | 5.112605128e-05 |
| calibration/2021-04 | reduce | restart | all | v127 | budget_matched | supported | 0 | 39153 | 34 | 0.005862502345 | 0.05328473541 | 0.0008557860379 | 1/2/33 | 3 | 0.3333333333 | 0.02941176471 | 0.05405405405 | 5.112605128e-05 |
| calibration/2021-04 | reduce | restart | all | v137 | original | supported | 0 | 39153 | 34 | 0.005955077053 | 0.03758980738 | 0.0008589451492 | 2/7/32 | 9 | 0.2222222222 | 0.05882352941 | 0.09302325581 | 0.0001789411795 |
| calibration/2021-04 | reduce | restart | all | v137 | budget_matched | supported | 0 | 39153 | 34 | 0.005955077053 | 0.03758980738 | 0.0008589451492 | 0/2/34 | 2 | 0 | 0 | 0 | 5.112605128e-05 |
| calibration/2021-04 | reduce | restart | below | v127 | original | supported | 0 | 3 | 1 | 2.059313164 | 0.3333333333 | 0.3319495496 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | below | v127 | budget_matched | supported | 0 | 3 | 1 | 2.059313164 | 0.3333333333 | 0.3319495496 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | below | v137 | original | supported | 0 | 3 | 1 | 2.152106519 | 0.3333333333 | 0.332286098 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | below | v137 | budget_matched | supported | 0 | 3 | 1 | 2.152106519 | 0.3333333333 | 0.332286098 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | capped | v127 | original | supported | 0 | 39150 | 33 | 0.005705149804 | 0.0545669498 | 0.0008304148682 | 1/2/32 | 3 | 0.3333333333 | 0.0303030303 | 0.05555555556 | 5.112866529e-05 |
| calibration/2021-04 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 39150 | 33 | 0.005705149804 | 0.0545669498 | 0.0008304148682 | 1/2/32 | 3 | 0.3333333333 | 0.0303030303 | 0.05555555556 | 5.112866529e-05 |
| calibration/2021-04 | reduce | restart | capped | v137 | original | supported | 0 | 39150 | 33 | 0.005790621004 | 0.03853917349 | 0.0008335484325 | 2/7/31 | 9 | 0.2222222222 | 0.06060606061 | 0.09523809524 | 0.0001789503285 |
| calibration/2021-04 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 39150 | 33 | 0.005790621004 | 0.03853917349 | 0.0008335484325 | 0/2/33 | 2 | 0 | 0 | 0 | 5.112866529e-05 |
| calibration/2021-04 | increase | quiet | all | v127 | original | supported | 0 | 42576 | 31 | 0.005115318687 | 0.01943812813 | 0.0007250961256 | 0/10/31 | 10 | 0 | 0 | 0 | 0.0002350452462 |
| calibration/2021-04 | increase | quiet | all | v127 | budget_matched | supported | 0 | 42576 | 31 | 0.005115318687 | 0.01943812813 | 0.0007250961256 | 1/3/30 | 4 | 0.25 | 0.03225806452 | 0.1063829787 | 7.051357386e-05 |
| calibration/2021-04 | increase | quiet | all | v137 | original | supported | 0 | 42576 | 31 | 0.005113081238 | 0.02680430826 | 0.0007228104797 | 1/45/30 | 46 | 0.02173913043 | 0.03225806452 | 0.02325581395 | 0.001057703608 |
| calibration/2021-04 | increase | quiet | all | v137 | budget_matched | supported | 0 | 42576 | 31 | 0.005113081238 | 0.02680430826 | 0.0007228104797 | 0/1/31 | 1 | 0 | 0 | 0 | 2.350452462e-05 |
| calibration/2021-04 | increase | quiet | below | v127 | original | supported | 0 | 2198 | 3 | 0.006552048683 | 0.3890137328 | 0.0012709325 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | quiet | below | v127 | budget_matched | supported | 0 | 2198 | 3 | 0.006552048683 | 0.3890137328 | 0.0012709325 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | quiet | below | v137 | original | supported | 0 | 2198 | 3 | 0.006614596856 | 0.5604575163 | 0.00127539958 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | quiet | below | v137 | budget_matched | supported | 0 | 2198 | 3 | 0.006614596856 | 0.5604575163 | 0.00127539958 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | quiet | capped | v127 | original | supported | 0 | 40378 | 28 | 0.005037109451 | 0.007019659384 | 0.000695383204 | 0/10/28 | 10 | 0 | 0 | 0 | 0.0002478314746 |
| calibration/2021-04 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 40378 | 28 | 0.005037109451 | 0.007019659384 | 0.000695383204 | 0/10/28 | 10 | 0 | 0 | 0 | 0.0002478314746 |
| calibration/2021-04 | increase | quiet | capped | v137 | original | supported | 0 | 40378 | 28 | 0.005031345359 | 0.007647286862 | 0.0006927299695 | 1/45/27 | 46 | 0.02173913043 | 0.03571428571 | 0.02358490566 | 0.001115241636 |
| calibration/2021-04 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 40378 | 28 | 0.005031345359 | 0.007647286862 | 0.0006927299695 | 0/5/28 | 5 | 0 | 0 | 0 | 0.0001239157373 |
| calibration/2021-04 | increase | first | all | v127 | original | supported | 0 | 4 | 2 | 1.74279178 | 0.75 | 0.4482476009 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | all | v127 | budget_matched | supported | 0 | 4 | 2 | 1.74279178 | 0.75 | 0.4482476009 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | all | v137 | original | supported | 0 | 4 | 2 | 1.617582574 | 0.8333333333 | 0.4506542521 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | all | v137 | budget_matched | supported | 0 | 4 | 2 | 1.617582574 | 0.8333333333 | 0.4506542521 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | below | v127 | original | supported | 0 | 4 | 2 | 1.74279178 | 0.75 | 0.4482476009 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | below | v127 | budget_matched | supported | 0 | 4 | 2 | 1.74279178 | 0.75 | 0.4482476009 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | below | v137 | original | supported | 0 | 4 | 2 | 1.617582574 | 0.8333333333 | 0.4506542521 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | below | v137 | budget_matched | supported | 0 | 4 | 2 | 1.617582574 | 0.8333333333 | 0.4506542521 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-04 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-04 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-04 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-04 | increase | restart | all | v127 | original | supported | 0 | 42572 | 29 | 0.004952049265 | 0.006882709965 | 0.0006830475956 | 0/10/29 | 10 | 0 | 0 | 0 | 0.000235056296 |
| calibration/2021-04 | increase | restart | all | v127 | budget_matched | supported | 0 | 42572 | 29 | 0.004952049265 | 0.006882709965 | 0.0006830475956 | 0/10/29 | 10 | 0 | 0 | 0 | 0.000235056296 |
| calibration/2021-04 | increase | restart | all | v137 | original | supported | 0 | 42572 | 29 | 0.004961576071 | 0.007350249199 | 0.0006805356097 | 1/45/28 | 46 | 0.02173913043 | 0.03448275862 | 0.0234741784 | 0.001057753332 |
| calibration/2021-04 | increase | restart | all | v137 | budget_matched | supported | 0 | 42572 | 29 | 0.004961576071 | 0.007350249199 | 0.0006805356097 | 0/5/29 | 5 | 0 | 0 | 0 | 0.000117528148 |
| calibration/2021-04 | increase | restart | below | v127 | original | supported | 0 | 2194 | 1 | 0.003386616174 | 0.01176470588 | 0.0004560251735 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | below | v127 | budget_matched | supported | 0 | 2194 | 1 | 0.003386616174 | 0.01176470588 | 0.0004560251735 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | below | v137 | original | supported | 0 | 2194 | 1 | 0.003677554053 | 0.005 | 0.0004561127022 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | below | v137 | budget_matched | supported | 0 | 2194 | 1 | 0.003677554053 | 0.005 | 0.0004561127022 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | capped | v127 | original | supported | 0 | 40378 | 28 | 0.005037109451 | 0.007019659384 | 0.000695383204 | 0/10/28 | 10 | 0 | 0 | 0 | 0.0002478314746 |
| calibration/2021-04 | increase | restart | capped | v127 | budget_matched | supported | 0 | 40378 | 28 | 0.005037109451 | 0.007019659384 | 0.000695383204 | 0/10/28 | 10 | 0 | 0 | 0 | 0.0002478314746 |
| calibration/2021-04 | increase | restart | capped | v137 | original | supported | 0 | 40378 | 28 | 0.005031345359 | 0.007647286862 | 0.0006927299695 | 1/45/27 | 46 | 0.02173913043 | 0.03571428571 | 0.02358490566 | 0.001115241636 |
| calibration/2021-04 | increase | restart | capped | v137 | budget_matched | supported | 0 | 40378 | 28 | 0.005031345359 | 0.007647286862 | 0.0006927299695 | 0/5/28 | 5 | 0 | 0 | 0 | 0.0001239157373 |
| calibration/2021-05 | reduce | quiet | all | v127 | original | supported | 0 | 43202 | 75 | 0.01045826146 | 0.03616458073 | 0.001710636721 | 4/37/71 | 41 | 0.09756097561 | 0.05333333333 | 0.06896551724 | 0.0008579312264 |
| calibration/2021-05 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 43202 | 75 | 0.01045826146 | 0.03616458073 | 0.001710636721 | 1/28/74 | 29 | 0.03448275862 | 0.01333333333 | 0.01923076923 | 0.0006492452524 |
| calibration/2021-05 | reduce | quiet | all | v137 | original | supported | 0 | 43202 | 75 | 0.01062704694 | 0.03588881571 | 0.001712438596 | 9/195/66 | 204 | 0.04411764706 | 0.12 | 0.06451612903 | 0.004521529436 |
| calibration/2021-05 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 43202 | 75 | 0.01062704694 | 0.03588881571 | 0.001712438596 | 3/36/72 | 39 | 0.07692307692 | 0.04 | 0.05263157895 | 0.0008347438959 |
| calibration/2021-05 | reduce | quiet | below | v127 | original | supported | 0 | 26564 | 25 | 0.006253334997 | 0.02931704549 | 0.0009310360912 | 3/22/22 | 25 | 0.12 | 0.12 | 0.12 | 0.0008289686876 |
| calibration/2021-05 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 26564 | 25 | 0.006253334997 | 0.02931704549 | 0.0009310360912 | 1/12/24 | 13 | 0.07692307692 | 0.04 | 0.05263157895 | 0.0004521647387 |
| calibration/2021-05 | reduce | quiet | below | v137 | original | supported | 0 | 26564 | 25 | 0.006331059416 | 0.01687091479 | 0.0009339761695 | 5/137/20 | 142 | 0.03521126761 | 0.2 | 0.05988023952 | 0.0051622141 |
| calibration/2021-05 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 26564 | 25 | 0.006331059416 | 0.01687091479 | 0.0009339761695 | 0/11/25 | 11 | 0 | 0 | 0 | 0.0004144843438 |
| calibration/2021-05 | reduce | quiet | capped | v127 | original | supported | 0 | 16638 | 50 | 0.01717178872 | 0.04644681905 | 0.002955336272 | 1/15/49 | 16 | 0.0625 | 0.02 | 0.0303030303 | 0.0009042681456 |
| calibration/2021-05 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 16638 | 50 | 0.01717178872 | 0.04644681905 | 0.002955336272 | 1/22/49 | 23 | 0.04347826087 | 0.02 | 0.02739726027 | 0.001326259947 |
| calibration/2021-05 | reduce | quiet | capped | v137 | original | supported | 0 | 16638 | 50 | 0.01748596103 | 0.0493998289 | 0.002955320908 | 4/58/46 | 62 | 0.06451612903 | 0.08 | 0.07142857143 | 0.003496503497 |
| calibration/2021-05 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 16638 | 50 | 0.01748596103 | 0.0493998289 | 0.002955320908 | 3/32/47 | 35 | 0.08571428571 | 0.06 | 0.07058823529 | 0.001929105377 |
| calibration/2021-05 | reduce | first | all | v127 | original | supported | 0 | 22026 | 21 | 0.006085552 | 0.0468593461 | 0.0009391531977 | 3/22/18 | 25 | 0.12 | 0.1428571429 | 0.1304347826 | 0.0009997727789 |
| calibration/2021-05 | reduce | first | all | v127 | budget_matched | supported | 0 | 22026 | 21 | 0.006085552 | 0.0468593461 | 0.0009391531977 | 3/22/18 | 25 | 0.12 | 0.1428571429 | 0.1304347826 | 0.0009997727789 |
| calibration/2021-05 | reduce | first | all | v137 | original | supported | 0 | 22026 | 21 | 0.006145280174 | 0.02121907548 | 0.0009438387894 | 5/137/16 | 142 | 0.03521126761 | 0.2380952381 | 0.06134969325 | 0.00622585776 |
| calibration/2021-05 | reduce | first | all | v137 | budget_matched | supported | 0 | 22026 | 21 | 0.006145280174 | 0.02121907548 | 0.0009438387894 | 2/33/19 | 35 | 0.05714285714 | 0.09523809524 | 0.07142857143 | 0.001499659168 |
| calibration/2021-05 | reduce | first | below | v127 | original | supported | 0 | 21113 | 18 | 0.005603866582 | 0.04783874558 | 0.0008406989578 | 3/22/15 | 25 | 0.12 | 0.1666666667 | 0.1395348837 | 0.001042901161 |
| calibration/2021-05 | reduce | first | below | v127 | budget_matched | supported | 0 | 21113 | 18 | 0.005603866582 | 0.04783874558 | 0.0008406989578 | 3/22/15 | 25 | 0.12 | 0.1666666667 | 0.1395348837 | 0.001042901161 |
| calibration/2021-05 | reduce | first | below | v137 | original | supported | 0 | 21113 | 18 | 0.005602579234 | 0.02198285184 | 0.000843976863 | 5/136/13 | 141 | 0.03546099291 | 0.2777777778 | 0.06289308176 | 0.006447025361 |
| calibration/2021-05 | reduce | first | below | v137 | budget_matched | supported | 0 | 21113 | 18 | 0.005602579234 | 0.02198285184 | 0.000843976863 | 2/35/16 | 37 | 0.05405405405 | 0.1111111111 | 0.07272727273 | 0.001659160939 |
| calibration/2021-05 | reduce | first | capped | v127 | original | supported | 0 | 913 | 3 | 0.01722446135 | 0.08639665419 | 0.003215894015 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | first | capped | v127 | budget_matched | supported | 0 | 913 | 3 | 0.01722446135 | 0.08639665419 | 0.003215894015 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | first | capped | v137 | original | supported | 0 | 913 | 3 | 0.01869516512 | 0.09393939394 | 0.003253132165 | 0/1/3 | 1 | 0 | 0 | 0 | 0.001098901099 |
| calibration/2021-05 | reduce | first | capped | v137 | budget_matched | supported | 0 | 913 | 3 | 0.01869516512 | 0.09393939394 | 0.003253132165 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | restart | all | v127 | original | supported | 0 | 21176 | 54 | 0.01500649052 | 0.04128617741 | 0.002513087424 | 1/15/53 | 16 | 0.0625 | 0.01851851852 | 0.02857142857 | 0.0007101600227 |
| calibration/2021-05 | reduce | restart | all | v127 | budget_matched | supported | 0 | 21176 | 54 | 0.01500649052 | 0.04128617741 | 0.002513087424 | 1/15/53 | 16 | 0.0625 | 0.01851851852 | 0.02857142857 | 0.0007101600227 |
| calibration/2021-05 | reduce | restart | all | v137 | original | supported | 0 | 21176 | 54 | 0.01528871084 | 0.04337158232 | 0.002511889831 | 4/58/50 | 62 | 0.06451612903 | 0.07407407407 | 0.06896551724 | 0.002745952088 |
| calibration/2021-05 | reduce | restart | all | v137 | budget_matched | supported | 0 | 21176 | 54 | 0.01528871084 | 0.04337158232 | 0.002511889831 | 2/19/52 | 21 | 0.09523809524 | 0.03703703704 | 0.05333333333 | 0.0008995360288 |
| calibration/2021-05 | reduce | restart | below | v127 | original | supported | 0 | 5451 | 7 | 0.008768878318 | 0.01176485219 | 0.001280932972 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | restart | below | v127 | budget_matched | supported | 0 | 5451 | 7 | 0.008768878318 | 0.01176485219 | 0.001280932972 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | restart | below | v137 | original | supported | 0 | 5451 | 7 | 0.00915263382 | 0.009075573509 | 0.001282564568 | 0/1/7 | 1 | 0 | 0 | 0 | 0.0001836884644 |
| calibration/2021-05 | reduce | restart | below | v137 | budget_matched | supported | 0 | 5451 | 7 | 0.00915263382 | 0.009075573509 | 0.001282564568 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | restart | capped | v127 | original | supported | 0 | 15725 | 47 | 0.01716873052 | 0.04659253983 | 0.002940208182 | 1/15/46 | 16 | 0.0625 | 0.02127659574 | 0.03174603175 | 0.0009567546881 |
| calibration/2021-05 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 15725 | 47 | 0.01716873052 | 0.04659253983 | 0.002940208182 | 1/15/46 | 16 | 0.0625 | 0.02127659574 | 0.03174603175 | 0.0009567546881 |
| calibration/2021-05 | reduce | restart | capped | v137 | original | supported | 0 | 15725 | 47 | 0.01741575414 | 0.05080443704 | 0.002938029864 | 4/57/43 | 61 | 0.06557377049 | 0.08510638298 | 0.07407407407 | 0.003635667815 |
| calibration/2021-05 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 15725 | 47 | 0.01741575414 | 0.05080443704 | 0.002938029864 | 3/20/44 | 23 | 0.1304347826 | 0.06382978723 | 0.08571428571 | 0.001275672917 |
| calibration/2021-05 | increase | quiet | all | v127 | original | supported | 0 | 43226 | 68 | 0.009835972738 | 0.04180221479 | 0.001540961174 | 3/13/65 | 16 | 0.1875 | 0.04411764706 | 0.1136363636 | 0.0003012187775 |
| calibration/2021-05 | increase | quiet | all | v127 | budget_matched | supported | 0 | 43226 | 68 | 0.009835972738 | 0.04180221479 | 0.001540961174 | 3/13/65 | 16 | 0.1875 | 0.04411764706 | 0.1136363636 | 0.0003012187775 |
| calibration/2021-05 | increase | quiet | all | v137 | original | supported | 0 | 43226 | 68 | 0.009948757484 | 0.05723756769 | 0.001535142753 | 10/125/58 | 135 | 0.07407407407 | 0.1470588235 | 0.08223684211 | 0.002896334399 |
| calibration/2021-05 | increase | quiet | all | v137 | budget_matched | supported | 0 | 43226 | 68 | 0.009948757484 | 0.05723756769 | 0.001535142753 | 3/17/65 | 20 | 0.15 | 0.04411764706 | 0.1013513514 | 0.0003939014783 |
| calibration/2021-05 | increase | quiet | below | v127 | original | supported | 0 | 26427 | 34 | 0.007704093968 | 0.06645134058 | 0.001245992356 | 3/13/31 | 16 | 0.1875 | 0.08823529412 | 0.1530612245 | 0.0004925548441 |
| calibration/2021-05 | increase | quiet | below | v127 | budget_matched | supported | 0 | 26427 | 34 | 0.007704093968 | 0.06645134058 | 0.001245992356 | 3/9/31 | 12 | 0.25 | 0.08823529412 | 0.1829268293 | 0.0003409995074 |
| calibration/2021-05 | increase | quiet | below | v137 | original | supported | 0 | 26427 | 34 | 0.007754689804 | 0.09597613536 | 0.0012373158 | 6/63/28 | 69 | 0.08695652174 | 0.1764705882 | 0.09677419355 | 0.002386996552 |
| calibration/2021-05 | increase | quiet | below | v137 | budget_matched | supported | 0 | 26427 | 34 | 0.007754689804 | 0.09597613536 | 0.0012373158 | 3/11/31 | 14 | 0.2142857143 | 0.08823529412 | 0.1666666667 | 0.0004167771758 |
| calibration/2021-05 | increase | quiet | capped | v127 | original | supported | 0 | 16799 | 34 | 0.01318969381 | 0.02146196086 | 0.002004985281 | 0/0/34 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 16799 | 34 | 0.01318969381 | 0.02146196086 | 0.002004985281 | 0/0/34 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | quiet | capped | v137 | original | supported | 0 | 16799 | 34 | 0.01340030975 | 0.02110790357 | 0.002003663075 | 4/62/30 | 66 | 0.06060606061 | 0.1176470588 | 0.06711409396 | 0.003698180734 |
| calibration/2021-05 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 16799 | 34 | 0.01340030975 | 0.02110790357 | 0.002003663075 | 0/6/34 | 6 | 0 | 0 | 0 | 0.0003578884581 |
| calibration/2021-05 | increase | first | all | v127 | original | supported | 0 | 5124 | 16 | 0.01536900085 | 0.1288442668 | 0.002935131921 | 3/7/13 | 10 | 0.3 | 0.1875 | 0.2678571429 | 0.001370399374 |
| calibration/2021-05 | increase | first | all | v127 | budget_matched | supported | 0 | 5124 | 16 | 0.01536900085 | 0.1288442668 | 0.002935131921 | 3/7/13 | 10 | 0.3 | 0.1875 | 0.2678571429 | 0.001370399374 |
| calibration/2021-05 | increase | first | all | v137 | original | supported | 0 | 5124 | 16 | 0.0153886203 | 0.1825216968 | 0.002901242943 | 3/7/13 | 10 | 0.3 | 0.1875 | 0.2678571429 | 0.001370399374 |
| calibration/2021-05 | increase | first | all | v137 | budget_matched | supported | 0 | 5124 | 16 | 0.0153886203 | 0.1825216968 | 0.002901242943 | 3/9/13 | 12 | 0.25 | 0.1875 | 0.234375 | 0.001761942052 |
| calibration/2021-05 | increase | first | below | v127 | original | supported | 0 | 5124 | 16 | 0.01536900085 | 0.1288442668 | 0.002935131921 | 3/7/13 | 10 | 0.3 | 0.1875 | 0.2678571429 | 0.001370399374 |
| calibration/2021-05 | increase | first | below | v127 | budget_matched | supported | 0 | 5124 | 16 | 0.01536900085 | 0.1288442668 | 0.002935131921 | 3/7/13 | 10 | 0.3 | 0.1875 | 0.2678571429 | 0.001370399374 |
| calibration/2021-05 | increase | first | below | v137 | original | supported | 0 | 5124 | 16 | 0.0153886203 | 0.1825216968 | 0.002901242943 | 3/7/13 | 10 | 0.3 | 0.1875 | 0.2678571429 | 0.001370399374 |
| calibration/2021-05 | increase | first | below | v137 | budget_matched | supported | 0 | 5124 | 16 | 0.0153886203 | 0.1825216968 | 0.002901242943 | 3/9/13 | 12 | 0.25 | 0.1875 | 0.234375 | 0.001761942052 |
| calibration/2021-05 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-05 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-05 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-05 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-05 | increase | restart | all | v127 | original | supported | 0 | 38102 | 52 | 0.009091884867 | 0.01671365113 | 0.001353471517 | 0/6/52 | 6 | 0 | 0 | 0 | 0.0001576872536 |
| calibration/2021-05 | increase | restart | all | v127 | budget_matched | supported | 0 | 38102 | 52 | 0.009091884867 | 0.01671365113 | 0.001353471517 | 0/6/52 | 6 | 0 | 0 | 0 | 0.0001576872536 |
| calibration/2021-05 | increase | restart | all | v137 | original | supported | 0 | 38102 | 52 | 0.009217198587 | 0.02070983433 | 0.001351428056 | 7/118/45 | 125 | 0.056 | 0.1346153846 | 0.0634057971 | 0.003101182654 |
| calibration/2021-05 | increase | restart | all | v137 | budget_matched | supported | 0 | 38102 | 52 | 0.009217198587 | 0.02070983433 | 0.001351428056 | 1/10/51 | 11 | 0.09090909091 | 0.01923076923 | 0.05208333333 | 0.0002628120894 |
| calibration/2021-05 | increase | restart | below | v127 | original | supported | 0 | 21303 | 18 | 0.005860457724 | 0.01999816503 | 0.0008397044558 | 0/6/18 | 6 | 0 | 0 | 0 | 0.000281888654 |
| calibration/2021-05 | increase | restart | below | v127 | budget_matched | supported | 0 | 21303 | 18 | 0.005860457724 | 0.01999816503 | 0.0008397044558 | 0/6/18 | 6 | 0 | 0 | 0 | 0.000281888654 |
| calibration/2021-05 | increase | restart | below | v137 | original | supported | 0 | 21303 | 18 | 0.005918504296 | 0.0259148831 | 0.0008370922314 | 3/56/15 | 59 | 0.05084745763 | 0.1666666667 | 0.05905511811 | 0.00263096077 |
| calibration/2021-05 | increase | restart | below | v137 | budget_matched | supported | 0 | 21303 | 18 | 0.005918504296 | 0.0259148831 | 0.0008370922314 | 1/4/17 | 5 | 0.2 | 0.05555555556 | 0.1315789474 | 0.0001879257693 |
| calibration/2021-05 | increase | restart | capped | v127 | original | supported | 0 | 16799 | 34 | 0.01318969381 | 0.02146196086 | 0.002004985281 | 0/0/34 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | restart | capped | v127 | budget_matched | supported | 0 | 16799 | 34 | 0.01318969381 | 0.02146196086 | 0.002004985281 | 0/0/34 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | restart | capped | v137 | original | supported | 0 | 16799 | 34 | 0.01340030975 | 0.02110790357 | 0.002003663075 | 4/62/30 | 66 | 0.06060606061 | 0.1176470588 | 0.06711409396 | 0.003698180734 |
| calibration/2021-05 | increase | restart | capped | v137 | budget_matched | supported | 0 | 16799 | 34 | 0.01340030975 | 0.02110790357 | 0.002003663075 | 0/6/34 | 6 | 0 | 0 | 0 | 0.0003578884581 |
| calibration/2021-06 | reduce | quiet | all | v127 | original | supported | 0 | 40583 | 69 | 0.01092245655 | 0.01256573786 | 0.001694397293 | 1/47/68 | 48 | 0.02083333333 | 0.01449275362 | 0.01709401709 | 0.001160092807 |
| calibration/2021-06 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 40583 | 69 | 0.01092245655 | 0.01256573786 | 0.001694397293 | 0/30/69 | 30 | 0 | 0 | 0 | 0.0007404847707 |
| calibration/2021-06 | reduce | quiet | all | v137 | original | supported | 0 | 40583 | 69 | 0.0109873655 | 0.01223571194 | 0.001694417334 | 4/138/65 | 142 | 0.02816901408 | 0.05797101449 | 0.03791469194 | 0.003406229945 |
| calibration/2021-06 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 40583 | 69 | 0.0109873655 | 0.01223571194 | 0.001694417334 | 0/27/69 | 27 | 0 | 0 | 0 | 0.0006664362936 |
| calibration/2021-06 | reduce | quiet | below | v127 | original | supported | 0 | 9911 | 9 | 0.005740605289 | 0.0249652285 | 0.0009021720855 | 1/10/8 | 11 | 0.09090909091 | 0.1111111111 | 0.1 | 0.001009896991 |
| calibration/2021-06 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 9911 | 9 | 0.005740605289 | 0.0249652285 | 0.0009021720855 | 0/8/9 | 8 | 0 | 0 | 0 | 0.0008079175924 |
| calibration/2021-06 | reduce | quiet | below | v137 | original | supported | 0 | 9911 | 9 | 0.005615028223 | 0.04236204598 | 0.0008967407846 | 2/33/7 | 35 | 0.05714285714 | 0.2222222222 | 0.09090909091 | 0.003332660069 |
| calibration/2021-06 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 9911 | 9 | 0.005615028223 | 0.04236204598 | 0.0008967407846 | 1/6/8 | 7 | 0.1428571429 | 0.1111111111 | 0.125 | 0.0006059381943 |
| calibration/2021-06 | reduce | quiet | capped | v127 | original | supported | 0 | 30672 | 60 | 0.01259686082 | 0.01249466279 | 0.001950387905 | 0/37/60 | 37 | 0 | 0 | 0 | 0.001208676336 |
| calibration/2021-06 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 30672 | 60 | 0.01259686082 | 0.01249466279 | 0.001950387905 | 0/20/60 | 20 | 0 | 0 | 0 | 0.00065333856 |
| calibration/2021-06 | reduce | quiet | capped | v137 | original | supported | 0 | 30672 | 60 | 0.01272332125 | 0.01167829603 | 0.001952169429 | 2/105/58 | 107 | 0.01869158879 | 0.03333333333 | 0.02395209581 | 0.00343002744 |
| calibration/2021-06 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 30672 | 60 | 0.01272332125 | 0.01167829603 | 0.001952169429 | 0/22/60 | 22 | 0 | 0 | 0 | 0.000718672416 |
| calibration/2021-06 | reduce | first | all | v127 | original | supported | 0 | 17813 | 17 | 0.006517740656 | 0.0124902143 | 0.0009541791749 | 1/40/16 | 41 | 0.0243902439 | 0.05882352941 | 0.03448275862 | 0.002247696111 |
| calibration/2021-06 | reduce | first | all | v127 | budget_matched | supported | 0 | 17813 | 17 | 0.006517740656 | 0.0124902143 | 0.0009541791749 | 1/40/16 | 41 | 0.0243902439 | 0.05882352941 | 0.03448275862 | 0.002247696111 |
| calibration/2021-06 | reduce | first | all | v137 | original | supported | 0 | 17813 | 17 | 0.006435769617 | 0.01508664914 | 0.0009505107563 | 4/112/13 | 116 | 0.03448275862 | 0.2352941176 | 0.06015037594 | 0.006293549112 |
| calibration/2021-06 | reduce | first | all | v137 | budget_matched | supported | 0 | 17813 | 17 | 0.006435769617 | 0.01508664914 | 0.0009505107563 | 1/34/16 | 35 | 0.02857142857 | 0.05882352941 | 0.03846153846 | 0.001910541695 |
| calibration/2021-06 | reduce | first | below | v127 | original | supported | 0 | 9632 | 7 | 0.004519805471 | 0.06577944939 | 0.000716711005 | 1/9/6 | 10 | 0.1 | 0.1428571429 | 0.1176470588 | 0.0009350649351 |
| calibration/2021-06 | reduce | first | below | v127 | budget_matched | supported | 0 | 9632 | 7 | 0.004519805471 | 0.06577944939 | 0.000716711005 | 1/9/6 | 10 | 0.1 | 0.1428571429 | 0.1176470588 | 0.0009350649351 |
| calibration/2021-06 | reduce | first | below | v137 | original | supported | 0 | 9632 | 7 | 0.004425186279 | 0.0605515399 | 0.0007129867227 | 2/30/5 | 32 | 0.0625 | 0.2857142857 | 0.1025641026 | 0.003116883117 |
| calibration/2021-06 | reduce | first | below | v137 | budget_matched | supported | 0 | 9632 | 7 | 0.004425186279 | 0.0605515399 | 0.0007129867227 | 1/6/6 | 7 | 0.1428571429 | 0.1428571429 | 0.1428571429 | 0.0006233766234 |
| calibration/2021-06 | reduce | first | capped | v127 | original | supported | 0 | 8181 | 10 | 0.008870033982 | 0.007246615642 | 0.001233765217 | 0/31/10 | 31 | 0 | 0 | 0 | 0.003793905275 |
| calibration/2021-06 | reduce | first | capped | v127 | budget_matched | supported | 0 | 8181 | 10 | 0.008870033982 | 0.007246615642 | 0.001233765217 | 0/31/10 | 31 | 0 | 0 | 0 | 0.003793905275 |
| calibration/2021-06 | reduce | first | capped | v137 | original | supported | 0 | 8181 | 10 | 0.008802954401 | 0.008559318222 | 0.001230162571 | 2/82/8 | 84 | 0.02380952381 | 0.2 | 0.04255319149 | 0.01003549137 |
| calibration/2021-06 | reduce | first | capped | v137 | budget_matched | supported | 0 | 8181 | 10 | 0.008802954401 | 0.008559318222 | 0.001230162571 | 0/27/10 | 27 | 0 | 0 | 0 | 0.00330436911 |
| calibration/2021-06 | reduce | restart | all | v127 | original | supported | 0 | 22770 | 52 | 0.0143682714 | 0.01353609662 | 0.00227347087 | 0/7/52 | 7 | 0 | 0 | 0 | 0.0003081257153 |
| calibration/2021-06 | reduce | restart | all | v127 | budget_matched | supported | 0 | 22770 | 52 | 0.0143682714 | 0.01353609662 | 0.00227347087 | 0/7/52 | 7 | 0 | 0 | 0 | 0.0003081257153 |
| calibration/2021-06 | reduce | restart | all | v137 | original | supported | 0 | 22770 | 52 | 0.01454808476 | 0.01249739505 | 0.002276376396 | 0/26/52 | 26 | 0 | 0 | 0 | 0.001144466943 |
| calibration/2021-06 | reduce | restart | all | v137 | budget_matched | supported | 0 | 22770 | 52 | 0.01454808476 | 0.01249739505 | 0.002276376396 | 0/6/52 | 6 | 0 | 0 | 0 | 0.000264107756 |
| calibration/2021-06 | reduce | restart | below | v127 | original | supported | 0 | 279 | 2 | 0.04788664059 | 0.008418482103 | 0.007304900141 | 0/1/2 | 1 | 0 | 0 | 0 | 0.003610108303 |
| calibration/2021-06 | reduce | restart | below | v127 | budget_matched | supported | 0 | 279 | 2 | 0.04788664059 | 0.008418482103 | 0.007304900141 | 0/1/2 | 1 | 0 | 0 | 0 | 0.003610108303 |
| calibration/2021-06 | reduce | restart | below | v137 | original | supported | 0 | 279 | 2 | 0.04669229562 | 0.008964429145 | 0.007240536929 | 0/3/2 | 3 | 0 | 0 | 0 | 0.01083032491 |
| calibration/2021-06 | reduce | restart | below | v137 | budget_matched | supported | 0 | 279 | 2 | 0.04669229562 | 0.008964429145 | 0.007240536929 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | reduce | restart | capped | v127 | original | supported | 0 | 22491 | 50 | 0.01395247731 | 0.01475378443 | 0.002211056181 | 0/6/50 | 6 | 0 | 0 | 0 | 0.0002673677644 |
| calibration/2021-06 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 22491 | 50 | 0.01395247731 | 0.01475378443 | 0.002211056181 | 0/6/50 | 6 | 0 | 0 | 0 | 0.0002673677644 |
| calibration/2021-06 | reduce | restart | capped | v137 | original | supported | 0 | 22491 | 50 | 0.01414933704 | 0.01357476064 | 0.002214796174 | 0/23/50 | 23 | 0 | 0 | 0 | 0.001024909763 |
| calibration/2021-06 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 22491 | 50 | 0.01414933704 | 0.01357476064 | 0.002214796174 | 0/8/50 | 8 | 0 | 0 | 0 | 0.0003564903525 |
| calibration/2021-06 | increase | quiet | all | v127 | original | supported | 0 | 40278 | 78 | 0.01199832831 | 0.02053898064 | 0.001923953936 | 1/24/77 | 25 | 0.04 | 0.01282051282 | 0.02808988764 | 0.0005970149254 |
| calibration/2021-06 | increase | quiet | all | v127 | budget_matched | supported | 0 | 40278 | 78 | 0.01199832831 | 0.02053898064 | 0.001923953936 | 1/25/77 | 26 | 0.03846153846 | 0.01282051282 | 0.02747252747 | 0.0006218905473 |
| calibration/2021-06 | increase | quiet | all | v137 | original | supported | 0 | 40278 | 78 | 0.01205107109 | 0.02097973338 | 0.001923542787 | 4/68/74 | 72 | 0.05555555556 | 0.05128205128 | 0.05464480874 | 0.001691542289 |
| calibration/2021-06 | increase | quiet | all | v137 | budget_matched | supported | 0 | 40278 | 78 | 0.01205107109 | 0.02097973338 | 0.001923542787 | 0/17/78 | 17 | 0 | 0 | 0 | 0.0004228855721 |
| calibration/2021-06 | increase | quiet | below | v127 | original | supported | 0 | 9575 | 29 | 0.01733278358 | 0.02763028507 | 0.00302096106 | 0/14/29 | 14 | 0 | 0 | 0 | 0.001466582862 |
| calibration/2021-06 | increase | quiet | below | v127 | budget_matched | supported | 0 | 9575 | 29 | 0.01733278358 | 0.02763028507 | 0.00302096106 | 0/17/29 | 17 | 0 | 0 | 0 | 0.001780850618 |
| calibration/2021-06 | increase | quiet | below | v137 | original | supported | 0 | 9575 | 29 | 0.01737043292 | 0.02866148601 | 0.003019522295 | 1/23/28 | 24 | 0.04166666667 | 0.03448275862 | 0.04 | 0.00240938613 |
| calibration/2021-06 | increase | quiet | below | v137 | budget_matched | supported | 0 | 9575 | 29 | 0.01737043292 | 0.02866148601 | 0.003019522295 | 0/11/29 | 11 | 0 | 0 | 0 | 0.001152315106 |
| calibration/2021-06 | increase | quiet | capped | v127 | original | supported | 0 | 30703 | 49 | 0.01033473161 | 0.02616326963 | 0.001581842637 | 1/10/48 | 11 | 0.09090909091 | 0.02040816327 | 0.05376344086 | 0.0003262217003 |
| calibration/2021-06 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 30703 | 49 | 0.01033473161 | 0.02616326963 | 0.001581842637 | 1/10/48 | 11 | 0.09090909091 | 0.02040816327 | 0.05376344086 | 0.0003262217003 |
| calibration/2021-06 | increase | quiet | capped | v137 | original | supported | 0 | 30703 | 49 | 0.01039218141 | 0.01886464327 | 0.001581751959 | 3/45/46 | 48 | 0.0625 | 0.0612244898 | 0.0622406639 | 0.001467997651 |
| calibration/2021-06 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 30703 | 49 | 0.01039218141 | 0.01886464327 | 0.001581751959 | 0/5/49 | 5 | 0 | 0 | 0 | 0.0001631108501 |
| calibration/2021-06 | increase | first | all | v127 | original | supported | 0 | 4632 | 14 | 0.01727981453 | 0.02721809472 | 0.003049794812 | 0/12/14 | 12 | 0 | 0 | 0 | 0.002598527501 |
| calibration/2021-06 | increase | first | all | v127 | budget_matched | supported | 0 | 4632 | 14 | 0.01727981453 | 0.02721809472 | 0.003049794812 | 0/12/14 | 12 | 0 | 0 | 0 | 0.002598527501 |
| calibration/2021-06 | increase | first | all | v137 | original | supported | 0 | 4632 | 14 | 0.01662611836 | 0.03140005121 | 0.003042322394 | 0/7/14 | 7 | 0 | 0 | 0 | 0.001515807709 |
| calibration/2021-06 | increase | first | all | v137 | budget_matched | supported | 0 | 4632 | 14 | 0.01662611836 | 0.03140005121 | 0.003042322394 | 0/8/14 | 8 | 0 | 0 | 0 | 0.001732351667 |
| calibration/2021-06 | increase | first | below | v127 | original | supported | 0 | 4632 | 14 | 0.01727981453 | 0.02721809472 | 0.003049794812 | 0/12/14 | 12 | 0 | 0 | 0 | 0.002598527501 |
| calibration/2021-06 | increase | first | below | v127 | budget_matched | supported | 0 | 4632 | 14 | 0.01727981453 | 0.02721809472 | 0.003049794812 | 0/12/14 | 12 | 0 | 0 | 0 | 0.002598527501 |
| calibration/2021-06 | increase | first | below | v137 | original | supported | 0 | 4632 | 14 | 0.01662611836 | 0.03140005121 | 0.003042322394 | 0/7/14 | 7 | 0 | 0 | 0 | 0.001515807709 |
| calibration/2021-06 | increase | first | below | v137 | budget_matched | supported | 0 | 4632 | 14 | 0.01662611836 | 0.03140005121 | 0.003042322394 | 0/8/14 | 8 | 0 | 0 | 0 | 0.001732351667 |
| calibration/2021-06 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-06 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-06 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-06 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-06 | increase | restart | all | v127 | original | supported | 0 | 35646 | 64 | 0.01131202846 | 0.02676477204 | 0.001777657159 | 1/12/63 | 13 | 0.07692307692 | 0.015625 | 0.04310344828 | 0.0003372491709 |
| calibration/2021-06 | increase | restart | all | v127 | budget_matched | supported | 0 | 35646 | 64 | 0.01131202846 | 0.02676477204 | 0.001777657159 | 1/12/63 | 13 | 0.07692307692 | 0.015625 | 0.04310344828 | 0.0003372491709 |
| calibration/2021-06 | increase | restart | all | v137 | original | supported | 0 | 35646 | 64 | 0.01145656907 | 0.02030866112 | 0.001778163582 | 4/61/60 | 65 | 0.06153846154 | 0.0625 | 0.06172839506 | 0.001714349952 |
| calibration/2021-06 | increase | restart | all | v137 | budget_matched | supported | 0 | 35646 | 64 | 0.01145656907 | 0.02030866112 | 0.001778163582 | 0/7/64 | 7 | 0 | 0 | 0 | 0.000196728683 |
| calibration/2021-06 | increase | restart | below | v127 | original | supported | 0 | 4943 | 15 | 0.01738241997 | 0.03700984399 | 0.002993941449 | 0/2/15 | 2 | 0 | 0 | 0 | 0.0004058441558 |
| calibration/2021-06 | increase | restart | below | v127 | budget_matched | supported | 0 | 4943 | 15 | 0.01738241997 | 0.03700984399 | 0.002993941449 | 0/2/15 | 2 | 0 | 0 | 0 | 0.0004058441558 |
| calibration/2021-06 | increase | restart | below | v137 | original | supported | 0 | 4943 | 15 | 0.01806791725 | 0.03416826876 | 0.002998156716 | 1/16/14 | 17 | 0.05882352941 | 0.06666666667 | 0.06024096386 | 0.003246753247 |
| calibration/2021-06 | increase | restart | below | v137 | budget_matched | supported | 0 | 4943 | 15 | 0.01806791725 | 0.03416826876 | 0.002998156716 | 0/2/15 | 2 | 0 | 0 | 0 | 0.0004058441558 |
| calibration/2021-06 | increase | restart | capped | v127 | original | supported | 0 | 30703 | 49 | 0.01033473161 | 0.02616326963 | 0.001581842637 | 1/10/48 | 11 | 0.09090909091 | 0.02040816327 | 0.05376344086 | 0.0003262217003 |
| calibration/2021-06 | increase | restart | capped | v127 | budget_matched | supported | 0 | 30703 | 49 | 0.01033473161 | 0.02616326963 | 0.001581842637 | 1/10/48 | 11 | 0.09090909091 | 0.02040816327 | 0.05376344086 | 0.0003262217003 |
| calibration/2021-06 | increase | restart | capped | v137 | original | supported | 0 | 30703 | 49 | 0.01039218141 | 0.01886464327 | 0.001581751959 | 3/45/46 | 48 | 0.0625 | 0.0612244898 | 0.0622406639 | 0.001467997651 |
| calibration/2021-06 | increase | restart | capped | v137 | budget_matched | supported | 0 | 30703 | 49 | 0.01039218141 | 0.01886464327 | 0.001581751959 | 0/5/49 | 5 | 0 | 0 | 0 | 0.0001631108501 |
| diagnosis/2021-07 | reduce | quiet | all | v127 | original | supported | 0 | 42679 | 28 | 0.004768863509 | 0.06284724385 | 0.0006455478205 | 2/3/26 | 5 | 0.4 | 0.07142857143 | 0.1212121212 | 7.033832735e-05 |
| diagnosis/2021-07 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 42679 | 28 | 0.004768863509 | 0.06284724385 | 0.0006455478205 | 1/9/27 | 10 | 0.1 | 0.03571428571 | 0.05263157895 | 0.0002110149821 |
| diagnosis/2021-07 | reduce | quiet | all | v137 | original | supported | 0 | 42679 | 28 | 0.004763831243 | 0.02499046596 | 0.0006506306864 | 3/42/25 | 45 | 0.06666666667 | 0.1071428571 | 0.08219178082 | 0.000984736583 |
| diagnosis/2021-07 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 42679 | 28 | 0.004763831243 | 0.02499046596 | 0.0006506306864 | 1/16/27 | 17 | 0.05882352941 | 0.03571428571 | 0.04444444444 | 0.0003751377459 |
| diagnosis/2021-07 | reduce | quiet | below | v127 | original | supported | 0 | 18073 | 10 | 0.004126326462 | 0.1241029614 | 0.0005397875274 | 2/3/8 | 5 | 0.4 | 0.2 | 0.2666666667 | 0.0001660853679 |
| diagnosis/2021-07 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 18073 | 10 | 0.004126326462 | 0.1241029614 | 0.0005397875274 | 2/8/8 | 10 | 0.2 | 0.2 | 0.2 | 0.0004428943143 |
| diagnosis/2021-07 | reduce | quiet | below | v137 | original | supported | 0 | 18073 | 10 | 0.004157838306 | 0.0285278384 | 0.0005518814096 | 2/24/8 | 26 | 0.07692307692 | 0.2 | 0.1111111111 | 0.001328682943 |
| diagnosis/2021-07 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 18073 | 10 | 0.004157838306 | 0.0285278384 | 0.0005518814096 | 1/19/9 | 20 | 0.05 | 0.1 | 0.06666666667 | 0.001051873997 |
| diagnosis/2021-07 | reduce | quiet | capped | v127 | original | supported | 0 | 24606 | 18 | 0.005240804176 | 0.02922004776 | 0.000723228296 | 0/0/18 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 24606 | 18 | 0.005240804176 | 0.02922004776 | 0.000723228296 | 0/2/18 | 2 | 0 | 0 | 0 | 8.13404913e-05 |
| diagnosis/2021-07 | reduce | quiet | capped | v137 | original | supported | 0 | 24606 | 18 | 0.00520893042 | 0.0300245391 | 0.0007231616008 | 1/18/17 | 19 | 0.05263157895 | 0.05555555556 | 0.05405405405 | 0.0007320644217 |
| diagnosis/2021-07 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 24606 | 18 | 0.00520893042 | 0.0300245391 | 0.0007231616008 | 0/2/18 | 2 | 0 | 0 | 0 | 8.13404913e-05 |
| diagnosis/2021-07 | reduce | first | all | v127 | original | supported | 0 | 14413 | 7 | 0.003482698595 | 0.2874269931 | 0.0004752315242 | 1/0/6 | 1 | 1 | 0.1428571429 | 0.25 | 0 |
| diagnosis/2021-07 | reduce | first | all | v127 | budget_matched | supported | 0 | 14413 | 7 | 0.003482698595 | 0.2874269931 | 0.0004752315242 | 1/0/6 | 1 | 1 | 0.1428571429 | 0.25 | 0 |
| diagnosis/2021-07 | reduce | first | all | v137 | original | supported | 0 | 14413 | 7 | 0.003563554256 | 0.07143442816 | 0.0004802988244 | 2/11/5 | 13 | 0.1538461538 | 0.2857142857 | 0.2 | 0.0007635707344 |
| diagnosis/2021-07 | reduce | first | all | v137 | budget_matched | supported | 0 | 14413 | 7 | 0.003563554256 | 0.07143442816 | 0.0004802988244 | 0/3/7 | 3 | 0 | 0 | 0 | 0.0002082465639 |
| diagnosis/2021-07 | reduce | first | below | v127 | original | supported | 0 | 9405 | 4 | 0.003146129599 | 0.2512393818 | 0.0004154041946 | 1/0/3 | 1 | 1 | 0.25 | 0.4 | 0 |
| diagnosis/2021-07 | reduce | first | below | v127 | budget_matched | supported | 0 | 9405 | 4 | 0.003146129599 | 0.2512393818 | 0.0004154041946 | 1/0/3 | 1 | 1 | 0.25 | 0.4 | 0 |
| diagnosis/2021-07 | reduce | first | below | v137 | original | supported | 0 | 9405 | 4 | 0.003162382883 | 0.1271217095 | 0.0004212984867 | 1/5/3 | 6 | 0.1666666667 | 0.25 | 0.2 | 0.0005318583129 |
| diagnosis/2021-07 | reduce | first | below | v137 | budget_matched | supported | 0 | 9405 | 4 | 0.003162382883 | 0.1271217095 | 0.0004212984867 | 0/1/4 | 1 | 0 | 0 | 0 | 0.0001063716626 |
| diagnosis/2021-07 | reduce | first | capped | v127 | original | supported | 0 | 5008 | 3 | 0.004114773556 | 0.3373206668 | 0.0005875869626 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | first | capped | v127 | budget_matched | supported | 0 | 5008 | 3 | 0.004114773556 | 0.3373206668 | 0.0005875869626 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | first | capped | v137 | original | supported | 0 | 5008 | 3 | 0.004316952172 | 0.08614152791 | 0.0005911011759 | 1/6/2 | 7 | 0.1428571429 | 0.3333333333 | 0.2 | 0.001198801199 |
| diagnosis/2021-07 | reduce | first | capped | v137 | budget_matched | supported | 0 | 5008 | 3 | 0.004316952172 | 0.08614152791 | 0.0005911011759 | 0/1/3 | 1 | 0 | 0 | 0 | 0.0001998001998 |
| diagnosis/2021-07 | reduce | restart | all | v127 | original | supported | 0 | 28266 | 21 | 0.005424686579 | 0.06532925814 | 0.0007323931039 | 1/3/20 | 4 | 0.25 | 0.04761904762 | 0.08 | 0.0001062134891 |
| diagnosis/2021-07 | reduce | restart | all | v127 | budget_matched | supported | 0 | 28266 | 21 | 0.005424686579 | 0.06532925814 | 0.0007323931039 | 1/3/20 | 4 | 0.25 | 0.04761904762 | 0.08 | 0.0001062134891 |
| diagnosis/2021-07 | reduce | restart | all | v137 | original | supported | 0 | 28266 | 21 | 0.005375859553 | 0.02549293409 | 0.0007374839067 | 1/31/20 | 32 | 0.03125 | 0.04761904762 | 0.03773584906 | 0.001097539388 |
| diagnosis/2021-07 | reduce | restart | all | v137 | budget_matched | supported | 0 | 28266 | 21 | 0.005375859553 | 0.02549293409 | 0.0007374839067 | 0/4/21 | 4 | 0 | 0 | 0 | 0.0001416179855 |
| diagnosis/2021-07 | reduce | restart | below | v127 | original | supported | 0 | 8668 | 6 | 0.005189864936 | 0.1690858497 | 0.0006747466005 | 1/3/5 | 4 | 0.25 | 0.1666666667 | 0.2 | 0.0003463403371 |
| diagnosis/2021-07 | reduce | restart | below | v127 | budget_matched | supported | 0 | 8668 | 6 | 0.005189864936 | 0.1690858497 | 0.0006747466005 | 1/3/5 | 4 | 0.25 | 0.1666666667 | 0.2 | 0.0003463403371 |
| diagnosis/2021-07 | reduce | restart | below | v137 | original | supported | 0 | 8668 | 6 | 0.005237932705 | 0.0378315902 | 0.0006935671952 | 1/19/5 | 20 | 0.05 | 0.1666666667 | 0.07692307692 | 0.002193488802 |
| diagnosis/2021-07 | reduce | restart | below | v137 | budget_matched | supported | 0 | 8668 | 6 | 0.005237932705 | 0.0378315902 | 0.0006935671952 | 0/4/6 | 4 | 0 | 0 | 0 | 0.0004617871161 |
| diagnosis/2021-07 | reduce | restart | capped | v127 | original | supported | 0 | 19598 | 15 | 0.005528545851 | 0.02628908025 | 0.0007578895776 | 0/0/15 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 19598 | 15 | 0.005528545851 | 0.02628908025 | 0.0007578895776 | 0/0/15 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | restart | capped | v137 | original | supported | 0 | 19598 | 15 | 0.005436863223 | 0.02903934072 | 0.0007569078304 | 0/12/15 | 12 | 0 | 0 | 0 | 0.0006127763877 |
| diagnosis/2021-07 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 19598 | 15 | 0.005436863223 | 0.02903934072 | 0.0007569078304 | 0/0/15 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | quiet | all | v127 | original | supported | 0 | 42564 | 33 | 0.005210052889 | 0.02239356599 | 0.0007687967296 | 0/4/33 | 4 | 0 | 0 | 0 | 9.404904658e-05 |
| diagnosis/2021-07 | increase | quiet | all | v127 | budget_matched | supported | 0 | 42564 | 33 | 0.005210052889 | 0.02239356599 | 0.0007687967296 | 1/7/32 | 8 | 0.125 | 0.0303030303 | 0.07692307692 | 0.0001645858315 |
| diagnosis/2021-07 | increase | quiet | all | v137 | original | supported | 0 | 42564 | 33 | 0.005397406371 | 0.01271254017 | 0.0007755803199 | 0/19/33 | 19 | 0 | 0 | 0 | 0.0004467329712 |
| diagnosis/2021-07 | increase | quiet | all | v137 | budget_matched | supported | 0 | 42564 | 33 | 0.005397406371 | 0.01271254017 | 0.0007755803199 | 0/8/33 | 8 | 0 | 0 | 0 | 0.0001880980932 |
| diagnosis/2021-07 | increase | quiet | below | v127 | original | supported | 0 | 18025 | 14 | 0.005505517696 | 0.02895093865 | 0.0007696940812 | 0/4/14 | 4 | 0 | 0 | 0 | 0.0002220865027 |
| diagnosis/2021-07 | increase | quiet | below | v127 | budget_matched | supported | 0 | 18025 | 14 | 0.005505517696 | 0.02895093865 | 0.0007696940812 | 0/7/14 | 7 | 0 | 0 | 0 | 0.0003886513797 |
| diagnosis/2021-07 | increase | quiet | below | v137 | original | supported | 0 | 18025 | 14 | 0.00561202152 | 0.01653670267 | 0.0007816445614 | 0/7/14 | 7 | 0 | 0 | 0 | 0.0003886513797 |
| diagnosis/2021-07 | increase | quiet | below | v137 | budget_matched | supported | 0 | 18025 | 14 | 0.00561202152 | 0.01653670267 | 0.0007816445614 | 0/6/14 | 6 | 0 | 0 | 0 | 0.000333129754 |
| diagnosis/2021-07 | increase | quiet | capped | v127 | original | supported | 0 | 24539 | 19 | 0.004993020689 | 0.01947707803 | 0.0007681375844 | 0/0/19 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 24539 | 19 | 0.004993020689 | 0.01947707803 | 0.0007681375844 | 0/0/19 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | quiet | capped | v137 | original | supported | 0 | 24539 | 19 | 0.005239761885 | 0.01192797203 | 0.0007711258615 | 0/12/19 | 12 | 0 | 0 | 0 | 0.0004893964111 |
| diagnosis/2021-07 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 24539 | 19 | 0.005239761885 | 0.01192797203 | 0.0007711258615 | 0/0/19 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | first | all | v127 | original | supported | 0 | 3107 | 5 | 0.01028737306 | 0.06157257818 | 0.001578938531 | 0/4/5 | 4 | 0 | 0 | 0 | 0.001289490651 |
| diagnosis/2021-07 | increase | first | all | v127 | budget_matched | supported | 0 | 3107 | 5 | 0.01028737306 | 0.06157257818 | 0.001578938531 | 0/4/5 | 4 | 0 | 0 | 0 | 0.001289490651 |
| diagnosis/2021-07 | increase | first | all | v137 | original | supported | 0 | 3107 | 5 | 0.01067815688 | 0.03671903793 | 0.00164639498 | 0/3/5 | 3 | 0 | 0 | 0 | 0.0009671179884 |
| diagnosis/2021-07 | increase | first | all | v137 | budget_matched | supported | 0 | 3107 | 5 | 0.01067815688 | 0.03671903793 | 0.00164639498 | 0/4/5 | 4 | 0 | 0 | 0 | 0.001289490651 |
| diagnosis/2021-07 | increase | first | below | v127 | original | supported | 0 | 3107 | 5 | 0.01028737306 | 0.06157257818 | 0.001578938531 | 0/4/5 | 4 | 0 | 0 | 0 | 0.001289490651 |
| diagnosis/2021-07 | increase | first | below | v127 | budget_matched | supported | 0 | 3107 | 5 | 0.01028737306 | 0.06157257818 | 0.001578938531 | 0/4/5 | 4 | 0 | 0 | 0 | 0.001289490651 |
| diagnosis/2021-07 | increase | first | below | v137 | original | supported | 0 | 3107 | 5 | 0.01067815688 | 0.03671903793 | 0.00164639498 | 0/3/5 | 3 | 0 | 0 | 0 | 0.0009671179884 |
| diagnosis/2021-07 | increase | first | below | v137 | budget_matched | supported | 0 | 3107 | 5 | 0.01067815688 | 0.03671903793 | 0.00164639498 | 0/4/5 | 4 | 0 | 0 | 0 | 0.001289490651 |
| diagnosis/2021-07 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-07 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-07 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-07 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-07 | increase | restart | all | v127 | original | supported | 0 | 39457 | 28 | 0.004810244647 | 0.01483413639 | 0.0007050029648 | 0/0/28 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | all | v127 | budget_matched | supported | 0 | 39457 | 28 | 0.004810244647 | 0.01483413639 | 0.0007050029648 | 0/0/28 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | all | v137 | original | supported | 0 | 39457 | 28 | 0.004981579222 | 0.009413792966 | 0.0007070089346 | 0/16/28 | 16 | 0 | 0 | 0 | 0.0004057926907 |
| diagnosis/2021-07 | increase | restart | all | v137 | budget_matched | supported | 0 | 39457 | 28 | 0.004981579222 | 0.009413792966 | 0.0007070089346 | 0/0/28 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | below | v127 | original | supported | 0 | 14918 | 9 | 0.004509591659 | 0.013049572 | 0.0006011512133 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | below | v127 | budget_matched | supported | 0 | 14918 | 9 | 0.004509591659 | 0.013049572 | 0.0006011512133 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | below | v137 | original | supported | 0 | 14918 | 9 | 0.004556887952 | 0.008791583361 | 0.0006015413605 | 0/4/9 | 4 | 0 | 0 | 0 | 0.0002682943189 |
| diagnosis/2021-07 | increase | restart | below | v137 | budget_matched | supported | 0 | 14918 | 9 | 0.004556887952 | 0.008791583361 | 0.0006015413605 | 0/0/9 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | capped | v127 | original | supported | 0 | 24539 | 19 | 0.004993020689 | 0.01947707803 | 0.0007681375844 | 0/0/19 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | capped | v127 | budget_matched | supported | 0 | 24539 | 19 | 0.004993020689 | 0.01947707803 | 0.0007681375844 | 0/0/19 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | capped | v137 | original | supported | 0 | 24539 | 19 | 0.005239761885 | 0.01192797203 | 0.0007711258615 | 0/12/19 | 12 | 0 | 0 | 0 | 0.0004893964111 |
| diagnosis/2021-07 | increase | restart | capped | v137 | budget_matched | supported | 0 | 24539 | 19 | 0.005239761885 | 0.01192797203 | 0.0007711258615 | 0/0/19 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | quiet | all | v127 | original | supported | 0 | 43613 | 51 | 0.007687525962 | 0.01819928523 | 0.001160784911 | 0/5/51 | 5 | 0 | 0 | 0 | 0.0001147789358 |
| diagnosis/2021-08 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 43613 | 51 | 0.007687525962 | 0.01819928523 | 0.001160784911 | 0/3/51 | 3 | 0 | 0 | 0 | 6.886736146e-05 |
| diagnosis/2021-08 | reduce | quiet | all | v137 | original | supported | 0 | 43613 | 51 | 0.007835144141 | 0.01856410463 | 0.0011619633 | 0/16/51 | 16 | 0 | 0 | 0 | 0.0003672925945 |
| diagnosis/2021-08 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 43613 | 51 | 0.007835144141 | 0.01856410463 | 0.0011619633 | 0/5/51 | 5 | 0 | 0 | 0 | 0.0001147789358 |
| diagnosis/2021-08 | reduce | quiet | below | v127 | original | supported | 0 | 4991 | 5 | 0.005848800335 | 0.03775341755 | 0.0009921146056 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 4991 | 5 | 0.005848800335 | 0.03775341755 | 0.0009921146056 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | quiet | below | v137 | original | supported | 0 | 4991 | 5 | 0.005889540005 | 0.04346151631 | 0.0009926205547 | 0/2/5 | 2 | 0 | 0 | 0 | 0.0004011231448 |
| diagnosis/2021-08 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 4991 | 5 | 0.005889540005 | 0.04346151631 | 0.0009926205547 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | quiet | capped | v127 | original | supported | 0 | 38622 | 46 | 0.007925138712 | 0.01871452574 | 0.001182581646 | 0/5/46 | 5 | 0 | 0 | 0 | 0.0001296142679 |
| diagnosis/2021-08 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 38622 | 46 | 0.007925138712 | 0.01871452574 | 0.001182581646 | 0/2/46 | 2 | 0 | 0 | 0 | 5.184570718e-05 |
| diagnosis/2021-08 | reduce | quiet | capped | v137 | original | supported | 0 | 38622 | 46 | 0.008086568465 | 0.01858235231 | 0.001183846932 | 0/14/46 | 14 | 0 | 0 | 0 | 0.0003629199502 |
| diagnosis/2021-08 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 38622 | 46 | 0.008086568465 | 0.01858235231 | 0.001183846932 | 0/5/46 | 5 | 0 | 0 | 0 | 0.0001296142679 |
| diagnosis/2021-08 | reduce | first | all | v127 | original | supported | 0 | 9966 | 7 | 0.004847873615 | 0.01236849261 | 0.0006999795844 | 0/3/7 | 3 | 0 | 0 | 0 | 0.0003012350638 |
| diagnosis/2021-08 | reduce | first | all | v127 | budget_matched | supported | 0 | 9966 | 7 | 0.004847873615 | 0.01236849261 | 0.0006999795844 | 0/3/7 | 3 | 0 | 0 | 0 | 0.0003012350638 |
| diagnosis/2021-08 | reduce | first | all | v137 | original | supported | 0 | 9966 | 7 | 0.004923099302 | 0.01141449544 | 0.0007020949177 | 0/11/7 | 11 | 0 | 0 | 0 | 0.001104528567 |
| diagnosis/2021-08 | reduce | first | all | v137 | budget_matched | supported | 0 | 9966 | 7 | 0.004923099302 | 0.01141449544 | 0.0007020949177 | 0/4/7 | 4 | 0 | 0 | 0 | 0.0004016467517 |
| diagnosis/2021-08 | reduce | first | below | v127 | original | supported | 0 | 4080 | 3 | 0.004455331173 | 0.0359354226 | 0.0007284808972 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | first | below | v127 | budget_matched | supported | 0 | 4080 | 3 | 0.004455331173 | 0.0359354226 | 0.0007284808972 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | first | below | v137 | original | supported | 0 | 4080 | 3 | 0.004263984315 | 0.05659309564 | 0.0007267748207 | 0/2/3 | 2 | 0 | 0 | 0 | 0.0004905567819 |
| diagnosis/2021-08 | reduce | first | below | v137 | budget_matched | supported | 0 | 4080 | 3 | 0.004263984315 | 0.05659309564 | 0.0007267748207 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | first | capped | v127 | original | supported | 0 | 5886 | 4 | 0.005119972351 | 0.005058895109 | 0.0006802233227 | 0/3/4 | 3 | 0 | 0 | 0 | 0.0005100306018 |
| diagnosis/2021-08 | reduce | first | capped | v127 | budget_matched | supported | 0 | 5886 | 4 | 0.005119972351 | 0.005058895109 | 0.0006802233227 | 0/3/4 | 3 | 0 | 0 | 0 | 0.0005100306018 |
| diagnosis/2021-08 | reduce | first | capped | v137 | original | supported | 0 | 5886 | 4 | 0.005379978191 | 0.003096112901 | 0.0006849875436 | 0/9/4 | 9 | 0 | 0 | 0 | 0.001530091806 |
| diagnosis/2021-08 | reduce | first | capped | v137 | budget_matched | supported | 0 | 5886 | 4 | 0.005379978191 | 0.003096112901 | 0.0006849875436 | 0/3/4 | 3 | 0 | 0 | 0 | 0.0005100306018 |
| diagnosis/2021-08 | reduce | restart | all | v127 | original | supported | 0 | 33647 | 44 | 0.008528610615 | 0.02184439376 | 0.001297272143 | 0/2/44 | 2 | 0 | 0 | 0 | 5.951849537e-05 |
| diagnosis/2021-08 | reduce | restart | all | v127 | budget_matched | supported | 0 | 33647 | 44 | 0.008528610615 | 0.02184439376 | 0.001297272143 | 0/2/44 | 2 | 0 | 0 | 0 | 5.951849537e-05 |
| diagnosis/2021-08 | reduce | restart | all | v137 | original | supported | 0 | 33647 | 44 | 0.00869767093 | 0.02203726391 | 0.001298173016 | 0/5/44 | 5 | 0 | 0 | 0 | 0.0001487962384 |
| diagnosis/2021-08 | reduce | restart | all | v137 | budget_matched | supported | 0 | 33647 | 44 | 0.00869767093 | 0.02203726391 | 0.001298173016 | 0/1/44 | 1 | 0 | 0 | 0 | 2.975924769e-05 |
| diagnosis/2021-08 | reduce | restart | below | v127 | original | supported | 0 | 911 | 2 | 0.01208958429 | 0.07472826087 | 0.002172823201 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | restart | below | v127 | budget_matched | supported | 0 | 911 | 2 | 0.01208958429 | 0.07472826087 | 0.002172823201 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | restart | below | v137 | original | supported | 0 | 911 | 2 | 0.01316974551 | 0.04598308668 | 0.002183235917 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | restart | below | v137 | budget_matched | supported | 0 | 911 | 2 | 0.01316974551 | 0.04598308668 | 0.002183235917 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | restart | capped | v127 | original | supported | 0 | 32736 | 42 | 0.008429513382 | 0.02221708497 | 0.001272906704 | 0/2/42 | 2 | 0 | 0 | 0 | 6.117330397e-05 |
| diagnosis/2021-08 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 32736 | 42 | 0.008429513382 | 0.02221708497 | 0.001272906704 | 0/2/42 | 2 | 0 | 0 | 0 | 6.117330397e-05 |
| diagnosis/2021-08 | reduce | restart | capped | v137 | original | supported | 0 | 32736 | 42 | 0.008573218952 | 0.02346525698 | 0.001273542874 | 0/5/42 | 5 | 0 | 0 | 0 | 0.0001529332599 |
| diagnosis/2021-08 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 32736 | 42 | 0.008573218952 | 0.02346525698 | 0.001273542874 | 0/1/42 | 1 | 0 | 0 | 0 | 3.058665199e-05 |
| diagnosis/2021-08 | increase | quiet | all | v127 | original | supported | 0 | 43461 | 60 | 0.008946173146 | 0.01788071769 | 0.001372712459 | 0/3/60 | 3 | 0 | 0 | 0 | 6.912283127e-05 |
| diagnosis/2021-08 | increase | quiet | all | v127 | budget_matched | supported | 0 | 43461 | 60 | 0.008946173146 | 0.01788071769 | 0.001372712459 | 0/6/60 | 6 | 0 | 0 | 0 | 0.0001382456625 |
| diagnosis/2021-08 | increase | quiet | all | v137 | original | supported | 0 | 43461 | 60 | 0.008944750148 | 0.02015445451 | 0.001372553353 | 1/9/59 | 10 | 0.1 | 0.01666666667 | 0.05 | 0.0002073684938 |
| diagnosis/2021-08 | increase | quiet | all | v137 | budget_matched | supported | 0 | 43461 | 60 | 0.008944750148 | 0.02015445451 | 0.001372553353 | 0/5/60 | 5 | 0 | 0 | 0 | 0.0001152047188 |
| diagnosis/2021-08 | increase | quiet | below | v127 | original | supported | 0 | 4782 | 20 | 0.02389477079 | 0.02995751025 | 0.004153363131 | 0/3/20 | 3 | 0 | 0 | 0 | 0.0006299874003 |
| diagnosis/2021-08 | increase | quiet | below | v127 | budget_matched | supported | 0 | 4782 | 20 | 0.02389477079 | 0.02995751025 | 0.004153363131 | 0/3/20 | 3 | 0 | 0 | 0 | 0.0006299874003 |
| diagnosis/2021-08 | increase | quiet | below | v137 | original | supported | 0 | 4782 | 20 | 0.02409842836 | 0.03571717184 | 0.00415088619 | 1/5/19 | 6 | 0.1666666667 | 0.05 | 0.1136363636 | 0.001049979 |
| diagnosis/2021-08 | increase | quiet | below | v137 | budget_matched | supported | 0 | 4782 | 20 | 0.02409842836 | 0.03571717184 | 0.00415088619 | 0/5/20 | 5 | 0 | 0 | 0 | 0.001049979 |
| diagnosis/2021-08 | increase | quiet | capped | v127 | original | supported | 0 | 38679 | 40 | 0.007098033485 | 0.0127911383 | 0.001028932333 | 0/0/40 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 38679 | 40 | 0.007098033485 | 0.0127911383 | 0.001028932333 | 0/0/40 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | quiet | capped | v137 | original | supported | 0 | 38679 | 40 | 0.007071255765 | 0.01303734677 | 0.001029059787 | 0/4/40 | 4 | 0 | 0 | 0 | 0.0001035223479 |
| diagnosis/2021-08 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 38679 | 40 | 0.007071255765 | 0.01303734677 | 0.001029059787 | 0/0/40 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | first | all | v127 | original | supported | 0 | 1045 | 6 | 0.0317800048 | 0.03473564616 | 0.005777614755 | 0/3/6 | 3 | 0 | 0 | 0 | 0.002887391723 |
| diagnosis/2021-08 | increase | first | all | v127 | budget_matched | supported | 0 | 1045 | 6 | 0.0317800048 | 0.03473564616 | 0.005777614755 | 0/3/6 | 3 | 0 | 0 | 0 | 0.002887391723 |
| diagnosis/2021-08 | increase | first | all | v137 | original | supported | 0 | 1045 | 6 | 0.02971616903 | 0.04245911117 | 0.005757058707 | 0/3/6 | 3 | 0 | 0 | 0 | 0.002887391723 |
| diagnosis/2021-08 | increase | first | all | v137 | budget_matched | supported | 0 | 1045 | 6 | 0.02971616903 | 0.04245911117 | 0.005757058707 | 0/4/6 | 4 | 0 | 0 | 0 | 0.00384985563 |
| diagnosis/2021-08 | increase | first | below | v127 | original | supported | 0 | 1045 | 6 | 0.0317800048 | 0.03473564616 | 0.005777614755 | 0/3/6 | 3 | 0 | 0 | 0 | 0.002887391723 |
| diagnosis/2021-08 | increase | first | below | v127 | budget_matched | supported | 0 | 1045 | 6 | 0.0317800048 | 0.03473564616 | 0.005777614755 | 0/3/6 | 3 | 0 | 0 | 0 | 0.002887391723 |
| diagnosis/2021-08 | increase | first | below | v137 | original | supported | 0 | 1045 | 6 | 0.02971616903 | 0.04245911117 | 0.005757058707 | 0/3/6 | 3 | 0 | 0 | 0 | 0.002887391723 |
| diagnosis/2021-08 | increase | first | below | v137 | budget_matched | supported | 0 | 1045 | 6 | 0.02971616903 | 0.04245911117 | 0.005757058707 | 0/4/6 | 4 | 0 | 0 | 0 | 0.00384985563 |
| diagnosis/2021-08 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-08 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-08 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-08 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-08 | increase | restart | all | v127 | original | supported | 0 | 42416 | 54 | 0.008383617646 | 0.02475828751 | 0.001264189192 | 0/0/54 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | all | v127 | budget_matched | supported | 0 | 42416 | 54 | 0.008383617646 | 0.02475828751 | 0.001264189192 | 0/0/54 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | all | v137 | original | supported | 0 | 42416 | 54 | 0.008433006166 | 0.02277719867 | 0.001264532604 | 1/6/53 | 7 | 0.1428571429 | 0.01851851852 | 0.06097560976 | 0.0001416363722 |
| diagnosis/2021-08 | increase | restart | all | v137 | budget_matched | supported | 0 | 42416 | 54 | 0.008433006166 | 0.02277719867 | 0.001264532604 | 0/0/54 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | below | v127 | original | supported | 0 | 3737 | 14 | 0.02168977493 | 0.06219242338 | 0.003699163787 | 0/0/14 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | below | v127 | budget_matched | supported | 0 | 3737 | 14 | 0.02168977493 | 0.06219242338 | 0.003699163787 | 0/0/14 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | below | v137 | original | supported | 0 | 3737 | 14 | 0.02252750543 | 0.0713246866 | 0.003701742417 | 1/2/13 | 3 | 0.3333333333 | 0.07142857143 | 0.1923076923 | 0.0005372011818 |
| diagnosis/2021-08 | increase | restart | below | v137 | budget_matched | supported | 0 | 3737 | 14 | 0.02252750543 | 0.0713246866 | 0.003701742417 | 0/0/14 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | capped | v127 | original | supported | 0 | 38679 | 40 | 0.007098033485 | 0.0127911383 | 0.001028932333 | 0/0/40 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | capped | v127 | budget_matched | supported | 0 | 38679 | 40 | 0.007098033485 | 0.0127911383 | 0.001028932333 | 0/0/40 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | capped | v137 | original | supported | 0 | 38679 | 40 | 0.007071255765 | 0.01303734677 | 0.001029059787 | 0/4/40 | 4 | 0 | 0 | 0 | 0.0001035223479 |
| diagnosis/2021-08 | increase | restart | capped | v137 | budget_matched | supported | 0 | 38679 | 40 | 0.007071255765 | 0.01303734677 | 0.001029059787 | 0/0/40 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | quiet | all | v127 | original | supported | 0 | 42563 | 38 | 0.00612488626 | 0.02792644792 | 0.0008821978344 | 0/6/38 | 6 | 0 | 0 | 0 | 0.0001410934744 |
| diagnosis/2021-09 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 42563 | 38 | 0.00612488626 | 0.02792644792 | 0.0008821978344 | 0/4/38 | 4 | 0 | 0 | 0 | 9.406231628e-05 |
| diagnosis/2021-09 | reduce | quiet | all | v137 | original | supported | 0 | 42563 | 38 | 0.006114636226 | 0.02636699134 | 0.0008832701977 | 3/54/35 | 57 | 0.05263157895 | 0.07894736842 | 0.06315789474 | 0.00126984127 |
| diagnosis/2021-09 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 42563 | 38 | 0.006114636226 | 0.02636699134 | 0.0008832701977 | 1/10/37 | 11 | 0.09090909091 | 0.02631578947 | 0.04081632653 | 0.0002351557907 |
| diagnosis/2021-09 | reduce | quiet | below | v127 | original | supported | 0 | 9766 | 11 | 0.007793298266 | 0.01145823124 | 0.001121825354 | 0/5/11 | 5 | 0 | 0 | 0 | 0.0005125576627 |
| diagnosis/2021-09 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 9766 | 11 | 0.007793298266 | 0.01145823124 | 0.001121825354 | 0/2/11 | 2 | 0 | 0 | 0 | 0.0002050230651 |
| diagnosis/2021-09 | reduce | quiet | below | v137 | original | supported | 0 | 9766 | 11 | 0.00764188748 | 0.0201071745 | 0.001119315671 | 1/35/10 | 36 | 0.02777777778 | 0.09090909091 | 0.04255319149 | 0.003587903639 |
| diagnosis/2021-09 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 9766 | 11 | 0.00764188748 | 0.0201071745 | 0.001119315671 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | quiet | capped | v127 | original | supported | 0 | 32797 | 27 | 0.005628081319 | 0.0377263891 | 0.0008108436753 | 0/1/27 | 1 | 0 | 0 | 0 | 3.051571559e-05 |
| diagnosis/2021-09 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 32797 | 27 | 0.005628081319 | 0.0377263891 | 0.0008108436753 | 0/1/27 | 1 | 0 | 0 | 0 | 3.051571559e-05 |
| diagnosis/2021-09 | reduce | quiet | capped | v137 | original | supported | 0 | 32797 | 27 | 0.005659864882 | 0.03339961574 | 0.0008129826687 | 2/19/25 | 21 | 0.09523809524 | 0.07407407407 | 0.08333333333 | 0.0005797985963 |
| diagnosis/2021-09 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 32797 | 27 | 0.005659864882 | 0.03339961574 | 0.0008129826687 | 1/9/26 | 10 | 0.1 | 0.03703703704 | 0.05405405405 | 0.0002746414403 |
| diagnosis/2021-09 | reduce | first | all | v127 | original | supported | 0 | 14355 | 11 | 0.005491446116 | 0.01859668156 | 0.0007608715241 | 0/6/11 | 6 | 0 | 0 | 0 | 0.0004182933631 |
| diagnosis/2021-09 | reduce | first | all | v127 | budget_matched | supported | 0 | 14355 | 11 | 0.005491446116 | 0.01859668156 | 0.0007608715241 | 0/6/11 | 6 | 0 | 0 | 0 | 0.0004182933631 |
| diagnosis/2021-09 | reduce | first | all | v137 | original | supported | 0 | 14355 | 11 | 0.005328577631 | 0.02708099902 | 0.0007592465755 | 2/38/9 | 40 | 0.05 | 0.1818181818 | 0.07843137255 | 0.002649191299 |
| diagnosis/2021-09 | reduce | first | all | v137 | budget_matched | supported | 0 | 14355 | 11 | 0.005328577631 | 0.02708099902 | 0.0007592465755 | 0/2/11 | 2 | 0 | 0 | 0 | 0.000139431121 |
| diagnosis/2021-09 | reduce | first | below | v127 | original | supported | 0 | 7708 | 6 | 0.005839574586 | 0.01131703548 | 0.0007771994003 | 0/5/6 | 5 | 0 | 0 | 0 | 0.0006491820306 |
| diagnosis/2021-09 | reduce | first | below | v127 | budget_matched | supported | 0 | 7708 | 6 | 0.005839574586 | 0.01131703548 | 0.0007771994003 | 0/5/6 | 5 | 0 | 0 | 0 | 0.0006491820306 |
| diagnosis/2021-09 | reduce | first | below | v137 | original | supported | 0 | 7708 | 6 | 0.00554370233 | 0.02969936681 | 0.0007729380226 | 1/35/5 | 36 | 0.02777777778 | 0.1666666667 | 0.04761904762 | 0.004544274214 |
| diagnosis/2021-09 | reduce | first | below | v137 | budget_matched | supported | 0 | 7708 | 6 | 0.00554370233 | 0.02969936681 | 0.0007729380226 | 0/2/6 | 2 | 0 | 0 | 0 | 0.0002596728123 |
| diagnosis/2021-09 | reduce | first | capped | v127 | original | supported | 0 | 6647 | 5 | 0.005087749072 | 0.08306000565 | 0.0007419373778 | 0/1/5 | 1 | 0 | 0 | 0 | 0.0001505570611 |
| diagnosis/2021-09 | reduce | first | capped | v127 | budget_matched | supported | 0 | 6647 | 5 | 0.005087749072 | 0.08306000565 | 0.0007419373778 | 0/1/5 | 1 | 0 | 0 | 0 | 0.0001505570611 |
| diagnosis/2021-09 | reduce | first | capped | v137 | original | supported | 0 | 6647 | 5 | 0.005079114537 | 0.07501227255 | 0.0007433696874 | 1/3/4 | 4 | 0.25 | 0.2 | 0.2222222222 | 0.0004516711834 |
| diagnosis/2021-09 | reduce | first | capped | v137 | budget_matched | supported | 0 | 6647 | 5 | 0.005079114537 | 0.07501227255 | 0.0007433696874 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | all | v127 | original | supported | 0 | 28208 | 27 | 0.0064472428 | 0.03367180698 | 0.0009439405735 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | all | v127 | budget_matched | supported | 0 | 28208 | 27 | 0.0064472428 | 0.03367180698 | 0.0009439405735 | 0/0/27 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | all | v137 | original | supported | 0 | 28208 | 27 | 0.006514660018 | 0.02876127519 | 0.000946385594 | 1/16/26 | 17 | 0.05882352941 | 0.03703703704 | 0.04545454545 | 0.0005677584188 |
| diagnosis/2021-09 | reduce | restart | all | v137 | budget_matched | supported | 0 | 28208 | 27 | 0.006514660018 | 0.02876127519 | 0.000946385594 | 0/4/27 | 4 | 0 | 0 | 0 | 0.0001419396047 |
| diagnosis/2021-09 | reduce | restart | below | v127 | original | supported | 0 | 2058 | 5 | 0.01511074342 | 0.0374302603 | 0.002412581841 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | below | v127 | budget_matched | supported | 0 | 2058 | 5 | 0.01511074342 | 0.0374302603 | 0.002412581841 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | below | v137 | original | supported | 0 | 2058 | 5 | 0.01550039629 | 0.0217347791 | 0.002416632926 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | below | v137 | budget_matched | supported | 0 | 2058 | 5 | 0.01550039629 | 0.0217347791 | 0.002416632926 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | capped | v127 | original | supported | 0 | 26150 | 22 | 0.005765426957 | 0.03789249864 | 0.0008283587866 | 0/0/22 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 26150 | 22 | 0.005765426957 | 0.03789249864 | 0.0008283587866 | 0/0/22 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | capped | v137 | original | supported | 0 | 26150 | 22 | 0.005807484291 | 0.03326580357 | 0.0008306774101 | 1/16/21 | 17 | 0.05882352941 | 0.04545454545 | 0.05128205128 | 0.0006123698714 |
| diagnosis/2021-09 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 26150 | 22 | 0.005807484291 | 0.03326580357 | 0.0008306774101 | 1/4/21 | 5 | 0.2 | 0.04545454545 | 0.07407407407 | 0.0001530924679 |
| diagnosis/2021-09 | increase | quiet | all | v127 | original | supported | 0 | 42487 | 36 | 0.005574740086 | 0.06502096071 | 0.0008231185764 | 1/4/35 | 5 | 0.2 | 0.02777777778 | 0.08928571429 | 9.422628442e-05 |
| diagnosis/2021-09 | increase | quiet | all | v127 | budget_matched | supported | 0 | 42487 | 36 | 0.005574740086 | 0.06502096071 | 0.0008231185764 | 1/7/35 | 8 | 0.125 | 0.02777777778 | 0.07352941176 | 0.0001648959977 |
| diagnosis/2021-09 | increase | quiet | all | v137 | original | supported | 0 | 42487 | 36 | 0.005732418579 | 0.05755068327 | 0.0008297679819 | 2/30/34 | 32 | 0.0625 | 0.05555555556 | 0.06097560976 | 0.0007066971332 |
| diagnosis/2021-09 | increase | quiet | all | v137 | budget_matched | supported | 0 | 42487 | 36 | 0.005732418579 | 0.05755068327 | 0.0008297679819 | 1/9/35 | 10 | 0.1 | 0.02777777778 | 0.06578947368 | 0.0002120091399 |
| diagnosis/2021-09 | increase | quiet | below | v127 | original | supported | 0 | 9612 | 20 | 0.01077389059 | 0.1132700257 | 0.001979058683 | 1/4/19 | 5 | 0.2 | 0.05 | 0.125 | 0.0004170141785 |
| diagnosis/2021-09 | increase | quiet | below | v127 | budget_matched | supported | 0 | 9612 | 20 | 0.01077389059 | 0.1132700257 | 0.001979058683 | 1/7/19 | 8 | 0.125 | 0.05 | 0.09615384615 | 0.0007297748123 |
| diagnosis/2021-09 | increase | quiet | below | v137 | original | supported | 0 | 9612 | 20 | 0.01166648683 | 0.09828095184 | 0.00201010524 | 1/17/19 | 18 | 0.05555555556 | 0.05 | 0.05434782609 | 0.001772310259 |
| diagnosis/2021-09 | increase | quiet | below | v137 | budget_matched | supported | 0 | 9612 | 20 | 0.01166648683 | 0.09828095184 | 0.00201010524 | 1/7/19 | 8 | 0.125 | 0.05 | 0.09615384615 | 0.0007297748123 |
| diagnosis/2021-09 | increase | quiet | capped | v127 | original | supported | 0 | 32875 | 16 | 0.004054611275 | 0.01278026526 | 0.0004851445442 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 32875 | 16 | 0.004054611275 | 0.01278026526 | 0.0004851445442 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | quiet | capped | v137 | original | supported | 0 | 32875 | 16 | 0.003997414349 | 0.0189053336 | 0.0004846607052 | 1/13/15 | 14 | 0.07142857143 | 0.0625 | 0.06944444444 | 0.0003956298122 |
| diagnosis/2021-09 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 32875 | 16 | 0.003997414349 | 0.0189053336 | 0.0004846607052 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | first | all | v127 | original | supported | 0 | 2603 | 10 | 0.01649611007 | 0.2197973677 | 0.003498101639 | 1/3/9 | 4 | 0.25 | 0.1 | 0.1923076923 | 0.001156961049 |
| diagnosis/2021-09 | increase | first | all | v127 | budget_matched | supported | 0 | 2603 | 10 | 0.01649611007 | 0.2197973677 | 0.003498101639 | 1/3/9 | 4 | 0.25 | 0.1 | 0.1923076923 | 0.001156961049 |
| diagnosis/2021-09 | increase | first | all | v137 | original | supported | 0 | 2603 | 10 | 0.0177255242 | 0.192859727 | 0.003600595089 | 1/5/9 | 6 | 0.1666666667 | 0.1 | 0.1470588235 | 0.001928268415 |
| diagnosis/2021-09 | increase | first | all | v137 | budget_matched | supported | 0 | 2603 | 10 | 0.0177255242 | 0.192859727 | 0.003600595089 | 1/6/9 | 7 | 0.1428571429 | 0.1 | 0.1315789474 | 0.002313922098 |
| diagnosis/2021-09 | increase | first | below | v127 | original | supported | 0 | 2603 | 10 | 0.01649611007 | 0.2197973677 | 0.003498101639 | 1/3/9 | 4 | 0.25 | 0.1 | 0.1923076923 | 0.001156961049 |
| diagnosis/2021-09 | increase | first | below | v127 | budget_matched | supported | 0 | 2603 | 10 | 0.01649611007 | 0.2197973677 | 0.003498101639 | 1/3/9 | 4 | 0.25 | 0.1 | 0.1923076923 | 0.001156961049 |
| diagnosis/2021-09 | increase | first | below | v137 | original | supported | 0 | 2603 | 10 | 0.0177255242 | 0.192859727 | 0.003600595089 | 1/5/9 | 6 | 0.1666666667 | 0.1 | 0.1470588235 | 0.001928268415 |
| diagnosis/2021-09 | increase | first | below | v137 | budget_matched | supported | 0 | 2603 | 10 | 0.0177255242 | 0.192859727 | 0.003600595089 | 1/6/9 | 7 | 0.1428571429 | 0.1 | 0.1315789474 | 0.002313922098 |
| diagnosis/2021-09 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-09 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-09 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-09 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-09 | increase | restart | all | v127 | original | supported | 0 | 39884 | 26 | 0.004861964886 | 0.01181152073 | 0.0006485377693 | 0/1/26 | 1 | 0 | 0 | 0 | 2.508906618e-05 |
| diagnosis/2021-09 | increase | restart | all | v127 | budget_matched | supported | 0 | 39884 | 26 | 0.004861964886 | 0.01181152073 | 0.0006485377693 | 0/1/26 | 1 | 0 | 0 | 0 | 2.508906618e-05 |
| diagnosis/2021-09 | increase | restart | all | v137 | original | supported | 0 | 39884 | 26 | 0.004949697339 | 0.01291041188 | 0.0006489319835 | 1/25/25 | 26 | 0.03846153846 | 0.03846153846 | 0.03846153846 | 0.0006272266546 |
| diagnosis/2021-09 | increase | restart | all | v137 | budget_matched | supported | 0 | 39884 | 26 | 0.004949697339 | 0.01291041188 | 0.0006489319835 | 0/1/26 | 1 | 0 | 0 | 0 | 2.508906618e-05 |
| diagnosis/2021-09 | increase | restart | below | v127 | original | supported | 0 | 7009 | 10 | 0.008648774695 | 0.02192511593 | 0.001414917035 | 0/1/10 | 1 | 0 | 0 | 0 | 0.0001428775539 |
| diagnosis/2021-09 | increase | restart | below | v127 | budget_matched | supported | 0 | 7009 | 10 | 0.008648774695 | 0.02192511593 | 0.001414917035 | 0/1/10 | 1 | 0 | 0 | 0 | 0.0001428775539 |
| diagnosis/2021-09 | increase | restart | below | v137 | original | supported | 0 | 7009 | 10 | 0.009416283629 | 0.01345986693 | 0.001419429668 | 0/12/10 | 12 | 0 | 0 | 0 | 0.001714530647 |
| diagnosis/2021-09 | increase | restart | below | v137 | budget_matched | supported | 0 | 7009 | 10 | 0.009416283629 | 0.01345986693 | 0.001419429668 | 0/1/10 | 1 | 0 | 0 | 0 | 0.0001428775539 |
| diagnosis/2021-09 | increase | restart | capped | v127 | original | supported | 0 | 32875 | 16 | 0.004054611275 | 0.01278026526 | 0.0004851445442 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | restart | capped | v127 | budget_matched | supported | 0 | 32875 | 16 | 0.004054611275 | 0.01278026526 | 0.0004851445442 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | restart | capped | v137 | original | supported | 0 | 32875 | 16 | 0.003997414349 | 0.0189053336 | 0.0004846607052 | 1/13/15 | 14 | 0.07142857143 | 0.0625 | 0.06944444444 | 0.0003956298122 |
| diagnosis/2021-09 | increase | restart | capped | v137 | budget_matched | supported | 0 | 32875 | 16 | 0.003997414349 | 0.0189053336 | 0.0004846607052 | 0/0/16 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | all | v127 | original | supported | 0 | 44478 | 12 | 0.002232908487 | 0.006186117527 | 0.0002695888327 | 0/0/12 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 44478 | 12 | 0.002232908487 | 0.006186117527 | 0.0002695888327 | 0/0/12 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | all | v137 | original | supported | 0 | 44478 | 12 | 0.002322792992 | 0.003631556657 | 0.0002696926165 | 0/0/12 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 44478 | 12 | 0.002322792992 | 0.003631556657 | 0.0002696926165 | 0/0/12 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | below | v127 | original | supported | 0 | 23417 | 1 | 0.0006310530207 | 0.00325732899 | 4.30953178e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 23417 | 1 | 0.0006310530207 | 0.00325732899 | 4.30953178e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | below | v137 | original | supported | 0 | 23417 | 1 | 0.0007242662862 | 0.001055966209 | 4.337590437e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 23417 | 1 | 0.0007242662862 | 0.001055966209 | 4.337590437e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | capped | v127 | original | supported | 0 | 21061 | 11 | 0.004013956369 | 0.008065781918 | 0.0005214191654 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 21061 | 11 | 0.004013956369 | 0.008065781918 | 0.0005214191654 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | capped | v137 | original | supported | 0 | 21061 | 11 | 0.004100139741 | 0.005614625587 | 0.0005213263683 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 21061 | 11 | 0.004100139741 | 0.005614625587 | 0.0005213263683 | 0/0/11 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | all | v127 | original | supported | 0 | 18173 | 2 | 0.0008866069315 | 0.03123486683 | 0.0001096760828 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | all | v127 | budget_matched | supported | 0 | 18173 | 2 | 0.0008866069315 | 0.03123486683 | 0.0001096760828 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | all | v137 | original | supported | 0 | 18173 | 2 | 0.0009693979849 | 0.008129421691 | 0.0001098418608 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | all | v137 | budget_matched | supported | 0 | 18173 | 2 | 0.0009693979849 | 0.008129421691 | 0.0001098418608 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | below | v127 | original | supported | 0 | 16981 | 1 | 0.0005528782961 | 0.05882352941 | 5.868763771e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | below | v127 | budget_matched | supported | 0 | 16981 | 1 | 0.0005528782961 | 0.05882352941 | 5.868763771e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | below | v137 | original | supported | 0 | 16981 | 1 | 0.0006429970168 | 0.00641025641 | 5.886874888e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | below | v137 | budget_matched | supported | 0 | 16981 | 1 | 0.0006429970168 | 0.00641025641 | 5.886874888e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | capped | v127 | original | supported | 0 | 1192 | 1 | 0.005640840119 | 0.03571428571 | 0.0008360475479 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | capped | v127 | budget_matched | supported | 0 | 1192 | 1 | 0.005640840119 | 0.03571428571 | 0.0008360475479 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | capped | v137 | original | supported | 0 | 1192 | 1 | 0.005619242649 | 0.02380952381 | 0.000835994893 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | capped | v137 | budget_matched | supported | 0 | 1192 | 1 | 0.005619242649 | 0.02380952381 | 0.000835994893 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | all | v127 | original | supported | 0 | 26305 | 10 | 0.003163010679 | 0.00643273911 | 0.0003800657155 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | all | v127 | budget_matched | supported | 0 | 26305 | 10 | 0.003163010679 | 0.00643273911 | 0.0003800657155 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | all | v137 | original | supported | 0 | 26305 | 10 | 0.003257795747 | 0.003782071262 | 0.0003801266702 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | all | v137 | budget_matched | supported | 0 | 26305 | 10 | 0.003257795747 | 0.003782071262 | 0.0003801266702 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | below | v127 | original | supported | 0 | 6436 | 0 | 0.0008373123429 | 없음 | 1.955916865e-06 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | below | v127 | budget_matched | supported | 0 | 6436 | 0 | 0.0008373123429 | 없음 | 1.955916865e-06 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | below | v137 | original | supported | 0 | 6436 | 0 | 0.0009386903796 | 없음 | 2.498963316e-06 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | below | v137 | budget_matched | supported | 0 | 6436 | 0 | 0.0009386903796 | 없음 | 2.498963316e-06 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | capped | v127 | original | supported | 0 | 19869 | 10 | 0.003916354807 | 0.00835575266 | 0.0005025436794 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 19869 | 10 | 0.003916354807 | 0.00835575266 | 0.0005025436794 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | capped | v137 | original | supported | 0 | 19869 | 10 | 0.00400900427 | 0.005709443261 | 0.0005024484741 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 19869 | 10 | 0.00400900427 | 0.005709443261 | 0.0005024484741 | 0/0/10 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | all | v127 | original | supported | 0 | 44484 | 8 | 0.001614643531 | 0.02643002415 | 0.0001792596015 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | all | v127 | budget_matched | supported | 0 | 44484 | 8 | 0.001614643531 | 0.02643002415 | 0.0001792596015 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | all | v137 | original | supported | 0 | 44484 | 8 | 0.001677361289 | 0.01299471276 | 0.0001795666591 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | all | v137 | budget_matched | supported | 0 | 44484 | 8 | 0.001677361289 | 0.01299471276 | 0.0001795666591 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | below | v127 | original | supported | 0 | 23408 | 3 | 0.001130769781 | 0.1118362772 | 0.0001269096086 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | below | v127 | budget_matched | supported | 0 | 23408 | 3 | 0.001130769781 | 0.1118362772 | 0.0001269096086 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | below | v137 | original | supported | 0 | 23408 | 3 | 0.001237737971 | 0.04303851968 | 0.0001274879645 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | below | v137 | budget_matched | supported | 0 | 23408 | 3 | 0.001237737971 | 0.04303851968 | 0.0001274879645 | 0/0/3 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | capped | v127 | original | supported | 0 | 21076 | 5 | 0.002152056549 | 0.001691974258 | 0.0002374019735 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 21076 | 5 | 0.002152056549 | 0.001691974258 | 0.0002374019735 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | capped | v137 | original | supported | 0 | 21076 | 5 | 0.002165627689 | 0.002088300889 | 0.0002374077144 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 21076 | 5 | 0.002165627689 | 0.002088300889 | 0.0002374077144 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | all | v127 | original | supported | 0 | 1110 | 1 | 0.004679824362 | 0.3333333333 | 0.0008711690692 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | all | v127 | budget_matched | supported | 0 | 1110 | 1 | 0.004679824362 | 0.3333333333 | 0.0008711690692 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | all | v137 | original | supported | 0 | 1110 | 1 | 0.004450287354 | 0.3333333333 | 0.0008759122115 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | all | v137 | budget_matched | supported | 0 | 1110 | 1 | 0.004450287354 | 0.3333333333 | 0.0008759122115 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | below | v127 | original | supported | 0 | 1110 | 1 | 0.004679824362 | 0.3333333333 | 0.0008711690692 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | below | v127 | budget_matched | supported | 0 | 1110 | 1 | 0.004679824362 | 0.3333333333 | 0.0008711690692 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | below | v137 | original | supported | 0 | 1110 | 1 | 0.004450287354 | 0.3333333333 | 0.0008759122115 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | below | v137 | budget_matched | supported | 0 | 1110 | 1 | 0.004450287354 | 0.3333333333 | 0.0008759122115 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-10 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-10 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-10 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-10 | increase | restart | all | v127 | original | supported | 0 | 43374 | 7 | 0.00153620136 | 0.001117872568 | 0.0001615526916 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | all | v127 | budget_matched | supported | 0 | 43374 | 7 | 0.00153620136 | 0.001117872568 | 0.0001615526916 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | all | v137 | original | supported | 0 | 43374 | 7 | 0.001606398318 | 0.001216551946 | 0.0001617462237 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | all | v137 | budget_matched | supported | 0 | 43374 | 7 | 0.001606398318 | 0.001216551946 | 0.0001617462237 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | below | v127 | original | supported | 0 | 22298 | 2 | 0.0009540969588 | 0.0008584188027 | 8.986018712e-05 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | below | v127 | budget_matched | supported | 0 | 22298 | 2 | 0.0009540969588 | 0.0008584188027 | 8.986018712e-05 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | below | v137 | original | supported | 0 | 22298 | 2 | 0.001077816461 | 0.001213443185 | 9.02312189e-05 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | below | v137 | budget_matched | supported | 0 | 22298 | 2 | 0.001077816461 | 0.001213443185 | 9.02312189e-05 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | capped | v127 | original | supported | 0 | 21076 | 5 | 0.002152056549 | 0.001691974258 | 0.0002374019735 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | capped | v127 | budget_matched | supported | 0 | 21076 | 5 | 0.002152056549 | 0.001691974258 | 0.0002374019735 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | capped | v137 | original | supported | 0 | 21076 | 5 | 0.002165627689 | 0.002088300889 | 0.0002374077144 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | capped | v137 | budget_matched | supported | 0 | 21076 | 5 | 0.002165627689 | 0.002088300889 | 0.0002374077144 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | all | v127 | original | supported | 0 | 43101 | 7 | 0.001578070292 | 0.003385896275 | 0.0001628817989 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 43101 | 7 | 0.001578070292 | 0.003385896275 | 0.0001628817989 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | all | v137 | original | supported | 0 | 43101 | 7 | 0.001578914773 | 0.00328701852 | 0.0001625052263 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 43101 | 7 | 0.001578914773 | 0.00328701852 | 0.0001625052263 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | below | v127 | original | supported | 0 | 14269 | 1 | 0.0007825010246 | 0.0001417434444 | 7.009349272e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 14269 | 1 | 0.0007825010246 | 0.0001417434444 | 7.009349272e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | below | v137 | original | supported | 0 | 14269 | 1 | 0.0007721442363 | 0.0001359804188 | 7.008119624e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 14269 | 1 | 0.0007721442363 | 0.0001359804188 | 7.008119624e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | capped | v127 | original | supported | 0 | 28832 | 6 | 0.001971798714 | 0.003925867454 | 0.0002088028706 | 0/0/6 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 28832 | 6 | 0.001971798714 | 0.003925867454 | 0.0002088028706 | 0/0/6 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | capped | v137 | original | supported | 0 | 28832 | 6 | 0.001978186721 | 0.003802686727 | 0.0002082460172 | 0/0/6 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 28832 | 6 | 0.001978186721 | 0.003802686727 | 0.0002082460172 | 0/0/6 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | all | v127 | original | supported | 0 | 22257 | 2 | 0.0008699505275 | 0.01477235845 | 8.967232127e-05 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | all | v127 | budget_matched | supported | 0 | 22257 | 2 | 0.0008699505275 | 0.01477235845 | 8.967232127e-05 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | all | v137 | original | supported | 0 | 22257 | 2 | 0.0008564126066 | 0.02180431098 | 8.957107503e-05 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | all | v137 | budget_matched | supported | 0 | 22257 | 2 | 0.0008564126066 | 0.02180431098 | 8.957107503e-05 | 0/0/2 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | below | v127 | original | supported | 0 | 14269 | 1 | 0.0007825010246 | 0.0001417434444 | 7.009349272e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | below | v127 | budget_matched | supported | 0 | 14269 | 1 | 0.0007825010246 | 0.0001417434444 | 7.009349272e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | below | v137 | original | supported | 0 | 14269 | 1 | 0.0007721442363 | 0.0001359804188 | 7.008119624e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | below | v137 | budget_matched | supported | 0 | 14269 | 1 | 0.0007721442363 | 0.0001359804188 | 7.008119624e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | capped | v127 | original | supported | 0 | 7988 | 1 | 0.001026161964 | 0.02941176471 | 0.0001246460699 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | capped | v127 | budget_matched | supported | 0 | 7988 | 1 | 0.001026161964 | 0.02941176471 | 0.0001246460699 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | capped | v137 | original | supported | 0 | 7988 | 1 | 0.001006941572 | 0.04347826087 | 0.0001243859324 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | capped | v137 | budget_matched | supported | 0 | 7988 | 1 | 0.001006941572 | 0.04347826087 | 0.0001243859324 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | all | v127 | original | supported | 0 | 20844 | 5 | 0.002334192993 | 0.002954295697 | 0.0002410540951 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | all | v127 | budget_matched | supported | 0 | 20844 | 5 | 0.002334192993 | 0.002954295697 | 0.0002410540951 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | all | v137 | original | supported | 0 | 20844 | 5 | 0.00235039485 | 0.002785263575 | 0.000240383532 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | all | v137 | budget_matched | supported | 0 | 20844 | 5 | 0.00235039485 | 0.002785263575 | 0.000240383532 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | below | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | reduce | restart | below | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | reduce | restart | below | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | reduce | restart | below | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | reduce | restart | capped | v127 | original | supported | 0 | 20844 | 5 | 0.002334192993 | 0.002954295697 | 0.0002410540951 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 20844 | 5 | 0.002334192993 | 0.002954295697 | 0.0002410540951 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | capped | v137 | original | supported | 0 | 20844 | 5 | 0.00235039485 | 0.002785263575 | 0.000240383532 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 20844 | 5 | 0.00235039485 | 0.002785263575 | 0.000240383532 | 0/0/5 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | all | v127 | original | supported | 0 | 43068 | 8 | 0.001703693575 | 0.001067329083 | 0.0001859323947 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | all | v127 | budget_matched | supported | 0 | 43068 | 8 | 0.001703693575 | 0.001067329083 | 0.0001859323947 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | all | v137 | original | supported | 0 | 43068 | 8 | 0.001623812168 | 0.002967760193 | 0.0001864122285 | 0/9/8 | 9 | 0 | 0 | 0 | 0.0002090106828 |
| diagnosis/2021-11 | increase | quiet | all | v137 | budget_matched | supported | 0 | 43068 | 8 | 0.001623812168 | 0.002967760193 | 0.0001864122285 | 0/0/8 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | below | v127 | original | supported | 0 | 14270 | 1 | 0.001057032706 | 0.0004943153732 | 7.088522264e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | below | v127 | budget_matched | supported | 0 | 14270 | 1 | 0.001057032706 | 0.0004943153732 | 7.088522264e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | below | v137 | original | supported | 0 | 14270 | 1 | 0.0008072220407 | 0.001769911504 | 7.04484689e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | below | v137 | budget_matched | supported | 0 | 14270 | 1 | 0.0008072220407 | 0.001769911504 | 7.04484689e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | capped | v127 | original | supported | 0 | 28798 | 7 | 0.002024127307 | 0.002502686589 | 0.0002429406294 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 28798 | 7 | 0.002024127307 | 0.002502686589 | 0.0002429406294 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | capped | v137 | original | supported | 0 | 28798 | 7 | 0.002028449334 | 0.003965166537 | 0.0002438746512 | 0/9/7 | 9 | 0 | 0 | 0 | 0.0003125976868 |
| diagnosis/2021-11 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 28798 | 7 | 0.002028449334 | 0.003965166537 | 0.0002438746512 | 0/4/7 | 4 | 0 | 0 | 0 | 0.0001389323052 |
| diagnosis/2021-11 | increase | first | all | v127 | original | supported | 0 | 14270 | 1 | 0.001057032706 | 0.0004943153732 | 7.088522264e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | all | v127 | budget_matched | supported | 0 | 14270 | 1 | 0.001057032706 | 0.0004943153732 | 7.088522264e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | all | v137 | original | supported | 0 | 14270 | 1 | 0.0008072220407 | 0.001769911504 | 7.04484689e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | all | v137 | budget_matched | supported | 0 | 14270 | 1 | 0.0008072220407 | 0.001769911504 | 7.04484689e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | below | v127 | original | supported | 0 | 14270 | 1 | 0.001057032706 | 0.0004943153732 | 7.088522264e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | below | v127 | budget_matched | supported | 0 | 14270 | 1 | 0.001057032706 | 0.0004943153732 | 7.088522264e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | below | v137 | original | supported | 0 | 14270 | 1 | 0.0008072220407 | 0.001769911504 | 7.04484689e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | below | v137 | budget_matched | supported | 0 | 14270 | 1 | 0.0008072220407 | 0.001769911504 | 7.04484689e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | restart | all | v127 | original | supported | 0 | 28798 | 7 | 0.002024127307 | 0.002502686589 | 0.0002429406294 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | all | v127 | budget_matched | supported | 0 | 28798 | 7 | 0.002024127307 | 0.002502686589 | 0.0002429406294 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | all | v137 | original | supported | 0 | 28798 | 7 | 0.002028449334 | 0.003965166537 | 0.0002438746512 | 0/9/7 | 9 | 0 | 0 | 0 | 0.0003125976868 |
| diagnosis/2021-11 | increase | restart | all | v137 | budget_matched | supported | 0 | 28798 | 7 | 0.002028449334 | 0.003965166537 | 0.0002438746512 | 0/4/7 | 4 | 0 | 0 | 0 | 0.0001389323052 |
| diagnosis/2021-11 | increase | restart | below | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | restart | below | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | restart | below | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | restart | below | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | restart | capped | v127 | original | supported | 0 | 28798 | 7 | 0.002024127307 | 0.002502686589 | 0.0002429406294 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | capped | v127 | budget_matched | supported | 0 | 28798 | 7 | 0.002024127307 | 0.002502686589 | 0.0002429406294 | 0/0/7 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | capped | v137 | original | supported | 0 | 28798 | 7 | 0.002028449334 | 0.003965166537 | 0.0002438746512 | 0/9/7 | 9 | 0 | 0 | 0 | 0.0003125976868 |
| diagnosis/2021-11 | increase | restart | capped | v137 | budget_matched | supported | 0 | 28798 | 7 | 0.002028449334 | 0.003965166537 | 0.0002438746512 | 0/4/7 | 4 | 0 | 0 | 0 | 0.0001389323052 |
| diagnosis/2021-12 | reduce | quiet | all | v127 | original | supported | 0 | 33267 | 4 | 0.001224384112 | 0.001088758987 | 0.0001204134545 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | all | v127 | budget_matched | supported | 0 | 33267 | 4 | 0.001224384112 | 0.001088758987 | 0.0001204134545 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | all | v137 | original | supported | 0 | 33267 | 4 | 0.001384838286 | 0.0005047366585 | 0.0001205892783 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | all | v137 | budget_matched | supported | 0 | 33267 | 4 | 0.001384838286 | 0.0005047366585 | 0.0001205892783 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | below | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | quiet | below | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | quiet | below | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | quiet | below | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | quiet | capped | v127 | original | supported | 0 | 33267 | 4 | 0.001224384112 | 0.001088758987 | 0.0001204134545 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | capped | v127 | budget_matched | supported | 0 | 33267 | 4 | 0.001224384112 | 0.001088758987 | 0.0001204134545 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | capped | v137 | original | supported | 0 | 33267 | 4 | 0.001384838286 | 0.0005047366585 | 0.0001205892783 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | capped | v137 | budget_matched | supported | 0 | 33267 | 4 | 0.001384838286 | 0.0005047366585 | 0.0001205892783 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | all | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | all | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | all | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | all | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | below | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | below | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | below | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | below | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | capped | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | capped | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | capped | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | first | capped | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | restart | all | v127 | original | supported | 0 | 33267 | 4 | 0.001224384112 | 0.001088758987 | 0.0001204134545 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | all | v127 | budget_matched | supported | 0 | 33267 | 4 | 0.001224384112 | 0.001088758987 | 0.0001204134545 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | all | v137 | original | supported | 0 | 33267 | 4 | 0.001384838286 | 0.0005047366585 | 0.0001205892783 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | all | v137 | budget_matched | supported | 0 | 33267 | 4 | 0.001384838286 | 0.0005047366585 | 0.0001205892783 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | below | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | restart | below | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | restart | below | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | restart | below | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | reduce | restart | capped | v127 | original | supported | 0 | 33267 | 4 | 0.001224384112 | 0.001088758987 | 0.0001204134545 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | capped | v127 | budget_matched | supported | 0 | 33267 | 4 | 0.001224384112 | 0.001088758987 | 0.0001204134545 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | capped | v137 | original | supported | 0 | 33267 | 4 | 0.001384838286 | 0.0005047366585 | 0.0001205892783 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | capped | v137 | budget_matched | supported | 0 | 33267 | 4 | 0.001384838286 | 0.0005047366585 | 0.0001205892783 | 0/0/4 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | all | v127 | original | supported | 0 | 33314 | 1 | 0.0005180515319 | 5.274539796e-05 | 3.016423187e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | all | v127 | budget_matched | supported | 0 | 33314 | 1 | 0.0005180515319 | 5.274539796e-05 | 3.016423187e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | all | v137 | original | supported | 0 | 33314 | 1 | 0.0006443200232 | 4.561835683e-05 | 3.042758501e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | all | v137 | budget_matched | supported | 0 | 33314 | 1 | 0.0006443200232 | 4.561835683e-05 | 3.042758501e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | below | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | quiet | below | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | quiet | below | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | quiet | below | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | quiet | capped | v127 | original | supported | 0 | 33314 | 1 | 0.0005180515319 | 5.274539796e-05 | 3.016423187e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | capped | v127 | budget_matched | supported | 0 | 33314 | 1 | 0.0005180515319 | 5.274539796e-05 | 3.016423187e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | capped | v137 | original | supported | 0 | 33314 | 1 | 0.0006443200232 | 4.561835683e-05 | 3.042758501e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | capped | v137 | budget_matched | supported | 0 | 33314 | 1 | 0.0006443200232 | 4.561835683e-05 | 3.042758501e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | first | all | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | all | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | all | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | all | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | below | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | below | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | below | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | below | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | capped | v127 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | capped | v127 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | capped | v137 | original | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | capped | v137 | budget_matched | no_calibration_rows | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | restart | all | v127 | original | supported | 0 | 33314 | 1 | 0.0005180515319 | 5.274539796e-05 | 3.016423187e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | all | v127 | budget_matched | supported | 0 | 33314 | 1 | 0.0005180515319 | 5.274539796e-05 | 3.016423187e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | all | v137 | original | supported | 0 | 33314 | 1 | 0.0006443200232 | 4.561835683e-05 | 3.042758501e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | all | v137 | budget_matched | supported | 0 | 33314 | 1 | 0.0006443200232 | 4.561835683e-05 | 3.042758501e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | below | v127 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | restart | below | v127 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | restart | below | v137 | original | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | restart | below | v137 | budget_matched | supported | 0 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | restart | capped | v127 | original | supported | 0 | 33314 | 1 | 0.0005180515319 | 5.274539796e-05 | 3.016423187e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | capped | v127 | budget_matched | supported | 0 | 33314 | 1 | 0.0005180515319 | 5.274539796e-05 | 3.016423187e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | capped | v137 | original | supported | 0 | 33314 | 1 | 0.0006443200232 | 4.561835683e-05 | 3.042758501e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | capped | v137 | budget_matched | supported | 0 | 33314 | 1 | 0.0006443200232 | 4.561835683e-05 | 3.042758501e-05 | 0/0/1 | 0 | 0 | 0 | 0 | 0 |

## 모든 집단의 요청 교집합

| 기간·월 | 행동 | 상태 | 횟수 | 모델 | 교집합 TP | 교집합 FP | 추가 TP | 추가 FP | 제거 TP | 제거 FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training | reduce | quiet | all | v127 | 30 | 102 | 18 | 103 | 10 | 45 |
| training | reduce | quiet | all | v137 | 40 | 188 | 0 | 0 | 37 | 271 |
| training | reduce | quiet | below | v127 | 12 | 41 | 7 | 57 | 3 | 19 |
| training | reduce | quiet | below | v137 | 14 | 42 | 0 | 0 | 13 | 93 |
| training | reduce | quiet | capped | v127 | 24 | 74 | 13 | 62 | 1 | 13 |
| training | reduce | quiet | capped | v137 | 25 | 132 | 0 | 0 | 25 | 192 |
| training | reduce | first | all | v127 | 15 | 63 | 0 | 1 | 0 | 0 |
| training | reduce | first | all | v137 | 7 | 21 | 0 | 0 | 15 | 109 |
| training | reduce | first | below | v127 | 12 | 46 | 0 | 1 | 0 | 0 |
| training | reduce | first | below | v137 | 4 | 8 | 0 | 0 | 12 | 82 |
| training | reduce | first | capped | v127 | 3 | 17 | 0 | 0 | 0 | 0 |
| training | reduce | first | capped | v137 | 2 | 13 | 0 | 0 | 4 | 27 |
| training | reduce | restart | all | v127 | 25 | 84 | 0 | 1 | 0 | 0 |
| training | reduce | restart | all | v137 | 22 | 85 | 0 | 0 | 33 | 244 |
| training | reduce | restart | below | v127 | 3 | 14 | 0 | 0 | 0 | 0 |
| training | reduce | restart | below | v137 | 2 | 2 | 0 | 0 | 9 | 43 |
| training | reduce | restart | capped | v127 | 22 | 70 | 0 | 1 | 0 | 0 |
| training | reduce | restart | capped | v137 | 20 | 81 | 0 | 0 | 24 | 203 |
| training | increase | quiet | all | v127 | 25 | 76 | 10 | 37 | 3 | 20 |
| training | increase | quiet | all | v137 | 24 | 44 | 14 | 66 | 57 | 581 |
| training | increase | quiet | below | v127 | 19 | 62 | 6 | 17 | 4 | 22 |
| training | increase | quiet | below | v137 | 19 | 29 | 9 | 41 | 29 | 208 |
| training | increase | quiet | capped | v127 | 5 | 12 | 0 | 1 | 0 | 0 |
| training | increase | quiet | capped | v137 | 12 | 56 | 0 | 0 | 21 | 332 |
| training | increase | first | all | v127 | 16 | 54 | 0 | 0 | 0 | 0 |
| training | increase | first | all | v137 | 19 | 29 | 0 | 13 | 0 | 0 |
| training | increase | first | below | v127 | 16 | 54 | 0 | 0 | 0 | 0 |
| training | increase | first | below | v137 | 19 | 29 | 0 | 13 | 0 | 0 |
| training | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training | increase | restart | all | v127 | 12 | 42 | 0 | 2 | 0 | 0 |
| training | increase | restart | all | v137 | 18 | 73 | 0 | 0 | 44 | 523 |
| training | increase | restart | below | v127 | 7 | 30 | 0 | 1 | 0 | 0 |
| training | increase | restart | below | v137 | 6 | 17 | 0 | 0 | 23 | 191 |
| training | increase | restart | capped | v127 | 5 | 12 | 0 | 1 | 0 | 0 |
| training | increase | restart | capped | v137 | 12 | 56 | 0 | 0 | 21 | 332 |
| calibration | reduce | quiet | all | v127 | 14 | 101 | 2 | 87 | 6 | 88 |
| calibration | reduce | quiet | all | v137 | 15 | 186 | 0 | 0 | 24 | 546 |
| calibration | reduce | quiet | below | v127 | 1 | 21 | 0 | 29 | 3 | 29 |
| calibration | reduce | quiet | below | v137 | 2 | 49 | 0 | 0 | 10 | 226 |
| calibration | reduce | quiet | capped | v127 | 14 | 89 | 1 | 50 | 2 | 50 |
| calibration | reduce | quiet | capped | v137 | 14 | 139 | 0 | 0 | 13 | 318 |
| calibration | reduce | first | all | v127 | 6 | 108 | 0 | 0 | 0 | 0 |
| calibration | reduce | first | all | v137 | 4 | 108 | 0 | 0 | 11 | 365 |
| calibration | reduce | first | below | v127 | 4 | 47 | 0 | 0 | 0 | 0 |
| calibration | reduce | first | below | v137 | 4 | 47 | 0 | 0 | 7 | 193 |
| calibration | reduce | first | capped | v127 | 2 | 61 | 0 | 0 | 0 | 0 |
| calibration | reduce | first | capped | v137 | 0 | 61 | 0 | 0 | 4 | 172 |
| calibration | reduce | restart | all | v127 | 14 | 81 | 0 | 0 | 0 | 0 |
| calibration | reduce | restart | all | v137 | 9 | 81 | 0 | 0 | 15 | 178 |
| calibration | reduce | restart | below | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| calibration | reduce | restart | below | v137 | 0 | 3 | 0 | 0 | 1 | 32 |
| calibration | reduce | restart | capped | v127 | 14 | 78 | 0 | 0 | 0 | 0 |
| calibration | reduce | restart | capped | v137 | 10 | 78 | 0 | 0 | 13 | 146 |
| calibration | increase | quiet | all | v127 | 5 | 48 | 2 | 32 | 3 | 32 |
| calibration | increase | quiet | all | v137 | 5 | 38 | 2 | 42 | 18 | 446 |
| calibration | increase | quiet | below | v127 | 4 | 35 | 0 | 12 | 3 | 12 |
| calibration | increase | quiet | below | v137 | 5 | 28 | 1 | 19 | 7 | 141 |
| calibration | increase | quiet | capped | v127 | 1 | 33 | 0 | 0 | 0 | 0 |
| calibration | increase | quiet | capped | v137 | 0 | 33 | 0 | 0 | 11 | 282 |
| calibration | increase | first | all | v127 | 4 | 32 | 0 | 0 | 0 | 0 |
| calibration | increase | first | all | v137 | 5 | 28 | 0 | 4 | 0 | 0 |
| calibration | increase | first | below | v127 | 4 | 32 | 0 | 0 | 0 | 0 |
| calibration | increase | first | below | v137 | 5 | 28 | 0 | 4 | 0 | 0 |
| calibration | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration | increase | restart | all | v127 | 4 | 48 | 0 | 0 | 0 | 0 |
| calibration | increase | restart | all | v137 | 2 | 48 | 0 | 0 | 16 | 408 |
| calibration | increase | restart | below | v127 | 3 | 15 | 0 | 0 | 0 | 0 |
| calibration | increase | restart | below | v137 | 2 | 15 | 0 | 0 | 5 | 126 |
| calibration | increase | restart | capped | v127 | 1 | 33 | 0 | 0 | 0 | 0 |
| calibration | increase | restart | capped | v137 | 0 | 33 | 0 | 0 | 11 | 282 |
| diagnosis | reduce | quiet | all | v127 | 1 | 5 | 0 | 11 | 1 | 9 |
| diagnosis | reduce | quiet | all | v137 | 2 | 31 | 0 | 0 | 4 | 81 |
| diagnosis | reduce | quiet | below | v127 | 2 | 5 | 0 | 5 | 0 | 3 |
| diagnosis | reduce | quiet | below | v137 | 1 | 19 | 0 | 0 | 2 | 42 |
| diagnosis | reduce | quiet | capped | v127 | 0 | 2 | 0 | 3 | 0 | 4 |
| diagnosis | reduce | quiet | capped | v137 | 1 | 16 | 0 | 0 | 2 | 35 |
| diagnosis | reduce | first | all | v127 | 1 | 9 | 0 | 0 | 0 | 0 |
| diagnosis | reduce | first | all | v137 | 0 | 9 | 0 | 0 | 4 | 51 |
| diagnosis | reduce | first | below | v127 | 1 | 5 | 0 | 0 | 0 | 0 |
| diagnosis | reduce | first | below | v137 | 0 | 3 | 0 | 0 | 2 | 39 |
| diagnosis | reduce | first | capped | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| diagnosis | reduce | first | capped | v137 | 0 | 4 | 0 | 0 | 2 | 14 |
| diagnosis | reduce | restart | all | v127 | 1 | 5 | 0 | 0 | 0 | 0 |
| diagnosis | reduce | restart | all | v137 | 0 | 9 | 0 | 0 | 2 | 43 |
| diagnosis | reduce | restart | below | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| diagnosis | reduce | restart | below | v137 | 0 | 4 | 0 | 0 | 1 | 15 |
| diagnosis | reduce | restart | capped | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| diagnosis | reduce | restart | capped | v137 | 1 | 5 | 0 | 0 | 0 | 28 |
| diagnosis | increase | quiet | all | v127 | 1 | 10 | 1 | 10 | 0 | 1 |
| diagnosis | increase | quiet | all | v137 | 1 | 11 | 0 | 11 | 2 | 56 |
| diagnosis | increase | quiet | below | v127 | 1 | 10 | 0 | 7 | 0 | 1 |
| diagnosis | increase | quiet | below | v137 | 1 | 11 | 0 | 7 | 1 | 18 |
| diagnosis | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis | increase | quiet | capped | v137 | 0 | 4 | 0 | 0 | 1 | 34 |
| diagnosis | increase | first | all | v127 | 1 | 10 | 0 | 0 | 0 | 0 |
| diagnosis | increase | first | all | v137 | 1 | 11 | 0 | 3 | 0 | 0 |
| diagnosis | increase | first | below | v127 | 1 | 10 | 0 | 0 | 0 | 0 |
| diagnosis | increase | first | below | v137 | 1 | 11 | 0 | 3 | 0 | 0 |
| diagnosis | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis | increase | restart | all | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| diagnosis | increase | restart | all | v137 | 0 | 5 | 0 | 0 | 2 | 51 |
| diagnosis | increase | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| diagnosis | increase | restart | below | v137 | 0 | 1 | 0 | 0 | 1 | 17 |
| diagnosis | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis | increase | restart | capped | v137 | 0 | 4 | 0 | 0 | 1 | 34 |
| training/2020-01 | reduce | quiet | all | v127 | 1 | 3 | 3 | 6 | 0 | 2 |
| training/2020-01 | reduce | quiet | all | v137 | 2 | 10 | 0 | 0 | 1 | 13 |
| training/2020-01 | reduce | quiet | below | v127 | 0 | 0 | 0 | 6 | 0 | 1 |
| training/2020-01 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 4 |
| training/2020-01 | reduce | quiet | capped | v127 | 1 | 3 | 3 | 3 | 0 | 1 |
| training/2020-01 | reduce | quiet | capped | v137 | 2 | 8 | 0 | 0 | 1 | 11 |
| training/2020-01 | reduce | first | all | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 7 |
| training/2020-01 | reduce | first | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 4 |
| training/2020-01 | reduce | first | capped | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 3 |
| training/2020-01 | reduce | restart | all | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | all | v137 | 2 | 4 | 0 | 0 | 1 | 12 |
| training/2020-01 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | capped | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| training/2020-01 | reduce | restart | capped | v137 | 2 | 4 | 0 | 0 | 1 | 12 |
| training/2020-01 | increase | quiet | all | v127 | 2 | 10 | 1 | 5 | 0 | 4 |
| training/2020-01 | increase | quiet | all | v137 | 1 | 8 | 1 | 8 | 12 | 39 |
| training/2020-01 | increase | quiet | below | v127 | 0 | 4 | 1 | 2 | 1 | 4 |
| training/2020-01 | increase | quiet | below | v137 | 0 | 0 | 0 | 3 | 9 | 15 |
| training/2020-01 | increase | quiet | capped | v127 | 1 | 6 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | quiet | capped | v137 | 2 | 11 | 0 | 0 | 2 | 21 |
| training/2020-01 | increase | first | all | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | first | below | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-01 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-01 | increase | restart | all | v127 | 2 | 10 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | restart | all | v137 | 2 | 12 | 0 | 0 | 11 | 35 |
| training/2020-01 | increase | restart | below | v127 | 1 | 4 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | restart | below | v137 | 0 | 1 | 0 | 0 | 9 | 14 |
| training/2020-01 | increase | restart | capped | v127 | 1 | 6 | 0 | 0 | 0 | 0 |
| training/2020-01 | increase | restart | capped | v137 | 2 | 11 | 0 | 0 | 2 | 21 |
| training/2020-02 | reduce | quiet | all | v127 | 1 | 10 | 1 | 13 | 0 | 3 |
| training/2020-02 | reduce | quiet | all | v137 | 2 | 18 | 0 | 0 | 2 | 16 |
| training/2020-02 | reduce | quiet | below | v127 | 0 | 0 | 0 | 1 | 0 | 3 |
| training/2020-02 | reduce | quiet | below | v137 | 0 | 4 | 0 | 0 | 0 | 2 |
| training/2020-02 | reduce | quiet | capped | v127 | 1 | 10 | 1 | 7 | 0 | 0 |
| training/2020-02 | reduce | quiet | capped | v137 | 1 | 12 | 0 | 0 | 3 | 16 |
| training/2020-02 | reduce | first | all | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 2 |
| training/2020-02 | reduce | first | below | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 2 |
| training/2020-02 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | restart | all | v127 | 1 | 10 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | restart | all | v137 | 0 | 10 | 0 | 0 | 4 | 22 |
| training/2020-02 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | restart | below | v137 | 0 | 1 | 0 | 0 | 0 | 3 |
| training/2020-02 | reduce | restart | capped | v127 | 1 | 10 | 0 | 0 | 0 | 0 |
| training/2020-02 | reduce | restart | capped | v137 | 0 | 10 | 0 | 0 | 4 | 18 |
| training/2020-02 | increase | quiet | all | v127 | 0 | 3 | 0 | 2 | 0 | 1 |
| training/2020-02 | increase | quiet | all | v137 | 0 | 4 | 0 | 5 | 6 | 43 |
| training/2020-02 | increase | quiet | below | v127 | 0 | 3 | 0 | 1 | 0 | 0 |
| training/2020-02 | increase | quiet | below | v137 | 0 | 0 | 0 | 4 | 1 | 3 |
| training/2020-02 | increase | quiet | capped | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | quiet | capped | v137 | 0 | 14 | 0 | 0 | 5 | 30 |
| training/2020-02 | increase | first | all | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | first | all | v137 | 0 | 0 | 0 | 2 | 0 | 0 |
| training/2020-02 | increase | first | below | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | first | below | v137 | 0 | 0 | 0 | 2 | 0 | 0 |
| training/2020-02 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-02 | increase | restart | all | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | restart | all | v137 | 0 | 16 | 0 | 0 | 6 | 31 |
| training/2020-02 | increase | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | restart | below | v137 | 0 | 2 | 0 | 0 | 1 | 1 |
| training/2020-02 | increase | restart | capped | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-02 | increase | restart | capped | v137 | 0 | 14 | 0 | 0 | 5 | 30 |
| training/2020-03 | reduce | quiet | all | v127 | 14 | 32 | 6 | 32 | 3 | 19 |
| training/2020-03 | reduce | quiet | all | v137 | 17 | 49 | 0 | 0 | 14 | 106 |
| training/2020-03 | reduce | quiet | below | v127 | 3 | 10 | 2 | 10 | 1 | 5 |
| training/2020-03 | reduce | quiet | below | v137 | 4 | 5 | 0 | 0 | 6 | 31 |
| training/2020-03 | reduce | quiet | capped | v127 | 12 | 25 | 3 | 22 | 1 | 11 |
| training/2020-03 | reduce | quiet | capped | v137 | 10 | 41 | 0 | 0 | 11 | 78 |
| training/2020-03 | reduce | first | all | v127 | 6 | 26 | 0 | 0 | 0 | 0 |
| training/2020-03 | reduce | first | all | v137 | 4 | 19 | 0 | 0 | 8 | 39 |
| training/2020-03 | reduce | first | below | v127 | 3 | 11 | 0 | 0 | 0 | 0 |
| training/2020-03 | reduce | first | below | v137 | 3 | 6 | 0 | 0 | 5 | 27 |
| training/2020-03 | reduce | first | capped | v127 | 3 | 15 | 0 | 0 | 0 | 0 |
| training/2020-03 | reduce | first | capped | v137 | 1 | 13 | 0 | 0 | 3 | 12 |
| training/2020-03 | reduce | restart | all | v127 | 11 | 25 | 0 | 0 | 0 | 0 |
| training/2020-03 | reduce | restart | all | v137 | 7 | 22 | 0 | 0 | 12 | 75 |
| training/2020-03 | reduce | restart | below | v127 | 1 | 4 | 0 | 0 | 0 | 0 |
| training/2020-03 | reduce | restart | below | v137 | 0 | 1 | 0 | 0 | 2 | 2 |
| training/2020-03 | reduce | restart | capped | v127 | 10 | 21 | 0 | 0 | 0 | 0 |
| training/2020-03 | reduce | restart | capped | v137 | 7 | 23 | 0 | 0 | 10 | 71 |
| training/2020-03 | increase | quiet | all | v127 | 8 | 19 | 3 | 8 | 0 | 4 |
| training/2020-03 | increase | quiet | all | v137 | 8 | 8 | 3 | 17 | 14 | 209 |
| training/2020-03 | increase | quiet | below | v127 | 5 | 17 | 2 | 3 | 2 | 4 |
| training/2020-03 | increase | quiet | below | v137 | 7 | 7 | 2 | 13 | 6 | 62 |
| training/2020-03 | increase | quiet | capped | v127 | 1 | 2 | 0 | 1 | 0 | 0 |
| training/2020-03 | increase | quiet | capped | v137 | 4 | 9 | 0 | 0 | 5 | 139 |
| training/2020-03 | increase | first | all | v127 | 5 | 15 | 0 | 0 | 0 | 0 |
| training/2020-03 | increase | first | all | v137 | 7 | 7 | 0 | 4 | 0 | 0 |
| training/2020-03 | increase | first | below | v127 | 5 | 15 | 0 | 0 | 0 | 0 |
| training/2020-03 | increase | first | below | v137 | 7 | 7 | 0 | 4 | 0 | 0 |
| training/2020-03 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-03 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-03 | increase | restart | all | v127 | 3 | 8 | 0 | 2 | 0 | 0 |
| training/2020-03 | increase | restart | all | v137 | 6 | 14 | 0 | 0 | 9 | 196 |
| training/2020-03 | increase | restart | below | v127 | 2 | 6 | 0 | 1 | 0 | 0 |
| training/2020-03 | increase | restart | below | v137 | 2 | 5 | 0 | 0 | 4 | 57 |
| training/2020-03 | increase | restart | capped | v127 | 1 | 2 | 0 | 1 | 0 | 0 |
| training/2020-03 | increase | restart | capped | v137 | 4 | 9 | 0 | 0 | 5 | 139 |
| training/2020-04 | reduce | quiet | all | v127 | 1 | 4 | 0 | 5 | 0 | 2 |
| training/2020-04 | reduce | quiet | all | v137 | 1 | 8 | 0 | 0 | 1 | 12 |
| training/2020-04 | reduce | quiet | below | v127 | 0 | 4 | 0 | 5 | 0 | 2 |
| training/2020-04 | reduce | quiet | below | v137 | 0 | 7 | 0 | 0 | 1 | 1 |
| training/2020-04 | reduce | quiet | capped | v127 | 1 | 0 | 0 | 1 | 0 | 0 |
| training/2020-04 | reduce | quiet | capped | v137 | 1 | 2 | 0 | 0 | 0 | 10 |
| training/2020-04 | reduce | first | all | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 1 | 2 |
| training/2020-04 | reduce | first | below | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 1 | 1 |
| training/2020-04 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| training/2020-04 | reduce | restart | all | v127 | 1 | 2 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | restart | all | v137 | 0 | 1 | 0 | 0 | 1 | 17 |
| training/2020-04 | reduce | restart | below | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 7 |
| training/2020-04 | reduce | restart | capped | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | reduce | restart | capped | v137 | 1 | 0 | 0 | 0 | 0 | 11 |
| training/2020-04 | increase | quiet | all | v127 | 0 | 0 | 0 | 1 | 0 | 2 |
| training/2020-04 | increase | quiet | all | v137 | 0 | 0 | 2 | 3 | 2 | 26 |
| training/2020-04 | increase | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 2 |
| training/2020-04 | increase | quiet | below | v137 | 0 | 0 | 1 | 1 | 0 | 14 |
| training/2020-04 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | quiet | capped | v137 | 1 | 0 | 0 | 0 | 1 | 12 |
| training/2020-04 | increase | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-04 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-04 | increase | restart | all | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | restart | all | v137 | 1 | 0 | 0 | 0 | 1 | 26 |
| training/2020-04 | increase | restart | below | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 14 |
| training/2020-04 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-04 | increase | restart | capped | v137 | 1 | 0 | 0 | 0 | 1 | 12 |
| training/2020-05 | reduce | quiet | all | v127 | 3 | 10 | 2 | 10 | 0 | 10 |
| training/2020-05 | reduce | quiet | all | v137 | 6 | 11 | 0 | 0 | 2 | 39 |
| training/2020-05 | reduce | quiet | below | v127 | 2 | 11 | 2 | 11 | 0 | 6 |
| training/2020-05 | reduce | quiet | below | v137 | 4 | 5 | 0 | 0 | 1 | 29 |
| training/2020-05 | reduce | quiet | capped | v127 | 1 | 3 | 1 | 7 | 0 | 0 |
| training/2020-05 | reduce | quiet | capped | v137 | 2 | 5 | 0 | 0 | 1 | 11 |
| training/2020-05 | reduce | first | all | v127 | 0 | 17 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | first | all | v137 | 1 | 1 | 0 | 0 | 1 | 30 |
| training/2020-05 | reduce | first | below | v127 | 0 | 17 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | first | below | v137 | 0 | 2 | 0 | 0 | 1 | 27 |
| training/2020-05 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 1 | 2 |
| training/2020-05 | reduce | restart | all | v127 | 3 | 3 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | restart | all | v137 | 5 | 7 | 0 | 0 | 1 | 12 |
| training/2020-05 | reduce | restart | below | v127 | 2 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | restart | below | v137 | 2 | 0 | 0 | 0 | 2 | 5 |
| training/2020-05 | reduce | restart | capped | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| training/2020-05 | reduce | restart | capped | v137 | 2 | 4 | 0 | 0 | 0 | 10 |
| training/2020-05 | increase | quiet | all | v127 | 1 | 7 | 1 | 0 | 0 | 2 |
| training/2020-05 | increase | quiet | all | v137 | 2 | 2 | 0 | 2 | 3 | 46 |
| training/2020-05 | increase | quiet | below | v127 | 1 | 6 | 1 | 0 | 0 | 3 |
| training/2020-05 | increase | quiet | below | v137 | 2 | 2 | 0 | 1 | 1 | 24 |
| training/2020-05 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | quiet | capped | v137 | 0 | 4 | 0 | 0 | 2 | 18 |
| training/2020-05 | increase | first | all | v127 | 1 | 5 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | first | all | v137 | 2 | 2 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | first | below | v127 | 1 | 5 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | first | below | v137 | 2 | 2 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-05 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-05 | increase | restart | all | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | restart | all | v137 | 0 | 6 | 0 | 0 | 3 | 40 |
| training/2020-05 | increase | restart | below | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | restart | below | v137 | 0 | 1 | 0 | 0 | 1 | 23 |
| training/2020-05 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-05 | increase | restart | capped | v137 | 0 | 4 | 0 | 0 | 2 | 18 |
| training/2020-06 | reduce | quiet | all | v127 | 0 | 1 | 0 | 1 | 0 | 0 |
| training/2020-06 | reduce | quiet | all | v137 | 0 | 1 | 0 | 0 | 1 | 6 |
| training/2020-06 | reduce | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| training/2020-06 | reduce | quiet | capped | v127 | 0 | 1 | 0 | 1 | 0 | 0 |
| training/2020-06 | reduce | quiet | capped | v137 | 0 | 1 | 0 | 0 | 1 | 5 |
| training/2020-06 | reduce | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| training/2020-06 | reduce | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| training/2020-06 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | all | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | all | v137 | 0 | 0 | 0 | 0 | 1 | 6 |
| training/2020-06 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | capped | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-06 | reduce | restart | capped | v137 | 0 | 0 | 0 | 0 | 1 | 6 |
| training/2020-06 | increase | quiet | all | v127 | 1 | 4 | 0 | 2 | 0 | 2 |
| training/2020-06 | increase | quiet | all | v137 | 0 | 3 | 1 | 2 | 1 | 7 |
| training/2020-06 | increase | quiet | below | v127 | 1 | 3 | 0 | 2 | 0 | 3 |
| training/2020-06 | increase | quiet | below | v137 | 0 | 3 | 1 | 1 | 1 | 5 |
| training/2020-06 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 2 |
| training/2020-06 | increase | first | all | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | first | all | v137 | 0 | 3 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | first | below | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | first | below | v137 | 0 | 3 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-06 | increase | restart | all | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | restart | all | v137 | 0 | 1 | 0 | 0 | 1 | 6 |
| training/2020-06 | increase | restart | below | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | restart | below | v137 | 0 | 1 | 0 | 0 | 1 | 4 |
| training/2020-06 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-06 | increase | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 2 |
| training/2020-07 | reduce | quiet | all | v127 | 0 | 1 | 2 | 1 | 0 | 0 |
| training/2020-07 | reduce | quiet | all | v137 | 2 | 0 | 0 | 0 | 0 | 3 |
| training/2020-07 | reduce | quiet | below | v127 | 0 | 0 | 1 | 0 | 0 | 0 |
| training/2020-07 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | quiet | capped | v127 | 0 | 1 | 2 | 1 | 0 | 0 |
| training/2020-07 | reduce | quiet | capped | v137 | 2 | 0 | 0 | 0 | 0 | 3 |
| training/2020-07 | reduce | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | all | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | all | v137 | 1 | 0 | 0 | 0 | 1 | 3 |
| training/2020-07 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | capped | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-07 | reduce | restart | capped | v137 | 1 | 0 | 0 | 0 | 1 | 3 |
| training/2020-07 | increase | quiet | all | v127 | 0 | 0 | 0 | 2 | 0 | 1 |
| training/2020-07 | increase | quiet | all | v137 | 0 | 2 | 0 | 2 | 1 | 15 |
| training/2020-07 | increase | quiet | below | v127 | 0 | 0 | 0 | 1 | 0 | 1 |
| training/2020-07 | increase | quiet | below | v137 | 0 | 1 | 0 | 1 | 0 | 9 |
| training/2020-07 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | quiet | capped | v137 | 1 | 0 | 0 | 0 | 0 | 7 |
| training/2020-07 | increase | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | first | all | v137 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | first | below | v137 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-07 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-07 | increase | restart | all | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | restart | all | v137 | 1 | 2 | 0 | 0 | 0 | 14 |
| training/2020-07 | increase | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | restart | below | v137 | 0 | 2 | 0 | 0 | 0 | 7 |
| training/2020-07 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-07 | increase | restart | capped | v137 | 1 | 0 | 0 | 0 | 0 | 7 |
| training/2020-08 | reduce | quiet | all | v127 | 1 | 1 | 1 | 5 | 1 | 1 |
| training/2020-08 | reduce | quiet | all | v137 | 2 | 8 | 0 | 0 | 5 | 6 |
| training/2020-08 | reduce | quiet | below | v127 | 1 | 2 | 1 | 4 | 0 | 0 |
| training/2020-08 | reduce | quiet | below | v137 | 3 | 6 | 0 | 0 | 2 | 3 |
| training/2020-08 | reduce | quiet | capped | v127 | 1 | 0 | 1 | 1 | 0 | 0 |
| training/2020-08 | reduce | quiet | capped | v137 | 1 | 2 | 0 | 0 | 1 | 3 |
| training/2020-08 | reduce | first | all | v127 | 1 | 1 | 0 | 1 | 0 | 0 |
| training/2020-08 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 2 | 4 |
| training/2020-08 | reduce | first | below | v127 | 1 | 1 | 0 | 1 | 0 | 0 |
| training/2020-08 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 2 | 3 |
| training/2020-08 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| training/2020-08 | reduce | restart | all | v127 | 1 | 1 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | restart | all | v137 | 1 | 1 | 0 | 0 | 4 | 9 |
| training/2020-08 | reduce | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 3 | 6 |
| training/2020-08 | reduce | restart | capped | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | reduce | restart | capped | v137 | 1 | 1 | 0 | 0 | 1 | 3 |
| training/2020-08 | increase | quiet | all | v127 | 0 | 4 | 2 | 7 | 0 | 0 |
| training/2020-08 | increase | quiet | all | v137 | 1 | 2 | 2 | 9 | 1 | 34 |
| training/2020-08 | increase | quiet | below | v127 | 0 | 4 | 1 | 2 | 0 | 0 |
| training/2020-08 | increase | quiet | below | v137 | 1 | 2 | 1 | 8 | 1 | 15 |
| training/2020-08 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | quiet | capped | v137 | 0 | 3 | 0 | 0 | 0 | 16 |
| training/2020-08 | increase | first | all | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | first | all | v137 | 1 | 2 | 0 | 5 | 0 | 0 |
| training/2020-08 | increase | first | below | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | first | below | v137 | 1 | 2 | 0 | 5 | 0 | 0 |
| training/2020-08 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-08 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-08 | increase | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | all | v137 | 0 | 3 | 0 | 0 | 1 | 31 |
| training/2020-08 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 1 | 15 |
| training/2020-08 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-08 | increase | restart | capped | v137 | 0 | 3 | 0 | 0 | 0 | 16 |
| training/2020-09 | reduce | quiet | all | v127 | 2 | 11 | 3 | 6 | 1 | 0 |
| training/2020-09 | reduce | quiet | all | v137 | 4 | 25 | 0 | 0 | 1 | 9 |
| training/2020-09 | reduce | quiet | below | v127 | 0 | 5 | 1 | 6 | 1 | 0 |
| training/2020-09 | reduce | quiet | below | v137 | 0 | 13 | 0 | 0 | 0 | 1 |
| training/2020-09 | reduce | quiet | capped | v127 | 2 | 6 | 2 | 2 | 0 | 0 |
| training/2020-09 | reduce | quiet | capped | v137 | 3 | 14 | 0 | 0 | 2 | 6 |
| training/2020-09 | reduce | first | all | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | below | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | restart | all | v127 | 2 | 11 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | restart | all | v137 | 3 | 14 | 0 | 0 | 2 | 20 |
| training/2020-09 | reduce | restart | below | v127 | 0 | 5 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 14 |
| training/2020-09 | reduce | restart | capped | v127 | 2 | 6 | 0 | 0 | 0 | 0 |
| training/2020-09 | reduce | restart | capped | v137 | 3 | 10 | 0 | 0 | 2 | 10 |
| training/2020-09 | increase | quiet | all | v127 | 1 | 6 | 0 | 4 | 1 | 0 |
| training/2020-09 | increase | quiet | all | v137 | 2 | 4 | 0 | 3 | 2 | 31 |
| training/2020-09 | increase | quiet | below | v127 | 1 | 4 | 0 | 3 | 0 | 0 |
| training/2020-09 | increase | quiet | below | v137 | 1 | 3 | 0 | 1 | 0 | 7 |
| training/2020-09 | increase | quiet | capped | v127 | 1 | 2 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | quiet | capped | v137 | 1 | 3 | 0 | 0 | 2 | 22 |
| training/2020-09 | increase | first | all | v127 | 1 | 4 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | first | all | v137 | 1 | 3 | 0 | 1 | 0 | 0 |
| training/2020-09 | increase | first | below | v127 | 1 | 4 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | first | below | v137 | 1 | 3 | 0 | 1 | 0 | 0 |
| training/2020-09 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-09 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-09 | increase | restart | all | v127 | 1 | 2 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | restart | all | v137 | 1 | 2 | 0 | 0 | 2 | 30 |
| training/2020-09 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 7 |
| training/2020-09 | increase | restart | capped | v127 | 1 | 2 | 0 | 0 | 0 | 0 |
| training/2020-09 | increase | restart | capped | v137 | 1 | 3 | 0 | 0 | 2 | 22 |
| training/2020-10 | reduce | quiet | all | v127 | 1 | 2 | 0 | 2 | 0 | 2 |
| training/2020-10 | reduce | quiet | all | v137 | 0 | 3 | 0 | 0 | 1 | 20 |
| training/2020-10 | reduce | quiet | below | v127 | 1 | 3 | 0 | 4 | 0 | 0 |
| training/2020-10 | reduce | quiet | below | v137 | 0 | 1 | 0 | 0 | 1 | 5 |
| training/2020-10 | reduce | quiet | capped | v127 | 0 | 0 | 0 | 2 | 0 | 1 |
| training/2020-10 | reduce | quiet | capped | v137 | 0 | 2 | 0 | 0 | 0 | 15 |
| training/2020-10 | reduce | first | all | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | first | all | v137 | 0 | 1 | 0 | 0 | 1 | 9 |
| training/2020-10 | reduce | first | below | v127 | 1 | 2 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 1 | 4 |
| training/2020-10 | reduce | first | capped | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 6 |
| training/2020-10 | reduce | restart | all | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | restart | all | v137 | 0 | 0 | 0 | 0 | 0 | 13 |
| training/2020-10 | reduce | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 2 |
| training/2020-10 | reduce | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | reduce | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 11 |
| training/2020-10 | increase | quiet | all | v127 | 2 | 0 | 0 | 1 | 0 | 0 |
| training/2020-10 | increase | quiet | all | v137 | 1 | 1 | 0 | 0 | 1 | 7 |
| training/2020-10 | increase | quiet | below | v127 | 1 | 0 | 0 | 1 | 0 | 0 |
| training/2020-10 | increase | quiet | below | v137 | 1 | 1 | 0 | 0 | 0 | 4 |
| training/2020-10 | increase | quiet | capped | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | quiet | capped | v137 | 0 | 0 | 0 | 0 | 1 | 3 |
| training/2020-10 | increase | first | all | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | first | all | v137 | 1 | 1 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | first | below | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | first | below | v137 | 1 | 1 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-10 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-10 | increase | restart | all | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | restart | all | v137 | 0 | 0 | 0 | 0 | 1 | 7 |
| training/2020-10 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 4 |
| training/2020-10 | increase | restart | capped | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| training/2020-10 | increase | restart | capped | v137 | 0 | 0 | 0 | 0 | 1 | 3 |
| training/2020-11 | reduce | quiet | all | v127 | 3 | 8 | 0 | 9 | 4 | 5 |
| training/2020-11 | reduce | quiet | all | v137 | 2 | 15 | 0 | 0 | 6 | 22 |
| training/2020-11 | reduce | quiet | below | v127 | 3 | 6 | 0 | 9 | 1 | 1 |
| training/2020-11 | reduce | quiet | below | v137 | 3 | 1 | 0 | 0 | 1 | 13 |
| training/2020-11 | reduce | quiet | capped | v127 | 3 | 6 | 0 | 5 | 0 | 0 |
| training/2020-11 | reduce | quiet | capped | v137 | 1 | 9 | 0 | 0 | 3 | 14 |
| training/2020-11 | reduce | first | all | v127 | 4 | 6 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | first | all | v137 | 1 | 0 | 0 | 0 | 1 | 12 |
| training/2020-11 | reduce | first | below | v127 | 4 | 6 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | first | below | v137 | 1 | 0 | 0 | 0 | 1 | 11 |
| training/2020-11 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| training/2020-11 | reduce | restart | all | v127 | 3 | 7 | 0 | 1 | 0 | 0 |
| training/2020-11 | reduce | restart | all | v137 | 1 | 6 | 0 | 0 | 5 | 19 |
| training/2020-11 | reduce | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-11 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 2 | 3 |
| training/2020-11 | reduce | restart | capped | v127 | 3 | 6 | 0 | 1 | 0 | 0 |
| training/2020-11 | reduce | restart | capped | v137 | 1 | 6 | 0 | 0 | 3 | 16 |
| training/2020-11 | increase | quiet | all | v127 | 8 | 19 | 1 | 1 | 2 | 3 |
| training/2020-11 | increase | quiet | all | v137 | 8 | 8 | 3 | 7 | 9 | 70 |
| training/2020-11 | increase | quiet | below | v127 | 8 | 17 | 0 | 0 | 1 | 4 |
| training/2020-11 | increase | quiet | below | v137 | 6 | 8 | 2 | 4 | 7 | 28 |
| training/2020-11 | increase | quiet | capped | v127 | 1 | 1 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | quiet | capped | v137 | 2 | 12 | 0 | 0 | 2 | 30 |
| training/2020-11 | increase | first | all | v127 | 5 | 13 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | first | all | v137 | 6 | 8 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | first | below | v127 | 5 | 13 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | first | below | v137 | 6 | 8 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-11 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-11 | increase | restart | all | v127 | 5 | 9 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | restart | all | v137 | 5 | 15 | 0 | 0 | 6 | 55 |
| training/2020-11 | increase | restart | below | v127 | 4 | 8 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | restart | below | v137 | 3 | 3 | 0 | 0 | 4 | 25 |
| training/2020-11 | increase | restart | capped | v127 | 1 | 1 | 0 | 0 | 0 | 0 |
| training/2020-11 | increase | restart | capped | v137 | 2 | 12 | 0 | 0 | 2 | 30 |
| training/2020-12 | reduce | quiet | all | v127 | 3 | 19 | 0 | 13 | 1 | 1 |
| training/2020-12 | reduce | quiet | all | v137 | 2 | 40 | 0 | 0 | 3 | 19 |
| training/2020-12 | reduce | quiet | below | v127 | 2 | 0 | 0 | 1 | 0 | 1 |
| training/2020-12 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 1 | 3 |
| training/2020-12 | reduce | quiet | capped | v127 | 2 | 19 | 0 | 10 | 0 | 0 |
| training/2020-12 | reduce | quiet | capped | v137 | 2 | 36 | 0 | 0 | 2 | 20 |
| training/2020-12 | reduce | first | all | v127 | 2 | 1 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | first | all | v137 | 1 | 0 | 0 | 0 | 1 | 3 |
| training/2020-12 | reduce | first | below | v127 | 2 | 1 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 1 | 2 |
| training/2020-12 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | first | capped | v137 | 1 | 0 | 0 | 0 | 0 | 1 |
| training/2020-12 | reduce | restart | all | v127 | 2 | 19 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | restart | all | v137 | 2 | 20 | 0 | 0 | 1 | 36 |
| training/2020-12 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| training/2020-12 | reduce | restart | capped | v127 | 2 | 19 | 0 | 0 | 0 | 0 |
| training/2020-12 | reduce | restart | capped | v137 | 2 | 23 | 0 | 0 | 1 | 32 |
| training/2020-12 | increase | quiet | all | v127 | 2 | 4 | 2 | 4 | 0 | 1 |
| training/2020-12 | increase | quiet | all | v137 | 1 | 2 | 2 | 8 | 5 | 54 |
| training/2020-12 | increase | quiet | below | v127 | 2 | 4 | 1 | 2 | 0 | 1 |
| training/2020-12 | increase | quiet | below | v137 | 1 | 2 | 2 | 4 | 3 | 22 |
| training/2020-12 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | quiet | capped | v137 | 1 | 0 | 0 | 0 | 1 | 32 |
| training/2020-12 | increase | first | all | v127 | 2 | 4 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | first | all | v137 | 1 | 2 | 0 | 1 | 0 | 0 |
| training/2020-12 | increase | first | below | v127 | 2 | 4 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | first | below | v137 | 1 | 2 | 0 | 1 | 0 | 0 |
| training/2020-12 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-12 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| training/2020-12 | increase | restart | all | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | restart | all | v137 | 2 | 2 | 0 | 0 | 3 | 52 |
| training/2020-12 | increase | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | restart | below | v137 | 1 | 2 | 0 | 0 | 2 | 20 |
| training/2020-12 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| training/2020-12 | increase | restart | capped | v137 | 1 | 0 | 0 | 0 | 1 | 32 |
| calibration/2021-01 | reduce | quiet | all | v127 | 9 | 44 | 2 | 42 | 1 | 34 |
| calibration/2021-01 | reduce | quiet | all | v137 | 8 | 97 | 0 | 0 | 12 | 219 |
| calibration/2021-01 | reduce | quiet | below | v127 | 0 | 9 | 0 | 20 | 0 | 7 |
| calibration/2021-01 | reduce | quiet | below | v137 | 1 | 31 | 0 | 0 | 4 | 65 |
| calibration/2021-01 | reduce | quiet | capped | v127 | 9 | 39 | 1 | 20 | 1 | 23 |
| calibration/2021-01 | reduce | quiet | capped | v137 | 8 | 64 | 0 | 0 | 7 | 156 |
| calibration/2021-01 | reduce | first | all | v127 | 1 | 37 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | reduce | first | all | v137 | 1 | 34 | 0 | 0 | 4 | 149 |
| calibration/2021-01 | reduce | first | below | v127 | 0 | 14 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | reduce | first | below | v137 | 1 | 4 | 0 | 0 | 3 | 61 |
| calibration/2021-01 | reduce | first | capped | v127 | 1 | 23 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | reduce | first | capped | v137 | 0 | 29 | 0 | 0 | 1 | 89 |
| calibration/2021-01 | reduce | restart | all | v127 | 9 | 41 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | reduce | restart | all | v137 | 5 | 45 | 0 | 0 | 10 | 88 |
| calibration/2021-01 | reduce | restart | below | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | reduce | restart | below | v137 | 0 | 3 | 0 | 0 | 1 | 28 |
| calibration/2021-01 | reduce | restart | capped | v127 | 9 | 39 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | reduce | restart | capped | v137 | 5 | 38 | 0 | 0 | 9 | 64 |
| calibration/2021-01 | increase | quiet | all | v127 | 1 | 14 | 1 | 17 | 3 | 9 |
| calibration/2021-01 | increase | quiet | all | v137 | 2 | 16 | 2 | 16 | 5 | 166 |
| calibration/2021-01 | increase | quiet | below | v127 | 1 | 10 | 0 | 5 | 3 | 5 |
| calibration/2021-01 | increase | quiet | below | v137 | 2 | 12 | 1 | 9 | 3 | 66 |
| calibration/2021-01 | increase | quiet | capped | v127 | 0 | 8 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | increase | quiet | capped | v137 | 0 | 10 | 0 | 0 | 2 | 94 |
| calibration/2021-01 | increase | first | all | v127 | 1 | 8 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | increase | first | all | v137 | 2 | 12 | 0 | 1 | 0 | 0 |
| calibration/2021-01 | increase | first | below | v127 | 1 | 8 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | increase | first | below | v137 | 2 | 12 | 0 | 1 | 0 | 0 |
| calibration/2021-01 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-01 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-01 | increase | restart | all | v127 | 3 | 15 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | increase | restart | all | v137 | 1 | 19 | 0 | 0 | 4 | 151 |
| calibration/2021-01 | increase | restart | below | v127 | 3 | 7 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | increase | restart | below | v137 | 1 | 9 | 0 | 0 | 2 | 57 |
| calibration/2021-01 | increase | restart | capped | v127 | 0 | 8 | 0 | 0 | 0 | 0 |
| calibration/2021-01 | increase | restart | capped | v137 | 0 | 10 | 0 | 0 | 2 | 94 |
| calibration/2021-02 | reduce | quiet | all | v127 | 3 | 20 | 0 | 18 | 1 | 5 |
| calibration/2021-02 | reduce | quiet | all | v137 | 2 | 22 | 0 | 0 | 2 | 53 |
| calibration/2021-02 | reduce | quiet | below | v127 | 0 | 1 | 0 | 0 | 0 | 1 |
| calibration/2021-02 | reduce | quiet | below | v137 | 0 | 1 | 0 | 0 | 0 | 8 |
| calibration/2021-02 | reduce | quiet | capped | v127 | 3 | 18 | 0 | 16 | 1 | 5 |
| calibration/2021-02 | reduce | quiet | capped | v137 | 2 | 18 | 0 | 0 | 2 | 48 |
| calibration/2021-02 | reduce | first | all | v127 | 1 | 9 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | first | all | v137 | 0 | 7 | 0 | 0 | 1 | 34 |
| calibration/2021-02 | reduce | first | below | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | first | below | v137 | 0 | 2 | 0 | 0 | 0 | 7 |
| calibration/2021-02 | reduce | first | capped | v127 | 1 | 7 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | first | capped | v137 | 0 | 5 | 0 | 0 | 1 | 27 |
| calibration/2021-02 | reduce | restart | all | v127 | 3 | 16 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | restart | all | v137 | 2 | 9 | 0 | 0 | 1 | 25 |
| calibration/2021-02 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | restart | capped | v127 | 3 | 16 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | reduce | restart | capped | v137 | 2 | 10 | 0 | 0 | 1 | 24 |
| calibration/2021-02 | increase | quiet | all | v127 | 0 | 5 | 0 | 0 | 0 | 2 |
| calibration/2021-02 | increase | quiet | all | v137 | 0 | 3 | 0 | 8 | 1 | 51 |
| calibration/2021-02 | increase | quiet | below | v127 | 0 | 5 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | quiet | below | v137 | 0 | 1 | 0 | 2 | 0 | 3 |
| calibration/2021-02 | increase | quiet | capped | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | quiet | capped | v137 | 0 | 5 | 0 | 0 | 1 | 45 |
| calibration/2021-02 | increase | first | all | v127 | 0 | 5 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | first | all | v137 | 0 | 1 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | first | below | v127 | 0 | 5 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | first | below | v137 | 0 | 1 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-02 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-02 | increase | restart | all | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | restart | all | v137 | 0 | 5 | 0 | 0 | 1 | 48 |
| calibration/2021-02 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 3 |
| calibration/2021-02 | increase | restart | capped | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| calibration/2021-02 | increase | restart | capped | v137 | 0 | 5 | 0 | 0 | 1 | 45 |
| calibration/2021-03 | reduce | quiet | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | all | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| calibration/2021-03 | reduce | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| calibration/2021-03 | reduce | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | all | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| calibration/2021-03 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | reduce | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| calibration/2021-03 | increase | quiet | all | v127 | 0 | 2 | 0 | 1 | 0 | 1 |
| calibration/2021-03 | increase | quiet | all | v137 | 0 | 2 | 0 | 0 | 0 | 8 |
| calibration/2021-03 | increase | quiet | below | v127 | 0 | 0 | 0 | 1 | 0 | 0 |
| calibration/2021-03 | increase | quiet | below | v137 | 0 | 1 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | quiet | capped | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | quiet | capped | v137 | 0 | 2 | 0 | 0 | 0 | 7 |
| calibration/2021-03 | increase | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | first | all | v137 | 0 | 1 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | first | below | v137 | 0 | 1 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-03 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-03 | increase | restart | all | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | all | v137 | 0 | 2 | 0 | 0 | 0 | 7 |
| calibration/2021-03 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | capped | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| calibration/2021-03 | increase | restart | capped | v137 | 0 | 2 | 0 | 0 | 0 | 7 |
| calibration/2021-04 | reduce | quiet | all | v127 | 1 | 2 | 0 | 4 | 0 | 0 |
| calibration/2021-04 | reduce | quiet | all | v137 | 2 | 4 | 0 | 0 | 0 | 3 |
| calibration/2021-04 | reduce | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | quiet | capped | v127 | 1 | 2 | 0 | 2 | 0 | 0 |
| calibration/2021-04 | reduce | quiet | capped | v137 | 1 | 3 | 0 | 0 | 1 | 4 |
| calibration/2021-04 | reduce | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | all | v127 | 1 | 2 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | all | v137 | 0 | 2 | 0 | 0 | 2 | 5 |
| calibration/2021-04 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | capped | v127 | 1 | 2 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | reduce | restart | capped | v137 | 0 | 2 | 0 | 0 | 2 | 5 |
| calibration/2021-04 | increase | quiet | all | v127 | 0 | 3 | 1 | 0 | 0 | 7 |
| calibration/2021-04 | increase | quiet | all | v137 | 0 | 1 | 0 | 0 | 1 | 44 |
| calibration/2021-04 | increase | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | quiet | capped | v127 | 0 | 10 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | quiet | capped | v137 | 0 | 5 | 0 | 0 | 1 | 40 |
| calibration/2021-04 | increase | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-04 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-04 | increase | restart | all | v127 | 0 | 10 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | all | v137 | 0 | 5 | 0 | 0 | 1 | 40 |
| calibration/2021-04 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | capped | v127 | 0 | 10 | 0 | 0 | 0 | 0 |
| calibration/2021-04 | increase | restart | capped | v137 | 0 | 5 | 0 | 0 | 1 | 40 |
| calibration/2021-05 | reduce | quiet | all | v127 | 1 | 16 | 0 | 12 | 3 | 21 |
| calibration/2021-05 | reduce | quiet | all | v137 | 3 | 36 | 0 | 0 | 6 | 159 |
| calibration/2021-05 | reduce | quiet | below | v127 | 1 | 8 | 0 | 4 | 2 | 14 |
| calibration/2021-05 | reduce | quiet | below | v137 | 0 | 11 | 0 | 0 | 5 | 126 |
| calibration/2021-05 | reduce | quiet | capped | v127 | 1 | 15 | 0 | 7 | 0 | 0 |
| calibration/2021-05 | reduce | quiet | capped | v137 | 3 | 32 | 0 | 0 | 1 | 26 |
| calibration/2021-05 | reduce | first | all | v127 | 3 | 22 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | first | all | v137 | 2 | 33 | 0 | 0 | 3 | 104 |
| calibration/2021-05 | reduce | first | below | v127 | 3 | 22 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | first | below | v137 | 2 | 35 | 0 | 0 | 3 | 101 |
| calibration/2021-05 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| calibration/2021-05 | reduce | restart | all | v127 | 1 | 15 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | restart | all | v137 | 2 | 19 | 0 | 0 | 2 | 39 |
| calibration/2021-05 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 1 |
| calibration/2021-05 | reduce | restart | capped | v127 | 1 | 15 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | reduce | restart | capped | v137 | 3 | 20 | 0 | 0 | 1 | 37 |
| calibration/2021-05 | increase | quiet | all | v127 | 3 | 8 | 0 | 5 | 0 | 5 |
| calibration/2021-05 | increase | quiet | all | v137 | 3 | 9 | 0 | 8 | 7 | 116 |
| calibration/2021-05 | increase | quiet | below | v127 | 3 | 8 | 0 | 1 | 0 | 5 |
| calibration/2021-05 | increase | quiet | below | v137 | 3 | 7 | 0 | 4 | 3 | 56 |
| calibration/2021-05 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | quiet | capped | v137 | 0 | 6 | 0 | 0 | 4 | 56 |
| calibration/2021-05 | increase | first | all | v127 | 3 | 7 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | first | all | v137 | 3 | 7 | 0 | 2 | 0 | 0 |
| calibration/2021-05 | increase | first | below | v127 | 3 | 7 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | first | below | v137 | 3 | 7 | 0 | 2 | 0 | 0 |
| calibration/2021-05 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-05 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-05 | increase | restart | all | v127 | 0 | 6 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | restart | all | v137 | 1 | 10 | 0 | 0 | 6 | 108 |
| calibration/2021-05 | increase | restart | below | v127 | 0 | 6 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | restart | below | v137 | 1 | 4 | 0 | 0 | 2 | 52 |
| calibration/2021-05 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| calibration/2021-05 | increase | restart | capped | v137 | 0 | 6 | 0 | 0 | 4 | 56 |
| calibration/2021-06 | reduce | quiet | all | v127 | 0 | 19 | 0 | 11 | 1 | 28 |
| calibration/2021-06 | reduce | quiet | all | v137 | 0 | 27 | 0 | 0 | 4 | 111 |
| calibration/2021-06 | reduce | quiet | below | v127 | 0 | 3 | 0 | 5 | 1 | 7 |
| calibration/2021-06 | reduce | quiet | below | v137 | 1 | 6 | 0 | 0 | 1 | 27 |
| calibration/2021-06 | reduce | quiet | capped | v127 | 0 | 15 | 0 | 5 | 0 | 22 |
| calibration/2021-06 | reduce | quiet | capped | v137 | 0 | 22 | 0 | 0 | 2 | 83 |
| calibration/2021-06 | reduce | first | all | v127 | 1 | 40 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | reduce | first | all | v137 | 1 | 34 | 0 | 0 | 3 | 78 |
| calibration/2021-06 | reduce | first | below | v127 | 1 | 9 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | reduce | first | below | v137 | 1 | 6 | 0 | 0 | 1 | 24 |
| calibration/2021-06 | reduce | first | capped | v127 | 0 | 31 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | reduce | first | capped | v137 | 0 | 27 | 0 | 0 | 2 | 55 |
| calibration/2021-06 | reduce | restart | all | v127 | 0 | 7 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | reduce | restart | all | v137 | 0 | 6 | 0 | 0 | 0 | 20 |
| calibration/2021-06 | reduce | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 3 |
| calibration/2021-06 | reduce | restart | capped | v127 | 0 | 6 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | reduce | restart | capped | v137 | 0 | 8 | 0 | 0 | 0 | 15 |
| calibration/2021-06 | increase | quiet | all | v127 | 1 | 16 | 0 | 9 | 0 | 8 |
| calibration/2021-06 | increase | quiet | all | v137 | 0 | 7 | 0 | 10 | 4 | 61 |
| calibration/2021-06 | increase | quiet | below | v127 | 0 | 12 | 0 | 5 | 0 | 2 |
| calibration/2021-06 | increase | quiet | below | v137 | 0 | 7 | 0 | 4 | 1 | 16 |
| calibration/2021-06 | increase | quiet | capped | v127 | 1 | 10 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | increase | quiet | capped | v137 | 0 | 5 | 0 | 0 | 3 | 40 |
| calibration/2021-06 | increase | first | all | v127 | 0 | 12 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | increase | first | all | v137 | 0 | 7 | 0 | 1 | 0 | 0 |
| calibration/2021-06 | increase | first | below | v127 | 0 | 12 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | increase | first | below | v137 | 0 | 7 | 0 | 1 | 0 | 0 |
| calibration/2021-06 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-06 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| calibration/2021-06 | increase | restart | all | v127 | 1 | 12 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | increase | restart | all | v137 | 0 | 7 | 0 | 0 | 4 | 54 |
| calibration/2021-06 | increase | restart | below | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | increase | restart | below | v137 | 0 | 2 | 0 | 0 | 1 | 14 |
| calibration/2021-06 | increase | restart | capped | v127 | 1 | 10 | 0 | 0 | 0 | 0 |
| calibration/2021-06 | increase | restart | capped | v137 | 0 | 5 | 0 | 0 | 3 | 40 |
| diagnosis/2021-07 | reduce | quiet | all | v127 | 1 | 3 | 0 | 6 | 1 | 0 |
| diagnosis/2021-07 | reduce | quiet | all | v137 | 1 | 16 | 0 | 0 | 2 | 26 |
| diagnosis/2021-07 | reduce | quiet | below | v127 | 2 | 3 | 0 | 5 | 0 | 0 |
| diagnosis/2021-07 | reduce | quiet | below | v137 | 1 | 19 | 0 | 0 | 1 | 5 |
| diagnosis/2021-07 | reduce | quiet | capped | v127 | 0 | 0 | 0 | 2 | 0 | 0 |
| diagnosis/2021-07 | reduce | quiet | capped | v137 | 0 | 2 | 0 | 0 | 1 | 16 |
| diagnosis/2021-07 | reduce | first | all | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | first | all | v137 | 0 | 3 | 0 | 0 | 2 | 8 |
| diagnosis/2021-07 | reduce | first | below | v127 | 1 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | first | below | v137 | 0 | 1 | 0 | 0 | 1 | 4 |
| diagnosis/2021-07 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | first | capped | v137 | 0 | 1 | 0 | 0 | 1 | 5 |
| diagnosis/2021-07 | reduce | restart | all | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | restart | all | v137 | 0 | 4 | 0 | 0 | 1 | 27 |
| diagnosis/2021-07 | reduce | restart | below | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | restart | below | v137 | 0 | 4 | 0 | 0 | 1 | 15 |
| diagnosis/2021-07 | reduce | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | reduce | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 12 |
| diagnosis/2021-07 | increase | quiet | all | v127 | 0 | 4 | 1 | 3 | 0 | 0 |
| diagnosis/2021-07 | increase | quiet | all | v137 | 0 | 3 | 0 | 5 | 0 | 16 |
| diagnosis/2021-07 | increase | quiet | below | v127 | 0 | 4 | 0 | 3 | 0 | 0 |
| diagnosis/2021-07 | increase | quiet | below | v137 | 0 | 3 | 0 | 3 | 0 | 4 |
| diagnosis/2021-07 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 12 |
| diagnosis/2021-07 | increase | first | all | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | first | all | v137 | 0 | 3 | 0 | 1 | 0 | 0 |
| diagnosis/2021-07 | increase | first | below | v127 | 0 | 4 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | first | below | v137 | 0 | 3 | 0 | 1 | 0 | 0 |
| diagnosis/2021-07 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-07 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-07 | increase | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | all | v137 | 0 | 0 | 0 | 0 | 0 | 16 |
| diagnosis/2021-07 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 4 |
| diagnosis/2021-07 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-07 | increase | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 12 |
| diagnosis/2021-08 | reduce | quiet | all | v127 | 0 | 2 | 0 | 1 | 0 | 3 |
| diagnosis/2021-08 | reduce | quiet | all | v137 | 0 | 5 | 0 | 0 | 0 | 11 |
| diagnosis/2021-08 | reduce | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 2 |
| diagnosis/2021-08 | reduce | quiet | capped | v127 | 0 | 2 | 0 | 0 | 0 | 3 |
| diagnosis/2021-08 | reduce | quiet | capped | v137 | 0 | 5 | 0 | 0 | 0 | 9 |
| diagnosis/2021-08 | reduce | first | all | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | first | all | v137 | 0 | 4 | 0 | 0 | 0 | 7 |
| diagnosis/2021-08 | reduce | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 2 |
| diagnosis/2021-08 | reduce | first | capped | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | first | capped | v137 | 0 | 3 | 0 | 0 | 0 | 6 |
| diagnosis/2021-08 | reduce | restart | all | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | restart | all | v137 | 0 | 1 | 0 | 0 | 0 | 4 |
| diagnosis/2021-08 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | restart | capped | v127 | 0 | 2 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | reduce | restart | capped | v137 | 0 | 1 | 0 | 0 | 0 | 4 |
| diagnosis/2021-08 | increase | quiet | all | v127 | 0 | 3 | 0 | 3 | 0 | 0 |
| diagnosis/2021-08 | increase | quiet | all | v137 | 0 | 3 | 0 | 2 | 1 | 6 |
| diagnosis/2021-08 | increase | quiet | below | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | quiet | below | v137 | 0 | 3 | 0 | 2 | 1 | 2 |
| diagnosis/2021-08 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 4 |
| diagnosis/2021-08 | increase | first | all | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | first | all | v137 | 0 | 3 | 0 | 1 | 0 | 0 |
| diagnosis/2021-08 | increase | first | below | v127 | 0 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | first | below | v137 | 0 | 3 | 0 | 1 | 0 | 0 |
| diagnosis/2021-08 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-08 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-08 | increase | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | all | v137 | 0 | 0 | 0 | 0 | 1 | 6 |
| diagnosis/2021-08 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 1 | 2 |
| diagnosis/2021-08 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-08 | increase | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 4 |
| diagnosis/2021-09 | reduce | quiet | all | v127 | 0 | 0 | 0 | 4 | 0 | 6 |
| diagnosis/2021-09 | reduce | quiet | all | v137 | 1 | 10 | 0 | 0 | 2 | 44 |
| diagnosis/2021-09 | reduce | quiet | below | v127 | 0 | 2 | 0 | 0 | 0 | 3 |
| diagnosis/2021-09 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 1 | 35 |
| diagnosis/2021-09 | reduce | quiet | capped | v127 | 0 | 0 | 0 | 1 | 0 | 1 |
| diagnosis/2021-09 | reduce | quiet | capped | v137 | 1 | 9 | 0 | 0 | 1 | 10 |
| diagnosis/2021-09 | reduce | first | all | v127 | 0 | 6 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | first | all | v137 | 0 | 2 | 0 | 0 | 2 | 36 |
| diagnosis/2021-09 | reduce | first | below | v127 | 0 | 5 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | first | below | v137 | 0 | 2 | 0 | 0 | 1 | 33 |
| diagnosis/2021-09 | reduce | first | capped | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 1 | 3 |
| diagnosis/2021-09 | reduce | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | all | v137 | 0 | 4 | 0 | 0 | 1 | 12 |
| diagnosis/2021-09 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | reduce | restart | capped | v137 | 1 | 4 | 0 | 0 | 0 | 12 |
| diagnosis/2021-09 | increase | quiet | all | v127 | 1 | 3 | 0 | 4 | 0 | 1 |
| diagnosis/2021-09 | increase | quiet | all | v137 | 1 | 5 | 0 | 4 | 1 | 25 |
| diagnosis/2021-09 | increase | quiet | below | v127 | 1 | 3 | 0 | 4 | 0 | 1 |
| diagnosis/2021-09 | increase | quiet | below | v137 | 1 | 5 | 0 | 2 | 0 | 12 |
| diagnosis/2021-09 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | quiet | capped | v137 | 0 | 0 | 0 | 0 | 1 | 13 |
| diagnosis/2021-09 | increase | first | all | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | first | all | v137 | 1 | 5 | 0 | 1 | 0 | 0 |
| diagnosis/2021-09 | increase | first | below | v127 | 1 | 3 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | first | below | v137 | 1 | 5 | 0 | 1 | 0 | 0 |
| diagnosis/2021-09 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-09 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-09 | increase | restart | all | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | restart | all | v137 | 0 | 1 | 0 | 0 | 1 | 24 |
| diagnosis/2021-09 | increase | restart | below | v127 | 0 | 1 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | restart | below | v137 | 0 | 1 | 0 | 0 | 0 | 11 |
| diagnosis/2021-09 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-09 | increase | restart | capped | v137 | 0 | 0 | 0 | 0 | 1 | 13 |
| diagnosis/2021-10 | reduce | quiet | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | reduce | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-10 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-10 | increase | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-10 | increase | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | reduce | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | all | v137 | 0 | 0 | 0 | 0 | 0 | 9 |
| diagnosis/2021-11 | increase | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | quiet | capped | v137 | 0 | 4 | 0 | 0 | 0 | 5 |
| diagnosis/2021-11 | increase | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-11 | increase | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | all | v137 | 0 | 4 | 0 | 0 | 0 | 5 |
| diagnosis/2021-11 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-11 | increase | restart | capped | v137 | 0 | 4 | 0 | 0 | 0 | 5 |
| diagnosis/2021-12 | reduce | quiet | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | first | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | reduce | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | quiet | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | first | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | first | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | first | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | first | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | first | capped | v127 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | first | capped | v137 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| diagnosis/2021-12 | increase | restart | all | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | all | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | below | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | below | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | capped | v127 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnosis/2021-12 | increase | restart | capped | v137 | 0 | 0 | 0 | 0 | 0 | 0 |
