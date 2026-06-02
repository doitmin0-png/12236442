# 스마트모빌리티공학실험2 Final Project

학번: 12236442
이름: 이민영

# RTT 기반 실내 위치측위 성능 향상을 위한 Hybrid ML 알고리즘

# 1. Motivation & Intro

본 프로젝트의 목표는 RTT 기반 거리 측정 데이터를 이용하여 사용자의 2차원 위치를 추정하는 것이다.
제공된 데이터는 18개의 기지국 좌표와 각 사용자별 RTT 기반 거리값으로 구성되어 있으며, 최종적으로 각 사용자에 대한 x, y 좌표를 반환해야 한다.

초기에는 RTT 거리값과 기지국 좌표를 이용한 기하학적 위치측위 방식을 적용하였다.
그러나 실내 환경에서는 벽면 반사, 다중경로, 장애물, 특정 기지국의 신호 편향 등으로 인해 RTT 거리값이 실제 거리와 다르게 측정되는 경우가 많았다.

특히 일부 앵커의 거리값이 비정상적으로 커지는 이상치가 존재하였고, 이러한 값이 위치 계산 과정에 그대로 반영되면서 최종 위치가 실제 위치에서 크게 벗어나는 문제가 발생하였다.
따라서 단순한 Trilateration 방식만으로는 안정적인 위치 추정을 수행하기 어렵다고 판단하였다.

Week6에서는 이러한 문제를 줄이기 위해 RANSAC과 WLS를 결합한 기하학적 알고리즘을 구현하였다.
RANSAC은 이상치 앵커를 제거하는 데 효과적이었고, WLS는 거리값에 따라 앵커별 가중치를 다르게 부여하여 초기 위치 추정의 안정성을 높였다.

하지만 RANSAC과 WLS만으로는 RTT 데이터 내부의 비선형 오차나 특정 방향으로 반복되는 편향을 완전히 보정하기 어려웠다.
따라서 최종 알고리즘에서는 기존 위치측위 알고리즘을 완전히 버리는 방식이 아니라, RANSAC/WLS로 얻은 초기 위치와 거리 기반 통계값을 머신러닝 모델의 입력 feature로 사용하는 Hybrid 구조를 선택하였다.

이 구조는 기하학적 알고리즘의 안정성과 머신러닝의 비선형 보정 능력을 함께 활용하기 위한 방식이다.

# 2. 알고리즘 설명

최종 알고리즘은 크게 두 단계로 구성된다.

첫 번째 단계는 RTT 거리값을 이용하여 물리적으로 가능한 초기 위치와 보조 feature를 만드는 과정이다.
두 번째 단계는 생성된 feature를 Gradient Boosting 기반 회귀 모델에 입력하여 최종 위치를 예측하는 과정이다.

입력으로는 사용자 한 명에 대한 18개 RTT 거리값과 18개 기지국의 2차원 좌표를 사용한다.
먼저 거리값에 NaN, inf, 0 이하의 값이 포함될 수 있으므로 이를 안전한 양수 값으로 변환하였다.

이후 원본 거리값뿐만 아니라 다음과 같은 feature를 추가하였다.

Feature 종류 | 설명
원본 RTT 거리값 | 18개 기지국에서 측정된 사용자별 거리 정보   
정렬된 RTT 거리값 | 앵커 순서와 무관한 거리 분포 반영 
RTT 통계값 | 평균, 표준편차, 최솟값, 최댓값, 중앙값, 사분위수 
Weighted centroid 좌표 | 가까운 앵커에 높은 가중치를 부여한 초기 위치  
RANSAC/WLS 위치 좌표 | 이상치 제거와 가중최소제곱 기반 초기 위치
Inlier ratio | RANSAC에서 정상치로 판단된 앵커 비율 
Residual feature | 예측 거리와 측정 거리의 차이 정보  
가까운 앵커 index | 거리값이 작은 상위 앵커 정보 

Weighted centroid는 가까운 앵커일수록 더 큰 영향을 주도록 거리의 역제곱을 가중치로 사용하였다.
이는 실제 사용자 위치와 가까운 앵커의 거리 정보가 상대적으로 더 신뢰도 높을 가능성이 있기 때문이다.

이후 RANSAC과 WLS를 이용하여 물리 기반 위치를 한 번 더 추정하였다.
RANSAC은 일부 앵커 조합을 반복적으로 선택하여 임시 위치를 계산하고, 전체 앵커에 대한 잔차가 작은 조합을 찾는 방식으로 사용하였다.

잔차가 큰 앵커는 이상치일 가능성이 높기 때문에 낮은 신뢰도로 반영하였다.
이후 WLS를 적용하여 선택된 앵커와 잔차 정보를 바탕으로 위치를 반복 보정하였다.

최종 feature는 train.py에서 Gradient Boosting 모델 학습에 사용된다.
main.py에서는 동일한 방식으로 사용자별 feature를 생성한 뒤, 저장된 model.pkl을 불러와 최종 위치를 예측한다.

제출 규격을 맞추기 위해 main.py는 main() 함수를 포함하도록 작성하였다.
main()에서는 DH_FR1.mat 파일을 읽고, 사용자 수는 d_hat.shape[1]에서 동적으로 가져오도록 하였다.

이후 p_hat 배열을 (2, num_user) 형태로 생성하고, 각 사용자 u에 대해 your_algorithm(d_hat[:, u], BS_positions)을 호출하여 결과를 저장하였다.
따라서 최종 반환값은 첫 번째 행이 x 좌표, 두 번째 행이 y 좌표인 numpy 배열이다.

main.py에서는 학습을 다시 수행하지 않는다.
학습은 train.py에서 수행하고, train.py가 저장한 model.pkl을 main.py에서 불러와 추론만 수행하도록 구성하였다.
이렇게 구성한 이유는 hidden test에서는 정답 좌표 p가 제공되지 않을 수 있고, main.py는 채점 시 제한 시간 안에 결과만 반환해야 하기 때문이다.

# 3. Agent AI 활용 방안

이번 프로젝트에서는 ChatGPT를 단순히 코드를 대신 작성하는 용도보다는, 이전 주차 보고서와 최종 제출 규격을 하나의 구조로 정리하는 보조 도구로 활용하였다.

먼저 Week6에서 작성한 RANSAC/WLS 알고리즘, Week7에서 설계한 머신러닝 기반 보정 구조, Week8에서 비교한 Ridge, Random Forest, Gradient Boosting 결과가 서로 자연스럽게 이어지도록 최종 알고리즘 흐름을 정리하는 데 활용하였다.
특히 기존 알고리즘을 완전히 버리는 것이 아니라, RANSAC/WLS 기반 초기 위치 결과를 머신러닝 feature로 활용하는 Hybrid 구조가 보고서 흐름과 가장 잘 맞는다는 점을 확인하였다.

또한 논문 reference를 정리하는 과정에서도 AI를 활용하였다.
처음에는 “RTT indoor positioning NLOS”, “UWB NLOS mitigation WLS”, “WiFi RTT machine learning indoor positioning”, “Random Forest indoor localization”, “Gradient Boosting indoor positioning”과 같은 검색 키워드를 바탕으로 관련 논문 후보를 찾았다.

이후 AI를 이용해 논문 제목, 초록, 핵심 내용을 한국어로 번역하고 요약하였다.
그 과정에서 과제와 직접 관련성이 낮은 논문은 제외하였다.
예를 들어 단순히 센서 종류만 비슷하거나, 딥러닝 구조가 너무 복잡하여 본 과제 데이터 규모와 맞지 않는 논문은 최종 reference에서 제외하였다.                                           

또한 main.py 작성 규격을 해석하는 과정에서도 AI를 활용하였다.
처음에는 scikit-learn 모델의 일반적인 사용 방식에 맞추어 전체 사용자 데이터를 한 번에 예측하는 구조를 검토하였다.
그러나 과제 안내문에서 사용자별로 your_algorithm(d_hat[:, u], BS_positions)을 호출하는 구조를 제시하고 있었기 때문에, 최종적으로는 main() 함수 내부에서 사용자별 반복 구조를 유지하도록 수정하였다.

이 과정에서 model.pkl을 매번 다시 불러오면 실행 시간이 크게 증가할 수 있으므로, 최초 한 번만 모델을 로드하고 이후에는 재사용하는 방식으로 구현하였다.

AI는 코드 오류 가능성을 줄이기 위한 점검에도 활용하였다.
예를 들어 hidden test에서는 정답 좌표 p가 제공되지 않을 수 있으므로 main.py에서는 p를 사용하지 않도록 하였다.
또한 Colab 환경에서 train.py를 실행하여 model.pkl을 생성하고, main.py가 반환하는 p_hat의 자료형과 shape가 규격에 맞는지 확인하는 테스트 절차를 정리하였다.

다만 최종 알고리즘의 방향은 이전 주차에서 직접 진행한 실험 흐름을 기준으로 결정하였다.
AI는 논문 후보 탐색, 번역, 요약, 비교, 제출 파일 구성, 코드 구조 점검에 활용하였고, 최종 모델 선택과 제출 구조는 기존 실험 결과와 과제 규격에 맞추어 판단하였다.

# 4. 결과 도출 & Discussion

학습은 DH_FR1.mat에 포함된 d_hat, BS_positions, p를 이용하여 수행하였다.
train.py에서는 전체 데이터를 학습용과 검증용으로 나누고, Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor를 비교하였다.

세 모델은 동일한 feature를 사용하도록 하여 모델 간 비교가 최대한 공정하게 이루어지도록 하였다.

모델 | 검증 평균 위치 오차
Ridge Regression | 7.094207 
Random Forest Regressor | 7.064307 
Gradient Boosting Regressor | 6.046496 

검증 결과 Gradient Boosting Regressor가 세 모델 중 가장 낮은 평균 위치 오차를 보였다.
Ridge Regression은 선형 모델이기 때문에 RTT 데이터의 비선형 오차를 충분히 반영하기 어려웠다.

Random Forest는 안정적인 결과를 보였지만 Gradient Boosting보다 오차가 조금 크게 나타났다.
Gradient Boosting은 이전 단계에서 남은 오차를 다음 단계의 모델이 순차적으로 보정하는 방식이기 때문에, RTT 데이터에 포함된 복잡한 비선형 오차 패턴을 상대적으로 잘 반영한 것으로 판단된다.

최종적으로 train.py에서는 Gradient Boosting 모델을 전체 데이터로 다시 학습한 후 model.pkl로 저장하였다.
main.py에서는 이 model.pkl을 불러와 각 사용자별로 위치를 예측하였다.

Colab에서 main.py를 실행한 결과는 다음과 같다.

항목 | 결과
반환 자료형 | numpy.ndarray
반환 shape | (2, 700)
main.py 실행 시간 | 약 177초
제한 시간 | 600초

실행 시간은 약 177초로 측정되어 10분 제한을 만족하였다.
사용자별 반복 구조를 사용했음에도 model.pkl을 매번 새로 불러오지 않고 최초 한 번만 로드하도록 구성했기 때문에 실행 시간을 줄일 수 있었다.

이번 결과에서 중요한 점은 머신러닝이 기존 RANSAC/WLS 알고리즘을 완전히 대체한 것이 아니라는 점이다.
RANSAC/WLS는 여전히 feature 생성 과정에서 초기 위치와 residual 정보를 제공하며, Gradient Boosting은 이를 바탕으로 최종 위치를 보정한다.

따라서 최종 시스템은 단순 ML 모델이 아니라, 기하학적 위치측위와 데이터 기반 보정을 결합한 Hybrid Positioning Framework라고 볼 수 있다.

다만 한계점도 존재한다.
현재 모델은 제공된 학습 데이터에서 학습된 패턴을 기반으로 예측하기 때문에 hidden test의 분포가 학습 데이터와 크게 다를 경우 성능이 낮아질 수 있다.

또한 RANSAC/WLS feature를 사용자별로 계산하기 때문에 단순 ML 예측보다 실행 시간이 길어지는 문제가 있다.
향후에는 RANSAC 반복 횟수와 feature 구성을 조정하여 실행 시간을 더 줄이고, 다양한 검증 분할 방식을 적용하여 hidden test에 대한 일반화 성능을 더 안정적으로 확보할 필요가 있다.

# 5. Reference

본 프로젝트에서는 단순히 참고문헌 제목만 가져오는 방식이 아니라, 각 논문에서 어떤 내용을 참고했고, 본 프로젝트에서는 그 내용을 어떻게 변형하여 적용했는지를 구분하여 정리하였다.
전체적으로는 RTT/UWB 기반 실내 측위에서 발생하는 NLOS, multipath, 거리 오차 문제를 먼저 확인하고, 이를 RANSAC/WLS와 머신러닝 기반 보정 구조로 연결하는 방향으로 참고하였다.

# 5.1 참고문헌 비교 및 선정 과정

AI를 활용하여 실내 위치측위 관련 논문 후보를 먼저 찾은 뒤, 초록과 주요 내용을 번역하고 요약하였다.
이후 본 과제와의 관련성을 기준으로 논문을 비교하였다.

후보 주제 | 검토 결과 | 최종 사용 여부
UWB NLOS identification survey | NLOS, residual, WLS, ML 기반 오차 완화 방법을 폭넓게 정리하고 있어 과제 배경 설명에 적합 -> 사용
Wi-Fi FTM two-step positioning | RTT 기반 ranging과 기하학적 위치 계산 이후 보정 구조를 설명하고 있어 Hybrid 구조 근거로 적합 -> 사용
WiFi RTT/RSS NLOS classification | RTT feature와 ML을 이용해 NLOS 패턴을 학습하는 내용이 있어 feature engineering 근거로 적합 -> 사용
Random Forest WiFi localization |  실내 측위에서 Random Forest를 학습 모델로 사용하는 사례라 ML 모델 비교 근거로 적합 -> 사용
Deep learning indoor localization | 구조가 복잡하고 본 과제 데이터 규모와 실행 시간 제한에 비해 과도하다고 판단 -> 일부 참고
Gradient Boosting indoor positioning | feature augmentation과 boosting 기반 위치 예측 방향이 본 프로젝트의 최종 모델과 연결 가능 -> 사용

최종적으로 선택한 reference는 다음과 같다.

### [1] Wang et al., “Survey on NLOS Identification and Error Mitigation for UWB Indoor Positioning,” Electronics, 2023

링크: https://www.mdpi.com/2079-9292/12/7/1678

이 논문은 UWB 기반 실내 측위에서 NLOS가 위치 오차의 주요 원인이며, 이를 식별하거나 완화하기 위한 여러 방법을 survey 형태로 정리한 논문이다.
논문에서는 residual analysis, statistical feature, machine learning, geometric feature 기반 NLOS 식별 방법을 분류하고, NLOS 오차를 줄이기 위한 weighting 기반 방법도 함께 다룬다.

본 프로젝트에서는 이 논문을 통해 RTT/UWB 기반 실내 측위에서 단순 거리값만 사용하면 NLOS나 multipath 때문에 위치 오차가 커질 수 있다는 점을 참고하였다.
이에 따라 RTT 거리값을 그대로 사용하는 대신, residual이 큰 앵커를 이상치로 보고 RANSAC/WLS feature를 추가하였다.

논문에서 직접 가져온 부분은 NLOS 오차가 위치 정확도를 크게 떨어뜨릴 수 있다는 문제의식과 residual/statistical feature/ML 기반 접근이 가능하다는 아이디어이다.
본인이 직접 구현한 부분은 해당 개념을 DH_FR1.mat 데이터에 맞게 RANSAC/WLS 위치 feature, residual feature, inlier ratio feature로 변환한 것이다.

### [2] Xu et al., “A Two-Step Fusion Method of Wi-Fi FTM for Indoor Positioning,” Sensors, 2022

링크: https://www.mdpi.com/1424-8220/22/9/3593

이 논문은 Wi-Fi FTM이 RTT 기반의 거리 측정 방식이라는 점을 설명하고, 실시간 range measurement를 이용한 single-point positioning 이후 fusion을 통해 위치 성능을 개선하는 구조를 제안한다.
즉, 한 번의 거리 기반 계산만으로 끝내는 것이 아니라, 기하학적 위치 계산 결과를 추가 단계에서 보정하는 two-step 구조를 사용한다.

본 프로젝트에서는 이 논문의 two-step 구조를 참고하였다.
먼저 RANSAC/WLS를 통해 기하학적 초기 위치를 계산하고, 이후 Gradient Boosting을 이용하여 최종 위치를 보정하는 방식으로 구성하였다.

논문에서 참고한 부분은 “기하학적 positioning 이후 추가 보정 단계가 필요하다”는 전체 구조이다.
다만 논문에서는 fusion method를 사용하였고, 본 프로젝트에서는 이를 머신러닝 기반 회귀 보정으로 바꾸었다는 차이가 있다.

### [3] Dong, Arslan, and Yang, “Real-Time NLOS/LOS Identification for Smartphone-Based Indoor Positioning Systems Using WiFi RTT and RSS,” IEEE Sensors Journal, 2022

링크: https://doi.org/10.1109/JSEN.2021.3119234
보조 링크: https://arxiv.org/abs/2104.11316

이 논문은 스마트폰 기반 WiFi RTT 실내 측위에서 NLOS 조건이 거리 오차를 발생시키며, WiFi RTT와 RSS feature를 이용해 NLOS/LOS를 식별하는 방법을 제안한다.
특히 Random Forest를 이용해 RTT/RSS feature에서 NLOS 패턴을 학습할 수 있음을 보여준다.

본 프로젝트에서는 NLOS/LOS를 직접 분류하지는 않았다.
하지만 RTT 원본값만 사용하는 대신 평균, 표준편차, 정렬된 거리값, residual 통계값 등 다양한 feature를 구성한 점에서 이 논문의 feature 기반 접근을 참고하였다.

논문에서 참고한 부분은 RTT 기반 측위에서 NLOS 여부가 중요한 오차 원인이며, 이를 machine learning feature로 학습할 수 있다는 점이다.
본인이 구현한 부분은 NLOS 분류 대신 최종 x, y 좌표를 예측하는 회귀 문제로 바꾸고, RANSAC/WLS residual 정보를 feature로 추가한 것이다.

### [4] Fan and Du, “NLOS Error Mitigation Using Weighted Least Squares and Kalman Filter in UWB Positioning,” 2022

링크: https://arxiv.org/abs/2205.05939

이 논문은 UWB positioning에서 NLOS가 큰 ranging bias와 위치 오차를 만든다고 설명하고, Weighted Least Squares와 Kalman Filter를 활용하여 NLOS 오차를 완화하는 방법을 제안한다.
이 논문은 WLS가 NLOS 완화 과정에서 사용할 수 있는 실질적인 위치 계산 방법이라는 점을 보여준다.

본 프로젝트에서는 Kalman Filter까지는 사용하지 않았다.
하지만 앵커별 거리값의 신뢰도가 동일하지 않다는 점을 반영하기 위해 WLS 기반 위치 보정 구조를 사용하였다.

논문에서 참고한 부분은 NLOS 환경에서 WLS가 오차 완화에 활용될 수 있다는 점이다.
본인이 직접 구현한 부분은 Kalman Filter 대신 RANSAC과 Huber 기반 robust weighting을 결합하여 한 사용자 단위의 위치 feature를 생성한 것이다.

### [5] Wang et al., “WiFi Indoor Localization with CSI Fingerprinting-Based Random Forest,” Sensors, 2018

링크: https://www.mdpi.com/1424-8220/18/9/2869

이 논문은 WiFi indoor localization에서 Random Forest 기반 fingerprinting 방법을 제안한다.
논문에서는 offline stage에서 Random Forest 모델을 학습하고, 학습된 모델을 이용해 위치를 추정하는 구조를 사용한다.
또한 Random Forest가 multipath 환경에서 비교적 강인한 성능을 보일 수 있음을 설명한다.

본 프로젝트에서는 이 논문을 통해 실내 위치측위 문제에 Random Forest와 같은 ensemble 기반 ML 모델을 적용할 수 있다는 점을 참고하였다.
또한 train.py에서 모델을 학습하고 model.pkl로 저장한 뒤, main.py에서 저장된 모델을 불러와 추론하는 구조를 구성하는 데 참고하였다.

논문에서 참고한 부분은 offline training과 online inference를 분리하는 구조이다.
본인이 직접 구현한 부분은 WiFi CSI fingerprint가 아니라 RTT 거리값과 RANSAC/WLS feature를 입력으로 사용하고, Random Forest뿐만 아니라 Ridge와 Gradient Boosting까지 비교한 것이다.

### [6] Bellavista-Parent, Torres-Sospedra, and Perez-Navarro, “New trends in indoor positioning based on WiFi and machine learning: A systematic review,” 2021

링크: https://arxiv.org/abs/2107.14356

이 논문은 WiFi 기반 indoor positioning과 machine learning을 사용한 연구들을 정리한 systematic review이다.
여러 연구에서 ML 기반 위치 추정이 사용되고 있지만, 실험 공간이 작거나 특정 환경에 치우친 경우 결과가 과대평가될 수 있다는 점도 지적한다.

본 프로젝트에서는 이 논문을 통해 hidden test에 대한 일반화 성능이 중요하다는 점을 참고하였다.
따라서 복잡한 deep learning 모델을 바로 사용하는 대신, Ridge Regression, Random Forest, Gradient Boosting처럼 데이터 수가 많지 않은 상황에서도 비교적 안정적으로 사용할 수 있는 회귀 모델을 비교하였다.

논문에서 참고한 부분은 indoor positioning에서 ML 모델의 성능을 평가할 때 일반화 가능성을 함께 고려해야 한다는 점이다.
본인이 직접 구현한 부분은 동일한 feature set으로 여러 회귀 모델을 비교하고, 검증 평균 위치 오차가 가장 낮은 Gradient Boosting을 최종 선택한 것이다.

### [7] Goharfar et al., “Indoor Positioning via Gradient Boosting Enhanced with Feature Augmentation using Deep Learning,” 2022

링크: https://arxiv.org/abs/2211.08752

이 논문은 indoor positioning 문제에서 Gradient Boosting과 feature augmentation을 결합하는 접근을 제안한다.
실내 측위에서는 단순 원본 신호만 사용하는 것보다 위치 추정에 도움이 되는 feature를 추가로 구성하는 것이 중요하다는 점을 보여준다.

본 프로젝트에서는 이 논문을 통해 Gradient Boosting이 실내 위치 예측 문제에서 활용될 수 있고, feature augmentation이 성능 향상에 도움이 될 수 있다는 점을 참고하였다.
다만 본 프로젝트에서는 deep learning 기반 feature augmentation은 사용하지 않았다.

본인이 직접 구현한 부분은 원본 RTT 거리값만 쓰는 것이 아니라 weighted centroid, RANSAC/WLS 위치, residual 통계값, 가까운 앵커 index 등을 추가 feature로 구성한 것이다.
또한 실행 시간 제한을 고려하여 딥러닝 모델 대신 Gradient Boosting Regressor를 사용하였다.

# 5.2 Reference를 바탕으로 한 본 프로젝트의 차별점

위 참고문헌들을 바탕으로 본 프로젝트의 최종 알고리즘은 다음과 같은 방향으로 정리된다.

먼저 [1], [2], [4]를 통해 RTT/UWB/FTM 기반 실내 측위에서 NLOS와 거리 오차가 주요 문제이며, 기하학적 위치 계산만으로는 한계가 있다는 점을 확인하였다.
이 내용을 바탕으로 본 프로젝트에서는 RANSAC과 WLS를 사용하여 이상치 앵커와 거리 기반 신뢰도 문제를 먼저 처리하였다.

다음으로 [3], [5], [6], [7]을 통해 indoor positioning에서 machine learning을 활용할 수 있으며, 원본 신호뿐만 아니라 통계 feature나 보조 feature를 추가하는 것이 중요하다는 점을 참고하였다.
이 내용을 바탕으로 본 프로젝트에서는 원본 RTT 거리값, 정렬 거리값, RTT 통계값, RANSAC/WLS 위치값, residual feature를 함께 사용하였다.

본 프로젝트에서 직접 구현한 부분은 다음과 같다.

구분 | 본인이 구현한 내용 

데이터 처리 | DH_FR1.mat의 d_hat과 BS_positions를 이용하여 사용자별 RTT feature 생성 
기하학적 위치 추정 | RANSAC과 WLS를 이용하여 초기 위치 및 residual feature 계산  
머신러닝 학습 | Ridge, Random Forest, Gradient Boosting을 동일 feature 조건에서 비교   
최종 모델 선택 | 검증 평균 위치 오차가 가장 낮았던 Gradient Boosting 선택   
제출 구조 | train.py에서 model.pkl 생성, main.py에서 model.pkl 로드 후 p_hat 반환 
규격 대응 | main() 함수, your_algorithm(d_hat[:, u], BS_positions), (2, num_user) 반환 구조 유지

따라서 본 프로젝트는 기존 논문을 그대로 복제한 것이 아니라, 최근 실내 측위 연구에서 공통적으로 다루는 NLOS 완화, weighted positioning, ML 기반 보정, feature engineering 아이디어를 과제 데이터 형식과 제출 규격에 맞게 재구성한 것이다.
