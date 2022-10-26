# NAVER AI RUSH 2022 - Unknown Documents Detection
**🥉 Top 3 in "Round 2: Unknown Documents Detection"**

## About Competition
![Unknown_Detection_Task](https://user-images.githubusercontent.com/53552847/197756342-34f9b21a-a930-4be4-9703-127eff610399.png)

- 모델이 학습하지 않은 종류의 문서를 탐지하는 Task
- 6개의 Class로 이루어진 Text Data를 학습
- **Test 시, 학습한 Data는 옳게 분류하고, 학습하지 않은 Data는 Unknown으로 분류**
- **Multiclass Classification와 Unknown Detection이 결합된 형태의 Task**

## Overall Process
- [**Presentation**](https://github.com/jjonhwa/clova-airush-2022/blob/main/AIRUSH_ROUND_2/AIRUSH_Unknown_Detection_3%EC%9C%84.pdf) 참고

## Core Idea

### 1. Pretrained Transformer Model 활용
- Pretrained Transformer MOdel이 OOD Task에서 높은 성능을 나타낸다는 연구 결과. => ***Backbone Model로서 BERT 계열 Model 채택***
- 많은 Data로 Pre-training을 진행할 경우, OOD Detection에서 더 좋은 성능을 나타낸다는 연구 결과. => ***KPF-Bert를 최종 Backbone Model로 선정*** 

### 2. Diverse Loss Function
- **Large Margin Softmax**
    - "정답 Class"와 "정답이 아닌 Class" 사이의 Margin을 줌으로서 "정답 Class"를 좀 더 확실하게 맞추도록 하는 방법론
    - class간의 차이를 보다 확실하게 함으로써, ***"In-Distribution" Dataset은 더 높은 Logit (더 확실하게 예측 가능)으로 맞추고, "Out-Distribution" Data은 좀 더 낮은 Logit (덜 확실하게 예측)을 가질 것으로 기대***

- **Relax Loss**
    - Gradient Descent와 Ascent를 혼합하여 학습함으로써, Classification은 잘 수행하면서 MIAs task에서 Member or not을 더 잘 수행할 수 있는 기법
    - "In-Distribution Dataset", "Out-Distribution Dataset" 모두에서 "Confidence Logit"을 완화
    - ***In-Distribution Data를 덜 정확하게 예측하기 때문에, Out-of-Distribution Data는 그보다 덜 정확하게 예측될 수 밖에 없을 것으로 기대 (In: 95%(CE) -> 70%(Relax) 예측했을 경우, Out: 90%(CE) -> 60%(Relax)일 것으로 기대)***

### 3. Preprocessing
- ***Hanspell을 활용하여 맞춤법(띄어쓰기 및 맞춤법) 검사 수행***
- Train & Test Data 모두에 적용

### 4. Ensemble
- 다양한 Backbone Model을 활용
- Logit Ensemble 수행

## Ablation Study

### 1. Outlier Exposure
![Outlier_explosure](https://user-images.githubusercontent.com/53552847/197760135-cfa2be1d-1c8b-4e85-bee7-90aae45e6fa2.png)

- ***학습에 활용하지 않는 Dataset을 Another Class에 대한 Dataset으로 활용함으로써, Unknown Data Detection Performance를 개선***한다.
- 이 때, 학습에 활용하지 않는 Dataset이란 Wiki Text Data와 같이, 특정 class로 target 되지 않은 public data를 의미한다.
- 본 대회에서는 외부데이터를 활용할 수 없었기 때문에, ***Max Token Length를 넘어서 활용하지 못한 Data를 Random Shuffling하여 Another Class로 활용하여 학습을 수행***하였다.
- 성능 하락: Known Class에 대한 예측 성능 하락, Unknown Class에 대한 성능도 크게 개선되지 않음.
    - 기존 Class의 Data를 추가로 활용하였기 때문에 모델이 기존 Class를 예측할 때 혼동한 것으로 보임.

### 2. MASKER
![MASKER](https://user-images.githubusercontent.com/53552847/197760346-dbf5b44b-1e92-49d4-a4b4-4df392723557.png)

- 유의미한 Token을 Masking한 후, Token-wise Classification Task를 추가적으로 수행.
- Model이 특정 Token만을 보고 의사결정을 내리는 경향이 있으며, 이러한 경향에 제약을 주기 위한 방법으로써, ***Unknown Text가 Keyword Token을 가지고 있을 때, Known으로 판단을 내리는 것을 방지***
- 큰 성능 개선을 보이지 않음.
    - 기존 Model에서 이미 높은 예측 확률 (97~8%의 정확도)를 보였기 때문에, 눈에 띄는 개선으로 이어지지 않은 것으로 보임.

### 3. Membership Inference Attacks 응용
![MIAs](https://user-images.githubusercontent.com/53552847/197767703-f419bdb6-9f08-45b8-8a9c-fd7a68d15ec7.png)

- Membership Inference Attacks
    - Test Data가 Train Dataset에 포함되는지 확인하고자 하는 Task
- Shadow Model과 Attack Model 활용
    - Shadow Model
        - ***Stratified KFold를 활용하여 5개의 Fold로 분할***
        - 각 Fold의 Train Data를 활용하여 Bert Model 학습
    - Attack Model
        - Shadow Model을 활용하여, 기존 Stratified KFold로 분할된 Dataset에 대하여 Inference 수행
        - ***Train Data는 Label 1 (in-Data), Valid Data는 Label 0 (Out-Data)로 하여, Attack Model을 위한 Data 제작***
        - ***생성된 Data를 활용하여 Attack Model (In or Out을 Classification하는 Model)을 학습***
- Test 시, Attack Model을 활용하여, Test Data가 In-Out인지를 판단하여, Unknown Text을 분류하는 것에 활용하고자 함.
- 성능이 좋지 않음
    - K-Fold를 수행하여 나뉜 Train, Valid Data의 형태가 거의 동일하기 때문에, In-Out Data의 특성을 잡아내지 못하는 것으로 보임. 
