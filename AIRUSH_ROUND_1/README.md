# 엉터리 문서 탐지

- **엉터리 문서: 문맥이 맞지 않은 단어들로 구성된 문서 혹은 단어의 순서를 바꾸어도 전혀 말이 되지 않는 문서를 의미**한다.
- **엉터리 문서를 Classification하는 Task**

## Data
- 구체적인 Data의 형태를 공개할 수는 없음.
- **불균형이 심한 Data (정상: 4000, 엉터리: 130000)**

## Core Idea

### 1. Pretrained Model 활용
- KLUE/RoBERTa-large
- KPF-bert
- KoBigBird

### 2. Hyper Parameter Tuning
- Long Document에 대한 분류 Task였기 때문에, Max_length를 최대한 길게 사용 (KLUE/RoBERTa: 512, KPF: 512, KoBigBird: 1024)
- AdamW, linear scheduler

### 3. Underfitting
- 130000:4000이라는 **Imbalanced Dataset 문제를 해결하기 위하여 Underfitting 활용**
- 엉터리 Data를 앞에서 부터 4000개를 Sampling하여, 총 8000개의 Dataset으로 학습을 수행

### 4. Layer Summation
- **Backbone Model의 마지막 4개의 Layer의 CLS Token에 대한 hidden state vector들을 Summatio**n하여 classification에 활용

### 5. Test Overflow Max
- Long Document 특성을 해결하기 위하여, **Overflow를 통해 Document를 Max_length 길이의 Token 단위로 쪼갠다.**
- **각각의 쪼개진 Document에 대하여, 엉터리 문서인지 판별**
- **하나의 Document Segment에 대하여 엉터리 문서라고 판단했다면, 엉터리 문서로 최종 분류**

### 6. Ensemble
- KLUE/RoBERTa-large, KPF-bert, KoBigBird를 Ensemble하여 최종 예측 수행

### 특이점
- **KoBigBird를 활용할 때, Classification Head를 부착하지 않은 채로 예측을 수행.**
- 768개(BERT의 output layer의 dimension)의 Multi-class classification으로 잘못 예측을 수행하였으나, 학습시 0과 1의 데이터로만 학습을 하였기 때문에, 768개의 class 중 0과 1로 대부분 예측을 수행함.
- 예측의 결과를 보았을 때, **Classification Head를 부착하지 않은 채로 예측할 때, 성능이 다소 높게 나옴. (Classification Head vs not = `0.94` : `0.947`)**

## 시도했지만 성능 향상으로 이어지지 않음.

### 1. Pre-processing
- 데이터를 자세히 확인할 수 없었기 때문에, 전처리가 필요한지 구체적으로 알 수 없었지만, 다양한 Preprocessing 방법론 적용 (개인정보 삭제, 띄어쓰기 교정, 반복 문자 제거 등)
- 성능 향상으로 이어지지 않음. => Data 자체적인 특성으로서 Noise가 많지 않는 것으로 보임

### 2. Post-Processing
- 정상 문서와 엉터리 문서를 명확하게 구분하지 못할 경우 (ex, 정상: 0.55, 엉터리: 0.45), 문서를 back translation 수행
- 엉터리 문서의 경우, "영어 번역 -> 한국어 번역"을 수행하면, 기존 문서와 back translation 후의 문서의 내용 차이가 클 것이라고 기대 (Data의 결과를 직접 출력해서 볼 수 없었다.)
- Long Text이기 때문에, 정상 문서 역시 back translation 후에 내용이 많이 변질된 것으로 예상

### 2. Curriculum Learning
- Data의 Imbalanced 문제로 인해 엉터리 문서의 개수가 부족하다고 판단하여 Curriculum Learning을 수행
- 정상 Data에서 오직 단어들의 순서만 바꿔서 Easy Data를 생성 (엉터리 문서의 경우, 단어들의 순서 뿐만 아니라, 문맥적으로 이해할 수 없는 문장으로 구성되어 있기 때문에, Difficult Data라고 명명하고, 오직 단어들의 순서만 바꾼 Data를 Easy Data라고 명명한다.)
- Easy Data를 Random으로 4000개 정도 Sampling하여 학습을 우선적으로 수행.
- Easy Data에 fine-tuning된 Model을 실제 Data (Difficult Data)로 다시 한 번 Fine-tuning 수행

### 3. R-Drop
- 엉터리 문서의 경우, 부분적인 엉터리가 존재할 수 있고, Dropout에 의해 엉터리인 부분을 학습 중에 제대로 판단할 수 없는 경우가 발생할 수 있다고 판단.
- R-Drop을 통해, 같은 Data가 Dropout을 통과한 후 출력된 서로 다른 두 결과물의 분포가 일치하도록 강제.