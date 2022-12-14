# NAVER AI RUSH 2022 - Unknown Documents Detection
**๐ฅ Top 3 in "Round 2: Unknown Documents Detection"**

## About Competition
![Unknown_Detection_Task](https://user-images.githubusercontent.com/53552847/197756342-34f9b21a-a930-4be4-9703-127eff610399.png)

- ๋ชจ๋ธ์ด ํ์ตํ์ง ์์ ์ข๋ฅ์ ๋ฌธ์๋ฅผ ํ์งํ๋ Task
- 6๊ฐ์ Class๋ก ์ด๋ฃจ์ด์ง Text Data๋ฅผ ํ์ต
- **Test ์, ํ์ตํ Data๋ ์ณ๊ฒ ๋ถ๋ฅํ๊ณ , ํ์ตํ์ง ์์ Data๋ Unknown์ผ๋ก ๋ถ๋ฅ**
- **Multiclass Classification์ Unknown Detection์ด ๊ฒฐํฉ๋ ํํ์ Task**

## Overall Process
- [**Presentation**](https://github.com/jjonhwa/clova-airush-2022/blob/main/AIRUSH_Unknown_Detection_3%EC%9C%84.pdf) ์ฐธ๊ณ 

## Core Idea

### 1. Pretrained Transformer Model ํ์ฉ
- Pretrained Transformer MOdel์ด OOD Task์์ ๋์ ์ฑ๋ฅ์ ๋ํ๋ธ๋ค๋ ์ฐ๊ตฌ ๊ฒฐ๊ณผ. => ***Backbone Model๋ก์ BERT ๊ณ์ด Model ์ฑํ***
- ๋ง์ Data๋ก Pre-training์ ์งํํ  ๊ฒฝ์ฐ, OOD Detection์์ ๋ ์ข์ ์ฑ๋ฅ์ ๋ํ๋ธ๋ค๋ ์ฐ๊ตฌ ๊ฒฐ๊ณผ. => ***KPF-Bert๋ฅผ ์ต์ข Backbone Model๋ก ์ ์ *** 

### 2. Diverse Loss Function
- **Large Margin Softmax**
    - "์ ๋ต Class"์ "์ ๋ต์ด ์๋ Class" ์ฌ์ด์ Margin์ ์ค์ผ๋ก์ "์ ๋ต Class"๋ฅผ ์ข ๋ ํ์คํ๊ฒ ๋ง์ถ๋๋ก ํ๋ ๋ฐฉ๋ฒ๋ก 
    - class๊ฐ์ ์ฐจ์ด๋ฅผ ๋ณด๋ค ํ์คํ๊ฒ ํจ์ผ๋ก์จ, ***"In-Distribution" Dataset์ ๋ ๋์ Logit (๋ ํ์คํ๊ฒ ์์ธก ๊ฐ๋ฅ)์ผ๋ก ๋ง์ถ๊ณ , "Out-Distribution" Data์ ์ข ๋ ๋ฎ์ Logit (๋ ํ์คํ๊ฒ ์์ธก)์ ๊ฐ์ง ๊ฒ์ผ๋ก ๊ธฐ๋***

- **Relax Loss**
    - Gradient Descent์ Ascent๋ฅผ ํผํฉํ์ฌ ํ์ตํจ์ผ๋ก์จ, Classification์ ์ ์ํํ๋ฉด์ MIAs task์์ Member or not์ ๋ ์ ์ํํ  ์ ์๋ ๊ธฐ๋ฒ
    - "In-Distribution Dataset", "Out-Distribution Dataset" ๋ชจ๋์์ "Confidence Logit"์ ์ํ
    - ***In-Distribution Data๋ฅผ ๋ ์ ํํ๊ฒ ์์ธกํ๊ธฐ ๋๋ฌธ์, Out-of-Distribution Data๋ ๊ทธ๋ณด๋ค ๋ ์ ํํ๊ฒ ์์ธก๋  ์ ๋ฐ์ ์์ ๊ฒ์ผ๋ก ๊ธฐ๋ (In: 95%(CE) -> 70%(Relax) ์์ธกํ์ ๊ฒฝ์ฐ, Out: 90%(CE) -> 60%(Relax)์ผ ๊ฒ์ผ๋ก ๊ธฐ๋)***

### 3. Preprocessing
- ***Hanspell์ ํ์ฉํ์ฌ ๋ง์ถค๋ฒ(๋์ด์ฐ๊ธฐ ๋ฐ ๋ง์ถค๋ฒ) ๊ฒ์ฌ ์ํ***
- Train & Test Data ๋ชจ๋์ ์ ์ฉ

### 4. Ensemble
- ๋ค์ํ Backbone Model์ ํ์ฉ
- Logit Ensemble ์ํ

## Ablation Study

### 1. Outlier Exposure
![Outlier_explosure](https://user-images.githubusercontent.com/53552847/197760135-cfa2be1d-1c8b-4e85-bee7-90aae45e6fa2.png)

- ***ํ์ต์ ํ์ฉํ์ง ์๋ Dataset์ Another Class์ ๋ํ Dataset์ผ๋ก ํ์ฉํจ์ผ๋ก์จ, Unknown Data Detection Performance๋ฅผ ๊ฐ์ ***ํ๋ค.
- ์ด ๋, ํ์ต์ ํ์ฉํ์ง ์๋ Dataset์ด๋ Wiki Text Data์ ๊ฐ์ด, ํน์  class๋ก target ๋์ง ์์ public data๋ฅผ ์๋ฏธํ๋ค.
- ๋ณธ ๋ํ์์๋ ์ธ๋ถ๋ฐ์ดํฐ๋ฅผ ํ์ฉํ  ์ ์์๊ธฐ ๋๋ฌธ์, ***Max Token Length๋ฅผ ๋์ด์ ํ์ฉํ์ง ๋ชปํ Data๋ฅผ Random Shufflingํ์ฌ Another Class๋ก ํ์ฉํ์ฌ ํ์ต์ ์ํ***ํ์๋ค.
- ์ฑ๋ฅ ํ๋ฝ: Known Class์ ๋ํ ์์ธก ์ฑ๋ฅ ํ๋ฝ, Unknown Class์ ๋ํ ์ฑ๋ฅ๋ ํฌ๊ฒ ๊ฐ์ ๋์ง ์์.
    - ๊ธฐ์กด Class์ Data๋ฅผ ์ถ๊ฐ๋ก ํ์ฉํ์๊ธฐ ๋๋ฌธ์ ๋ชจ๋ธ์ด ๊ธฐ์กด Class๋ฅผ ์์ธกํ  ๋ ํผ๋ํ ๊ฒ์ผ๋ก ๋ณด์.

### 2. MASKER
![MASKER](https://user-images.githubusercontent.com/53552847/197760346-dbf5b44b-1e92-49d4-a4b4-4df392723557.png)

- ์ ์๋ฏธํ Token์ Maskingํ ํ, Token-wise Classification Task๋ฅผ ์ถ๊ฐ์ ์ผ๋ก ์ํ.
- Model์ด ํน์  Token๋ง์ ๋ณด๊ณ  ์์ฌ๊ฒฐ์ ์ ๋ด๋ฆฌ๋ ๊ฒฝํฅ์ด ์์ผ๋ฉฐ, ์ด๋ฌํ ๊ฒฝํฅ์ ์ ์ฝ์ ์ฃผ๊ธฐ ์ํ ๋ฐฉ๋ฒ์ผ๋ก์จ, ***Unknown Text๊ฐ Keyword Token์ ๊ฐ์ง๊ณ  ์์ ๋, Known์ผ๋ก ํ๋จ์ ๋ด๋ฆฌ๋ ๊ฒ์ ๋ฐฉ์ง***
- ํฐ ์ฑ๋ฅ ๊ฐ์ ์ ๋ณด์ด์ง ์์.
    - ๊ธฐ์กด Model์์ ์ด๋ฏธ ๋์ ์์ธก ํ๋ฅ  (97~8%์ ์ ํ๋)๋ฅผ ๋ณด์๊ธฐ ๋๋ฌธ์, ๋์ ๋๋ ๊ฐ์ ์ผ๋ก ์ด์ด์ง์ง ์์ ๊ฒ์ผ๋ก ๋ณด์.

### 3. Membership Inference Attacks ์์ฉ
![MIAs](https://user-images.githubusercontent.com/53552847/197767703-f419bdb6-9f08-45b8-8a9c-fd7a68d15ec7.png)

- Membership Inference Attacks
    - Test Data๊ฐ Train Dataset์ ํฌํจ๋๋์ง ํ์ธํ๊ณ ์ ํ๋ Task
- Shadow Model๊ณผ Attack Model ํ์ฉ
    - Shadow Model
        - ***Stratified KFold๋ฅผ ํ์ฉํ์ฌ 5๊ฐ์ Fold๋ก ๋ถํ ***
        - ๊ฐ Fold์ Train Data๋ฅผ ํ์ฉํ์ฌ Bert Model ํ์ต
    - Attack Model
        - Shadow Model์ ํ์ฉํ์ฌ, ๊ธฐ์กด Stratified KFold๋ก ๋ถํ ๋ Dataset์ ๋ํ์ฌ Inference ์ํ
        - ***Train Data๋ Label 1 (in-Data), Valid Data๋ Label 0 (Out-Data)๋ก ํ์ฌ, Attack Model์ ์ํ Data ์ ์***
        - ***์์ฑ๋ Data๋ฅผ ํ์ฉํ์ฌ Attack Model (In or Out์ Classificationํ๋ Model)์ ํ์ต***
- Test ์, Attack Model์ ํ์ฉํ์ฌ, Test Data๊ฐ In-Out์ธ์ง๋ฅผ ํ๋จํ์ฌ, Unknown Text์ ๋ถ๋ฅํ๋ ๊ฒ์ ํ์ฉํ๊ณ ์ ํจ.
- ์ฑ๋ฅ์ด ์ข์ง ์์
    - K-Fold๋ฅผ ์ํํ์ฌ ๋๋ Train, Valid Data์ ํํ๊ฐ ๊ฑฐ์ ๋์ผํ๊ธฐ ๋๋ฌธ์, In-Out Data์ ํน์ฑ์ ์ก์๋ด์ง ๋ชปํ๋ ๊ฒ์ผ๋ก ๋ณด์. 
