# NAVER AI RUSH 2022
![airush](https://user-images.githubusercontent.com/53552847/198039191-0116de37-e764-4b95-8702-fd342de723d7.png)(#https://campaign.naver.com/clova_airush/)

**🥈 Top 2 in Round 1: Nonsense Documents Detection**

**🥉 Top 3 in Round 2: Unknown Documents Detection**

**Award Poster: [Poster](https://github.com/jjonhwa/clova-airush-2022/blob/main/AIRUSH_Unknown_Detection_3%EC%9C%84.pdf)**

## Round1: Nonsens Documents Detection
![Nonsense_docuement_detection](https://user-images.githubusercontent.com/53552847/197984010-aace21e8-c56a-43a8-8ca4-2c92487a161d.png)
- **엉터리 문서 (Nonsense Documents)**
    - **문맥이 맞지 않은 단어들로 구성**된 문서 혹은 **단어의 순서를 바꾸어도 전혀 말이 되지 않는 문서**를 의미한다.
- **엉터리 문서를 Classification하는 Task**

**[자세한 내용 보기](https://github.com/jjonhwa/clova-airush-2022/tree/main/AIRUSH_ROUND_1)**
## Round2: Unknown Documents Detection
![Unknown_Detection_Task](https://user-images.githubusercontent.com/53552847/197756342-34f9b21a-a930-4be4-9703-127eff610399.png)


- 모델이 학습하지 않은 종류의 문서를 탐지하는 Task
- 6개의 Class로 이루어진 Text Data를 학습
- **Test 시, 학습한 Data는 옳게 분류하고, 학습하지 않은 Data는 Unknown으로 분류**
- **Multiclass Classification와 Unknown Detection이 결합된 형태의 Task**

**[자세한 내용 보기](https://github.com/jjonhwa/clova-airush-2022/tree/main/AIRUSH_ROUND_2)**
