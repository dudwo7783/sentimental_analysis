# ELECTRA Model based Natural Language Sentimental Analysis



## 실행 방법

**NSMC**



모델에 state dict를 저장하기에는 pickle 파일이 너무 커서 업로드 할 수 없었습니다.

따라서 모델은 없는 상태이며 학습을 진행해야 test와 leaderboard test가 가능합니다.

만약 train이 종료되고 이어서 test와 leaderboard를 동시에 수행하면 이전 checkpoint의 모델이 아닌 학습이 진행된 모델로부터 예측합니다.



학습 & 테스트

```
python ./nsmc.py --train yes --test yes --leaderboard_test yes
```



테스트

```
python ./nsmc.py --train no --test yes --leaderboard_test yes
```



**Friends**

학습 & 테스트

```
python ./friends.py --train yes --test yes --leaderboard_test yes
```



테스트

```
python ./friends.py --train no --test yes --leaderboard_test yes
```





## 참고



**Google ELECTRA Model**

Friends의 경우 google의 ELECTRA의 pre-trained small model을 사용하였습니다.



**KoELECTRA Model**

NSMC의 경우 박장원님의 KoELECTRA의 pre-trained base model을 사용하였습니다.

https://github.com/monologg/KoELECTRA



**모델 학습 코드**

모델 학습의 코드는

https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1

를 참고하였습니다.

## 기타

Data Augumentaion을 위한 코드는 개인적으로 google cloud api를 결제하여 사용하였습니다.
해당 코드를 이용하고자하면 Google API Credential을 발급받아 코드안에서 수정바랍니다.


Input으로 Bert WordEmbedding, 분류 모델로 TCN을 이용한 sample 코드를 추가하였습니다.
