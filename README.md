## level1*imageclassification-cv-12*모아이

### 마스크 착용 상태 분류

##### - 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task

<img width="80%" src="https://github.com/boostcampaitech5/level1_imageclassification-cv-12/assets/70469008/bdf07fa9-41ab-49e0-ad3f-0a82da3c1cc9"/>

COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다.
COVID-19 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크를 반드시 착용하여 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 무엇보다도 코와 입을 완전히 가릴 수 있도록 마스크를 올바르게 착용하는 것이 중요하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.
따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다. 🌎

#### Team Members

강혜지
김용환
박혜나
신중현
이종휘

#### 실험 내용

- pretrained model : AlexNet / VGGNet / GoogleNet / ResNet / vision transformer
- Ensemble : Multi Label classification / Multi Model
- Loss : Label Smoothing / Focal Loss / F1 Loss / Weighted Cross Entropy Loss
- Data Augmentation : torchvision / Albumentations / CLAHE
- Hyperparameter : optuna (epoch / learning rate / batch size / optimizer)
- stratified K-fold cross validation

#### 최종 반영된 내용

- pretrained model : vision transformer
- Ensemble : Multi Label classification
  (그 외 사항은 성능 하락으로 최종 버전에 미반영)

#### 최종 파일

- train_final / dataset_final / model_final / loss_final / inference_final

#### Wrap-up Report

https://drive.google.com/file/d/1R3XxrkhqjrIZ_2g5fbKal0MGWWbPAY-e/view?usp=sharing

#### 평가 Metric

- F1 Score (in "macro" F1, a separate F1 score is calculated for each classes value and then averaged)

#### Dataset

- 이미지 수 : 31,500
  - 전체 사람 명 수 : 4,500
  - 한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]
  - train 60% / test 40% (public 20% + private 20%)
- 이미지 크기 : (384, 512)
- 클래스 수 : 마스크 착용여부, 성별, 나이를 기준으로 총 18개의 클래스
  - Mask : Wear / Incorrect / Not Wear
  - Gender : Male / Female
  - Age : <30 / >=30 and <60 / >=60

#### Input & Output

- Input
  - 마스크 착용 사진, 미착용 사진, 혹은 이상하게 착용한 사진(코스크, 턱스크)
- Output
  - 총 18개 클래스에 대해 각 이미지 당 0 ~ 17에 해당되는 예측값을 포함한 csv 파일
  - Ex 7 (the class of cfe1268.jpg), 2 (the class of 3a2662c.jpg), ...
