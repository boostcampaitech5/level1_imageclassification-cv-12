# level1_imageclassification-cv-12
level1_imageclassification-cv-12 created by GitHub Classroom

- 최종 버전 : train_final / dataset_final / model_final / loss_final / inference_final

- 시도한 내용
    pretrained model : AlexNet / VGGNet / GoogleNet / ResNet / vision transformer
    Ensemble : Multi Label classification / Multi Model
    Loss : Label Smoothing / Focal Loss / F1 Loss / Weighted Cross Entropy Loss
    Data Augmentation : torchvision / Albumentations / CLAHE
    Hyperparameter : optuna (epoch / learning rate / batch size / optimizer)
    stratified K-fold cross validation

- 최종 반영된 내용
    pretrained model : vision transformer
    Ensemble : Multi Label classification
    (그 외 사항은 성능 하락으로 최종 버전에 미반영)