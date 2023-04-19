import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        self.resnet18_pretrained = models.resnet18(pretrained=True)
        num_features = self.resnet18_pretrained.fc.in_features
        self.resnet18_pretrained.fc = nn.Linear(num_features, num_classes)

        # self.resnet34_pretrained = models.resnet34(pretrained=True)
        # num_features = self.resnet34_pretrained.fc.in_features
        # self.resnet34_pretrained.fc = nn.Linear(num_features, num_classes)

        # self.resnet50_pretrained = models.resnet50(pretrained=True)
        # num_features = self.resnet50_pretrained.fc.in_features
        # self.resnet50_pretrained.fc = nn.Linear(num_features, num_classes)

        # self.resnet101_pretrained = models.resnet101(pretrained=True)
        # num_features = self.resnet101_pretrained.fc.in_features
        # self.resnet101_pretrained.fc = nn.Linear(num_features, num_classes)

        # self.resnet152_pretrained = models.resnet152(pretrained=True)
        # num_features = self.resnet152_pretrained.fc.in_features
        # self.resnet152_pretrained.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet18_pretrained(x)
        # x = self.resnet34_pretrained(x)
        # x = self.resnet50_pretrained(x)
        # x = self.resnet101_pretrained(x)
        # x = self.resnet152_pretrained(x)

        return x