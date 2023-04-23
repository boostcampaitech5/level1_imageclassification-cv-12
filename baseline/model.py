import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torchvision.models import resnet50, resnet101, resnet152, resnet18
from efficientnet_pytorch import EfficientNet


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


# Custom Model Template
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

        # self.resnet18_pretrained = models.resnet18(pretrained=True)
        # num_features = self.resnet18_pretrained.fc.in_features
        # self.resnet18_pretrained.fc = nn.Linear(num_features, num_classes)

        # self.resnet34_pretrained = models.resnet34(pretrained=True)
        # num_features = self.resnet34_pretrained.fc.in_features
        # self.resnet34_pretrained.fc = nn.Linear(num_features, num_classes)

        self.resnet50_pretrained = models.resnet50(pretrained=True)
        num_features = self.resnet50_pretrained.fc.in_features
        self.resnet50_pretrained.fc = nn.Linear(num_features, num_classes)

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
        # x = self.resnet18_pretrained(x)
        # x = self.resnet34_pretrained(x)
        x = self.resnet50_pretrained(x)
        # x = self.resnet101_pretrained(x)
        # x = self.resnet152_pretrained(x)

        return x


class Res(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = resnet50(pretrained=True) # Change to ResNet-50
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1000, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, num_classes) # Update the number of input features for fc layer to 2048

    def forward(self, x):
        x = self.base_model(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    

class Effi(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b4')
        self.classifier = nn.Sequential(
            nn.Linear(1000, 4096), #self.base_model._fc.in_features
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    


class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_bn1 = nn.BatchNorm2d(64)
        self.block1_relu1 = nn.ReLU(inplace=True)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_bn2 = nn.BatchNorm2d(64)
        self.block1_relu2 = nn.ReLU(inplace=True)
        self.block1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_bn1 = nn.BatchNorm2d(128)
        self.block2_relu1 = nn.ReLU(inplace=True)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_bn2 = nn.BatchNorm2d(128)
        self.block2_relu2 = nn.ReLU(inplace=True)
        self.block2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_bn1 = nn.BatchNorm2d(256)
        self.block3_relu1 = nn.ReLU(inplace=True)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_bn2 = nn.BatchNorm2d(256)
        self.block3_relu2 = nn.ReLU(inplace=True)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_bn3 = nn.BatchNorm2d(256)
        self.block3_relu3 = nn.ReLU(inplace=True)
        self.block3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_bn1 = nn.BatchNorm2d(512)
        self.block4_relu1 = nn.ReLU(inplace=True)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_bn2 = nn.BatchNorm2d(512)
        self.block4_relu2 = nn.ReLU(inplace=True)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_bn3 = nn.BatchNorm2d(512)
        self.block4_relu3 = nn.ReLU(inplace=True)
        self.block4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_bn1 = nn.BatchNorm2d(512)
        self.block5_relu1 = nn.ReLU(inplace=True)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3)
        self.block5_bn2 = nn.BatchNorm2d(512)
        self.block5_relu2 = nn.ReLU(inplace=True)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_bn3 = nn.BatchNorm2d(512)
        self.block5_relu3 = nn.ReLU(inplace=True)
        self.block5_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(18432, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_relu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_relu2(x)
        x = self.block1_maxpool(x)

        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_relu2(x)
        x = self.block2_maxpool(x)

        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_conv3(x)
        x = self.block3_bn3(x)
        x = self.block3_relu3(x)
        x = self.block3_maxpool(x)

        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_relu2(x)
        x = self.block4_conv3(x)
        x = self.block4_bn3(x)
        x = self.block4_relu3(x)
        x = self.block4_maxpool(x)

        x = self.block5_conv1(x)
        x = self.block5_bn1(x)
        x = self.block5_relu1(x)
        x = self.block5_conv2(x)
        x = self.block5_bn2(x)
        x = self.block5_relu2(x)
        x = self.block5_conv3(x)
        x = self.block5_bn3(x)
        x = self.block5_relu3(x)
        x = self.block5_maxpool(x)

        x = torch.flatten(x, 1)
        print(x.size(),'xxxxxxx')
        x = self.classifier(x)
        return x
    

class June(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)


        self.classifier = nn.Sequential(
            nn.Linear(1000, 4096), #self.base_model._fc.in_features
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x