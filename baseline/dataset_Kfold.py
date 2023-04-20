import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


#파일 이름이 이미지파일인지 여부를 판단함(위에 IMG_EXTENSIONS안에 포함되면 TURE,아니면 FALSE반환)
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# 이미지 사이즈, 이미지의 평균과 표준편차등을 입력으로 받아 입력에 맞게 변환시킴
class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),#이미지 크기 조절
            ToTensor(),#이미지를 tensor형태로 변환 (이미지 데이터를 모델에 넣을 수 있게 해줌)
            Normalize(mean=mean, std=std), #mean과 std에 맞게 정규화
        ])

    def __call__(self, image):
        return self.transform(image)


# 이미지에 가우시안 노이즈를 추가함
class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """
    # 평균 0, std1이 기본값
    # tensor 입력을 받아 torch.randn으로 입력텐서 사이즈와 동일한 크기의 
    # 랜덤한 가우시안 분포의 값을 생성후 더해서 가우시안 노이즈가 들어간 텐서를 반환한다.
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


    
class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        # compose는 torchvision.transforms 모듈에서 제공하는 함수
        self.transform = Compose([
            CenterCrop((320, 256)), # 이미지 중앙부분을 320,256크기로 자름
            Resize(resize, Image.BILINEAR),
            # resize크기에 맞게 이미지를 바꿈
            # resize할때 resize매개변수가 224로 설정되어있으면 가로, 세로중 큰쪽을 224로 맞추고
            # 비율을 유지하면서 다른쪽을 조절함
            # (224,224)로 입력하면 입력이미지의 비율을 유지하면서 224,224로 맞춰서 출력함
            # BILINEAR은 이웃픽셀들의 값에 가중치를 주어 보간하는 방법
            ColorJitter(0.1, 0.1, 0.1, 0.1), # 밝기(brightness), 대비(contrast), 채도(saturation), 색조(hue)를 +- 0.1 범위에서 랜덤하게 변화시킴
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


# MASKLABALES 클래스 선언 (마스크쓴거 0 마스크 잘못쓴거 1 마스크안쓴거 2)
class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


# 성별 (남자 0 여자 1)
class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1
    
    @classmethod
    def from_str(cls, value: str) -> int:
        # 문자열로 된 성별값을 입력으로 받아 이를 0과 1로 변환해줌
        # 이에 해당하지 않을 경우 valueerror반환
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")

# 30미만 0 30이상 60미만 1 60이상 2
class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    #마스크 라벨링 (0,1,2)*성별(0,1)*나이(0,1,2)
    #기존코드
    #num_classes = 3 * 2 * 3
    #마스크 라벨링만 테스트하기 위해 바꿈
    # num_classes = 3 * 2 * 3
    num_classes = 3 + 2 + 3 # Multi Label Classification


    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    labels=[]

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
        
    def setup(self):
        # 데이터셋 경로에 있는 파일들의 이름을 가져와 파일명에서 라벨 추출함
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                #추가한거
                self.labels.append(mask_label * 6 + gender_label * 3 + age_label)
    
    def calc_statistics(self):
        # 데이터셋 이미지의 평균과 표준편차를 계산함 
        # 없는 경우 첫 3000개의 이미지에 대해 평균과 표준편차를 계산함
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform
    
    def __getitem__(self, index):
        # 이미지에 적용할 transform을 설정하고, 이미지를 변환함
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        # 마스크와 성별, 나이 라벨을 가져와 멀티클래스 라벨 생성(마스크,성별,나이 순)
        # 이걸 마스크, 성별, 나이 따로따로 가져와서 모델 결과 활용
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        
        # 이미지 변환
        image_transform = self.transform(image)
        # return image_transform, multi_class_label
        return image_transform, (mask_label, gender_label, age_label) # Multi Label Classification
    
        # 테스트 코드 (mask만 찾는 모델 만들기 위해)
        # return image_transform, mask_label
        
    # 데이터셋의 전체 이미지수 반환
    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    # 마스크라벨, 성별라벨, 나이라벨을 입력받아 합친 단일 클래스 라벨로 반환
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label
    # 위 라벨을 분해하여 마스크라벨,성별라벨,나이라벨을 반환함
    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label
    # 정규화한 이미지를 다시 역정규화 해서 원래의 이미지로 돌려놓는것
    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp
    
    
    #기존 split dataset
    def split_dataset(self, n_splits: int = 5) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        # val_ration 비율로 검증데이터와 학습데이터를 나눔 (기본값은 0.2)
        # n_val = int(len(self) * self.val_ratio)
        # n_train = len(self) - n_val
        # train_set, val_set = random_split(self, [n_train, n_val])
        # return train_set, val_set
        # 다른버전
        from typing import Tuple
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import Subset
        from torch.utils.data import ConcatDataset
     
        # StratifiedKFold 클래스를 사용하여 fold를 생성합니다.
        #데이터 잘 나눠지나 확인하기 위해서 shuffle option은 false
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        #skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # fold를 순회하면서 각 fold에 포함될 데이터 인덱스를 추출합니다.
        for i,(train_index, val_index) in enumerate(skf.split(self.image_paths, self.labels)):
            if i==0:
                train_set = Subset(self, train_index)
                val_set = Subset(self, val_index)
            else :
                break
                # train_set = ConcatDataset([train_set, Subset(self, train_index)])
                # val_set = ConcatDataset([val_set, Subset(self, val_index)])
        print(len(self.labels))
        print(train_set[0])
        print(val_set[0])
        print(len(train_set),len(val_set))
        return train_set,val_set
        # yield train_set, val_set
    
    
# maskbasedataset을 상속받아 구현
class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


# dataset의 클래스를 상속받아 test dataset을 정의함
class TestDataset(Dataset):
    # test데이터셋이 사용할 이미지경로, 리사이즈할 크기, 평균과 표춘편차 를 받아
    # transform할 compose함수를 저장함
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    # 해당 인덱스의 이미지를 불러와 전처리 시킨 이미지를 반환
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image
    # 이미지 경로 리스트의 길이를 반환
    def __len__(self):
        return len(self.img_paths)
