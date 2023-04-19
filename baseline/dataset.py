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


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


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
        
# age 세분화: 클래스 추가
class NewAgeLabels(int, Enum):
    EIGHTEEN = 0
    NINETEEN = 1
    TWENTY = 2
    TWENTYONE = 3
    TWENTYTWO = 4
    TWENTYTHREE = 5
    TWENTYFOUR = 6
    TWENTYFIVE = 7
    TWENTYSIX = 8
    TWENTYSEVEN = 9
    TWENTYEIGHT = 10
    TWENTYNINE = 11
    THIRTY = 12
    THIRTYONE = 13
    THIRTYTWO = 14
    THIRTYTHREE = 15
    THIRTYFOUR = 16
    THIRTYFIVE = 17
    THIRTYSIX = 18
    THIRTYSEVEN = 19
    THIRTYEIGHT = 20
    THIRTYNINE = 21
    FORTY = 22
    FORTYONE = 23
    FORTYTWO = 24
    FORTYTHREE = 25
    FORTYFOUR = 26
    FORTYFIVE = 27
    FORTYSIX = 28
    FORTYSEVEN = 29
    FORTYEIGHT = 30
    FORTYNINE = 31
    FIFTY = 32
    FIFTYONE = 33
    FIFTYTWO = 34
    FIFTYTHREE = 35
    FIFTYFOUR = 36
    FIFTYFIVE = 37
    FIFTYSIX = 38
    FIFTYSEVEN = 39
    FIFTYEIGHT = 40
    FIFTYNINE = 41
    SIXTY = 42
    
    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value == 18:
            return cls.EIGHTEEN
        elif value == 19:
            return cls.NINETEEN
        elif value == 20:
            return cls.TWENTY
        elif value == 21:
            return cls.TWENTYONE
        elif value == 22:
            return cls.TWENTYTWO
        elif value == 23:
            return cls.TWENTYTHREE
        elif value == 24:
            return cls.TWENTYFOUR
        elif value == 25:
            return cls.TWENTYFIVE
        elif value == 26:
            return cls.TWENTYSIX
        elif value == 27:
            return cls.TWENTYSEVEN
        elif value == 28:
            return cls.TWENTYEIGHT
        elif value == 29:
            return cls.TWENTYNINE
        elif value == 30:
            return cls.THIRTY
        elif value == 31:
            return cls.THIRTYONE
        elif value == 32:
            return cls.THIRTYTWO
        elif value == 33:
            return cls.THIRTYTHREE
        elif value == 34:
            return cls.THIRTYFOUR
        elif value == 35:
            return cls.THIRTYFIVE
        elif value == 36:
            return cls.THIRTYSIX
        elif value == 37:
            return cls.THIRTYSEVEN
        elif value == 38:
            return cls.THIRTYEIGHT
        elif value == 39:
            return cls.THIRTYNINE
        elif value == 40:
            return cls.FORTY
        elif value == 41:
            return cls.FORTYONE
        elif value == 42:
            return cls.FORTYTWO
        elif value == 43:
            return cls.FORTYTHREE
        elif value == 44:
            return cls.FORTYFOUR
        elif value == 45:
            return cls.FORTYFIVE
        elif value == 46:
            return cls.FORTYSIX
        elif value == 47:
            return cls.FORTYSEVEN
        elif value == 48:
            return cls.FORTYEIGHT
        elif value == 49:
            return cls.FORTYNINE
        elif value == 50:
            return cls.FIFTY
        elif value == 51:
            return cls.FIFTYONE
        elif value == 52:
            return cls.FIFTYTWO
        elif value == 53:
            return cls.FIFTYTHREE
        elif value == 54:
            return cls.FIFTYFOUR
        elif value == 55:
            return cls.FIFTYFIVE
        elif value == 56:
            return cls.FIFTYSIX
        elif value == 57:
            return cls.FIFTYSEVEN
        elif value == 58:
            return cls.FIFTYEIGHT
        elif value == 59:
            return cls.FIFTYNINE
        elif value == 60:
            return cls.SIXTY
        else:
            raise ValueError("Invalid number value: {}".format(value))




class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3
    num_age_classes = 43
    num_3labels_clases = 3 + 2 + 43
    
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
    # age 세분화: 라벨 추가
    new_age_labels = []
    
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
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
                # age 세분화: 라벨 추가
                new_age_label = NewAgeLabels.from_number(age)
                
                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                # age 세분화: 라벨 추가
                self.new_age_labels.append(new_age_label)
                
    def calc_statistics(self):
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
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        # age 세분화 작업
        new_age_label = self.get_new_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, (mask_label, gender_label, age_label, new_age_label), multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]
    # age세분화: 함수 추가
    def get_new_age_label(self, index) -> NewAgeLabels:
        return self.new_age_labels[index]
    

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label
# 여기 새 클래스 들어가야 되나?
    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


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
                    #age 세분화
                    new_age_label = NewAgeLabels.from_number(age)


                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    self.new_age_labels.append(age_label)
                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
