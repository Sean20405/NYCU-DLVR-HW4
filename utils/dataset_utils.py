import os
import random
from PIL import Image
import numpy as np
from glob import glob

from torch.utils.data import Dataset
from torchvision.transforms import (
    ToPILImage, Compose, RandomCrop, ToTensor, ColorJitter
)

from utils.image_utils import random_augmentation, crop_img


class PromptTrainDataset(Dataset):
    def __init__(self, args, transform=None):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rain_ids = []
        self.snow_ids = []
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.transform = transform
        print(self.de_type)

        self.de_dict = {'derain': 0, 'desnow': 1}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        """
        Initialize the dataset by loading the filename of the degraded images.

        To access the corresponding clean images, change the filename from
        `degraded/{type}-{id}.png` to `clean/{type}_clean-{id}.png`.
        """
        glob_pattern = os.path.join(self.args.train_dir, 'degraded/', '*')
        files = glob(glob_pattern)
        rain_files = []
        snow_files = []
        for file in files:
            if 'rain-' in file:
                rain_files.append(file)
            elif 'snow-' in file:
                snow_files.append(file)

        self.rain_ids = [{"degrad_id": x, "de_type": 0} for x in rain_files]
        self.snow_ids = [{"degrad_id": x, "de_type": 1} for x in snow_files]

        self.rain_counter = 0
        self.num_rain = len(self.rain_ids)
        print(f'Total Rain Ids : {self.num_rain}')

        self.snow_counter = 0
        self.num_snow = len(self.snow_ids)
        print(f'Total Snow Ids : {self.num_snow}')

        random.shuffle(self.de_type)

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[
            ind_H:ind_H + self.args.patch_size,
            ind_W:ind_W + self.args.patch_size
        ]
        patch_2 = img_2[
            ind_H:ind_H + self.args.patch_size,
            ind_W:ind_W + self.args.patch_size
        ]

        return patch_1, patch_2

    def _get_gt_name(self, degrad_name):
        """
        From `degraded/{type}-{id}.png` to `clean/{type}_clean-{id}.png`.
        """
        gt_name = degrad_name.replace('degraded', 'clean')
        gt_name_split = gt_name.split('-')
        gt_name_split[0] += '_clean'
        gt_name = '-'.join(gt_name_split)

        return gt_name

    def _merge_ids(self):
        self.sample_ids = self.rain_ids + self.snow_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        degrad_name = sample["degrad_id"]
        degrad_img = crop_img(
            np.array(Image.open(degrad_name).convert('RGB')), base=16
        )
        clean_name = self._get_gt_name(degrad_name)
        clean_img = crop_img(
            np.array(Image.open(clean_name).convert('RGB')), base=16
        )

        if self.transform is None:  # Default augmentation
            degrad_patch, clean_patch = random_augmentation(
                *self._crop_patch(degrad_img, clean_img)
            )
        else:
            wrapper = self.transform(image=degrad_img, image2=clean_img)
            degrad_patch, clean_patch = wrapper['image'], wrapper['image2']

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class PromptTrainDatasetColorJitter(Dataset):
    def __init__(self, args):
        super(PromptTrainDatasetColorJitter, self).__init__()
        self.args = args
        self.rain_ids = []
        self.snow_ids = []
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'derain': 0, 'desnow': 1}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        self.toTensor = ToTensor()

    def _init_ids(self):
        """
        Initialize the dataset by loading the filename of the degraded images.

        To access the corresponding clean images, change the filename from
        `degraded/{type}-{id}.png` to `clean/{type}_clean-{id}.png`.
        """
        glob_pattern = os.path.join(self.args.train_dir, 'degraded/', '*')
        files = glob(glob_pattern)
        rain_files = []
        snow_files = []
        for file in files:
            if 'rain-' in file:
                rain_files.append(file)
            elif 'snow-' in file:
                snow_files.append(file)

        self.rain_ids = [{"degrad_id": x, "de_type": 0} for x in rain_files]
        self.snow_ids = [{"degrad_id": x, "de_type": 1} for x in snow_files]

        self.rain_counter = 0
        self.num_rain = len(self.rain_ids)
        print(f'Total Rain Ids : {self.num_rain}')

        self.snow_counter = 0
        self.num_snow = len(self.snow_ids)
        print(f'Total Snow Ids : {self.num_snow}')

        random.shuffle(self.de_type)

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[
            ind_H:ind_H + self.args.patch_size,
            ind_W:ind_W + self.args.patch_size
        ]
        patch_2 = img_2[
            ind_H:ind_H + self.args.patch_size,
            ind_W:ind_W + self.args.patch_size
        ]

        return patch_1, patch_2

    def _get_gt_name(self, degrad_name):
        """
        From `degraded/{type}-{id}.png` to `clean/{type}_clean-{id}.png`.
        """
        gt_name = degrad_name.replace('degraded', 'clean')
        gt_name_split = gt_name.split('-')
        gt_name_split[0] += '_clean'
        gt_name = '-'.join(gt_name_split)

        return gt_name

    def _merge_ids(self):
        self.sample_ids = self.rain_ids + self.snow_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        degrad_name = sample["degrad_id"]
        degrad_img = crop_img(
            np.array(Image.open(degrad_name).convert('RGB')), base=16
        )
        clean_name = self._get_gt_name(degrad_name)
        clean_img = crop_img(
            np.array(Image.open(clean_name).convert('RGB')), base=16
        )

        degrad_patch, clean_patch = random_augmentation(
            *self._crop_patch(degrad_img, clean_img)
        )
        degrad_patch_pil = Image.fromarray(degrad_patch.astype(np.uint8))
        degrad_patch_jittered = self.color_jitter(degrad_patch_pil)
        degrad_patch = np.array(degrad_patch_jittered)

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_dir)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        print(root)
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception(
                    'The input directory does not contain any image files'
                )
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print(f"Total Images: {len(self.degraded_ids)}")

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(
            np.array(
                Image.open(self.degraded_ids[idx]).convert('RGB')
            ),
            base=16
        )
        name = self.degraded_ids[idx]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
