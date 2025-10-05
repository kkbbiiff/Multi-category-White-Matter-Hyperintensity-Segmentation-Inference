from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image

def random_rot_flip(image, image_t1, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    image_t1 = np.rot90(image_t1, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image_t1 = np.flip(image_t1, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, image_t1, label


def random_rotate(image, image_t1, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    image_t1 = ndimage.rotate(image_t1, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, image_t1, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        image_t1 = sample['image_t1']
        has_label = 'label' in sample and sample['label'] is not None
        label = sample['label'] if has_label else None

        if has_label:
            if random.random() > 0.5:
                image, image_t1, label = random_rot_flip(image, image_t1, label)
            elif random.random() > 0.5:
                image, image_t1, label = random_rotate(image, image_t1, label)
            
            x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

            x, y = image_t1.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image_t1 = zoom(image_t1, (self.output_size[0] / x, self.output_size[1] / y), order=3)

            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            image_t1 = torch.from_numpy(image_t1.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'image_t1': image_t1, 'label': label.long()}
        else:
            x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            x, y = image_t1.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image_t1 = zoom(image_t1, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            image_t1 = torch.from_numpy(image_t1.astype(np.float32)).unsqueeze(0)
            sample = {'image': image, 'image_t1': image_t1}
        return sample



class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, transform=None):
        super(NPY_datasets, self)
        self.transform = transform  # using transform in torch!
        if train:
            images_list = sorted(os.listdir(path_Data + 'train/images/'))
            masks_list = sorted(os.listdir(path_Data + 'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            masks_list = sorted(os.listdir(path_Data + 'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        # img = np.array(Image.open(img_path).convert('RGB'))
        img = np.array(Image.open(img_path))
        # img = img[:, :, 0]
        print(img.shape)
        # msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        msk = np.array(Image.open(msk_path))
        # msk = np.stack([msk] * 3, axis=2)
        print("imageshape:",img.shape)
        sample = {'image': img, 'label': msk}
        sample = self.transform(sample)
        # sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

    def __len__(self):
        return len(self.data)


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image = data['image']
            image_t1 = data['image_t1']
            label = data['label'] if 'label' in data else None
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, "{}.npy.h5".format(vol_name))
            data = h5py.File(filepath)
            image = data['image'][:]
            image_t1 = data['image_t1'][:] if 'image_t1' in data else None
            label = data['label'][:] if 'label' in data else None

        sample = {'image': image, 'image_t1': image_t1}
        if label is not None:
            sample['label'] = label
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
