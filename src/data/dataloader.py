import sys
import warnings

sys.path.insert(0, r'./')
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, random_split


EMOTIONS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}


class EmotionDataset(Dataset):
    def __init__(self,
                 data_path:str=None,
                 split: Literal['train', 'test']='train',
                 transform=None):
        self.dataset = datasets.FER2013(root=data_path, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        if label is None:
            warnings.warn("\n Test dataset does not have label!")

        return image, label


class EmotionDataloader:
    def __init__(self,
                 data_path: str,
                 train_transform=None,
                 val_transform=None,
                 train_batch_size: int=3000,
                 val_batch_size: int=500,
                 train_size: float=0.8,
                 num_worker: int=2,
                 seed: int=42):
        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_size = train_size
        self.num_worker = num_worker

        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def __call__(self):
        print(f"\n Loading train and eval dataset from {self.data_path.split('/')[-1]}")
        train_eval_dataset = EmotionDataset(self.data_path, split='train', transform=self.train_transform)
        train_dataset, eval_dataset = random_split(train_eval_dataset, lengths=[self.train_size, 1-self.train_size])

        print(f"\n Loading test dataset from {self.data_path.split('/')[-1]}")
        test_dataset = EmotionDataset(self.data_path, split='test', transform=self.val_transform)

        print(f"\n Dataset info from {self.data_path}: "
              f"\n Seed: {self.seed}"
              f"\n Number of training examples: {len(train_dataset)}"
              f"\n Number of evaluating examples: {len(eval_dataset)}"
              f"\n Number of test examples: {len(test_dataset.dataset)}\n")

        return {"train": self.get_dataloader(train_dataset, shuffle_flag=True, batch_size=self.train_batch_size),
                "eval": self.get_dataloader(eval_dataset, shuffle_flag=False, batch_size=self.val_batch_size),
                "test": self.get_dataloader(test_dataset, shuffle_flag=False, batch_size=self.val_batch_size)}

    @staticmethod
    def show_random_data(faces, emotions):
        idx = np.random.randint(len(faces))
        print(EMOTIONS[int(emotions[idx])])
        plt.imshow(faces[idx].reshape(48, 48), cmap='gray')
        plt.show()

    def get_dataloader(self, dataset, shuffle_flag: bool = False, batch_size: int = 1) -> DataLoader:
        sampler = RandomSampler(data_source=dataset, generator=self.generator) if shuffle_flag else \
            SequentialSampler(dataset)
        return DataLoader(dataset,
                          sampler=sampler,
                          batch_size=batch_size,
                          drop_last=True,
                          num_workers=self.num_worker)


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.Normalize((0.507395516207,), (0.255128989415,))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507395516207,), (0.255128989415,))
    ])
    dataloaders = EmotionDataloader(r"./fer_2013",
                                    train_transform=train_transform,
                                    val_transform=val_transform,
                                    num_worker=1)

    dataloaders = dataloaders.__call__()

    print(f"\n Number of training examples: {len(dataloaders['train'].dataset)}"
          f"\n Number of training batches: {len(dataloaders['train'])}"
          f"\n Number of validation examples: {len(dataloaders['eval'].dataset)}"
          f"\n Number of validation batches: {len(dataloaders['eval'])}"
          f"\n Number of test examples: {len(dataloaders['test'].dataset)}"
          f"\n Number of test batches: {len(dataloaders['test'])}")

    for face, emotion in iter(dataloaders['train']):
        EmotionDataloader.show_random_data(face, emotion)
