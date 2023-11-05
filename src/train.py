import os
import sys
import argparse
sys.path.insert(0, r'./') #Add root directory here
import torchvision.transforms as transforms
from src.data.dataloader import EmotionDataloader
from src.models.trainer import Trainer
from src.utils.utils import clear_cuda_cache


def parse_args(args):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--output_dir', type=str, default="./src/checkpoints",
                        help="The output directory to save")
    parser.add_argument('--data_path', type=str, default="./src/data/fer_2013/fer2013/fer2013.csv",
                        help="The path to eval dir")
    parser.add_argument('--train_batch_size', type=int, default=100, help="Batch size for the dataloader")
    parser.add_argument('--val_batch_size', type=int, default=300, help="Eval batch size for the dataloader")
    parser.add_argument('--train_size', type=float, default=0.8,
                        help="The size of the training data")
    parser.add_argument('--num_worker', type=int, default=2, help="Number of worker for dataloader")
    parser.add_argument('--seed', type=int, default=44, help="A seed for reproducible training.")

    # Training
    parser.add_argument('--num_train_epochs', type=int, default=20,
                        help="number training epochs")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Dropout arg for classifier (prevent Overfitting)")

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument('--optim_name', type=str, default="AdamW", help="The name of the optimizer")

    args = parser.parse_args(args)

    # Sanity check
    assert os.path.isdir(args.output_dir), "Invalid output dir path!"
    assert os.path.isfile(args.data_path), "Invalid content file path!"

    return args


def main(args):
    args = parse_args(args)
    clear_cuda_cache()

    dataloader_args = {
        "data_path": args.data_path,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "train_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507395516207,), (0.255128989415,)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(p=5, distortion_scale=0.3, fill=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomAutocontrast(),
            transforms.ColorJitter(brightness=.5, contrast=.5),
            transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            transforms.RandomRotation(20),
        ]),
        "val_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507395516207,), (0.255128989415,))
        ]),
        "train_size": args.train_size,
        "seed": args.seed,
        "num_worker": args.num_worker
    }
    dataloaders = EmotionDataloader(**dataloader_args)

    trainer_args = {
        "dataloaders": dataloaders,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "config": args,
        "optim_name": args.optim_name,
    }
    trainer = Trainer(**trainer_args)
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
