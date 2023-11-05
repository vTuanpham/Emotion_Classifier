import os
import torch
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from src.data.dataloader import EmotionDataloader
from src.models.classifier import EmotionClassifier
from src.utils.utils import set_seed, in_notebook


if in_notebook():
    try:
        from tqdm import tqdm_notebook as tqdm
    except ImportError as e:
        from tqdm.auto import tqdm
else:
    from tqdm.auto import tqdm


class Trainer:
    def __init__(self,
                 dataloaders,
                 output_dir: str,
                 config: Namespace,
                 seed: int = 42,
                 learning_rate: float=1e-3,
                 train_batch_size: int = 1000,
                 val_batch_size: int = 500,
                 num_train_epochs: int = 100,
                 dropout: float=0.3,
                 visualize_learning_curve: bool = True,
                 optim_name: str = "AdamW",
                 ):
        self.dataloaders = dataloaders.__call__()
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.visualize_learning_curve = visualize_learning_curve
        self.config = config
        self.seed = seed
        self.learning_rate = learning_rate
        self.optim_name = optim_name
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Since the Negative log likelyhood require the input
        # to be log probabilities rather than raw probabilities NLLLoss is chosen
        self.criterion = nn.NLLLoss()
        set_seed(self.seed)

    def build_model(self):
        emotion_classifier = EmotionClassifier(dropout=self.dropout)
        return emotion_classifier.to(self.device)

    def train(self):
        model = self.build_model()

        print(f"\n  Training init info: ")
        for key, value in vars(self.config).items():
            print(f" {key}: {value}")
        print("\n")

        # Define the optimizer and the learning rate scheduler
        optimizer = getattr(torch.optim, self.optim_name)(model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.85, 0.95))
        scheduler = lr_scheduler.LinearLR(optimizer)

        # Set valid loss to be inf so the first epoch loss would be saved
        valid_loss_min = np.Inf
        train_losses, test_losses = [], []
        for e in tqdm(range(self.num_train_epochs), colour='green', position=0, leave=False):
            model.train()
            running_loss = 0
            tr_accuracy = 0
            for images, labels in tqdm(self.dataloaders['train'], colour='blue', position=1, leave=True):
                # Send the images and labels to cuda(GPU memory) to process
                images = images.to(self.device)
                labels = labels.long().to(self.device)

                # Forward pass
                # Reset to zero using optimizer.zero_grad() to avoid accumulating gradients from previous iterations.
                optimizer.zero_grad()
                log_ps = model(images)
                loss = self.criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                tr_accuracy += torch.mean(equals.type(torch.FloatTensor))

            # Update the lr
            scheduler.step()

            test_loss, accuracy = self.eval(model)

            # Compute average loss over the entire epoch
            train_losses.append(running_loss / len(self.dataloaders['train']))
            test_losses.append(test_loss / len(self.dataloaders['eval']))

            tqdm.write(f"Epoch: {e + 1}/{self.num_train_epochs}" \
                       f"Training Loss: {train_losses[-1]} " \
                       f"Training Acc: {tr_accuracy / len(self.dataloaders['train'])} " \
                       f"Val Loss: {test_losses[-1]} " \
                       f"Val Acc: {accuracy / len(self.dataloaders['eval'])}")
            if test_loss / len(self.dataloaders['eval']) <= valid_loss_min:
                tqdm.write(f'Validation loss decreased ({valid_loss_min} --> {test_loss / len(self.dataloaders["eval"])}).  Saving model ...')
                self.save(model)
                valid_loss_min = test_loss / len(self.dataloaders['eval'])

        if self.visualize_learning_curve:
            fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
            ax.plot(torch.tensor(train_losses).cpu(), 'b', label='Training Loss')
            ax.plot(torch.tensor(test_losses).cpu(), 'r', label='Validation Loss')
            ax.legend(loc='best')  # Use ax.legend() instead of fig.legend()
            plt.show()
            plot_save_path = os.path.join(self.output_dir, "result_plot.png")
            fig.savefig(plot_save_path, bbox_inches='tight')
        return model

    def eval(self, model):
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in tqdm(self.dataloaders['eval'], colour='red', position=3, leave=True):
                images = images.to(self.device)
                labels = labels.long().to(self.device)
                log_ps = model(images)
                test_loss += self.criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        return test_loss, accuracy

    def save(self, model):
        model_path = os.path.join(self.output_dir, 'best_model.pt')
        torch.save(model.state_dict(), model_path)
