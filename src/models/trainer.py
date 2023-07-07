import os
import torch
import numpy as np
from tqdm.auto import tqdm
from argparse import Namespace
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from src.data.dataloader import EmotionDataloader
from src.models.classifier import EmotionClassifier
from src.utils.utils import set_seed


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
                 visualize_learning_curve: bool = True
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
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        def lambda_function(epoch):
            # Define your custom learning rate schedule
            if epoch % 3 != 0:
                return 1.0  # Update every 3 epochs
            else:
                return 0.5  # After the 10th epoch, reduce the learning rate by half

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_function)

        valid_loss_min = np.Inf
        train_losses, test_losses = [], []
        for e in tqdm(range(self.num_train_epochs), colour='green', position=0, leave=False):
            model.train()
            running_loss = 0
            tr_accuracy = 0
            for images, labels in tqdm(self.dataloaders['train'], colour='blue', position=1, leave=True):
                images = images.to(self.device)
                labels = labels.long().to(self.device)
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

            train_losses.append(running_loss / len(self.dataloaders['train']))
            test_losses.append(test_loss / len(self.dataloaders['eval']))

            print("Epoch: {}/{} ".format(e + 1, self.num_train_epochs),
                  "Training Loss: {:.3f} ".format(train_losses[-1]),
                  "Training Acc: {:.3f} ".format(tr_accuracy / len(self.dataloaders['train'])),
                  "Val Loss: {:.3f} ".format(test_losses[-1]),
                  "Val Acc: {:.3f}".format(accuracy / len(self.dataloaders['eval'])))
            if test_loss / len(self.dataloaders['eval']) <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                        test_loss / len(self.dataloaders['eval'])))
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
