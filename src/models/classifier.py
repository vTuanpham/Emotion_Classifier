import torch.nn as nn
import torch.nn.functional as F
import torch


class EmotionClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3, num_features: int=64, num_labels: int=7, width=48, height=48):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(num_features, 2 * num_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(2 * num_features)
        self.conv3 = nn.Conv2d(2 * num_features, 2 * num_features, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(2 * num_features)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(dropout)

        self.conv4 = nn.Conv2d(2 * num_features, 4 * num_features, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(4 * num_features)
        self.conv5 = nn.Conv2d(4 * num_features, 4 * num_features, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(4 * num_features)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(dropout)

        self.conv6 = nn.Conv2d(4 * num_features, 8 * num_features, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(8 * num_features)
        self.conv7 = nn.Conv2d(8 * num_features, 8 * num_features, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2d(8 * num_features)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(dropout)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(8 * num_features * (width // 16) * (height // 16), 8 * num_features)
        self.relu8 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(8 * num_features, 4 * num_features)
        self.relu9 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(4 * num_features, 2 * num_features)
        self.relu10 = nn.ReLU()
        self.dropout7 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(2 * num_features, num_labels)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.bn7(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu8(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.relu9(x)
        x = self.dropout6(x)
        x = self.fc3(x)
        x = self.relu10(x)
        x = self.dropout7(x)
        x = self.fc4(x)
        x = self.softmax(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    bn_model = EmotionClassifier()
    x = torch.randn(1, 1, 48, 48)
    print('Shape of output = ', bn_model(x).shape)
    print('No of Parameters of the BatchNorm-CNN Model =', bn_model.count_parameters())