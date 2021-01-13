import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=3)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv4_drop = nn.Dropout()
        self.fc1 = nn.Linear(5696, 32)
        self.fc2 = nn.Linear(32, num_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 3))
        x = F.relu(F.max_pool1d(self.conv2(x), 3))
        x = F.relu(F.avg_pool1d(self.conv3(x), 3))
        x = F.relu(self.conv4_drop(self.conv4(x)))
        x = x.view(-1, 5696)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        return x