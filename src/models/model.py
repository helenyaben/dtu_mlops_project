import torch.nn.functional as F
from torch import nn
class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3), # [N, 64, 26]
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 6, 4), # [N, 32, 24]
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 128, 3), # [N, 16, 22]
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 8, 2), # [N, 8, 20]
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(32, 12),
            nn.LogSoftmax(dim=1)
        )
        # self.fc1 = nn.Linear(16384, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 12)
        # self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        # x = x.view(x.shape[0], -1)
        
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        # x = F.log_softmax(self.fc4(x), dim=1)
        
        return self.classifier(self.backbone(x))
        