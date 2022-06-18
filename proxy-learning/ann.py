import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.vars = nn.ParameterList()
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
        )

        self.conv = nn.Sequential(
            nn.ReLU(),           

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64 * 4 * 5, bias=False),
            nn.ReLU(),
            nn.Linear(64 * 4 * 5, 64 * 3 * 3, bias=False),
            nn.ReLU(),
            nn.Linear(64 * 3 * 3, 64 * 2 * 1, bias=False),
            nn.ReLU (),
            nn.Linear(64 * 2 * 1, 10, bias=False),
        )


    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        return self.fc(self.conv(self.static_conv(x)))
    
    def parameters(self):
        return self.vars