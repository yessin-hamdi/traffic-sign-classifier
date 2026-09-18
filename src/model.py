import torch
import torch.nn as nn


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = x.view(x.size(0), -1)                 # flatten (keep batch dim)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)                       # only active during model.train()
        x = self.fc2(x)                           # raw logits, no softmax here
        return x


if __name__ == '__main__':
    model = TrafficSignCNN()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f'Output shape: {output.shape}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
