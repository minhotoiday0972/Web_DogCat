import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 32 * 32, num_classes) # Giáº£ Ä‘á»‹nh image_size=128, sau pooling 2 láº§n cÃ²n 32x32

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32) # Flatten tensor (náº¿u image_size=128)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # Kiá»ƒm tra nhanh mÃ´ hÃ¬nh
    model = SimpleCNN(num_classes=2)
    print(model) # In kiáº¿n trÃºc mÃ´ hÃ¬nh Ä‘á»ƒ kiá»ƒm tra