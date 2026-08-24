import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # always call this first
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = MyModel()

    # All parameters across all submodules, recursively
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    # Just the submodules
    # for name, module in model.named_modules():
    #     print(name, module)
    # for item in model.parameters():
    #     print(item)
    for module in model.named_modules():
        print(module)