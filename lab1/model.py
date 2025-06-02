import torch
import torch.nn as nn
import torch.ao.quantization as tq


class VGG(nn.Module):
    """ Implement your model here """
    def __init__(self, in_channels=3, in_size=32, num_classes=10) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,  64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,  192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(192,  384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384,  256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256,  256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc6 = nn.Sequential(
            nn.Linear(4*4*256, 256),
            nn.ReLU(),
        )

        self.fc7 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.fc8 = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x




if __name__ == "__main__":
    model = VGG()
    inputs = torch.randn(1, 3, 32, 32)
    print(model)

    from torchsummary import summary

    summary(model, (3, 32, 32), device="cpu")


