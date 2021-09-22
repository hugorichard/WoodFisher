import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def __main__():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5)),]
    )
    batch_size = 4
    classes = list(range(10))
    trainset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    import torch.optim as optim

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(30):
        running_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = torch.flatten(inputs, start_dim=1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(
                    "[%d, %5d] loss %.3f"
                    % (epoch + 1, i + 1, running_loss / 2000)
                )
                running_loss = 0

    PATH = "./trained_model_mnist.pth"
    torch.save(model.state_dict(), PATH)

    print("Finished trianing")
