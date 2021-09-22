import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from train_model_mnist import Model

batch_size = 4
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5)),]
)
testset = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

PATH = "./trained_model_mnist.pth"
model = Model()
model.load_state_dict(torch.load(PATH))


criterion = nn.CrossEntropyLoss()

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = torch.flatten(inputs, start_dim=1)
        # calculate outputs by running images through the network
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    "Accuracy of the network on the 10000 test images: %d %%"
    % (100 * correct / total)
)
