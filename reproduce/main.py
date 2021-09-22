import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from train_model_mnist import Model
from tqdm import tqdm


batch_size = 1
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5)),]
)
testset = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

trainset = torchvision.datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)


PATH = "./trained_model_mnist.pth"
model = Model()
model.load_state_dict(torch.load(PATH))


criterion = nn.CrossEntropyLoss()


def flatten_tensor_list(tensors, return_shapes=False):
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.view(-1))
    return torch.cat(flattened, 0).detach().numpy()


def inverse_fisher(data, model, criterion, init_lambda=1):
    all_params = flatten_tensor_list(model.parameters())
    inv_F = init_lambda * np.eye(len(all_params))
    N = len(data)
    for inputs, labels in tqdm(data):
        inputs = torch.flatten(inputs, start_dim=1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, model.parameters())
        grads = flatten_tensor_list(grads)
        v = inv_F.dot(grads)
        inv_F = inv_F - v[:, None].dot(v[None, :]) / (N + v.dot(grads))
    return inv_F


def update(data, model, criterion, init_lambda=1, target_sparsity=0.5):
    all_params = flatten_tensor_list(model.parameters(), return_shapes=True)
    inv_F = inverse_fisher(data, model, criterion, init_lambda)
    statistics = all_params ** 2 / (2 * np.diag(inv_F))
    index = np.argsort(statistics)
    to_mask_params = index[: int(len(index) * target_sparsity)]
    all_params = all_params - np.sum(inv_F * statistics, axis=0)
    all_params[to_mask_params] = 0

    with torch.no_grad():
        current_index = 0
        for params in model.parameters():
            params.copy_(
                torch.Tensor(
                    all_params[
                        current_index : current_index + np.prod(params.shape)
                    ]
                ).view_as(params)
            )
            current_index = current_index + np.prod(params.shape)
    return model


# flatten = flatten_tensor_list(model.parameters())
# flatten2 = flatten_tensor_list(
#     inverse_flatten_tensor_list(model.parameters(), flatten * 2)
# )

# print(flatten[:30].shape)
# print(flatten2[:30].shape)

data = trainloader
model = update(data, model, criterion)
flat = flatten_tensor_list(model.parameters())
print("number of zero in the model params", np.sum(flat == 0))


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
