import torch
x = torch.rand(3, 1, requires_grad=True)
a = torch.rand(5, requires_grad=True)
y = (x * a)

print(torch.autograd.grad(y, a))
