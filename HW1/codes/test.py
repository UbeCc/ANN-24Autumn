import torch
import torch.nn as nn
import numpy as np
epsilon = 1e-8
# Custom KL Divergence Loss
class KLDivLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        log_h = np.where(target == 0, 0, np.log(h + epsilon))
        log_target = np.where(target == 0, 0, np.log(target + epsilon))
        loss = np.mean(target * (log_target - log_h))
        return loss

    def backward(self, input, target):
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        grad = h - target
        return grad / target.shape[0]  # Ensure mean reduction

# PyTorch setup
logits = torch.randn(10, 5, requires_grad=True)
log_probs = torch.log_softmax(logits, dim=1)

target_probs = torch.rand(10, 5)
target_probs /= target_probs.sum(dim=1, keepdim=True)
target_probs = target_probs.detach()  # No grad needed for target

# Custom and PyTorch loss calculation
my_loss = KLDivLoss("loss")
torch_loss = nn.KLDivLoss(reduction='batchmean')

my_forward = my_loss.forward(log_probs.detach().numpy(), target_probs.numpy())
torch_forward = torch_loss(log_probs, target_probs)

print("My Forward:", my_forward, "Torch Forward:", torch_forward.item())

torch_forward.backward()
my_backward = my_loss.backward(log_probs.detach().numpy(), target_probs.numpy())

print("My Backward:\n", my_backward)
print("Torch Gradient:\n", logits.grad.numpy())

# Additional debug info
print("Difference in gradients:\n", np.abs(my_backward - logits.grad.numpy()))