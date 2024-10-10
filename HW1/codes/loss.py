from __future__ import division
import numpy as np

epsilon = 1e-8

# KLDivLoss = SoftmaxCrossEntropyLoss, because log(1) = 0, 0log(0) = 0
class KLDivLoss(object):
    def __init__(self, name):
        self.name = name
    
    def forward(self, input, target): # (100 x 10), (100 x 10)
        # TODO START
        mval = np.max(input, axis=1, keepdims=True)
        h = np.exp(input - mval) / np.sum(np.exp(input - mval), axis=1, keepdims=True)
        log_h = np.where(target == 0, 0, np.log(h + epsilon))
        log_target = np.where(target == 0, 0, np.log(target + epsilon))
        loss = np.sum(target * (log_target - log_h)) / input.shape[0]
        # print("DEBUG:", target * (log_target - log_h))
        return loss
        # TODO END

    def backward(self, input, target):
		# TODO START
        mval = np.max(input, axis=1, keepdims=True)
        h = np.exp(input - mval) / np.sum(np.exp(input - mval), axis=1, keepdims=True)
        grad = h - target
        return grad
		# TODO END

class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        # print(input[0])
        mval = np.max(input, axis=1, keepdims=True)
        h = np.exp(input - mval) / np.sum(np.exp(input - mval), axis=1, keepdims=True)
        log_h = np.where(target == 0, 0, np.log(h + epsilon))
        loss = -np.sum(target * log_h) / input.shape[0]
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        mval = np.max(input, axis=1, keepdims=True)
        h = np.exp(input - mval) / np.sum(np.exp(input - mval), axis=1, keepdims=True)
        grad = h - target
        # CHECK: whether divide by bsize? (don't)
        return grad
        # TODO END

# TODO: check correctness
class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        # \text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}
        idx = np.argmax(target, axis=1) # prelim: every batch only have one label==1
        pred_gt = input[np.arange(input.shape[0]), idx].reshape(-1, 1)
        E = np.maximum(0, self.margin - pred_gt + input) # else
        E[np.arange(input.shape[0]), idx] = 0 # if $k=t_n$
        loss = np.sum(E) / input.shape[0]
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        idx = np.argmax(target, axis=1)
        pred_gt = input[np.arange(input.shape[0]), idx].reshape(-1, 1)
        E = np.maximum(0, self.margin - pred_gt + input) # else
        E[np.arange(input.shape[0]), idx] = 0 # if $k=t_n$
        grad = np.zeros_like(input)
        mask = E > 0
        grad[mask] = 1
        grad[np.arange(input.shape[0]), idx] -= np.sum(mask, axis=1)
        return grad
        # TODO END

class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        alpha = np.array(self.alpha)
        mval = np.max(input, axis=1, keepdims=True)
        h = np.exp(input - mval) / np.sum(np.exp(input - mval), axis=1, keepdims=True)
        cross_entropy = alpha * target + (1 - alpha) * (1 - target)
        E = cross_entropy * np.power(1 - h, self.gamma) * target * np.log(h)
        return -np.sum(E) / input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        # Reference: https://github.com/namdvt/Focal-loss-pytorch-implementation
        alpha = np.array(self.alpha)
        mval = np.max(input, axis=1, keepdims=True)
        h = np.exp(input - mval) / np.sum(np.exp(input - mval), axis=1, keepdims=True)
        cross_entropy = alpha * target + (1 - alpha) * (1 - target)
        E = cross_entropy * (self.gamma * np.power(1 - h, self.gamma - 1) * target * np.log(h) - np.power(1 - h, self.gamma) * target / h)
        bsize, lines = E.shape[0], E.shape[1]
        h_col, h_row, I = h.reshape(bsize, 1, lines), h.reshape(bsize, lines, 1), np.eye(lines)
        h_x = -h_row @ h_col + I * h_col
        return np.sum(h_x * E[:, np.newaxis, :], axis=2)
        # TODO END

if __name__ == "__main__":
    from torch.nn.modules.loss import KLDivLoss as TorchKLDivLoss
    import torch
    from torch import nn
    import torch.nn.functional as F
    import numpy as np
    
    with_label = False
    # inputs = [[-1.5072, -0.5475,  1.6552,  1.3270,  0.1383, -0.1262, -0.4392, -0.6543, 0.0127, -0.9532], [-1.5072, -0.5475,  1.6552,  1.3270,  0.1383, -0.1262, -0.4392, -0.6543, 0.0127, -0.9532]]
    # label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    # 5.34376
    inputs = [[-1.5072, -0.5475,  1.6552,  1.3270,  0.1383, -0.1262, -0.4392, -0.6543, 0.0127, -0.9532]]
    label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    idx = np.argmax(label, axis=1)
    inputs = np.array(inputs)
    label = np.array(label)

    logits = torch.randn(10, 5, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=1)

    target_probs = torch.rand(10, 5)
    target_probs /= target_probs.sum(dim=1, keepdim=True)
    target_probs = target_probs.detach()  # No grad needed for target
    idx = np.argmax(target_probs, axis=1)
    # KL
    # my_loss = KLDivLoss("loss")
    # torch_loss = nn.KLDivLoss(reduction='batchmean')
    
    # SoftmaxCrossEntropy
    # my_loss = SoftmaxCrossEntropyLoss("loss")
    # torch_loss = nn.CrossEntropyLoss()
    
    # HingeLoss
    my_loss = HingeLoss("loss")
    torch_loss = nn.MultiMarginLoss(p=1, margin=5, reduction="mean")

    # FocalLoss
    # my_loss = FocalLoss("loss")
    
    if not with_label:
        my_forward = my_loss.forward(log_probs.detach().numpy(), target_probs.numpy())
        torch_forward = torch_loss(log_probs, idx)
        # torch_forward = torch_loss(log_probs, target_probs)
        print("My Forward:", my_forward, "Torch Forward:", torch_forward.item())
        torch_forward.backward()
        my_backward = my_loss.backward(log_probs.detach().numpy(), target_probs.numpy())
        print("My Backward:\n", my_backward)
        print("Torch Gradient:\n", logits.grad.numpy())
        print("Difference in gradients:\n", np.abs(my_backward - logits.grad.numpy()))
        assert np.allclose(my_backward, logits.grad.numpy()), "Gradients are not equal!"
    else:
        my_forward = my_loss.forward(inputs, label)
        inputs = torch.tensor(inputs, requires_grad=True)
        torch_forward = torch_loss(inputs, torch.tensor(idx))
        print("My Forward:", my_forward, "Torch Forward:", torch_forward.item())
        torch_forward.backward()
        my_backward = my_loss.backward(inputs.detach().numpy(), label)
        print("My Backward:\n", my_backward)
        print("Torch Gradient:\n", inputs.grad.numpy())
        print("Difference in gradients:\n", np.abs(my_backward - inputs.grad.numpy()))
        assert np.allclose(my_backward, logits.grad.numpy()), "Gradients are not equal!"