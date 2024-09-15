from __future__ import division
import numpy as np

epsilon = 1e-8

# KLDivLoss = SoftmaxCrossEntropyLoss, because log(1) = 0, 0log(0) = 0
class KLDivLoss(object):
    def __init__(self, name):
        self.name = name
    
    def forward(self, input, target): # (100 x 10), (100 x 10)
        # TODO START
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        log_h = np.where(target == 0, 0, np.log(h + epsilon))
        log_target = np.where(target == 0, 0, np.log(target + epsilon))
        loss = np.mean(target * (log_target - log_h))
        # print("DEBUG:", target * (log_target - log_h))
        return loss
        # TODO END

    def backward(self, input, target):
		# TODO START
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        grad = h - target
        return grad / target.shape[0]
		# TODO END

class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        log_h = np.where(target == 0, 0, np.log(h + epsilon))
        loss = -np.mean(target * log_h)
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        grad = h - target
        return grad / target.shape[0]
        # TODO END

# class HingeLoss(object):
#     def __init__(self, name, margin=1):
#         self.name = name
#         self.margin = margin

#     def softmax(self, x):
#         exps = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return exps / np.sum(exps, axis=1, keepdims=True)

#     def forward(self, input, target):
#         probabilities = self.softmax(input)
#         correct_class_scores = probabilities[np.arange(len(target)), target].reshape(-1, 1)
        
#         margins = np.maximum(0, self.margin - correct_class_scores + probabilities)
#         margins[np.arange(len(target)), target] = 0  # Do not consider correct class in the loss calculation
#         loss = np.sum(margins) / len(target)
        
#         return loss

#     def backward(self, input, target):
#         probabilities = self.softmax(input)
#         dscores = probabilities
#         dscores[np.arange(len(target)), target] -= 1
#         dscores = dscores * (probabilities > 0)  # Only include positive contributions
#         return dscores / len(target)


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        # \text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}
        idx = np.argmax(target, axis=1)
        pred_gt = input[np.arange(input.shape[0]), idx].reshape(-1, 1)
        print(input, target)
        print(pred_gt)
        E = np.maximum(0, self.margin - pred_gt + input)
        E[np.arange(input.shape[0]), idx] = 0
        loss = np.sum(E) / input.shape[1]
        return loss
        # TODO: Add a softmax before the HingeLoss fn and set delta as 0.5
        # TODO END

    def backward(self, input, target):
        # TODO START
        idx = np.argmax(target, axis=1)
        pred_gt = input[np.arange(input.shape[0]), idx].reshape(-1, 1)
        E = self.margin - pred_gt + input
        grad = np.zeros_like(input)
        mask = E > 0
        grad[mask] = 1
        grad[np.arange(input.shape[0]), idx] -= np.sum(mask, axis=1)
        return grad / input.shape[1]
        # TODO END
        
# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

if __name__ == "__main__":
    from torch.nn.modules.loss import KLDivLoss as TorchKLDivLoss
    import torch
    from torch import nn
    import numpy as np
    randomization = False
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

    # KL
    # my_loss = KLDivLoss("loss")
    # torch_loss = nn.KLDivLoss(reduction='batchmean')
    
    # SoftmaxCrossEntropy
    # my_loss = SoftmaxCrossEntropyLoss("loss")
    # torch_loss = nn.CrossEntropyLoss()
    
    # HingeLoss
    # my_loss = HingeLoss("loss")
    # torch_loss = nn.MultiMarginLoss(p=1, margin=5)

    if randomization:
        my_forward = my_loss.forward(log_probs.detach().numpy(), target_probs.numpy())
        torch_forward = torch_loss(log_probs, target_probs)
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