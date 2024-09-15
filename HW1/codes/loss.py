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


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

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
    
    # inputs = [-1.5072, -0.5475,  1.6552,  1.3270,  0.1383, -0.1262, -0.4392, -0.6543, 0.0127, -0.9532]
    # label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # inputs = np.array(inputs)
    # label = np.array(label)
    # inputs = inputs.reshape(1, -1)
    # label = label.reshape(1, -1)

    logits = torch.randn(10, 5, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=1)

    target_probs = torch.rand(10, 5)
    target_probs /= target_probs.sum(dim=1, keepdim=True)
    target_probs = target_probs.detach()  # No grad needed for target

    # KL
    # my_loss = KLDivLoss("loss")
    # torch_loss = nn.KLDivLoss(reduction='batchmean')
    
    # SoftmaxCrossEntropy
    my_loss = SoftmaxCrossEntropyLoss("loss")
    torch_loss = nn.CrossEntropyLoss()

    my_forward = my_loss.forward(log_probs.detach().numpy(), target_probs.numpy())
    torch_forward = torch_loss(log_probs, target_probs)

    print("My Forward:", my_forward, "Torch Forward:", torch_forward.item())

    torch_forward.backward()
    my_backward = my_loss.backward(log_probs.detach().numpy(), target_probs.numpy())

    print("My Backward:\n", my_backward)
    print("Torch Gradient:\n", logits.grad.numpy())

    print("Difference in gradients:\n", np.abs(my_backward - logits.grad.numpy()))
    
    assert np.allclose(my_backward, logits.grad.numpy()), "Gradients are not equal!"