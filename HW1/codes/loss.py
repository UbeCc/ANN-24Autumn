from __future__ import division
import numpy as np

class KLDivLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target): # (100 x 10), (100 x 10)
        # TODO START
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        log_h = np.log(h)
        log_target = np.where(target == 0, 0, np.log(target))
        # loss 
        loss = np.mean(target * (log_target - log_h))
        # h = exp(x_k) / sum(exp(x))
        # h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        # print(target[0], h[0])
        # print((np.log(target) - np.log(h))[0])
        # loss = mean(target * (log(target) - log(h)))
        # loss = np.mean(target * (np.log(target) - np.log(h)))
        return loss
        # TODO END

    def backward(self, input, target):
		# TODO START
        '''Your codes here'''
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        grad = h - target
        return grad
		# TODO END

class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.softmax_output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        
        num_samples = input.shape[0]
        loss = -np.sum(target * np.log(self.softmax_output + 1e-8)) / num_samples
        return loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        num_samples = input.shape[0]
        grad = (self.softmax_output - target) / num_samples
        return grad
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