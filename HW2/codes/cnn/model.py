# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

device = "cuda:3"

class BatchNorm2d(nn.Module):
    # TODO START
    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    # Reference: https://blog.sailor.plus/deep-learning/optimization/
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Buffers
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        # input: [batch_size, num_features, height, width]
        if self.training:
            batch_mean = input.mean([0, 2, 3])
            batch_var = input.var([0, 2, 3], unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            x_normalized = (input - batch_mean[None, :, None, None]) / torch.sqrt(batch_var[None, :, None, None] + self.eps)
        else:
            x_normalized = (input - self.running_mean[None, :, None, None]) / torch.sqrt(self.running_var[None, :, None, None] + self.eps)
        return self.weight[None, :, None, None] * x_normalized + self.bias[None, :, None, None]
	# TODO END
    
class Dropout(nn.Module):
    # TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
		if self.training:
			# During training, scale the output by 1 / (1 - p) with **remaining** network
			mask = torch.rand(input.shape)
			mask = torch.where(mask > self.p, torch.tensor(1.0), torch.tensor(0.0))
			mask = mask.float().to(device)
			return mask * input * (1.0 / (1 - self.p))
		else:
			# During testing
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		# input - Conv - BN - ReLU - Dropout - MaxPool - Conv - BN - ReLU - Dropout - MaxPool - Linear - loss
		self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
		self.bn1 = BatchNorm2d(16)
		self.bn_torch = nn.BatchNorm2d(16)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		self.pool1 = nn.MaxPool2d(2, stride=2)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
		self.bn2 = BatchNorm2d(32)
		self.relu2 = nn.ReLU()
		self.dropout2 = Dropout(drop_rate)
		self.pool2 = nn.MaxPool2d(2, stride=2)
		self.fc = nn.Linear(32 * 8 * 8, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		logits = self.conv1(x)
		logits = self.bn1(logits)
		logits = self.relu1(logits)
		logits = self.dropout1(logits)
		logits = self.pool1(logits)
		logits = self.conv2(logits)
		logits = self.bn2(logits)
		logits = self.relu2(logits)
		logits = self.dropout2(logits)
		logits = self.pool2(logits)
		logits = logits.view(logits.size(0), -1)
		logits = self.fc(logits)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc