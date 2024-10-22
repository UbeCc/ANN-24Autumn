# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum

		# Parameters
		self.weight = nn.Parameter(torch.ones(num_features))
		self.bias = nn.Parameter(torch.zeros(num_features))

		# Buffers (not parameters)
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))

	def forward(self, input):
		# input: [batch_size, num_features]
		if self.training:
			batch_mean = input.mean(dim=0)
			batch_var = input.var(dim=0, unbiased=False)

			# with torch.no_grad():
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

			# Normalize
			x_normalized = (input - batch_mean) / torch.sqrt(batch_var + self.eps)
		else:
		# Use running mean and variance for inference
			x_normalized = (input - self.running_mean) / torch.sqrt(self.running_var + self.eps)

		# Scale and shift
		return self.weight * x_normalized + self.bias
	# TODO END

# In this code, we implement dropout in an alternative way. During the training process, we scale the 
# remaining network nodes' output by 1/(1-p),. At testing time, we do nothing in the dropout layer. It's 
# easy to find that this method has similar results to original dropout
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
			mask = mask.float().to("cuda:1")
			return mask * input * (1.0 / (1 - self.p))
		else:
			# During testing
			return input
	# TODO END

# model architecture: input -- Linear - BN - ReLU - Dropout - Linear - loss
class Model(nn.Module):
	def __init__(self, drop_rate=0.5, hidden_size=128):
		super(Model, self).__init__()
		# we train on cifar-10 dataset, the input shape is bsize * 3072
		# TODO START
		# Define your layers here
		self.linear1 = nn.Linear(3072, hidden_size)
		self.bn1 = BatchNorm1d(hidden_size)
		self.relu = nn.ReLU()
		self.dropout = Dropout(drop_rate)
		self.linear2 = nn.Linear(hidden_size, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		x = x.view(x.size(0), -1)
		logits = self.linear1(x)
		logits = self.bn1(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)
		logits = self.linear2(logits)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc