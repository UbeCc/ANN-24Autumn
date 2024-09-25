# ANN Autumn24 HW1 (MNIST)

王浩然 计23 2022010229

---

## 实验介绍

本次实验需要手工实现神经网络的前向和反向传播，涉及到链式法则。

Loss 方面：有 `KLDivLoss`、`SoftmaxCrossEntropy`、`HingeLoss`、`FocalLoss`。

函数层方面：有`Selu`、`HardSwish`、`Tanh`、`Linear`。

实验需求

- 画出训练集和测试集的所有 Average & Final Loss / Accuracy
- 单层 MLP，激活函数用 Selu / HardSwish / Tanh；loss函数用 KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss，检测指标是训练时间、收敛性和 Acc
- 双层 MLP，激活函数用 Selu / HardSwish / Tanh；loss函数用KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss，检测指标是训练时间、收敛性和 Acc
- 调参数，做 Grid Search
- 实现 FocalLoss， 和 SoftmaxCrossEntrophyLoss 比较

layers.py / loss.py
![alt text](result1.png)

03:04:45.028 Training @ 49 epoch...
03:04:46.205   Training iter 100, batch loss 0.0113, batch acc 0.9956
03:04:47.406   Training iter 200, batch loss 0.0147, batch acc 0.9950
03:04:48.507   Training iter 300, batch loss 0.0136, batch acc 0.9948
03:04:49.738   Training iter 400, batch loss 0.0264, batch acc 0.9925
03:04:51.138   Training iter 500, batch loss 0.0185, batch acc 0.9923
03:04:52.331   Training iter 600, batch loss 0.0183, batch acc 0.9933
03:04:53.596   Training iter 700, batch loss 0.0240, batch acc 0.9930
03:04:54.772   Training iter 800, batch loss 0.0192, batch acc 0.9936
03:04:55.960   Training iter 900, batch loss 0.0194, batch acc 0.9939
03:04:56.397 Testing @ 49 epoch...
03:04:56.752     Testing, total mean loss 0.13155, total acc 0.97273

model = Network()

# AlexNet-inspired architecture for MNIST
model.add(Linear('fc1', 784, 512, 0.005))  # Smaller std for initialization
model.add(Selu('relu1'))                   # Switch to Selu if needed

model.add(Linear('fc2', 512, 256, 0.005))
model.add(Selu('relu2'))

model.add(Linear('fc3', 256, 128, 0.005))
model.add(Selu('relu3'))

model.add(Linear('fc4', 128, 10, 0.005))   # Output layer

# Loss function
loss = SoftmaxCrossEntropyLoss(name='loss')