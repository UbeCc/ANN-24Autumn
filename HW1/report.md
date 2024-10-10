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

## 损失函数对比实验

### `KLDivLoss`

#### Forward
$$\begin{align*} h_k &= P(t_k = 1 \mid \mathbf{x}) = \frac{\exp(x_k)}{\sum_{j=1}^{K} \exp(x_j)} \\ E &= \frac{1}{N} \sum_{n=1}^{N} E^{(n)} = \frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} t_k^{(n)} \left( \ln t_k^{(n)} - \ln h_k^{(n)} \right) \end{align*}$$

#### Backward
$\frac{\partial E}{\partial x_k^{(n)}} = h_k^{(n)} - t_k^{(n)}$

### `SoftmaxCrossEntropyLoss`

#### Forward
$\begin{align}
P(t_k = 1 \mid \mathbf{x}) &= \frac{\exp(x_k)}{\sum_{j=1}^{K} \exp(x_j)} \tag{1} \\
E &= \frac{1}{N} \sum_{n=1}^{N} E^{(n)} \\
E^{(n)} &= -\sum_{k=1}^{K} t_k^{(n)} \ln h_k^{(n)} \tag{2} \\
h_k^{(n)} &= P(t_k^{(n)} = 1 \mid \mathbf{x}^{(n)}) = \frac{\exp(x_k^{(n)})}{\sum_{j=1}^{K} \exp(x_j^{(n)})}
\end{align}$

#### Backward
$\frac{\partial E}{\partial x_k^{(n)}} = h_k^{(n)} - t_k^{(n)}$

### `HingeLoss`

#### Forward
$$\begin{align*}
E &= \frac{1}{N} \sum_{n=1}^{N} E^{(n)} \\
E^{(n)} &= \sum_{k=1}^{K} h_k^{(n)} \\
h_k^{(n)} &= 
\begin{cases} 
0, & \text{if } k = t_n \\
\max(0, \Delta - x_{t_n}^{(n)} + x_k^{(n)}), & \text{otherwise}
\end{cases}
\end{align*}$$

#### Backward
$\frac{\partial E}{\partial x_k^{(n)}} = 
\begin{cases} 
0, & \text{if } k = t_n \text{ and } x_{t_n}^{(n)} - x_k^{(n)} \geq \Delta \\
-1, & \text{if } k = t_n \text{ and } x_{t_n}^{(n)} - x_k^{(n)} < \Delta \\
1, & \text{if } k \neq t_n \text{ and } x_{t_n}^{(n)} - x_k^{(n)} < \Delta \\
0, & \text{otherwise}
\end{cases}$

### `FocalLoss`

#### Forward
$\begin{align*}
h_k &= P(t_k = 1 \mid \mathbf{x}) = \frac{\exp(x_k)}{\sum_{j=1}^{K} \exp(x_j)} \\
E &= \frac{1}{N} \sum_{n=1}^{N} E^{(n)} \\
E^{(n)} &= -\sum_{k=1}^{K} \left( \alpha_k t_k^{(n)} + (1 - \alpha_k)(1 - t_k^{(n)}) \right) (1 - h_k^{(n)}) \gamma_{t_k}^{(n)} \ln h_k^{(n)}
\end{align*}$

#### Backward
$\frac{\partial E}{\partial x_k^{(n)}} = -\left( \alpha_k t_k^{(n)} + (1 - \alpha_k)(1 - t_k^{(n)}) \right) \gamma_{t_k}^{(n)} \left( \frac{t_k^{(n)}}{h_k^{(n)}} - 1 \right) (1 - h_k^{(n)})$

## 激活函数对比实验

### `Selu`

#### Forward
$f(u) = \lambda \begin{cases} 
u, & u > 0 \\ 
\alpha(e^u - 1), & u \leq 0 
\end{cases}$

#### Backward
$f'(u) = \lambda \begin{cases} 
1, & u > 0 \\ 
\alpha e^u, & u \leq 0 
\end{cases}$

### `HardSwish`

#### Forward 
$f(u) = \begin{cases} 
0, & u \leq -3 \\ 
u, & u \geq 3 \\ 
\frac{u(u+3)}{6}, & \text{otherwise} 
\end{cases}$

#### Backward
$f'(u) = \begin{cases} 
0, & u \leq -3 \\ 
1, & u \geq 3 \\ 
\frac{2u + 3}{6}, & -3 < u < 3 
\end{cases}$

### `Tanh`

#### Forward
$\dfrac{exp(u)-exp(-u)}{exp(u)+exp(-u)}$

#### Backward
$f'(u) = 1 - \left( \frac{\exp(u) - \exp(-u)}{\exp(u) + \exp(-u)} \right)^2$

## 网络架构对比实验

> 这里我进行了两部分实验，一部分以网络层数为变量，一部分以网络中的激活函数为种类变量

### 使用同一激活函数，改变网络层数

我使用了计算最快的 `HingeLoss` 和 `HardSwish` 函数，令网络层分别为 1, 2, 4, 8, 16，结果如下。

### 使用不同激活函数，固定网络层数

我固定网络层数为 4，尝试了不同激活函数组合（HS-HardSwish, SL-Selu, TH-Tanh）

- HS-HS-HS-HS
- SL-SL-SL-SL
- HS-HS-SL-SL
- SL-SL-HS-HS

这样选的原因是，我认为 `HardSwish` 属于简单激活函数，而 `Selu` 和 `Tanh` 属于复杂激活函数。

## 结论

- 不同函数对于训练时间影响很大，比如带指数运算的 `Selu` `Tanh`, `KLDivLoss`, `FocalLoss` 和 `SoftmaxCrossEntropyLoss` ，效果最好但耗时最长，有机会可以做下更 fine-grained 的实验，比如用三阶泰勒代替指数操作
- 最好的激活函数是 `Selu` / `Tanh`，
- 在所有实验中，更换损失函数对结果的影响比更换激活函数明显大。前者决定了你如何拟合真实数据与模型输出，后者只是让不同的 MLP 之间**有区别**。因为 $A\cdot B\cdot C\cdot X=WX(W=A\cdot B\cdot C)$
- 简单堆叠函数层数会提高模型性能（是因为 MNIST 太简单了，如果换成 ImageNet 这种效果差异会很大）。但堆叠更多后，模型表示能力也不会一直增加，会收敛，这也引发了后面 `GoogLenet` 这种并行结构的网络研究。