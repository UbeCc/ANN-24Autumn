import numpy as np

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        # TODO START
        # TODO: modified
        self.lambda_ = 1.0507
        self.alpha = 1.67326
        self._saved_for_backward(input)
        return self.lambda_ * np.where(input > 0, input, self.alpha * (np.exp(input) - 1))
        # TODO END

    def backward(self, grad_output):
        # TODO START
        # TODO: modified
        self.lambda_ = 1.0507
        self.alpha = 1.67326
        input = self._saved_tensor
        return self.lambda_ * np.where(input > 0, grad_output, grad_output * self.alpha * np.exp(input))
        # TODO END

class HardSwish(Layer):
    def __init__(self, name):
        super(HardSwish, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return input * np.clip(input + 3, 0, 6) / 6
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        grad_input = np.zeros_like(input)
        grad_input[input < -3] = 0
        mask = (input > -3) & (input < 3)
        grad_input[mask] = (2 * input[mask] + 3) / 6
        grad_input[input > 3] = 1
        return grad_output * grad_input
        # TODO END

class Tanh(Layer):
    def __init__(self, name):
        super(Tanh, self).__init__(name)

    def forward(self, input):
        # TODO START
        fwd_result = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        self._saved_for_backward(fwd_result)
        return fwd_result
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        fwd_result = self._saved_tensor
        return grad_output * (1 - np.square(fwd_result))
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        output = np.dot(input, self.W) + self.b
        self._saved_for_backward(input)
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        self.grad_W = np.dot(input.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b