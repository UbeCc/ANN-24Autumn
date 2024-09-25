from network import Network
from utils import LOG_INFO
from layers import Selu, HardSwish, Linear, Tanh
from loss import KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


train_data, test_data, train_label, test_label = load_mnist_2d('data')

model = Network()

# AlexNet-inspired architecture for MNIST
model.add(Linear('fc1', 784, 512, 0.005))  # Smaller std for initialization
model.add(Tanh('relu1'))                   # Switch to Selu if needed

model.add(Linear('fc2', 512, 256, 0.005))
model.add(Selu('relu2'))

model.add(Linear('fc3', 256, 128, 0.005))
model.add(Tanh('relu3'))

model.add(Linear('fc4', 128, 10, 0.005))   # Output layer

# Loss function
loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0001,  # Lower weight decay
    'momentum': 0.5,         # Higher momentum
    'batch_size': 32,        # Larger batch size
    'max_epoch': 50,         # Reduced number of epochs
    'disp_freq': 100,        # Less frequent display to speed up training
    'test_epoch': 1          # Frequent testing to monitor progress
}

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])






# from network import Network
# from utils import LOG_INFO
# from layers import Selu, HardSwish, Linear, Tanh
# from loss import KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss
# from solve_net import train_net, test_net
# from load_data import load_mnist_2d


# train_data, test_data, train_label, test_label = load_mnist_2d('data')

# # Your model defintion here
# # You should explore different model architecture
# model = Network()
# model.add(Linear('fc1', 784, 64, 0.01))
# model.add(HardSwish('ac'))
# model.add(Linear('fc1', 64, 10, 0.01))
# # model.add(Selu('ac'))
# # model.add(Linear('fc1', 64, 10, 0.01))
# # model.add(Selu('ac'))

# # loss = KLDivLoss(name='loss')
# loss = SoftmaxCrossEntropyLoss(name='loss')
# # loss = HingeLoss(name='loss')

# # Training configuration
# # You should adjust these hyperparameters
# # NOTE: one iteration means model forward-backwards one batch of samples.
# #       one epoch means model has gone through all the training samples.
# #       'disp_freq' denotes number of iterations in one epoch to display information.

# config = {
#     'learning_rate': 0.001,
#     'weight_decay': 0.01,
#     'momentum': 0.1,
#     'batch_size': 50,
#     'max_epoch': 100,
#     'disp_freq': 50,
#     'test_epoch': 5
# }


# for epoch in range(config['max_epoch']):
#     LOG_INFO('Training @ %d epoch...' % (epoch))
#     train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

#     if epoch % config['test_epoch'] == 0:
#         LOG_INFO('Testing @ %d epoch...' % (epoch))
#         test_net(model, loss, test_data, test_label, config['batch_size'])
