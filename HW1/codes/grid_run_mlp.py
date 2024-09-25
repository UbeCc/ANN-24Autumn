import wandb
from network import Network
from utils import LOG_INFO
from layers import Selu, HardSwish, Linear, Tanh
from loss import KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import numpy as np

def report(iter_counter, loss_list, acc_list):
    # use wandb
    wandb.log({"batch_loss": np.mean(loss_list), "batch_acc": np.mean(acc_list), "iter": iter_counter})

def main():
    # Initialize wandb
    wandb.init(
        project="ANN-HW1",
        name="mlp_test"
    )

    train_data, test_data, train_label, test_label = load_mnist_2d('data')

    model = Network()

    # AlexNet-inspired architecture for MNIST
    model.add(Linear('fc1', 784, 512, 0.005))
    model.add(Selu('relu1'))
    model.add(Linear('fc2', 512, 256, 0.005))
    model.add(Selu('relu2'))
    model.add(Linear('fc3', 256, 128, 0.005))
    model.add(Selu('relu3'))
    model.add(Linear('fc4', 128, 10, 0.005))

    # Loss function
    loss = SoftmaxCrossEntropyLoss(name='loss')

    # Training configuration
    config = {
        'learning_rate': wandb.config.learning_rate,
        'weight_decay': wandb.config.weight_decay,
        'momentum': wandb.config.momentum,
        'batch_size': wandb.config.batch_size,
        'max_epoch': wandb.config.num_epochs,
        'disp_freq': 100,
        'test_epoch': 1
    }

    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            accuracy = test_net(model, loss, test_data, test_label, config['batch_size'])
            wandb.log({"test_accuracy": accuracy, "epoch": epoch})

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()