import time
import argparse
from network import Network
from utils import LOG_INFO
from layers import Selu, HardSwish, Linear, Tanh
from loss import KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST data.")
    
    parser.add_argument('--project', type=str, default='ANN-HW1', help='WandB project name')
    parser.add_argument('--name', type=str, default='mlp_test', help='WandB run name')
    
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--max_epoch', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--disp_freq', type=int, default=100, help='Frequency of displaying training progress')
    parser.add_argument('--test_epoch', type=int, default=1, help='Frequency of testing the model')
    parser.add_argument('--activation', type=str, default='selu', help='Activation function to use')
    parser.add_argument('--loss', type=str, default='softmax', help='Loss function to use')
    return parser.parse_args()

def main():
    args = parse_arguments()
    start_time = time.time()
    wandb.init(
        project=args.project,
        name=f'{args.name}-{args.activation}-{args.loss}'
    )

    activation_dict = {
        'selu': Selu,
        'hardswish': HardSwish,
        'tanh': Tanh
    }
    
    loss_dict = {
        'kl': KLDivLoss,
        'softmax': SoftmaxCrossEntropyLoss,
        'hinge': HingeLoss,
        'focal': FocalLoss   
    }
    
    activation_fn = activation_dict[args.activation]
    loss_fn = loss_dict[args.loss]

    train_data, test_data, train_label, test_label = load_mnist_2d('data')

    model = Network()

    layers = [784, 128, 10]
    # layers = [784, 245, 10]
    # layers = [784, 245, 137, 77, 10]
    # layers = [784, 438, 245, 183, 137, 77, 43, 24, 10]
    # layers = [784, 586, 438, 328, 245, 227, 183, 137, 102, 77, 57, 43, 32, 24, 18, 13, 10]
    acs = []
    # acs = ['selu', 'selu', 'selu', 'selu']
    # acs = ['hardswish', 'hardswish', 'hardswish', 'hardswish']
    # acs = ['hardswish', 'hardswish', 'selu', 'selu']
    # acs = ['selu', 'selu', 'hardswish', 'hardswish']

    for i in range(len(layers) - 1):
        model.add(Linear('fc%d' % i, layers[i], layers[i + 1], 0.005))
        if i != len(layers) - 2:
            if not acs:
                print('No activation function specified, using default Tanh')
                model.add(activation_fn('ac%d' % i))
            else:
                model.add(activation_dict[acs[i]]('ac%d' % i))

    # model.add(Linear('fc1', 784, 128, 0.005))
    # model.add(activation_fn('ac'))
    # model.add(Linear('fc2', 128, 10, 0.005))
    # AlexNet-inspired architecture for MNIST
    # model.add(Linear('fc1', 784, 512, 0.005))  # Smaller std for initialization
    # model.add(Tanh('relu1'))                   # Switch to Selu if needed
    # model.add(Linear('fc2', 512, 256, 0.005))
    # model.add(Selu('relu2'))
    # model.add(Linear('fc3', 256, 128, 0.005))
    # model.add(Tanh('relu3'))
    # model.add(Linear('fc4', 128, 10, 0.005))   # Output layer

    # Loss function
    loss = loss_fn(name='loss')

    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'batch_size': args.batch_size,
        'max_epoch': args.max_epoch,
        'disp_freq': args.disp_freq,
        'test_epoch': args.test_epoch
    }

    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, test_data, test_label, config['batch_size'])
        
        # last epoch
        if epoch == config['max_epoch'] - 1:
            # save the final result to wandb; final loss / final acc / total time cost
            wandb.log({'total_time': time.time() - start_time})

if __name__ == "__main__":
    main()