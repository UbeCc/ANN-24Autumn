import wandb
from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np

def report(iter_counter, loss_list, acc_list):
    # use wandb
    wandb.log({"batch_loss": np.mean(loss_list), "batch_acc": np.mean(acc_list), "iter": iter_counter})

def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq):
    iter_counter = 0
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size): # (100 x 784), (100 x 1)
        target = onehot_encoding(label, 10) # (100 x 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

        if iter_counter % disp_freq == 0:
            if wandb.run:
                report(iter_counter, loss_list, acc_list)
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (iter_counter, np.mean(loss_list), np.mean(acc_list))
            loss_list = []
            acc_list = []
            LOG_INFO(msg)


def test_net(model, loss, inputs, labels, batch_size):
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)
