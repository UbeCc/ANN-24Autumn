import matplotlib.pyplot as plt
import json
import os
import re

def extract():
    training_pattern = re.compile(r'Training iter \d+, batch loss ([\d.]+), batch acc [\d.]+')
    testing_pattern = re.compile(r'Testing, total mean loss [\d.]+, total acc ([\d.]+)')

    for file in os.listdir('logs'):
        if not file.endswith("log"):
            continue
        with open(os.path.join('logs', file), 'r') as f:
            data = f.read()
        current_epoch = 0
        training_loss = []
        testing_accuracy = []
        epoch_loss = {}

        for line in data.split('\n'):
            if 'Training @' in line:
                line = line.replace("epoch...", "")
                current_epoch = int(line.split()[-1])
            elif 'Training iter' in line:
                match = training_pattern.search(line)
                if match:
                    loss = float(match.group(1))
                    epoch_loss[current_epoch] = loss
            elif 'Testing' in line and 'total acc' in line:
                match = testing_pattern.search(line)
                if match:
                    accuracy = float(match.group(1))
                    if current_epoch not in epoch_loss:
                        epoch_loss[current_epoch] = None  
                    training_loss.append(epoch_loss[current_epoch])
                    testing_accuracy.append(accuracy)
        print("Training Losses:", training_loss)
        print("Testing Accuracies:", testing_accuracy)
        with open(os.path.join('logs', file.split('.')[0]+".json"), 'w') as f:
            json.dump({
                "training_loss": training_loss,
                "testing_accuracy": testing_accuracy,
            }, f)

def draw(prefixes: list, name: str):
    # we need to combine all the data in files into one plot
    # plot loss and accuracy independently and save the figs
    
    # now we draw accuracy
    testing_accuracies = []
    for prefix in prefixes:
        for file in os.listdir('logs'):
            if file.startswith(prefix) and file.endswith('.json'):
                with open(os.path.join('logs', file), 'r') as f:
                    data = json.load(f)
                testing_accuracies.append(data['testing_accuracy'])
    # plot training loss
    plt.figure()
    for i, loss in enumerate(testing_accuracies):
        plt.plot(range(len(loss)), loss, label=f"{prefixes[i]}")
    plt.xlabel("Epoch")
    plt.ylabel("Testing Accuracy")
    plt.legend()
    plt.savefig(f"figs/{name}_testing_accuracy.png", dpi=300)

    # now we draw loss
    training_losses = []
    for prefix in prefixes:
        for file in os.listdir('logs'):
            if file.startswith(prefix) and file.endswith('.json'):
                with open(os.path.join('logs', file), 'r') as f:
                    data = json.load(f)
                training_losses.append(data['training_loss'])
    # plot training loss
    plt.figure()
    for i, loss in enumerate(training_losses):
        plt.plot(range(len(loss)), loss, label=f"{prefixes[i]}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.savefig(f"figs/{name}_training_loss.png", dpi=300)

extract()
draw(['layer2', 'layer4', 'layer8', 'layer16'], 'layer')
draw(['ssss', 'sshh', 'hhss', 'hhhh'], 'architecture')
draw(['selu-focal', 'selu-hinge', 'selu-softmax', 'selu-kl'], 'loss')
draw(['selu-hinge', 'hardswish-hinge', 'tanh-hinge'], 'activation')