import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        
        if self.train:
            for i in range(1, 6):
                file_path = os.path.join(self.root_dir, f'data_batch_{i}')
                batch_data = unpickle(file_path)
                self.data.append(batch_data[b'data'])
                self.targets.extend(batch_data[b'labels'])
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1)) 
        else:
            file_path = os.path.join(self.root_dir, 'test_batch')
            batch_data = unpickle(file_path)
            self.data = batch_data[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            self.targets = batch_data[b'labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # clip
    transforms.RandomHorizontalFlip(),     # flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # color shift
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 norm parmeter
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = CIFAR10Dataset(root_dir='./cifar-10_data', train=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=10000, shuffle=True, num_workers=2)

augmented_data = []
augmented_labels = []

for batch_idx, (images, labels) in enumerate(trainloader):

    batch_data = (images.numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1).reshape(-1, 3072)
    augmented_data.append(batch_data)
    augmented_labels.extend(labels.numpy())


    augmented_batch = {
        b'data': batch_data,
        b'labels': labels.numpy().tolist()
    }
    if not os.path.exists('./augmented_cifar-10_data'):
        os.mkdir('./augmented_cifar-10_data')
    
    if not os.path.exists('./augmented_cifar-10_data'):
        os.mkdir('./augmented_cifar-10_data')

    with open(f'./augmented_cifar-10_data/data_batch_{batch_idx+1}', 'wb') as f:
        pickle.dump(augmented_batch, f)

    print(f"Saved augmented batch {batch_idx+1}")

def visualize_augmentation():
    import matplotlib.pyplot as plt
    
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    def denormalize(tensor):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        return tensor * std + mean
    
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = denormalize(images[i]).permute(1, 2, 0).clamp(0, 1).numpy()
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("augmentation.png")

visualize_augmentation()