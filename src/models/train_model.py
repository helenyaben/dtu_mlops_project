import os
import pathlib
from typing import Dict, List, Tuple
import wandb
import click
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.pyplot import show
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, train:bool=True):
        
        # 3. Create class attributes
        self.classes = list(range(0, 12))
        self.class_to_idx = class_index = {'0R': 0, '1R': 1, '2R': 2, '3R': 3, '4R': 4, '5R': 5, '0L': 6, '1L':7, '2L':8, '3L':9, '4L':10, '5L':11}

        # Import preprocessed data
        train_test_data = torch.load(os.path.join(targ_dir, 'train_test_processed.pt'))
        if train:
            self.images = train_test_data['train_data']
            self.labels = train_test_data['train_labels']
        else:
            self.images = train_test_data['test_data']
            self.labels = train_test_data['test_labels']

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:

        return self.images.shape[0]
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.images[index]
        class_idx  = self.labels[index]
        
        # Transform if necessary
        return img, class_idx # return data, label (X, y)

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    model = MyAwesomeModel()
    epochs = 5
    batch_size = 64

    # logging with wandb
    args = {"epochs": epochs, "batch_size": batch_size,
            "learning_rate": lr}
    wandb.init(project = "count fingers", config = args)
    wandb.watch(model, log_freq=100)
    
    trainset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=True)
    # Train DataLoader
    trainloader = DataLoader(dataset=trainset, # use custom created train Dataset
                                        batch_size=batch_size, # how many samples per batch?
                                        shuffle=True) # shuffle the data?
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    print("Training day and night")
    print(lr)
    
    for e in range(epochs):

        epoch_losses = 0

        for images, labels in trainloader:
            # Flatten images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            # Reset gradients
            optimizer.zero_grad()
            # Obtain log probabilities
            log_ps = model(images)
            # Calculate loss
            # print(log_ps.shape)
            loss = criterion(log_ps, labels)
            # Apply backward
            loss.backward()
            # Move optimizer 
            optimizer.step()
            # Add batch loss to epoch losses list
            epoch_losses += loss.item()
            
            # Log loss to wandb
            wandb.log({"loss": loss})
        
        train_losses.append(epoch_losses/len(trainloader))
        print(f"Train loss in epoch {e}: {epoch_losses/len(trainloader)}")

    torch.save(model.state_dict(), os.path.join('src', 'models','my_trained_model.pt')) 
    
    plt.plot(train_losses)
    show()

    
if __name__ == "__main__":
    train()

    