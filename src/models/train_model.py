import os
import pathlib
from typing import Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.pyplot import show
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import wandb


# Define sweep config
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [2, 3, 5]},
        'lr': {'max': 0.001, 'min': 0.0001}
     }
}

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='the_most_final_sweep')


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, train:bool=True):
        
        # 3. Create class attributes
        self.classes = list(range(0, 12))
        self.class_to_idx = class_index = {'0R': 0, '1R': 1, '2R': 2, '3R': 3, '4R': 4, 
        '5R': 5, '0L': 6, '1L':7, '2L':8, '3L':9, '4L':10, '5L':11}

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

# @click.command()
# @click.option("--lr", default=1e-3, help='learning rate to use for training')

# def train(epochs,batch_size,lr):
def train():
    run = wandb.init()
    lr  =  wandb.config.lr
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs


    # # Set up your default parameters
    # config = {"epochs": 4, "batch_size": 32, "lr" : 1e-3}
    # wandb.init(project = "Fingers", config =config)
    # lr  =  wandb.config.lr
    # batch_size = wandb.config.batch_size
    # epochs = wandb.config.epochs

    model = MyAwesomeModel()
    wandb.watch(model, log_freq=100)

     # Set up your default hyperparameters
    # with open('src/models/sweep.yaml') as file:
    #    config = yaml.load(file, Loader=yaml.FullLoader)
    # sweep_id = wandb.sweep(sweep=config, project='my-first-sweep')
    # wandb.agent(sweep_id, function=train, count=4)

    
    trainset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=True)
    # Train DataLoader
    trainloader = DataLoader(dataset=trainset, # use custom created train Dataset
                                        batch_size=batch_size, # how many samples per batch?
                                        shuffle=True) # shuffle the data?

    valset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=False)
    # Test DataLoader
    valloader = DataLoader(dataset=valset, # use custom created train Dataset
                                        batch_size=64, # how many samples per batch?
                                        shuffle=True) # shuffle the data?
    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    print("Training day and night")
    print(lr)
    
    for e in range(epochs):

        epoch_losses = 0
        train_accuracies = []

        for images, labels in trainloader:
            # Flatten images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            # Reset gradients
            optimizer.zero_grad()
            # Obtain log probabilities
            log_ps = model(images)
            # Calculate loss
            # print(log_ps.shape)
            batch_loss = criterion(log_ps, labels)
            # Apply backward
            batch_loss.backward()
            # Move optimizer 
            optimizer.step()
            # Add batch loss to epoch losses list
            epoch_losses += batch_loss.item()
            # Obtain probabilities
            ps = torch.exp(log_ps)
            # Obtain top probabilities and corresponding classes
            top_p, top_class = ps.topk(1, dim=1)
            # Compare with true labels
            equals = top_class == labels.view(*top_class.shape)   
            # Obtain accuracy
            batch_acccuracy = torch.mean(equals.type(torch.FloatTensor)) 
            train_accuracies.append(batch_acccuracy)
            wandb.log({"batch_loss": batch_loss})
        
        # Calculate validation loss and accuracy
        val_accuracies = []
        model.eval() # set model to evaluation mode
        for images, labels in valloader:
            # Caculate log probabilities
            log_ps = model(images)
            # Calculate probabilities
            ps = torch.exp(log_ps)
            # Obtain top probabilities and corresponding classes
            top_p, top_class = ps.topk(1, dim=1)
            # Compare with true labels
            equals = top_class == labels.view(*top_class.shape)   
            # Obtain accuracy
            val_batch_acccuracy = torch.mean(equals.type(torch.FloatTensor)) 
            val_accuracies.append(val_batch_acccuracy)
            

        train_loss = epoch_losses/len(trainloader)
        train_losses.append(train_loss)
        val_accuracy = sum(val_accuracies)/len(valloader)

        print(f"Epoch {e} - Train loss: {epoch_losses/len(trainloader)}, Accuracy on val: {val_accuracy.item()*100}%")
        
        # log important metrics
        train_accuracy = sum(train_accuracies)/len(trainloader)
        # Log loss and train_acc to wandb
        wandb.log({
        'epoch': e, 
        'train_acc': train_accuracy,
        'train_loss': train_loss,
        'val_acc': val_accuracy,
        })
    # log images and its prediction from the last batch
    i = 0
    for image, label, prediction in zip(images, labels, top_class):
        class_index = {'0R': 0, '1R': 1, '2R': 2, '3R': 3, '4R': 4, '5R': 5, '0L': 6, '1L':7, '2L':8, '3L':9, '4L':10, '5L':11}
        class_index_inv = {v: k for k, v in class_index.items()}
        images = wandb.Image(image, 
                            caption=f'Prediction: {class_index_inv[prediction.item()]} \n True: {class_index_inv[label.item()]}')
        wandb.log({"examples": images})
        # image = image.view(128, 128)
        # class_index = {'0R': 0, '1R': 1, '2R': 2, '3R': 3, '4R': 4, '5R': 5, '0L': 6, '1L':7, '2L':8, '3L':9, '4L':10, '5L':11}
        # class_index_inv = {v: k for k, v in class_index.items()}
        # plt.imshow(image, cmap='gray')
        # plt.title(f'Prediction: {class_index_inv[prediction.item()]} \n True: {class_index_inv[label.item()]}')
        # wandb.log({"chart": plt})
        i += 1
        if i == 10:
            break
        
    torch.save(model.state_dict(), os.path.join('models','my_trained_model.pt')) 

if __name__ == "__main__":
    # Start sweep job.
    wandb.agent(sweep_id, function=train, count=4)
