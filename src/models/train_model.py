import os
import pathlib
from typing import Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import torch
import yaml
import torch.nn.functional as F
from matplotlib.pyplot import show
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pickle

import wandb


# Define sweep config
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32]},
        'epochs': {'values': [2,3]},
        'lr': {'max': 0.001, 'min': 0.0001}
     }
}

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


# def train(epochs,batch_size,lr):
def train(option = 'train'):
    if option == 'train':
        wandb.init()
    lr  =  wandb.config.lr
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs

    model = MyAwesomeModel()
    wandb.watch(model, log_freq=100)


    
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
    print("learning rate:", lr)
    
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
            wandb.log({"batch_accuracy": batch_acccuracy})
        
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

        print(f"Epoch {e+1 } - Train loss: {epoch_losses/len(trainloader)}, Val accuracy: {val_accuracy.item()*100}%")
        
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
        wandb.log({"run examples": images})
        i += 1
        if i == 4:
            break
    save_best_model(train_loss, model, lr, epochs, batch_size)
    # torch.save(model.state_dict(), os.path.join('models','my_trained_model.pt')) 
    # # save model as pickle
    # with open(os.path.join('models','my_trained_model.pkl'), 'wb') as f:
    #     pickle.dump(model, f)

def save_best_model(train_loss, model, lr, epochs, batch_size):
    #load best config and compare with current train loss
    with open('src/models/best_config.yaml') as file:
        config_settings = yaml.load(file, Loader=yaml.FullLoader)
    if config_settings['loss']['validation_loss'] > train_loss:
        # if better val loss update best config and save model
        config_settings['loss']['validation_loss'] = train_loss
        config_settings['parameters']['lr'] = lr
        config_settings['parameters']['epochs'] = epochs
        config_settings['parameters']['batch_size'] = batch_size
        with open('src/models/best_config.yaml', 'w') as file:
            yaml.dump(config_settings, file)
        torch.save(model.state_dict(), os.path.join('models','my_trained_model.pt')) 
        # save model to gcloud
        # BUCKET_NAME = 'fingers_model'
        # MODEL_FILE = 'my_trained_model.pt'
        # client = storage.Client()
        # bucket = client.get_bucket(BUCKET_NAME)
        # blob = bucket.get_blob(MODEL_FILE)
        # blob.save_to_filename(MODEL_FILE)


@click.command()
@click.option("--run", default="train", help="select between train or sweep")
def decide_run(run):
    if run == "sweep":
        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='test-sweep')
        # Start sweep job.
        wandb.agent(sweep_id, function=train, count=4)
    else:
        #load best run parameters
        with open('src/models/best_config.yaml') as file:
            config_settings = yaml.load(file, Loader=yaml.FullLoader)
        wandb.init(config = config_settings['parameters'])
        train("train")


if __name__ == "__main__":
    decide_run()


