from model_utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def main():
    # Load in data
    X, y = joblib.load('./data/processed/cnn_mfcc_data.pkl')
    print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")

    # Split data into train/val and test sets, then train and validation sets 
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2 / 0.8, stratify=y_temp, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # adds channel dimension 
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)  # adds channel dimension 
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    print(f"Train tensor shape: {X_train_tensor.shape}, Test tensor shape: {X_val_tensor.shape}")

    # Convert data to DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print("DataLoaders ready.")

    # Instantiate the model 
    model = CNN()

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training the model 
    num_epochs = 500
    best_val_loss = float('inf')  # best validation loss encountered
    patience = 10  # how long do we want to wait until we stop?
    counter = 0  # how long have we been waiting?

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        ### TRAINING PHASE ###
        model.train()  # set model to training mode
        train_loss = 0.0  # training loss for this epoch
        train_correct = 0  # number of correct predictions
        train_total = 0  # total number of predictions made

        for X_batch, y_batch in train_loader:  # loop over the batches of the training data
            optimizer.zero_grad()  # reset gradients from previous batch
            outputs = model(X_batch)  # forward pass - get model predictions
            loss = criterion(outputs, y_batch)  # compute the loss of those predictions
            loss.backward()  # backward pass - compute gradients
            optimizer.step()  # update model weights based on gradients to minimize loss

            train_loss += loss.item() * X_batch.size(0)  # add loss scaled by batch size

            _, predicted = torch.max(outputs, 1)  # get index of max logit (what its predicting)
            train_correct += (predicted == y_batch).sum().item()  # mark how many it got correct
            train_total += y_batch.size(0)  # mark how many total predictions were there

        train_epoch_loss = train_loss / len(train_loader.dataset)  # compute average loss for epoch
        train_accuracy = train_correct / train_total  # compute the accuracy

        train_losses.append(train_epoch_loss)  
        train_accuracies.append(train_accuracy)

        ### EVALUATION PHASE ####
        model.eval()  # set model to evaluation mode
        val_loss = 0.0
        val_correct = 0  
        val_total = 0  

        with torch.no_grad():  # we don't need gradients for evaluation
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                _, predicted = torch.max(outputs, 1)  
                val_correct += (predicted == y_batch).sum().item()  
                val_total += y_batch.size(0)   
            
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total 

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_accuracy) 
        
        ### EARLY STOPPING ###
        if val_epoch_loss < best_val_loss:  # if the val loss is improved
            best_val_loss = val_epoch_loss  # make this the new best val loss
            counter = 0  # reset the counter since we are improving
            torch.save(model.state_dict(), './models/best_cnn_model.pth')  # save the best model
            print(f"Validation loss improved — model saved.")

        else:  # if the val loss did not improve
            counter += 1  # increase the counter
            print(f"No improvement. Patience counter: {counter}/{patience}")

            if counter >= patience:  # if we have waiting long enough and no improvement
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break  # stop training 
        
        scheduler.step(val_epoch_loss)  # step the scheduler to adjust lr
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.6f}")
        
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

    plot_loss(train_losses, val_losses)
    plot_accuracy(train_accuracies, val_accuracies)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Input (batch_size, 1, 25, 32)
        # 1. Convolution1 = (16, 25, 32) --> ReLU --> Max Pool = (16, 12, 16)
        # 2. Convolution2 = (32, 12, 16) --> ReLU --> Max Pool = (32, 6, 8)
        # 3. Flatten (32 * 6 * 8 = 1536)
        # 4. FC1 = (1536 → 64) --> ReLU
        # 5. FC2 = (64 → 10) --> output (10)
        self.conv1 = nn.Conv2d(
            in_channels=1,  # single input channel ("greyscale" MFCC)
            out_channels=16,  # learn 16 feature maps
            kernel_size=3,  # 3x3 filters
            padding=1  # keeps output same as input 
        )
        self.pool = nn.MaxPool2d(2,2)  # halves height and width
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten_dim = 32 * 6 * 8  # channels * height * width

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_dim, 64)  # hidden layer with 64 units
        self.fc2 = nn.Linear(64, num_classes)       # output layer for classification

    def forward(self, x):# 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    main()