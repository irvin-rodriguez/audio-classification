from model_utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def main():
    # Load in data
    X, y = joblib.load('./data/processed/avg_mfcc_data.pkl')

    # Split data into train/val and test sets, then train and validation sets 
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2 / 0.8, stratify=y_temp, random_state=42)

    # Standarize features
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_val_scaled = scalar.transform(X_val)
    joblib.dump(scalar, './models/fnn_scaler.pkl')  # save the scalar for future use

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Convert data to DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Instantiate the model 
    model = Net()

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training the model 
    num_epochs = 500
    best_val_loss = float('inf')  # best validation loss encountered
    patience = 20  # how long do we want to wait until we stop?
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
            torch.save(model.state_dict(), './models/best_fnn_model.pth')  # save the best model
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

    return 0

# Define the neural network model
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(25, 128)  # 25 inputs
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)  
        self.fc5 = nn.Linear(16, num_classes)  # 10 output classes

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 20% dropout rate 
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

if __name__ == "__main__":
    main()