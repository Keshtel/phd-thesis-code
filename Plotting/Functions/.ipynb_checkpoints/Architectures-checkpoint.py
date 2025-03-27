from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

class SpeedPredictorCNN(nn.Module):
    #1 convolutional layer + ReLu + one fully connected layer + ReLu
    #output layer: prediction of speed from kernel_size time point in future for the first time point of each image(batch) till 
    # T-kernel_size timepoint in each batch
    def __init__(self, num_neurons, window_size, num_kernels=1, kernel_size=10):
        super(SpeedPredictorCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_neurons, 
                               out_channels=num_kernels, 
                               kernel_size=kernel_size, 
                               stride=1, 
                               padding=0)
        self.relu1 = nn.ReLU()
        
        # Compute output length after convolution
        conv_output_length = window_size - kernel_size + 1  # Since padding=0, stride=1
        self.flattened_size = num_kernels * conv_output_length

        # Fully Connected Layer
        self.fc = nn.Linear(self.flattened_size, window_size)# - kernel_size + 1)  # Predicting full sequence
        self.relu2 = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.flatten(start_dim=1)  # Flatten before FC layer
        x = self.fc(x)  # Fully connected layer
        return x



class SpeedPredictorCNN_GAP(nn.Module):
    #1 convolutional layer + ReLu + Average of each kernel after sliding through the whole images + one fully connected layer
    #output layer: prediction of speed from kernel_size time point in future for all time points in a batch 
    
    def __init__(self, num_neurons, window_size, num_kernels=3, kernel_size=10):
        super(SpeedPredictorCNN_GAP, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_neurons, 
                               out_channels=num_kernels, 
                               kernel_size=kernel_size, 
                               stride=1, 
                               padding=0)
        self.relu1 = nn.ReLU()

        # Global Average Pooling: Reduces (batch, num_kernels, output_length) to (batch, num_kernels, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layer: Predicts the final output using pooled features
        self.fc = nn.Linear(num_kernels, window_size)# - kernel_size + 1)  # Maps from num_kernels to window_size

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.global_pool(x)  # Global Average Pooling → (batch, num_kernels, 1)
        x = x.squeeze(2)  # Remove the last dimension to get (batch, num_kernels)

        x = self.fc(x)  # Fully connected layer maps kernels to output sequence
        return x


class SpeedPredictorCNN_Conv1D(nn.Module):
    #1 convolutional layer + ReLu + 1d convlutional layer with kernelsize 1 so the different time points from the previous layers are not mixed
    #output layer: prediction of speed from kernel_size time point in future for the first time point of each image(batch) till 
    #T-kernel_size timepoint in each batch
    def __init__(self, num_neurons, window_size, num_kernels=3, kernel_size=10):
        super(SpeedPredictorCNN_Conv1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_neurons, 
                               out_channels=num_kernels, 
                               kernel_size=kernel_size, 
                               stride=1, 
                               padding=0)
        self.relu1 = nn.ReLU()

        # Output length after first conv layer
        conv_output_length = window_size - kernel_size + 1

        # 1D Convolution for final prediction (instead of FC layer)
        self.conv2 = nn.Conv1d(in_channels=num_kernels, 
                               out_channels=1,  # Predict single speed value per time step
                               kernel_size=1,  # Kernel size of 1 keeps time alignment
                               stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)  # Second convolution layer instead of FC
        x = x.squeeze(1)  # Remove the singleton dimension to match (batch_size, time_steps)
        return x

