import torch.nn as nn
import torch

class PowerSystemNN(nn.Module):
    """
    Neural network model for electric power system branch overload prediction.

    Define your neural network architecture here. You should consider how many layers to include,
    the size of each layer, and the activation functions you will use.
    """
    
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize your neural network model.
        
        Parameters:
            input_dim (int): The dimensionality of the input data.
            output_dim (int): The dimensionality of the output data.
        """
        super(PowerSystemNN, self).__init__()
        # Define your neural network architecture here
        # Empirically determined values
        hidden1_dim = 64
        hidden2_dim = 32

        # Define fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output_layer = nn.Linear(hidden2_dim, output_dim)
        
        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model.
        
        Here, you should apply the layers and activation functions you defined in __init__ to the input tensor.
        
        Parameters:
            x (Tensor): The input tensor to the neural network.
        
        Returns:
            Tensor: The output of the network.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output_layer(x))
        return x
