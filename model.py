import torch.nn as nn
from utils import get_fc_input_size

class charCNN(nn.Module):
    """
    Defines the character level CNN architecture
    """
    def __init__(self,num_features,conv_channel_size,fc_size,num_class,max_seq_len,mean,std):
        """
        Initializes the model
        @params num_features (int): The number of features (the number of
         characters) considered
        @params conv_channel_size (int): Number of 1D Convolutional kernels used
        @params fc_size (int): Number of units in the fully-connected layers
        @num_class (int): Number of classes in the dataset
        @returns object of this class when implicitly called
        """
        super(charCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=num_features,out_channels=conv_channel_size,kernel_size=7,stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channel_size,out_channels=conv_channel_size,kernel_size=7,stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channel_size,out_channels=conv_channel_size,kernel_size=3,stride=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channel_size,out_channels=conv_channel_size,kernel_size=3,stride=1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channel_size,out_channels=conv_channel_size,kernel_size=3,stride=1),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channel_size,out_channels=conv_channel_size,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=get_fc_input_size(max_seq_len)*conv_channel_size,out_features=fc_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc_size,out_features=fc_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(in_features=fc_size,out_features=num_class)

        self.log_softmax = nn.LogSoftmax(dim=1)

        self._create_weights(mean,std) # weight initialization

    def forward(self,inputs):
        """
        Forward pass through CNN
        @params inputs (torch.Tensor): Tensor of shape
        (batch_size,num_characters,max_seq_len) representing a batch of
        sentences
        @returns x (torch.Tensor): Tensor of shape (batch_size,num_cls) that
        contains a batch of vectors having unnormalized log probabilities for
        each class computed according to the input text
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x


    def _create_weights(self, mean, std):
        """
        Initialization of weights using a Gaussian distribution
        @params mean (int): Mean of the distribution
        @params std (int): Standard deviation of the distribution
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)
