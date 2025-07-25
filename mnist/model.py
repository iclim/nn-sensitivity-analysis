import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    Simple feedforward neural network for MNIST digit classification
    """

    def __init__(
        self, input_size=784, hidden1=128, hidden2=64, num_classes=10, dropout_rate=0.2
    ):
        super(MNISTNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Flatten the image if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    def get_feature_maps(self, x):
        """Return intermediate representations for analysis"""
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        h1 = F.relu(self.fc1(x))
        h1_dropped = self.dropout(h1)
        h2 = F.relu(self.fc2(h1_dropped))
        h2_dropped = self.dropout(h2)
        output = self.fc3(h2_dropped)

        return {
            "input": x,
            "hidden1": h1,
            "hidden2": h2,
            "output": output,
            "log_softmax": F.log_softmax(output, dim=1),
        }
