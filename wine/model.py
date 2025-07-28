import torch.nn as nn
import torch.nn.functional as F


class WineNet(nn.Module):
    """
    Neural network for wine quality prediction from chemical features
    """

    def __init__(
        self,
        input_size=11,
        hidden1=64,
        hidden2=32,
        hidden3=16,
        num_classes=10,
        dropout_rate=0.3,
    ):
        super(WineNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, num_classes)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden1)
        self.batch_norm2 = nn.BatchNorm1d(hidden2)
        self.batch_norm3 = nn.BatchNorm1d(hidden3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    def get_feature_maps(self, x):
        """Return intermediate representations for analysis"""
        h1 = F.relu(self.batch_norm1(self.fc1(x)))
        h1_dropped = self.dropout1(h1)
        h2 = F.relu(self.batch_norm2(self.fc2(h1_dropped)))
        h2_dropped = self.dropout2(h2)
        h3 = F.relu(self.batch_norm3(self.fc3(h2_dropped)))
        h3_dropped = self.dropout3(h3)
        output = self.fc4(h3_dropped)

        return {
            "input": x,
            "hidden1": h1,
            "hidden2": h2,
            "hidden3": h3,
            "output": output,
            "log_softmax": F.log_softmax(output, dim=1),
        }
