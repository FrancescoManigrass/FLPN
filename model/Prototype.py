from torch import nn


class ProtoModel(nn.Module):
    def __init__(self, attr_dim=85, hid_dim = 2048):
        super().__init__()
        self.fc1 = nn.Linear(attr_dim, hid_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, hid_dim)




    def forward(self, attrs):
        protos = self.fc2(self.relu1(self.fc1(attrs)))
        return protos