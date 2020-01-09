import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio
import matplotlib.pyplot as plt

class RV1Torch(nn.Module):

    def __init__(self, output=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.dense1 = torch.nn.Linear(64, 256)  #64 Conv with (3*3) followed by a (3*3) max pooling
        self.dense2 = torch.nn.Linear(256, 256)
        self.dense3 = torch.nn.Linear(256, output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3)
        # add dropout (λ = 0.3)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 3)
        # add dropout (λ = 0.3)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = RV1Torch()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

context = 32
n_bins = 32
input = torch.randn(1, 1, context, n_bins)
out = net(input)
print(out)

target = torch.tensor([0, 1, 1, 0, 0], dtype=torch.float32)  #  dtype=torch.long a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output

# loss = nn.MSELoss()(out, target)
loss = nn.CrossEntropyLoss
