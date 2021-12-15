import torch.nn as nn
import torch.nn.functional as F

# NN model

class Fit(nn.Module):
    def __init__(self):
        super(Fit,self).__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc4 = nn.Linear(10,1)

    def forward(self,z):
        # z = z.view(z.size(0), -1)
        z=F.leaky_relu(self.fc1(z))
        z=self.fc4(z)
        return z

model = Fit()