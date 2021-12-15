# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.manual_seed(1)
samples = 1000

x = torch.randint(-250,250,(1,1000)).view(-1,1)
y = ((x**3 + x**2 + x)- np.random.randint(100)).view(-1,1)


x = x.type(torch.FloatTensor)
y = y.type(torch.FloatTensor)


# Response Curve
plt.figure(figsize=(10,8))
plt.scatter(x,y, color = "red")


#%%

# Data Split
def split(len_data,x,y):
    
    # To mix the datapoint index randomly
    rangee = np.random.permutation(len_data)
    train_validation = round(0.7*(len_data))

        
    # To split train and validation
    train_validation_range = rangee[:train_validation]
    train_val_splitter = round(0.7*len(train_validation_range))
    
    train_x = x[train_validation_range[:train_val_splitter]]
    train_y = y[train_validation_range[:train_val_splitter]]
    
    val_x = x[train_validation_range[train_val_splitter:]]
    val_y = y[train_validation_range[train_val_splitter:]]

    return train_x, train_y, val_x, val_y

train_x, train_y, val_x, val_y = split(len(x),x,y)

# Dataloader to complete the datafeed
from torch.utils.data import TensorDataset
train_ds = TensorDataset(train_x, train_y)
val_ds = TensorDataset(val_x, val_y)

batch_size = 120
from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size =batch_size, shuffle= True)
val_dl = DataLoader(val_ds, batch_size =batch_size, shuffle= True)

#%%

# NN model

class Fit(nn.Module):
    def __init__(self):
        super(Fit,self).__init__()
        self.fc1 = nn.Linear(1,80)
        self.fc2 = nn.Linear(80,60)
        self.fc3 = nn.Linear(60,40)
        self.fc4 = nn.Linear(40,20)
        self.fc5 = nn.Linear(20,1)


    def forward(self,z):
        z = z.view(z.size(0), -1)
        z=F.leaky_relu(self.fc1(z)) 
        z=F.leaky_relu(self.fc2(z))
        z=F.leaky_relu(self.fc3(z))
        z=F.leaky_relu(self.fc4(z))
        z=self.fc5(z)
        return z

model = Fit()
optimizer = torch.optim.Adam(model.parameters(), lr =1e-1)

#%%

# Training the model

losses =[]
def train(n_epochs, model, loss_fn, optimizer, batch_size):

    for i in range(n_epochs):
        print(i)
        print("\n")
        total_loss = 0
        
        for xb,yb in train_dl:
            xb = xb
            yb = yb
            print("x: ",xb)
            print("\n")
            
            print("yb: ",yb)
            print("\n")
            
            pred = model(xb)
            print(pred)
            
            loss = loss_fn(pred,yb)
            total_loss += loss
            print(loss)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            print("\n")
            
 

        print("Epoch:{}, Mean loss of training: {}" .format(i,(total_loss/ batch_size)))
        losses.append(total_loss/batch_size)
        print("\n")    
    
loss_fn = F.mse_loss
train(100, model, loss_fn, optimizer, batch_size)


plt.figure(figsize=(10,5))
plt.plot(losses)
plt.show()

#%%
f =torch.tensor(80).view(-1,1)
f = f.type(torch.FloatTensor)
print((f))
print(model(f))
