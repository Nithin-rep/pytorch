import torch
import numpy as np

samples = 1000
x=(torch.randint(-100,100,(1,samples))*1e-2).view(-1,1)
y=(torch.randint(-100,100,(1,samples))*1e-2).view(-1,1)

z_original=(x**3 - (3*(x*y**2)))
noise = (torch.randint(-10,10,(samples,1))*0.05)
z=(x**3 - (3*(x*y**2))) + noise
z= z.type(torch.FloatTensor)

print(z[:3],z_original[:3])
xy = []
for i in range(len(x)):
    xy.append(x[i])
    xy.append(y[i])
xy = torch.tensor(xy)
xy = np.reshape(xy,(-1,2))
xy =xy.type(torch.FloatTensor)


batch_size = 200

# Data Split
def split(len_data,xy,z):
    
    # To mix the datapoint index randomly
    rangee = np.random.permutation(len_data)
    train_validation = round(0.7*(len_data))
 
    # To split train and validation        
    train_xy = xy[rangee[:train_validation]]
    train_z = z[rangee[:train_validation]]
    
    val_xy = xy[rangee[train_validation:]]
    val_z = z[rangee[train_validation:]]

    return train_xy, val_xy, train_z, val_z

train_xy, val_xy, train_z, val_z = split(len(x),xy,z)

# Dataloader to complete the datafeed
from torch.utils.data import TensorDataset
train_ds = TensorDataset(train_xy, train_z)
val_ds = TensorDataset(val_xy, val_z)



from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size =batch_size)
val_dl = DataLoader(val_ds, batch_size =batch_size)

def get_dataloader():
    return train_dl, val_dl