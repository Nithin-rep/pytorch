import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

# data
import sys
sys.path.append("C:/Users/Nagireddy/Desktop/practice_projects/2_2input_curve_fitting")
from data import x,y,z,z_original,xy, train_dl, val_dl


# 3d plot
figure = plt.figure(figsize=(5,5))
ax = figure.add_subplot(111,projection='3d')
ax.scatter(x,y,z_original, color ="orange")
ax.scatter(x,y,z, color = "blue", marker = "*")
plt.show()



#%%

# NN model

from network import model
optimizer = torch.optim.Adam(model.parameters(), lr =1e-1)

#%%

# Training the model
from train import trai #,zb,xy, pred

batch_size = 200        
loss_fn = F.mse_loss
losses,val_losses, xy, zb, pred = trai(200, model, train_dl, val_dl, loss_fn, optimizer, batch_size)

#plot of the traning and validation loss
plt.figure(figsize=(10,5))
plt.plot(losses)
plt.plot(val_losses)
plt.legend(["train_loss","validation_loss"])
plt.show()

#%%

# plot original fn and predictions  
figure =plt.figure(figsize =(10,10))
ax = figure.add_subplot(111,projection="3d")
ax.scatter(xy[:,0],xy[:,1],zb, color="red",marker= "o")
ax.scatter(xy[:,0],xy[:,1], pred.detach().numpy(), color="black",marker= "*")
ax.legend(["Actual Response", "Predictions by MLP"])
plt.show()

