# from main_2input_curve_v1 import train_dl, val_dl
import torch
# import toch.nn.functional as F


def trai(n_epochs, model, train_dl, val_dl, loss_fn, optimizer, batch_size):

    losses =[]
    val_losses = []
    
    # global xy, zb, pred
    for i in range(n_epochs):

        total_loss = 0
        total_val_loss = 0
        
        # train loop 
        for xy,zb in train_dl:
            pred = model(xy)            
            loss = loss_fn(pred,zb)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            # val loop 
            for xy, zb in val_dl:
                pred = model(xy)            
                loss = loss_fn(pred,zb)
                val_loss = loss_fn(pred,zb)
                total_val_loss += val_loss  
  
        print("Epoch:{}, Mean loss of training: {} and validation: {}" .format(i+1,(total_loss.item()/ batch_size),(total_val_loss.item()/ batch_size)))
        losses.append(total_loss.item()/batch_size)
        val_losses.append(total_val_loss.item()/batch_size)
        print("\n")      
    
    return losses, val_losses, xy, zb, pred


