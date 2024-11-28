import pandas as pd
import numpy as np
from time import time
import torch
from torch.autograd import Variable
from FrEIA.framework import *
from FrEIA.modules import *
import configuration as c
import losses
import model
import data as d
from sklearn.model_selection import train_test_split
import numpy as np

x_train = d.x_train
y_train = d.y_train
x_test = d.x_test
y_test = d.y_test

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=c.batch_size, shuffle=True, drop_last=True)

mse_mmd = pd.DataFrame(columns=['epoch', 'train_avg_mse_y_loss', 'test_avg_mse_y_loss', 'train_avg_mmd_x_loss', 'test_avg_mmd_x_loss'])

def train_epoch():
    print(model.model)
    print("Starting training now")

    for epoch in range(c.n_epochs):
        # Set to Training Mode
        train_loss = 0
        train_mse_y_loss = 0
        train_mmd_x_loss = 0

        model.model.train()

        loss_factor = min(1., 2. * 0.002 ** (1. - (float(epoch) / c.n_epochs)))

        train_loss_history = []
        for j, (x, y) in enumerate(train_loader):
            batch_losses = []
            x, y = x.to(c.device), y.to(c.device)
            
            #  data pad cat #
            x_pad = c.add_pad_noise * torch.randn(c.batch_size, c.ndim_pad_x).to(c.device)
            x = torch.cat((x, x_pad), dim=1)

            # yz_pad，z_pad,y
            yz_pad = c.add_pad_noise * torch.randn(c.batch_size, c.ndim_pad_zy).to(c.device)
            z_pad = torch.randn(c.batch_size, c.ndim_z).to(c.device)
            y = torch.cat((z_pad, yz_pad, y), dim=1)

            #  Forward step  #
            model.optim.zero_grad()
            out_y = model.model(x)[0]
            out_x = model.model(y, rev=True)[0]

            if c.train_max_likelihood:
                batch_losses.append(losses.loss_max_likelihood(out_y, y))

            if c.train_forward_mmd:
                batch_losses.extend(losses.loss_forward_fit_mmd(out_y, y))

            if c.train_backward_mmd:
                batch_losses.append(losses.loss_backward_mmd(x, y))

            if c.train_reconstruction:
                batch_losses.append(losses.loss_reconstruction(out_y.data, y, x))

            loss_total = sum(batch_losses)
            train_loss_history.append([l.item() for l in batch_losses])

            # original y and x
            mse_y_loss = losses.l2_fit(out_y[:, -c.ndim_y:], y[:, -c.ndim_y:])
            mmd_x_loss = torch.mean(losses.backward_mmd(x[:, :c.ndim_x], out_x[:, :c.ndim_x]))

            loss_total.backward()
            
            #  Gradient Clipping #
            for parameter in model.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.data.clamp_(-c.grad_clamp, c.grad_clamp)

            # Descent gradient #
            model.optim.step()  

            # MLE training
            train_loss += loss_total
            train_mse_y_loss += mse_y_loss
            train_mmd_x_loss += mmd_x_loss

        # Calculate the avg loss of training
        train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)
        train_avg_mse_y_loss = train_mse_y_loss.cpu().data.numpy() / (j + 1)
        train_avg_mmd_x_loss = train_mmd_x_loss.cpu().data.numpy() / (j + 1)

        if epoch % c.eval_test == 0:
            model.model.eval()
            print("Doing Testing evaluation on the model now")

            test_loss = 0
            test_mse_y_loss = 0
            test_mmd_x_loss = 0

            test_loss_history = []
            
            for j, (x, y) in enumerate(test_loader):

                batch_losses = []

                x, y = Variable(x).to(c.device), Variable(y).to(c.device)

                #  data pad cat #
                x_pad = c.add_pad_noise * torch.randn(c.batch_size, c.ndim_pad_x).to(c.device)
                x = torch.cat((x, x_pad), dim=1)

                # yz_pad，z_pad,y
                yz_pad = c.add_pad_noise * torch.randn(c.batch_size, c.ndim_pad_zy).to(c.device)
                
                z_pad = torch.randn(c.batch_size, c.ndim_z).to(c.device)
                y = torch.cat((z_pad, yz_pad, y), dim=1)

                # Forward step #

                model.optim.zero_grad()
                out_y = model.model(x)[0]
                out_x = model.model(y, rev=True)[0]

                if c.train_max_likelihood:
                    batch_losses.append(losses.loss_max_likelihood(out_y, y))

                if c.train_forward_mmd:
                    batch_losses.extend(losses.loss_forward_fit_mmd(out_y, y))

                if c.train_backward_mmd:
                    batch_losses.append(losses.loss_backward_mmd(x, y))

                if c.train_reconstruction:
                    batch_losses.append(losses.loss_reconstruction(out_y.data, y, x))

                loss_total = sum(batch_losses)
                test_loss_history.append([l.item() for l in batch_losses])
                

                # original y and x
                mse_y_loss = losses.l2_fit(out_y[:, -c.ndim_y:], y[:, -c.ndim_y:])
                mmd_x_loss = torch.mean(losses.backward_mmd(x[:, :c.ndim_x], out_x[:, :c.ndim_x]))

                # MLE training
                test_loss += loss_total
                test_mse_y_loss += mse_y_loss
                test_mmd_x_loss += mmd_x_loss

            # Calculate the avg loss of test
            test_avg_loss = test_loss.cpu().data.numpy() / (j + 1)
            test_avg_mse_y_loss = test_mse_y_loss.cpu().data.numpy() / (j + 1)
            test_avg_mmd_x_loss = test_mmd_x_loss.cpu().data.numpy() / (j + 1)
            mse_mmd.loc[epoch] = [epoch, train_avg_mse_y_loss, test_avg_mse_y_loss, train_avg_mmd_x_loss, test_avg_mmd_x_loss]
           
            print('Training losses', np.mean(train_loss_history, axis=0))
            print('Testing losses', np.mean(test_loss_history, axis=0))
            print('Epoch %d, training loss %.4f, testing loss %.4f' % (epoch, train_avg_loss, test_avg_loss))
            print('Epoch %d, training mse y loss %.4f, testing mse y loss %.4f' % (epoch, train_avg_mse_y_loss, test_avg_mse_y_loss))
            print('Epoch %d, training mmd x loss %.4f, testing mmd x loss %.4f' % (epoch, train_avg_mmd_x_loss, test_avg_mmd_x_loss))
        
        mse_mmd.to_excel(d.mse_mmd, index=False)   
        model.scheduler_step()
    
if __name__ == "__main__":
    train_epoch()
    torch.save(model.model.state_dict(), d.pklname)
