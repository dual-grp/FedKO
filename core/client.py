import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
import copy
from utils.reservoir_koopman import *
from utils.ad_util import *
from utils.dataset_utils import TimeSeriesDataset

import tracemalloc
import pynvml
import time

class Client:
    def __init__(self, id, train_data, test_data, label, model = 0, alpha = 0.75, spectral_radius = 0.99,
                 feature_dim = 25, reservoir_dim = 512, output_dim = 128, train_rate = 0.9,
                 i_window = 256, o_window = 32, i_batch_size = 512, o_batch_size = 64, 
                 inner_lr = 1e-3, outer_lr = 1e-3, lr_decay_factor = 0.8, device = "cpu"):
        super(Client, self).__init__()

        self.id = id
        self.device = device
        self.input_dim = feature_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.spectral_radius = spectral_radius

        self.i_batch_size = i_batch_size
        self.o_batch_size = o_batch_size
        self.i_window = i_window
        self.o_window = o_window

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.lr_decay_factor = lr_decay_factor

        # Regularization strengths
        self.lambda_phi = 1e-4
        self.lambda_K = 1e-4
        self.lambda_Xi = 1e-4

        self.train_size = int(train_rate * len(train_data))

        self.train_data = train_data
        self.test_data = test_data

        # Initialize GPU monitoring
        #gpu_handle = start_gpu_monitoring()

        # Start monitoring CPU memory
        #tracemalloc.start()
        # Record CPU and GPU memory before function execution
        #cpu_start, _ = tracemalloc.get_traced_memory()
        #gpu_start = get_gpu_memory_usage(gpu_handle)

        self.train_inputs = torch.tensor(train_data[:self.train_size-1,:]).to(self.device)
        self.train_targets = torch.tensor(train_data[1:self.train_size,:]).to(self.device)
        # Create the dataset and dataloader
        self.train_dataset = TimeSeriesDataset(train_data[:self.train_size,:],
                                               sequence_length = self.i_window)
        #self.train_dataloader = DataLoader(self.train_dataset,
        #                                   batch_size=self.i_batch_size, shuffle=True)

        self.val_inputs = torch.tensor(train_data[self.train_size-1:,:]).to(self.device)
        self.val_targets = torch.tensor(train_data[self.train_size:,:]).to(self.device)
        # Create the dataset and dataloader        
        self.val_dataset = TimeSeriesDataset(train_data[self.train_size:,:],
                                             sequence_length = self.o_window)
        #self.val_dataloader = DataLoader(self.val_dataset,
        #                                 batch_size=self.o_batch_size, shuffle=True)

        self.test_inputs = torch.tensor(test_data[:-1,:]).to(self.device)
        self.test_targets = torch.tensor(test_data[1:,:]).to(self.device)

        self.target = label

        if model == 0:       
            self.phi_net = PsiNN0(input_size = self.input_dim,
                                  reservoir_size = self.reservoir_dim,
                                  output_size = self.output_dim,
                                  #device = device,
                                  alpha = alpha, spectral_radius = spectral_radius,
                                  device=self.device).to(self.device)
        elif model == 1: 
            #print("Here")      
            self.phi_net = PsiNN1(input_size = self.input_dim,
                                  reservoir_size = self.reservoir_dim,
                                  output_size = self.output_dim,
                                  #device = device,
                                  alpha = alpha, spectral_radius = spectral_radius,
                                  device=self.device).to(self.device)  
              
        #phi_net = EigenfunctionNN(input_dim, output_dim, hidden_dim).to(self.device) 
        # Defining Koopman operator as a linear layer without bias
        self.K = KoopmanOperator(self.output_dim, self.spectral_radius).to(self.device) 
        self.Xi = KoopmanMode(self.output_dim, self.input_dim).to(self.device)
 

        # Record CPU and GPU memory after function execution
        #cpu_end, _ = tracemalloc.get_traced_memory()
        #gpu_end = get_gpu_memory_usage(gpu_handle)

        # Stop the CPU memory monitoring
        #tracemalloc.stop()

        # Calculate the memory usage
        #cpu_memory_used = (cpu_end - cpu_start) / 1024  # Convert to KB
        #gpu_memory_used = gpu_end - gpu_start

        #print(f"CPU Memory Used: {cpu_memory_used} KB")
        #print(f"GPU Memory Used: {gpu_memory_used} MB")

        # Don't forget to shutdown NVML
        #pynvml.nvmlShutdown()

         # Optimization settings
        self.optimizer_inner = torch.optim.Adam([{'params': self.phi_net.parameters()},
                                                 {'params': self.K.parameters()},
                                                 {'params': self.Xi.parameters()}], lr=self.inner_lr)

        self.optimizer_outer = torch.optim.Adam([{'params': self.phi_net.parameters()},
                                                 {'params': self.Xi.parameters()},
                                                 {'params': self.K.parameters()}], lr=self.outer_lr)
        
        #self.opt_aut = torch.optim.Adam(self.AUTOENCODER.parameters(), lr=0.0001)
        #self.opt_kpm = torch.optim.Adam(self.KPM.parameters(), lr=0.00001)
        #self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
        self.factors = [0.25,0.1,10,5]

    def l2_regularization_loss(self, parameters):
        return torch.sum(torch.stack([torch.norm(p)**2 for p in parameters]))

    def linearity_loss(self, x_t, x_next):
        phi_xt = self.phi_net(x_t)
        phi_xnext_pred = torch.mm(phi_xt, self.K)
        phi_xnext_true = self.phi_net(x_next)
        loss = F.mse_loss(phi_xnext_pred, phi_xnext_true)
        reg_loss = self.lambda_K * self.l2_regularization_loss([self.K]) \
            + self.lambda_phi * self.l2_regularization_loss(self.phi_net.parameters())

        return loss + reg_loss

    def invariance_loss(self, x):
        phi_x = self.phi_net(x)
        x_reconstructed = torch.mm(phi_x, torch.transpose(self.Xi, 0, 1))
        loss = F.mse_loss(x, x_reconstructed)
        #reg = 0.0  
        return loss

    def invariance_loss_2(self, x, y):
        """Compute the invariance loss based on Koopman eigenfunctions and modes."""
        # Compute observables for current state
        phi_x = self.phi_net(x)
        # Reconstruct current state from observables
        x_reconstructed = torch.mm(phi_x.to(torch.complex128),
                                   torch.transpose(self.Xi, 0, 1)).real
        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(x, x_reconstructed)
        # Predict next state using Koopman operator
        # Here, we multiply the observables with Koopman modes first, then apply the Koopman operator (eigenvalues of K)
        #x_next_predicted = torch.mm(torch.mm(phi_x, Xi), K)
        #print(phi_X.shape)
        eigenvals, _  = torch.linalg.eig(self.K)
        phi_Y_pred = torch.mm(phi_x.to(torch.complex128), torch.diag(eigenvals).to(self.device))
        x_next_predicted = torch.mm(phi_Y_pred, torch.transpose(self.Xi, 0, 1)).real
        #phi_Y_pred = torch.mm(phi_x, self.K)
        #x_next_predicted = torch.mm(phi_Y_pred, torch.transpose(self.Xi, 0, 1))
        # Compute dynamics prediction loss
        dynamics_loss = F.mse_loss(y, x_next_predicted)
        # Regularization losses
        reg_loss_Xi = self.lambda_Xi * self.l2_regularization_loss([self.Xi])
        reg_loss_phi = self.lambda_phi * self.l2_regularization_loss(list(self.phi_net.parameters()))
        # Total loss
        total_loss = reconstruction_loss + dynamics_loss + reg_loss_Xi + reg_loss_phi
        
        return total_loss

    def inner_loss(self, x_prev, x_next):

        phi_xprev = self.phi_net(x_prev)
        phi_xnext = self.phi_net(x_next)
        
        phi_xnext_pred = self.K(phi_xprev)
        linear_loss = F.mse_loss(phi_xnext_pred, phi_xnext)

        xnext_reconstructed = self.Xi(phi_xnext_pred)
        recon_loss = F.mse_loss(x_next, xnext_reconstructed)
        
        reg_loss = self.lambda_K * self.l2_regularization_loss(self.K.parameters()) 
                   # + self.lambda_phi * self.l2_regularization_loss(self.phi_net.parameters()) \
                   # + self.lambda_Xi * self.l2_regularization_loss(self.Xi.parameters())

        return linear_loss + recon_loss + reg_loss
    
    def train_inner(self, epochs, decay_iter = 10):
        """Train the model on local data for a given number of epochs."""
        # This is a simplified training loop for demonstration.
        # In practice, you'd have batches, data loaders, etc.
        """Train the model on local data for a given number of epochs."""
        self.optimizer_inner = torch.optim.Adam([#{'params': self.phi_net.parameters()},
                                                 {'params': self.K.parameters()},
                                                 #{'params': self.Xi.parameters()}
                                                 ],
                                                 lr=self.inner_lr)
        self.inner_losses = []
        for epoch in range(epochs):

            #self.phi_net.train() 
            self.K.train() 
            #self.Xi.train() 
            running_loss = 0.0

            for i in range(0, len(self.train_inputs), self.batch_size):
                # Get the data; data is a list of [inputs, labels]
                #inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self.train_inputs[i:i+self.batch_size].to(self.device)
                labels = self.train_targets[i:i+self.batch_size].to(self.device)
                           
                # INNER OPTIMIZATION for phi_net and K
                self.optimizer_inner.zero_grad()
                loss_linearity = self.inner_loss(inputs, labels)
                loss_linearity.backward()
                self.optimizer_inner.step()

                running_loss += loss_linearity.item()

                #if epoch % 1 == 0:
            inner_train_loss = running_loss/len(self.train_inputs)

            if epoch % decay_iter == 0:
                self.inner_losses.append(inner_train_loss)
                if len(self.inner_losses) > 2 and self.inner_losses[-1] > self.inner_losses[-2]:
                    print("Error increased. Decay learning rate")
                    for g in self.optimizer_inner.param_groups:
                        g['lr'] *= self.lr_decay_factor    

        return inner_train_loss
    
    def train_outer(self, epochs, decay_iter = 10):
        """Train the model on local data for a given number of epochs."""
        # This is a simplified training loop for demonstration.
        # In practice, you'd have batches, data loaders, etc.
        self.optimizer_outer = torch.optim.Adam([{'params': self.phi_net.parameters()},
                                                #{'params': self.K.parameters()},
                                                {'params': self.Xi.parameters()}],
                                                lr=self.outer_lr)
        self.outer_losses = []
        for epoch in range(epochs):

            self.phi_net.train() 
            #self.K.train() 
            self.Xi.train() 
            running_loss = 0.0
             
            for i in range(0, len(self.train_inputs), self.batch_size):
                # Get the data; data is a list of [inputs, labels]
                #inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self.train_inputs[i:i+self.batch_size].to(self.device)
                labels = self.train_targets[i:i+self.batch_size].to(self.device)
                
                # OUTER OPTIMIZATION for phi_net and Xi
                self.K.requires_grad = False
                self.optimizer_outer.zero_grad()
                loss_invariance = self.invariance_loss_2(inputs, labels)
                loss_invariance.backward()
                self.optimizer_outer.step()
                self.K.requires_grad = True

                running_loss += loss_invariance.item()

                #if epoch % 1 == 0:
            outer_train_loss = running_loss/inputs.size(0)/len(self.train_inputs)

            if epoch % decay_iter == 0:
                self.outer_losses.append(outer_train_loss)
                if len(self.outer_losses) > 2 and self.outer_losses[-1] > self.outer_losses[-2]:
                    print("Error increased. Decay learning rate")
                    for g in self.optimizer_outer.param_groups:
                        g['lr'] *= self.lr_decay_factor   

        return outer_train_loss
        
    def train(self, epochs, decay_iter = 10):
        """Train the model on local data for a given number of epochs."""
        # This is a simplified training loop for demonstration.
        # In practice, you'd have batches, data loaders, etc.
        self.losses = []
        for epoch in range(epochs):

            self.phi_net.train() 
            running_loss = 0.0

            for i in range(0, len(self.train_data), self.batch_size):
                # Get the data; data is a list of [inputs, labels]
                #inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self.train_inputs[i:i+self.batch_size].to(self.device)
                labels = self.train_targets[i:i+self.batch_size].to(self.device)
                
                upper_bound = self.K.sum(dim=1).max().item()

                if upper_bound > 1 : 
                    #self.weight /= upper_bound
                    self.K.data /= upper_bound
            
                # INNER OPTIMIZATION for phi_net and K
                self.optimizer_inner.zero_grad()
                loss_linearity = self.linearity_loss(inputs, labels)
                loss_linearity.backward()
                self.optimizer_inner.step()

                # OUTER OPTIMIZATION for phi_net and Xi
                self.K.requires_grad = False
                self.optimizer_outer.zero_grad()
                loss_invariance = self.invariance_loss_2(inputs, labels)
                loss_invariance.backward()
                self.optimizer_outer.step()
                self.K.requires_grad = True

                running_loss += ((loss_linearity + loss_invariance)/2).item()

            total_train_loss = running_loss/inputs.size(0)/len(self.train_inputs)

            if epoch % decay_iter == 0:
                self.losses.append(total_train_loss)
                if len(self.losses) > 2 and self.losses[-1] > self.losses[-2]:
                    print("Error increased. Decay learning rate")
                    for g in self.optimizer_outer.param_groups:
                        g['lr'] *= self.lr_decay_factor    
                    for g in self.optimizer_inner.param_groups:
                        g['lr'] *= self.lr_decay_factor               
        
        return total_train_loss

    def train_new(self, epochs, decay_iter = 5, inner = True, outer = True):
        """Train the model on local data for a given number of epochs."""

        self.inner_losses = []
        self.outer_losses = []
        self.psi_list = []
        self.K_list = []
        self.Xi_list = []        

        for epoch in range(epochs):

            self.phi_net.train() 
            self.K.train()
            self.Xi.train()
            inner_loss = 0.0
            outer_loss = 0.0
            
            if inner:
                for i in range(0, len(self.train_inputs), self.i_window):
                    # Get the data; data is a list of [inputs, labels]
                    #inputs, labels = inputs.to(self.device), labels.to(self.device)
                    #print(i)
                    inputs = self.train_inputs[i:i+self.i_window].to(self.device)
                    labels = self.train_targets[i:i+self.i_window].to(self.device)
            
                    # INNER OPTIMIZATION for phi_net and K
                    self.optimizer_inner.zero_grad()
                    loss_linearity = self.inner_loss(inputs, labels)
                    loss_linearity.backward(retain_graph=True)
                    self.optimizer_inner.step()

                    inner_loss += loss_linearity.item()

                inner_loss = inner_loss/len(self.train_inputs)

            #print(f"Epoch [{epoch + 1}/{epochs}] Inner Loss: {inner_loss}")

            if outer:
            # OUTER OPTIMIZATION for phi_net and Xi
                input_0 = self.val_inputs[0:1].to(self.device)
                phi_x_0 = self.phi_net(input_0)
                for i in range(0, len(self.val_targets)):
                    koopman_pred = self.K(phi_x_0)
                    x_t_reconstructed = self.Xi(koopman_pred)
                    x_t = self.val_targets[i:i+1].to(self.device)
                    #print(x_t_reconstructed.shape)
                    #print(x_t.shape)
                    #labels_outer = self.train_targets[i+self.inner_size: i+self.batch_size].to(self.device)
                    # Compute reconstruction loss
                    self.optimizer_outer.zero_grad()
                    reg_loss = self.lambda_K * self.l2_regularization_loss(self.K.parameters()) \
                        + self.lambda_phi * self.l2_regularization_loss(self.phi_net.parameters()) \
                        + self.lambda_Xi * self.l2_regularization_loss(self.Xi.parameters())
                    reconstruction_loss = F.mse_loss(x_t, x_t_reconstructed) + reg_loss
                    reconstruction_loss.backward(retain_graph=True)
                    self.optimizer_outer.step()
                    #self.K.requires_grad = True
                    phi_x_0 = copy.copy(koopman_pred)
                    outer_loss += reconstruction_loss.item()
                
                outer_loss = outer_loss/len(self.val_targets)
            #else:
            #    continue
            #total_train_loss = running_loss/len(self.train_inputs)
            #print(f"Epoch [{epoch + 1}/{epochs}] Outer Loss: {outer_loss}")
            self.inner_losses.append(inner_loss)
            self.outer_losses.append(outer_loss)
            if epoch % decay_iter == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.K_list.append(copy.deepcopy(self.K))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                if len(self.inner_losses) > 2 and self.inner_losses[-1] > self.inner_losses[-2]:
                    for g in self.optimizer_inner.param_groups:
                        g['lr'] *= self.lr_decay_factor   
                if len(self.outer_losses) > 2 and self.outer_losses[-1] > self.outer_losses[-2]:
                    print("Error increased. Decay learning rate")
                    for g in self.optimizer_outer.param_groups:
                        g['lr'] *= self.lr_decay_factor        

        #predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.eval_new()                
        #return self.losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse

        return inner_loss, outer_loss

    def train_new_2(self, epochs, decay_iter = 5, inner = True, outer = True):
        """Train the model on local data for a given number of epochs."""

        # Initialize GPU monitoring
        #gpu_handle = start_gpu_monitoring()

        # Start monitoring CPU memory
        #tracemalloc.start()

        # Record CPU and GPU memory before function execution
        #cpu_start, _ = tracemalloc.get_traced_memory()
        #gpu_start = get_gpu_memory_usage(gpu_handle)

        self.inner_losses = []
        self.outer_losses = []
        self.psi_list = []
        self.K_list = []
        self.Xi_list = []        

        for epoch in range(epochs):

            inner_loss = 0.0
            outer_loss = 0.0
            
            if inner:
                self.K.train()
                for i in range(0, len(self.train_inputs), self.i_window):
                #for data in self.train_dataloader:
                    # Get the data; data is a list of [inputs, labels]
                    #inputs, labels = inputs.to(self.device), labels.to(self.device)
                    #print(i)
                    inputs = self.train_inputs[i:i+self.i_window].to(self.device)
                    labels = self.train_targets[i:i+self.i_window].to(self.device)
                    #inputs = data[:, :-1,:]
                    #labels = data[:, 1:,:]
            
                    # INNER OPTIMIZATION for phi_net and K
                    self.optimizer_inner.zero_grad()
                    loss_linearity = self.inner_loss(inputs, labels)
                    loss_linearity.backward(retain_graph=True)
                    self.optimizer_inner.step()

                    inner_loss += loss_linearity.item()

                inner_loss = inner_loss/len(self.train_inputs)

            #print(f"Epoch [{epoch + 1}/{epochs}] Inner Loss: {inner_loss}")

            if outer:
                self.phi_net.train() 
                self.Xi.train()
                eigenvals, _  = self.K.spectral_decomposition()
                #print(eigenvals)
                #print(eigenvals.real)
                lam = torch.diag(eigenvals.real).to(self.device)
                # OUTER OPTIMIZATION for phi_net and Xi
                for i in range(0, len(self.val_inputs) - self.o_window):
                #for data in self.val_dataloader:
                    #input_0 = self.val_inputs[i]
                    #val_labels = self.val_inputs[:, 1:,:]
                    input_0 = self.val_inputs[i:i+1].to(self.device)
                    phi_x_0 = self.phi_net(input_0)
                    self.optimizer_outer.zero_grad()
                    reconstruction_loss = 0.0
                    for j in range(i+1, self.o_window):
                        ####
                        lam_power = torch.matrix_power(lam, j)
                        koopman_pred = torch.mm(phi_x_0, lam_power)
                        ####
                        ##koopman_pred = self.K(phi_x_0)
                        x_t_reconstructed = self.Xi(koopman_pred)
                        x_t = self.val_inputs[j:j+1].to(self.device)
                        #print(x_t_reconstructed.shape)
                        #print(x_t.shape)
                        #labels_outer = self.train_targets[i+self.inner_size: i+self.batch_size].to(self.device)
                        # Compute reconstruction loss
                        reconstruction_loss += F.mse_loss(x_t, x_t_reconstructed)
                        #self.K.requires_grad = True
                        ##phi_x_0 = copy.copy(koopman_pred)
                        #outer_loss += reconstruction_loss.item()

                    #reg_loss = # self.lambda_K * self.l2_regularization_loss(self.K.parameters()) \
                    reg_loss = self.lambda_phi * self.l2_regularization_loss(self.phi_net.parameters()) \
                            + self.lambda_Xi * self.l2_regularization_loss(self.Xi.parameters())
                    reconstruction_loss += reg_loss
                    reconstruction_loss.backward(retain_graph=True)
                    self.optimizer_outer.step()

                    outer_loss = reconstruction_loss.item()/self.o_window
            #else:
            #    continue
            #total_train_loss = running_loss/len(self.train_inputs)
            #print(f"Epoch [{epoch + 1}/{epochs}] Outer Loss: {outer_loss}")
            self.inner_losses.append(inner_loss)
            self.outer_losses.append(outer_loss)
            if epoch % decay_iter == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.K_list.append(copy.deepcopy(self.K))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                if len(self.inner_losses) > 2 and self.inner_losses[-1] > self.inner_losses[-2]:
                    for g in self.optimizer_inner.param_groups:
                        g['lr'] *= self.lr_decay_factor   
                if len(self.outer_losses) > 2 and self.outer_losses[-1] > self.outer_losses[-2]:
                    print("Error increased. Decay learning rate")
                    for g in self.optimizer_outer.param_groups:
                        g['lr'] *= self.lr_decay_factor        

        #predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.eval_new()                
        #return self.losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse

        # Record CPU and GPU memory after function execution
        #cpu_end, _ = tracemalloc.get_traced_memory()
        #gpu_end = get_gpu_memory_usage(gpu_handle)

        # Stop the CPU memory monitoring
        #tracemalloc.stop()

        # Calculate the memory usage
        #cpu_memory_used = (cpu_end - cpu_start) / 1024  # Convert to KB
        #gpu_memory_used = gpu_end - gpu_start

        #print(f"CPU Memory Used: {cpu_memory_used} KB")
        #print(f"GPU Memory Used: {gpu_memory_used} MB")

        # Don't forget to shutdown NVML
        #pynvml.nvmlShutdown()

        return inner_loss, outer_loss
    
    def train_reservoir(self, epochs):
        # Training Loop
        self.losses = []
        self.psi_list = []
        self.optimizer = torch.optim.Adam(self.phi_net.parameters(), lr=self.inner_lr)

        for epoch in range(epochs):

            self.phi_net.train()
            running_loss = 0.0

            for i in range(0, len(self.train_inputs), self.batch_size):
                # Get mini-batch
                inputs = self.train_inputs[i:i+self.batch_size].to(self.device)
                targets = self.train_targets[i:i+self.batch_size].to(self.device)
                # Forward pass
                outputs = self.phi_net(inputs)
                # Loss computation
                mse_loss = self.loss_fn(outputs, targets)
                #mse_loss = criterion(outputs, targets[0])
                # Regularization penalty for all trainable parameters
                #ridge_penalty = 0
                #for param in model.parameters():
                #    ridge_penalty += torch.norm(param, 2)
                #ridge_penalty *= lambda_ridge
                ridge_penalty = self.lambda_phi * self.l2_regularization_loss(self.phi_net.parameters())
                #ridge_penalty = lambda_ridge * (torch.norm(model.fc1.weight, 2) + torch.norm(model.fc2.weight, 2))
                loss = mse_loss + ridge_penalty
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
            epoch_loss = running_loss/len(self.train_inputs)
            self.losses.append(epoch_loss)
        
        return epoch_loss
    
    def eval(self):

        self.phi_net.eval() 
        # Predict using the trained model
        with torch.no_grad():
            phi_predictions = self.phi_net(self.test_inputs)
            eigenvals, _  = torch.linalg.eig(self.K)
            phi_Y_pred = torch.mm(phi_predictions.to(torch.complex128),
                                  torch.diag(eigenvals).to(self.device))
            # Loss computation
            predictions = torch.mm(phi_Y_pred, torch.transpose(self.Xi, 0, 1)).real
            squared_diff = (predictions - self.test_targets.to(self.device)) ** 2
            mse = torch.mean(squared_diff, axis=1)
            rmse = torch.sqrt(mse)
            #mse = criterion(predictions, test_targets)

        avg_mse = torch.sum(mse)/len(mse)
        avg_rmse = torch.sum(rmse)/len(rmse)
        anomaly_score_fl = torch.linalg.norm(self.test_targets.to(self.device) - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy())
        result_rmse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy())

        return predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def eval_new(self):

        self.phi_net.eval() 
        self.K.eval() 
        self.Xi.eval()
        # Predict using the trained model
        with torch.no_grad():
            phi_predictions = self.phi_net(self.test_inputs)
            phi_Y_pred = self.K(phi_predictions)
            # Loss computation
            predictions = self.Xi(phi_Y_pred)
            squared_diff = (predictions - self.test_targets.to(self.device)) ** 2
            mse = torch.mean(squared_diff, axis=1)
            #print(mse.shape)
            #rmse = torch.sqrt(mse)
            #mse = criterion(predictions, test_targets)

        avg_mse = torch.sum(mse)/len(mse)
        #avg_rmse = torch.sum(rmse)/len(rmse)
        avg_rmse = 0
        #anomaly_score_fl = torch.linalg.norm(self.test_targets - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(),
                                     print_or_not=False)
        #result_rmse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)
        result_rmse = 0

        return predictions, avg_mse, avg_rmse, result_mse, result_rmse
    
    """
    def eval_new(self, score="mse"):

        self.phi_net.eval() 
        self.K.eval() 
        self.Xi.eval()
        # Predict using the trained model
        with torch.no_grad():
            phi_predictions = self.phi_net(self.test_inputs)
            phi_Y_pred = self.K(phi_predictions)
            # Loss computation
            predictions = self.Xi(phi_Y_pred)
            squared_diff = (predictions - self.test_targets.to(self.device)) ** 2
            if score == "mse":
                mse = torch.mean(squared_diff, axis=1)
            else:
                rmse = torch.sqrt(mse)

        if score == "mse":
            avg_mse = torch.sum(mse)/len(mse)
            result_mse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(),
                                         print_or_not=False)
            avg_rmse = 0
            result_rmse = 0
        else:
            avg_rmse = torch.sum(rmse)/len(rmse)
            result_rmse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(),
                                          print_or_not=False)
            avg_mse = 0
            result_mse = 0

        return predictions, avg_mse, avg_rmse, result_mse, result_rmse
    """
    def eval_reservoir(self):

        self.phi_net.eval()
        # Predict using the trained model
        with torch.no_grad():
            predictions = self.phi_net(self.test_inputs)
            # Loss computation
            squared_diff = (predictions - self.test_targets) ** 2
            mse = torch.mean(squared_diff, axis=1)
            #print(mse.shape)
            #rmse = torch.sqrt(mse)
            #mse = criterion(predictions, test_targets)

        avg_mse = torch.sum(mse)/len(mse)
        #avg_rmse = torch.sum(rmse)/len(rmse)
        avg_rmse = 0
        #anomaly_score_fl = torch.linalg.norm(self.test_targets - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:],
                                     mse.detach().cpu().numpy(),
                                     print_or_not=False)
        #result_rmse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)
        result_rmse = 0

        return predictions, avg_mse, avg_rmse, result_mse, result_rmse
    
    def get_weights(self):
        """Return the current model weights."""
        return self.phi_net.state_dict(), self.K.state_dict(), self.Xi.state_dict()

    def set_weights(self, phi_weights, k_weights, xi_weights):
        """Set the model weights."""
        self.phi_net.load_state_dict(phi_weights)
        self.K.load_state_dict(k_weights)
        self.Xi.load_state_dict(xi_weights)
        


def start_gpu_monitoring():
    """Initialize NVML to monitor GPU memory."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU. Adjust the index for multiple GPUs.
    return handle

def get_gpu_memory_usage(handle):
    """Retrieve the current GPU memory usage."""
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 ** 2)  # Convert bytes to MB

def monitor_function_cpu_gpu_memory(func, *args, **kwargs):
    """Monitor and report the CPU and GPU memory usage of a function."""
    # Initialize GPU monitoring
    gpu_handle = start_gpu_monitoring()

    # Start monitoring CPU memory
    tracemalloc.start()

    # Record CPU and GPU memory before function execution
    cpu_start, _ = tracemalloc.get_traced_memory()
    gpu_start = get_gpu_memory_usage(gpu_handle)

    # Execute the function
    result = func(*args, **kwargs)

    # Record CPU and GPU memory after function execution
    cpu_end, _ = tracemalloc.get_traced_memory()
    gpu_end = get_gpu_memory_usage(gpu_handle)

    # Stop the CPU memory monitoring
    tracemalloc.stop()

    # Calculate the memory usage
    cpu_memory_used = (cpu_end - cpu_start) / 1024  # Convert to KB
    gpu_memory_used = gpu_end - gpu_start

    print(f"CPU Memory Used: {cpu_memory_used} KB")
    print(f"GPU Memory Used: {gpu_memory_used} MB")

    # Don't forget to shutdown NVML
    pynvml.nvmlShutdown()

    return result
