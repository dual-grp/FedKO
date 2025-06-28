from collections import Counter
from tqdm import tqdm
import copy
from utils.reservoir_koopman import *
from utils.ad_util import *
#from utils.memory_tracking import *
from client import Client
import time

class Server:
    def __init__(self, clients, test_data, label, model = 0, device = "cpu",
                feature_dim = 25, reservoir_dim = 512, output_dim = 128,
                alpha = 0.75, spectral_radius = 0.99, beta = 0.0,
                i_window = 256, o_window = 32, i_batch_size = 512, o_batch_size = 64, 
                inner_lr = 1e-3, outer_lr = 1e-3, lr_decay_factor = 0.8,
                subsampling = 0.1, glob_iters = 100, local_epochs = 10):
        
        self.device = device
        self.clients = clients
        self.glob_iters = glob_iters
        self.local_epochs = local_epochs
        self.num_client = len(self.clients)

        self.input_dim = feature_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.spectral_radius = spectral_radius
        self.beta = beta

        self.i_batch_size = i_batch_size
        self.o_batch_size = o_batch_size
        self.i_window = i_window
        self.o_window = o_window

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.lr_decay_factor = lr_decay_factor

        print("Using ", self.device)

        if model == 0:
            self.phi_net = PsiNN0(input_size = self.input_dim,
                                reservoir_size = self.reservoir_dim,
                                output_size = self.output_dim,
                                #device = device,
                                alpha = alpha, spectral_radius = spectral_radius,
                                device = self.device).to(self.device)
        elif model == 1:       
            self.phi_net = PsiNN1(input_size = self.input_dim,
                                  reservoir_size = self.reservoir_dim,
                                  output_size = self.output_dim,
                                  #device = device,
                                  alpha = alpha, spectral_radius = spectral_radius,
                                  device=self.device).to(self.device) 
                        
        #phi_net = EigenfunctionNN(input_dim, output_dim, hidden_dim).to(self.device)
        """
        self.K = torch.nn.Parameter(torch.randn(self.output_dim, self.output_dim,
                                   dtype=torch.double, device = device),
                                   requires_grad=True)

        self.Xi = torch.nn.Parameter(torch.randn(self.input_dim, self.output_dim,
                                    dtype=torch.complex128, device = device),
                                    requires_grad=True)
        """
        # Defining Koopman operator as a linear layer without bias

        self.K = KoopmanOperator(self.output_dim, spectral_radius=self.spectral_radius).to(self.device) 
        self.Xi = KoopmanMode(self.output_dim,
                              self.input_dim).to(self.device)

        total_size_1 = self.model_size(self.K) 
        total_size_2 = self.model_size(self.phi_net) + self.model_size(self.Xi)
        print(total_size_1)
        print(total_size_2)
              
        self.loss_fn = nn.MSELoss()
        self.factors = [0.25,0.1,10,5]

        self.client_per_round = int(subsampling * self.num_client)
        self.client_train_samples = [len(client.train_data) for client in self.clients]
        #print(self.client_train_samples)
        total_samples = sum(self.client_train_samples)
        self.update_weights = [num_samples / total_samples for num_samples in self.client_train_samples]
        #print(self.update_weights)
        self.test_data = test_data
        self.test_inputs = torch.tensor(test_data[:-1,:]).to(self.device)
        self.test_targets = torch.tensor(test_data[1:,:]).to(self.device)
        self.target = label 


    def model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_kb = (param_size + buffer_size) / 1024
        return size_all_kb

    def l2_regularization_loss(parameters):
        return torch.sum(torch.stack([torch.norm(p)**2 for p in parameters]))

    def linearity_loss(self, x_t, x_next):
        phi_xt = self.phi_net(x_t)
        phi_xnext_pred = torch.mm(phi_xt, self.K)
        phi_xnext_true = self.phi_net(x_next)
        loss = F.mse_loss(phi_xnext_pred, phi_xnext_true)
        reg_loss = self.lambda_K * self.l2_regularization_loss([self.K])\
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
        x_reconstructed = torch.mm(phi_x, torch.transpose(self.Xi, 0, 1))
        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(x, x_reconstructed)
        # Predict next state using Koopman operator
        # Here, we multiply the observables with Koopman modes first, then apply the Koopman operator (eigenvalues of K)
        #x_next_predicted = torch.mm(torch.mm(phi_x, Xi), K)
        #print(phi_X.shape)
        phi_Y_pred = torch.mm(phi_x, self.K)
        x_next_predicted = torch.mm(phi_Y_pred, torch.transpose(self.Xi, 0, 1))
        # Compute dynamics prediction loss
        dynamics_loss = F.mse_loss(y, x_next_predicted)
        # Regularization losses
        reg_loss_Xi = self.lambda_Xi * self.l2_regularization_loss([self.Xi])
        reg_loss_phi = self.lambda_phi * self.l2_regularization_loss(list(self.phi_net.parameters()))

        # Total loss
        total_loss = reconstruction_loss + dynamics_loss + reg_loss_Xi + reg_loss_phi
        
        return total_loss
    
    def train(self):

        train_losses = []
        test_losses = []
        component_losses = []
        self.psi_list = []
        self.Xi_list = []
        self.K_list = []
        
        for i in tqdm(range(self.glob_iters)):
            print(f"-------------- Round {i+1} ----------------")
            # Train each client
            chosen_idx = np.random.choice(self.num_client,
                                          replace=False,
                                          size = self.client_per_round)
            c_train_losses = []
            for idx in chosen_idx:
                start_time = time.time()
                c_train_loss = self.clients[idx].train(self.local_epochs)
                print("--- %s seconds ---" % (time.time() - start_time))
                c_train_losses.append(c_train_loss)

            total_train_loss = sum(c_train_losses)/len(c_train_losses)

            #self.average_weights(chosen_idx)
            self.aggregate_weights(chosen_idx)
            #total_test_loss, losses = self.evaluate_global_model()
            # Update the clients with the global model's weights
            self.update_clients()

            #print('-'*50, 'TEST', '-'*50)
            print(f"Average Train Loss : {total_train_loss}")
            #print(f"Total Test Loss : {total_test_loss:4f}")
            #print(epoch, total_test_loss)
            #print(f"Individual Loss : {losses}")
            print('-'*50)
            #test_losses.append(total_test_loss)
            train_losses.append(total_train_loss)
            #component_losses.append(losses)

            if (i+1) % 10 == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                self.K_list.append(copy.deepcopy(self.K))

        predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.evaluate_global_model()

        return train_losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def train_2(self):

        train_losses = []
        test_losses = []
        component_losses = []
        self.psi_list = []
        self.Xi_list = []
        self.K_list = []
        
        for i in tqdm(range(self.glob_iters)):
            print(f"-------------- Round {i+1} ----------------")
            # Train each client
            chosen_idx = np.random.choice(self.num_client,
                                          replace=False,
                                          size = self.client_per_round)
            c_inner_train_losses = []
            for idx in chosen_idx:
                c_inner_train_loss = self.clients[idx].train_inner(self.local_epochs)
                c_inner_train_losses.append(c_inner_train_loss)

            inner_train_loss = sum(c_inner_train_losses) / len(c_inner_train_losses)
            print(f"Inner Train Loss : {inner_train_loss}")
            #self.average_weights(chosen_idx)
            self.aggregate_phi(chosen_idx)
            self.aggregate_K(chosen_idx)
            # Update the clients with the global model's weights
            self.update_clients(phi = True, K = True, Xi = False)


            c_outer_train_losses = []
            for idx in chosen_idx:
                c_outer_train_loss = self.clients[idx].train_outer(self.local_epochs)
                c_outer_train_losses.append(c_outer_train_loss)

            outer_train_loss = sum(c_outer_train_losses) / len(c_outer_train_losses)
            print(f"Outer Train Loss : {outer_train_loss}")

            self.aggregate_phi(chosen_idx)
            self.aggregate_Xi(chosen_idx)
            self.update_clients(phi = True, K = False, Xi = True)

            print('-'*50)
            train_losses.append(outer_train_loss)

            if (i+1) % 10 == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                self.K_list.append(copy.deepcopy(self.K))

        predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.evaluate_global_model()

        return train_losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def train_inner(self):

        train_losses = []
        test_losses = []
        component_losses = []
        self.psi_list = []
        self.Xi_list = []
        self.K_list = []
        
        for i in tqdm(range(self.glob_iters)):
            print(f"-------------- Round {i+1} ----------------")
            # Train each client
            chosen_idx = np.random.choice(self.num_client,
                                          replace=False,
                                          size = self.client_per_round)
            c_inner_train_losses = []
            for idx in chosen_idx:
                c_inner_train_loss = self.clients[idx].train_inner(self.local_epochs)
                c_inner_train_losses.append(c_inner_train_loss)

            inner_train_loss = sum(c_inner_train_losses) / len(c_inner_train_losses)
            print(f"Inner Train Loss : {inner_train_loss}")
            #self.average_weights(chosen_idx)
            self.aggregate_phi(chosen_idx)
            self.aggregate_K(chosen_idx)
            # Update the clients with the global model's weights
            self.update_clients(phi = True, K = True, Xi = False)

            print('-'*50)
            train_losses.append(inner_train_loss)

            if (i+1) % 10 == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                self.K_list.append(copy.deepcopy(self.K))

        predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.evaluate_global_model()

        return train_losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse
        
    def train_standalone(self):

        train_losses = []
        c_train_losses = []
        all_results = []
        
        for i in tqdm(range(self.num_client)):
            print(f"-------------- Client {i+1} ----------------")
            # Train each client
            c_train_loss = self.clients[i].train(self.local_epochs)
            c_train_losses.append(c_train_loss)
            
            total_train_loss = sum(c_train_losses) / len(c_train_losses)

            #print('-'*50, 'TEST', '-'*50)
            print(f"Average Train Loss : {total_train_loss}")
            #print(epoch, total_test_loss)
            print('-'*50)

            train_losses.append(total_train_loss)

            predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.clients[i].eval()

            all_results.append(result_mse)
            #component_losses.append(losses)
        
        return all_results

    def train_standalone_new(self):

        train_losses = []
        c_train_losses = []
        all_results = []
        
        for i in tqdm(range(self.num_client)):
            print(f"-------------- Client {i+1} ----------------")
            # Train each client
            c_train_loss = self.clients[i].train_new_2(self.local_epochs,
                                                       inner = True,
                                                       outer = True)
            #c_train_losses.append(c_train_loss)
            
            #total_train_loss = sum(c_train_losses) / len(c_train_losses)

            #print('-'*50, 'TEST', '-'*50)
            #print(f"Average Train Loss : {total_train_loss}")
            #print(epoch, total_test_loss)
            #print('-'*50)

            #train_losses.append(total_train_loss)

            predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.clients[i].eval_new()
            all_results.append(result_mse)
            print(result_mse)
            #component_losses.append(losses)
        
        return all_results

    def train_reservoir(self):

        train_losses = []
        component_losses = []
        self.psi_list = []
        
        self.set_reservoir()
        
        for i in tqdm(range(self.glob_iters)):
            print(f"-------------- Round {i+1} ----------------")
            # Train each client
            chosen_idx = np.random.choice(self.num_client,
                                          replace=False,
                                          size = self.client_per_round)
            c_train_losses = []
            for idx in chosen_idx:
                c_train_loss = self.clients[idx].train_reservoir(self.local_epochs)
                c_train_losses.append(c_train_loss)

            train_loss = sum(c_train_losses)/len(c_train_losses)
            train_losses.append(train_loss)

            #self.average_weights(chosen_idx)
            self.aggregate_phi(chosen_idx, win=False, w=False)
            # Update the clients with the global model's weights
            self.update_clients(phi = True, K = False, Xi = False)

            print(f"Train Loss : {train_loss}")
                        
            if (i+1) % 10 == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                #self.Xi_list.append(copy.deepcopy(self.Xi))
                #self.K_list.append(copy.deepcopy(self.K))

        predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.eval_reservoir()

        return train_losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse
        
    def train_new(self, inner = True, outer = True, print_results = True):

        inner_train_losses = []
        outer_train_losses = []
        component_losses = []
        self.psi_list = []
        self.Xi_list = []
        self.K_list = []

        self.set_reservoir()
        
        for i in tqdm(range(self.glob_iters)):
            print(f"-------------- Round {i+1} ----------------")
            # Train each client
            chosen_idx = np.random.choice(self.num_client,
                                          replace=False,
                                          size = self.client_per_round)
            c_inner_train_losses = []
            c_outer_train_losses = []
            for idx in chosen_idx:
                c_inner_train_loss, c_outer_train_loss = self.clients[idx].train_new_2(self.local_epochs,
                                                                                        inner = inner,
                                                                                        outer = outer)
                c_inner_train_losses.append(c_inner_train_loss)
                c_outer_train_losses.append(c_outer_train_loss)

            inner_train_loss = sum(c_inner_train_losses) / len(c_inner_train_losses)
            inner_train_losses.append(inner_train_loss)
            outer_train_loss = sum(c_outer_train_losses) / len(c_outer_train_losses)
            outer_train_losses.append(outer_train_loss)

            #self.average_weights(chosen_idx)
            self.aggregate_phi(chosen_idx, iter = i, win=False, w=False)
            self.aggregate_K(chosen_idx, iter = i)
            self.aggregate_Xi(chosen_idx, iter = i)
            # Update the clients with the global model's weights
            self.update_clients(phi = True, K = True, Xi = True)

            if print_results:
                print(f"Inner Train Loss : {inner_train_loss}")
                print(f"Outer Train Loss : {outer_train_loss}")
                        
            if (i+1) % 10 == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                self.K_list.append(copy.deepcopy(self.K))

        predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.eval_new()

        return inner_train_losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def train_new_2(self, inner = True, outer = True):

        inner_train_losses = []
        outer_train_losses = []
        component_losses = []
        self.psi_list = []
        self.Xi_list = []
        self.K_list = []

        self.set_reservoir()
        
        for i in tqdm(range(self.glob_iters)):
            print(f"-------------- Round {i+1} ----------------")
            # Train each client
            chosen_idx = np.random.choice(self.num_client,
                                          replace=False,
                                          size = self.client_per_round)
            c_inner_train_losses = []
            c_outer_train_losses = []
            for idx in chosen_idx:
                #start_time = time.time()
                c_inner_train_loss, c_outer_train_loss = self.clients[idx].train_new_2(self.local_epochs,
                                                                                       inner = True, outer = False)
                #     
                #c_inner_train_loss, c_outer_train_loss = monitor_function_cpu_gpu_memory(self.clients[idx].train_new_2, 
                #                                                                        self.local_epochs,
                #                                                                        inner = True, 
                #                                                                        outer = False)                                                                 outer = False)
                #print("--- %s seconds ---" % (time.time() - start_time))
                c_inner_train_losses.append(c_inner_train_loss)
                #c_outer_train_losses.append(c_outer_train_loss)

            inner_train_loss = sum(c_inner_train_losses) / len(c_inner_train_losses)
            inner_train_losses.append(inner_train_loss)
            #outer_train_loss = sum(c_outer_train_losses) / len(c_outer_train_losses)
            #outer_train_losses.append(outer_train_loss)
            print(f"Inner Train Loss : {inner_train_loss}")
            #self.average_weights(chosen_idx)
            #self.aggregate_phi(chosen_idx, iter = i, win=False, w=False)
            self.aggregate_K(chosen_idx, iter = i)
            #self.aggregate_Xi(chosen_idx, iter = i)
            # Update the clients with the global model's weights
            self.update_clients(phi = False, K = True, Xi = False)

            for idx in chosen_idx:
                #start_time = time.time()
                c_inner_train_loss, c_outer_train_loss = self.clients[idx].train_new_2(self.local_epochs,
                                                                                       inner = False,
                                                                                       outer = True)
                #print("--- %s seconds ---" % (time.time() - start_time))
                #c_inner_train_losses.append(c_inner_train_loss)
                c_outer_train_losses.append(c_outer_train_loss)  

            outer_train_loss = sum(c_outer_train_losses) / len(c_outer_train_losses)
            outer_train_losses.append(outer_train_loss)
            print(f"Outer Train Loss : {outer_train_loss}")

            self.aggregate_phi(chosen_idx, iter = i, win=False, w=False)
            #self.aggregate_K(chosen_idx, iter = i)
            self.aggregate_Xi(chosen_idx, iter = i)
            # Update the clients with the global model's weights
            self.update_clients(phi = True, K = False, Xi = True)
            
            if (i+1) % 10 == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                self.K_list.append(copy.deepcopy(self.K))

        predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.eval_new()

        print(f"Performance: {result_mse}")
        
        return inner_train_losses, outer_train_losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def train_new_inner(self, inner = True, outer = True):

        inner_train_losses = []
        outer_train_losses = []
        component_losses = []
        self.psi_list = []
        self.Xi_list = []
        self.K_list = []

        self.set_reservoir()
        
        for i in tqdm(range(self.glob_iters)):
            print(f"-------------- Round {i+1} ----------------")
            # Train each client
            chosen_idx = np.random.choice(self.num_client,
                                          replace=False,
                                          size = self.client_per_round)
            c_inner_train_losses = []
            c_outer_train_losses = []
            for idx in chosen_idx:
                #start_time = time.time()
                c_inner_train_loss, c_outer_train_loss = self.clients[idx].train_new_2(self.local_epochs,
                                                                                       inner = True, outer = False)
                #     
                #c_inner_train_loss, c_outer_train_loss = monitor_function_cpu_gpu_memory(self.clients[idx].train_new_2, 
                #                                                                        self.local_epochs,
                #                                                                        inner = True, 
                #                                                                        outer = False)                                                                 outer = False)
                #print("--- %s seconds ---" % (time.time() - start_time))
                c_inner_train_losses.append(c_inner_train_loss)
                #c_outer_train_losses.append(c_outer_train_loss)

            inner_train_loss = sum(c_inner_train_losses) / len(c_inner_train_losses)
            inner_train_losses.append(inner_train_loss)
            #outer_train_loss = sum(c_outer_train_losses) / len(c_outer_train_losses)
            #outer_train_losses.append(outer_train_loss)
            print(f"Inner Train Loss : {inner_train_loss}")
            #self.average_weights(chosen_idx)
            #self.aggregate_phi(chosen_idx, iter = i, win=False, w=False)
            self.aggregate_K(chosen_idx, iter = i)
            #self.aggregate_Xi(chosen_idx, iter = i)
            # Update the clients with the global model's weights
            self.update_clients(phi = False, K = True, Xi = False)

            """
            for idx in chosen_idx:
                #start_time = time.time()
                c_inner_train_loss, c_outer_train_loss = self.clients[idx].train_new_2(self.local_epochs,
                                                                                       inner = False,
                                                                                       outer = True)
                #print("--- %s seconds ---" % (time.time() - start_time))
                #c_inner_train_losses.append(c_inner_train_loss)
                c_outer_train_losses.append(c_outer_train_loss)  

            outer_train_loss = sum(c_outer_train_losses) / len(c_outer_train_losses)
            outer_train_losses.append(outer_train_loss)
            print(f"Outer Train Loss : {outer_train_loss}")

            self.aggregate_phi(chosen_idx, iter = i, win=False, w=False)
            #self.aggregate_K(chosen_idx, iter = i)
            self.aggregate_Xi(chosen_idx, iter = i)
            # Update the clients with the global model's weights
            self.update_clients(phi = True, K = False, Xi = True)
            """
            
            if (i+1) % 10 == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                self.K_list.append(copy.deepcopy(self.K))

        predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.eval_new()

        print(f"Performance: {result_mse}")

        return inner_train_losses, outer_train_losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse


    def train_new_outer(self, inner = False, outer = True):

        inner_train_losses = []
        outer_train_losses = []
        component_losses = []
        self.psi_list = []
        self.Xi_list = []
        self.K_list = []

        self.set_reservoir()
        
        for i in tqdm(range(self.glob_iters)):
            print(f"-------------- Round {i+1} ----------------")
            # Train each client
            chosen_idx = np.random.choice(self.num_client,
                                          replace=False,
                                          size = self.client_per_round)
            c_inner_train_losses = []
            c_outer_train_losses = []
            """
            for idx in chosen_idx:
                #start_time = time.time()
                c_inner_train_loss, c_outer_train_loss = self.clients[idx].train_new_2(self.local_epochs,
                                                                                       inner = True, outer = False)
                #     
                #c_inner_train_loss, c_outer_train_loss = monitor_function_cpu_gpu_memory(self.clients[idx].train_new_2, 
                #                                                                        self.local_epochs,
                #                                                                        inner = True, 
                #                                                                        outer = False)                                                                 outer = False)
                #print("--- %s seconds ---" % (time.time() - start_time))
                c_inner_train_losses.append(c_inner_train_loss)
                #c_outer_train_losses.append(c_outer_train_loss)

            inner_train_loss = sum(c_inner_train_losses) / len(c_inner_train_losses)
            inner_train_losses.append(inner_train_loss)
            #outer_train_loss = sum(c_outer_train_losses) / len(c_outer_train_losses)
            #outer_train_losses.append(outer_train_loss)
            print(f"Inner Train Loss : {inner_train_loss}")
            #self.average_weights(chosen_idx)
            #self.aggregate_phi(chosen_idx, iter = i, win=False, w=False)
            self.aggregate_K(chosen_idx, iter = i)
            #self.aggregate_Xi(chosen_idx, iter = i)
            # Update the clients with the global model's weights
            self.update_clients(phi = False, K = True, Xi = False)

            """
            for idx in chosen_idx:
                #start_time = time.time()
                c_inner_train_loss, c_outer_train_loss = self.clients[idx].train_new_2(self.local_epochs,
                                                                                       inner = False,
                                                                                       outer = True)
                #print("--- %s seconds ---" % (time.time() - start_time))
                #c_inner_train_losses.append(c_inner_train_loss)
                c_outer_train_losses.append(c_outer_train_loss)  

            outer_train_loss = sum(c_outer_train_losses) / len(c_outer_train_losses)
            outer_train_losses.append(outer_train_loss)
            print(f"Outer Train Loss : {outer_train_loss}")

            self.aggregate_phi(chosen_idx, iter = i, win=False, w=False)
            #self.aggregate_K(chosen_idx, iter = i)
            self.aggregate_Xi(chosen_idx, iter = i)
            # Update the clients with the global model's weights
            self.update_clients(phi = True, K = False, Xi = True)
            
            if (i+1) % 10 == 0:
                self.psi_list.append(copy.deepcopy(self.phi_net))
                self.Xi_list.append(copy.deepcopy(self.Xi))
                self.K_list.append(copy.deepcopy(self.K))

        predictions, avg_mse, avg_rmse, result_mse, result_rmse = self.eval_new()

        print(f"Performance: {result_mse}")

        return inner_train_losses, outer_train_losses, predictions, avg_mse, avg_rmse, result_mse, result_rmse

            
    def evaluate_global_model(self):

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
        anomaly_score_fl = torch.linalg.norm(self.test_targets - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)
        result_rmse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)


        return predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def eval_new(self):

        print("Evaluation....")
        self.phi_net.eval() 
        self.K.eval() 
        self.Xi.eval()
        # Predict using the trained model
        with torch.no_grad():
            infer_time = time.time()
            phi_predictions = self.phi_net(self.test_inputs)
            phi_Y_pred = self.K(phi_predictions)
            # Loss computation
            predictions = self.Xi(phi_Y_pred)
            #print(len(predictions))
            #print("--- %s seconds ---" % (time.time() - infer_time)*1000)
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

    def eval_eigenvalues(self):

        self.phi_net.eval() 
        self.K.eval() 
        self.Xi.eval()

        eigenvalues, _ = self.K.spectral_decomposition()
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
        #print(predictions.shape)
        #print(mse.shape)
        avg_mse = torch.sum(mse)/len(mse)
        #avg_rmse = torch.sum(rmse)/len(rmse)
        avg_rmse = 0
        #anomaly_score_fl = torch.linalg.norm(self.test_targets - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(),
                                     print_or_not=False)
        #result_rmse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)
        result_rmse = 0

        return predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def eval_reservoir_new(self, test_size = 2048):

        self.phi_net.eval()
        # Predict using the trained model
        with torch.no_grad():

            batch_mse = []
            batch_rmse = []

            for i in range(0, len(self.test_inputs), test_size):
                test_batch = self.test_inputs[i:i+test_size].to(self.device)
                test_target_batch = self.test_targets[i:i+test_size].to(self.device)
                predictions = self.phi_net(test_batch)
                # Loss computation
                squared_diff = (predictions - test_target_batch) ** 2
                b_mse = torch.mean(squared_diff, axis=1)
                batch_mse.append(b_mse)
                #print(mse.shape)
                #b_rmse = torch.sqrt(b_mse)
                #batch_rmse.append(b_rmse)
                #mse = criterion(predictions, test_targets)
            mse = torch.cat(batch_mse)
            #rmse = torch.cat(batch_rmse)
        #print(predictions.shape)
        #print(mse.shape)
        avg_mse = torch.sum(mse)/len(mse)
        #avg_rmse = torch.sum(rmse)/len(rmse)
        #anomaly_score_fl = torch.linalg.norm(self.test_targets - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)
        #result_rmse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)
        return predictions, avg_mse, result_mse
        
    def eval_iter(self, iter = -1):

        self.psi_list[iter].eval() 
        # Predict using the trained model
        with torch.no_grad():
            phi_predictions = self.psi_list[iter](self.test_inputs)
            eigenvals, _  = torch.linalg.eig(self.K_list[iter])
            phi_Y_pred = torch.mm(phi_predictions.to(torch.complex128),
                                  torch.diag(eigenvals).to(self.device))
            # Loss computation
            predictions = torch.mm(phi_Y_pred, torch.transpose(self.Xi_list[iter], 0, 1)).real
            squared_diff = (predictions - self.test_targets.to(self.device)) ** 2
            mse = torch.mean(squared_diff, axis=1)
            rmse = torch.sqrt(mse)
            #mse = criterion(predictions, test_targets)

        avg_mse = torch.sum(mse)/len(mse)
        avg_rmse = torch.sum(rmse)/len(rmse)
        anomaly_score_fl = torch.linalg.norm(self.test_targets.to(self.device) - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)
        result_rmse = get_threshold_2(self.target[1:], mse.detach().cpu().numpy(), print_or_not=False)

        return predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def eval_reservoir_long_term(self, num_step = 1000):

        self.phi_net.eval() 
        self.K.eval() 
        self.Xi.eval()
        # Predict using the trained model
        with torch.no_grad():
            test_0 = self.test_inputs[0:1]
            predictions = []
            squared_diff = []
            for i in range(0, num_step):
                y_pred = self.phi_net(test_0)
                #phi_y_pred = self.K(phi_x_pred)
                # Loss computation
                #y_pred = self.Xi(phi_y_pred)
                #print(y_pred.shape)
                predictions.append(y_pred[0])
                #test_0 = copy.copy(y_pred)
                #error = (y_pred - self.test_targets[i]) ** 2
                test_0 = copy.copy(y_pred)
                #squared_diff.append(error)
            predictions = torch.stack(predictions)
            print(predictions.shape)
            squared_diff = (predictions - self.test_targets[:num_step]) ** 2
            mse = torch.mean(squared_diff, axis=1)
            print(mse.shape)
            #print(mse.shape)
            rmse = torch.sqrt(mse)
            #mse = criterion(predictions, test_targets)

        avg_mse = torch.sum(mse)/len(mse)
        avg_rmse = torch.sum(rmse)/len(rmse)
        #anomaly_score_fl = torch.linalg.norm(self.test_targets - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:num_step+1], mse.detach().cpu().numpy(), print_or_not=False)
        result_rmse = get_threshold_2(self.target[1:num_step+1], mse.detach().cpu().numpy(), print_or_not=False)

        return predictions, avg_mse, avg_rmse, result_mse, result_rmse

    def eval_long_term(self, index = 0, num_step = 1000):

        self.phi_net.eval() 
        self.K.eval() 
        self.Xi.eval()
        # Predict using the trained model
        with torch.no_grad():
            test_0 = self.test_inputs[index:index+1]
            predictions = []
            for i in range(index, num_step):
                phi_x_pred = self.phi_net(test_0)
                phi_y_pred = self.K(phi_x_pred)
                # Loss computation
                y_pred = self.Xi(phi_y_pred)
                #print(y_pred.shape)
                predictions.append(y_pred[0])
                test_0 = copy.copy(y_pred)
            
            predictions = torch.stack(predictions)
            squared_diff = (predictions - self.test_targets[:num_step]) ** 2
            mse = torch.mean(squared_diff, axis=1)
            #print(mse.shape)
            rmse = torch.sqrt(mse)
            #mse = criterion(predictions, test_targets)
        print(predictions.shape)
        print(mse.shape)
        avg_mse = torch.sum(mse)/len(mse)
        avg_rmse = torch.sum(rmse)/len(rmse)
        #anomaly_score_fl = torch.linalg.norm(self.test_targets - predictions, axis=1)**2
        result_mse = get_threshold_2(self.target[1:num_step+1], mse.detach().cpu().numpy(), print_or_not=False)
        result_rmse = get_threshold_2(self.target[1:num_step+1], mse.detach().cpu().numpy(), print_or_not=False)

        return predictions, avg_mse, avg_rmse, result_mse, result_rmse


    def aggregate_phi(self, selected_clients, iter = 0, win = True, w = True):
        """
        Compute weighted average of model weights from selected clients based on the number of samples each client has.
        Args:
        - selected_clients: list of indices of clients that have been selected for training in the current round.
        """
        #print("Averaging.....")
        # Extract number of samples of selected clients
        ns = [self.client_train_samples[i] for i in selected_clients] 

        # Extract weights of selected clients
        client_ae_weights = [self.clients[i].phi_net.state_dict() for i in selected_clients]
        server_phi_weights = self.phi_net.state_dict()
        # Initialize w_avg with deep copy of first selected client's weights
        avg_ae_weights = copy.deepcopy(client_ae_weights[0]) 
        #print(avg_ae_weights.keys())   
        for key in avg_ae_weights.keys():
            #print(key)
            # Multiply first client's weights with its number of samples
            avg_ae_weights[key] *= ns[0]         
            # Accumulate weighted sum of weights from rest of the selected clients
            for i in range(1, len(client_ae_weights)):
                avg_ae_weights[key] += ns[i] * client_ae_weights[i][key]            
            # Divide by total number of samples of selected clients to get the weighted average
            avg_ae_weights[key] = torch.div(avg_ae_weights[key], sum(ns))       
        # Update the global model's weights with the computed weighted average

        if self.beta != 0.0 and iter != 0:
            for key in avg_ae_weights.keys():
                avg_ae_weights[key] = self.beta * server_phi_weights[key] + (1 - self.beta) * avg_ae_weights[key]

        self.phi_net.load_state_dict(avg_ae_weights)

        if win:
            # Aggregate Win Layer
            client_phi_Win = [self.clients[i].phi_net.Win.data for i in selected_clients]
            server_phi_Win = self.phi_net.Win.data
            avg_phi_Win = copy.deepcopy(client_phi_Win[0]) 
            # Multiply first client's weights with its number of samples
            avg_phi_Win *= ns[0]         
            # Accumulate weighted sum of weights from rest of the selected clients
            for i in range(1, len(client_phi_Win)):
                avg_phi_Win += ns[i] * client_phi_Win[i]       
            # Divide by total number of samples of selected clients to get the weighted average
            avg_phi_Win = torch.div(avg_phi_Win, sum(ns))

            if self.beta != 0.0 and iter != 0:
                avg_phi_Win = self.beta * server_phi_Win + (1- self.beta) * avg_phi_Win

            self.phi_net.Win.data = avg_phi_Win

        if w:
            ## Aggregate Reservoir Layer
            client_phi_W = [self.clients[i].phi_net.W.data for i in selected_clients]
            avg_phi_W = copy.deepcopy(client_phi_W[0])
            server_phi_W = self.phi_net.W.data
            # Multiply first client's weights with its number of samples
            avg_phi_W *= ns[0]         
            # Accumulate weighted sum of weights from rest of the selected clients
            for i in range(1, len(client_phi_W)):
                avg_phi_W += ns[i] * client_phi_W[i]       
            # Divide by total number of samples of selected clients to get the weighted average
            avg_phi_W = torch.div(avg_phi_W, sum(ns))

            if self.beta != 0.0 and iter != 0:
                avg_phi_W = self.beta * server_phi_W + (1- self.beta) * avg_phi_W
            self.phi_net.W.data = avg_phi_W

        return 0

    def aggregate_K(self, selected_clients, iter = 0):
        """
        Compute weighted average of model weights from selected clients based on the number of samples each client has.
        Args:
        - selected_clients: list of indices of clients that have been selected for training in the current round.
        """
        #print("Averaging.....")
        # Extract number of samples of selected clients
        ns = [self.client_train_samples[i] for i in selected_clients] 

        ################ Global Koopman #########################
        client_kpm_weights = [self.clients[i].K.state_dict() for i in selected_clients]
        server_kpm_weights = self.K.state_dict()

        avg_kpm_weights = copy.deepcopy(client_kpm_weights[0])
        #print(avg_kpm_weights.keys())
        #print(avg_kpm_weights['weights'][0])
        for key in avg_kpm_weights.keys():
            #print(key)
            avg_kpm_weights[key] *= ns[0]
            # Accumulate weighted sum of weights from rest of the selected clients
            for i in range(1, len(client_kpm_weights)):
                avg_kpm_weights[key] += ns[i] * client_kpm_weights[i][key]
            # Divide by total number of samples of selected clients to get the weighted average
            avg_kpm_weights[key] = torch.div(avg_kpm_weights[key], sum(ns))
        # Update the global model's weights with the computed weighted average
        if self.beta != 0.0 and iter != 0:
            for key in avg_kpm_weights.keys():
                avg_kpm_weights[key] = self.beta * server_kpm_weights[key] + (1 - self.beta) * avg_kpm_weights[key]

        self.K.load_state_dict(avg_kpm_weights)

        return 0

    def aggregate_Xi(self, selected_clients, iter = 0):
        """
        Compute weighted average of model weights from selected clients based on the number of samples each client has.
        Args:
        - selected_clients: list of indices of clients that have been selected for training in the current round.
        """
        #print("Averaging.....")
        # Extract number of samples of selected clients
        ns = [self.client_train_samples[i] for i in selected_clients] 

        ################ Global Koopman #########################
        client_xi_weights = [self.clients[i].Xi.state_dict() for i in selected_clients]
        server_xi_weights = self.Xi.state_dict()
        avg_xi_weights = copy.deepcopy(client_xi_weights[0])
        #print(avg_kpm_weights.keys())
        #print(avg_kpm_weights['weights'][0])
        for key in avg_xi_weights.keys():
            #print(key)
            avg_xi_weights[key] *= ns[0]
            # Accumulate weighted sum of weights from rest of the selected clients
            for i in range(1, len(client_xi_weights)):
                avg_xi_weights[key] += ns[i] * client_xi_weights[i][key]
            # Divide by total number of samples of selected clients to get the weighted average
            avg_xi_weights[key] = torch.div(avg_xi_weights[key], sum(ns))
            # Update the global model's weights with the computed weighted average
        if self.beta != 0.0 and iter != 0:
            for key in avg_xi_weights.keys():
                avg_xi_weights[key] = self.beta * server_xi_weights[key] + (1 - self.beta) * avg_xi_weights[key]

        self.Xi.load_state_dict(avg_xi_weights)

        return 0

    def aggregate_weights(self, selected_clients):
        """
        Compute weighted average of model weights from selected clients based on the number of samples each client has.
        Args:
        - selected_clients: list of indices of clients that have been selected for training in the current round.
        """
        #print("Averaging.....")
        # Extract number of samples of selected clients
        ns = [self.client_train_samples[i] for i in selected_clients] 

        # Extract weights of selected clients
        client_ae_weights = [self.clients[i].phi_net.state_dict() for i in selected_clients]
        # Initialize w_avg with deep copy of first selected client's weights
        avg_ae_weights = copy.deepcopy(client_ae_weights[0]) 
        #print(avg_ae_weights.keys())   
        for key in avg_ae_weights.keys():
            # Multiply first client's weights with its number of samples
            avg_ae_weights[key] *= ns[0]         
            # Accumulate weighted sum of weights from rest of the selected clients
            for i in range(1, len(client_ae_weights)):
                avg_ae_weights[key] += ns[i] * client_ae_weights[i][key]            
            # Divide by total number of samples of selected clients to get the weighted average
            avg_ae_weights[key] = torch.div(avg_ae_weights[key], sum(ns))       
        # Update the global model's weights with the computed weighted average
        self.phi_net.load_state_dict(avg_ae_weights)

        ################ Global Koopman #########################
        client_kpm_weights = [self.clients[i].K.data for i in selected_clients]
        avg_kpm_weights = copy.deepcopy(client_kpm_weights[0])
        #print(avg_kpm_weights.keys())
        #print(avg_kpm_weights['weights'][0])
        avg_kpm_weights *= ns[0]
        # Accumulate weighted sum of weights from rest of the selected clients
        for i in range(1, len(client_kpm_weights)):
            avg_kpm_weights += ns[i] * client_kpm_weights[i]
        # Divide by total number of samples of selected clients to get the weighted average
        avg_kpm_weights = torch.div(avg_kpm_weights, sum(ns))
        # Update the global model's weights with the computed weighted average
        self.K.data = avg_kpm_weights

        upper_bound = self.K.sum(dim=1).max().item()

        if upper_bound > 1 : 
            #self.weight /= upper_bound
            self.K.data /= upper_bound

        ################ Global Koopman #########################
        client_xi_weights = [self.clients[i].Xi.data for i in selected_clients]
        avg_xi_weights = copy.deepcopy(client_xi_weights[0])
        #print(avg_kpm_weights.keys())
        #print(avg_kpm_weights['weights'][0])
        avg_xi_weights *= ns[0]
        # Accumulate weighted sum of weights from rest of the selected clients
        for i in range(1, len(client_xi_weights)):
            avg_xi_weights += ns[i] * client_xi_weights[i]
        # Divide by total number of samples of selected clients to get the weighted average
        avg_xi_weights = torch.div(avg_xi_weights, sum(ns))
        # Update the global model's weights with the computed weighted average
        self.Xi.data = avg_xi_weights

    def update_clients(self, phi = True, K = True, Xi = True):
        """
        Update each client's model with the global model's weights.
        """
        ae_weights = self.phi_net.state_dict()
        kmp_weights = self.K.state_dict()
        xi_weights = self.Xi.state_dict()
        for client in self.clients:
            if phi:
                client.phi_net.load_state_dict(ae_weights)
                client.phi_net.W.data = self.phi_net.W.data
                client.phi_net.Win.data = self.phi_net.Win.data
            if K:
                client.K.load_state_dict(kmp_weights)
            if Xi:
                client.Xi.load_state_dict(xi_weights)

    def set_reservoir(self):
        """
        Update each client's model with the global reservoir's weights.
        """
        for client in self.clients:
            client.phi_net.W.data = self.phi_net.W.data
            client.phi_net.Win.data = self.phi_net.Win.data
# This is just the basic structure. Actual implementation might require more details based on the specific use-case.