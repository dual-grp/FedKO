import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PsiNN0(nn.Module):
    """
    A Reservoir of for Echo State Networks

    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation.
            Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix.
            Default: `'normal'`
        net_gain_and_bias: if ``True``, the network uses additional ``g`` (gain)
            and ``b`` (bias) parameters. Default: ``False``
    """
    def __init__(self, input_size = 25, reservoir_size = 512, output_size = 50, 
                 win_initializer= "uniform", res_initializer = "normal", 
                 alpha=0.5, spectral_radius=0.99, device = "cpu"):
        super(PsiNN0, self).__init__()
        
        # Parameters
        self.device = device
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.alpha = alpha

        # Input weights initialization
        self.Win = torch.Tensor(reservoir_size, input_size).to(self.device)
        if win_initializer == 'uniform':
            nn.init.uniform_(self.Win, -0.1, 0.1)
        elif win_initializer == 'normal':
            nn.init.normal_(self.Win, mean=0, std=0.1)
        elif win_initializer == 'xavier':
            nn.init.xavier_uniform_(self.Win)
        elif win_initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.Win, nonlinearity='relu')
        
        self.W = torch.Tensor(reservoir_size, reservoir_size).to(self.device)
        # Reservoir weights initialization
        if res_initializer == 'uniform':
            nn.init.uniform_(self.W, -0.1, 0.1)
        elif res_initializer == 'normal':
            nn.init.normal_(self.W, mean=0, std=0.1)
        elif res_initializer == 'xavier':
            nn.init.xavier_uniform_(self.W)
        elif res_initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.W, nonlinearity='relu')

        # Scaling the reservoir weights to have spectral radius near 1 (for echo state property)
        eigenvalues = torch.linalg.eigvals(self.W)
        spectral_radius_actual = torch.max(torch.abs(eigenvalues))
        self.W = self.W / spectral_radius_actual * spectral_radius
  
        # Output weights (only these will be trained)
        self.Wout = nn.Parameter(torch.randn(output_size, reservoir_size) * 0.1)
        
    def forward(self, x):
        """
        x: Input tensor of shape (sequence_length, input_size)
        """
        sequence_length, _ = x.size()
        
        # Initial reservoir state
        reservoir_state = torch.zeros(self.reservoir_size).to(self.device)
        
        # To store reservoir states for each time step
        reservoir_states = []
        
        for t in range(sequence_length):
            reservoir_state = (1 - self.alpha) * reservoir_state + self.alpha * torch.tanh(
                torch.mv(self.Win, x[t, :]) + torch.mv(self.W, reservoir_state)
            )
            reservoir_states.append(reservoir_state)
        
        # Convert reservoir states list to tensor
        reservoir_states_tensor = torch.stack(reservoir_states)
        
        # Output computation
        output = torch.mm(reservoir_states_tensor, self.Wout.T)
        
        return output

class PsiNN(nn.Module):
    """
    A Reservoir of for Echo State Networks

    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation.
            Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix.
            Default: `'normal'`
        net_gain_and_bias: if ``True``, the network uses additional ``g`` (gain)
            and ``b`` (bias) parameters. Default: ``False``
    """
    def __init__(self, input_size = 25, reservoir_size = 512, output_size = 50, 
                 win_initializer= "uniform", res_initializer = "normal", 
                 alpha=0.5, spectral_radius=0.99, device = "cpu"):
        super(PsiNN, self).__init__()
        
        # Parameters
        self.device = device
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.alpha = alpha

        # Input weights initialization
        self.Win = torch.Tensor(reservoir_size, input_size).to(self.device)
        if win_initializer == 'uniform':
            nn.init.uniform_(self.Win, -0.1, 0.1)
        elif win_initializer == 'normal':
            nn.init.normal_(self.Win, mean=0, std=0.1)
        elif win_initializer == 'xavier':
            nn.init.xavier_uniform_(self.Win)
        elif win_initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.Win, nonlinearity='relu')
        
        self.W = torch.Tensor(reservoir_size, reservoir_size).to(self.device)
        # Reservoir weights initialization
        if res_initializer == 'uniform':
            nn.init.uniform_(self.W, -0.1, 0.1)
        elif res_initializer == 'normal':
            nn.init.normal_(self.W, mean=0, std=0.1)
        elif res_initializer == 'xavier':
            nn.init.xavier_uniform_(self.W)
        elif res_initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.W, nonlinearity='relu')

        # Scaling the reservoir weights to have spectral radius near 1 (for echo state property)
        eigenvalues = torch.linalg.eigvals(self.W)
        spectral_radius_actual = torch.max(torch.abs(eigenvalues))
        self.W = self.W / spectral_radius_actual * spectral_radius
  
        # Output weights (only these will be trained)
        self.Wout = nn.Parameter(torch.randn(output_size, reservoir_size) * 0.1)
        
    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, sequence_length, input_size)
        """
        batch_size, sequence_length, _ = x.size()
        
        # Initialize reservoir states for the batch
        reservoir_states_batch = torch.zeros(batch_size, self.reservoir_size).to(self.device)
        
        # To store reservoir states for each time step and each batch item
        reservoir_states_all = []
        
        for t in range(sequence_length):
            updated_states = []
            for b in range(batch_size):
                reservoir_state = reservoir_states_batch[b, :]
                reservoir_state = (1 - self.alpha) * reservoir_state + self.alpha * torch.tanh(
                    torch.mv(self.Win, x[b, t, :]) + torch.mv(self.W, reservoir_state)
                )
                updated_states.append(reservoir_state)

            # Update the batch states
            reservoir_states_batch = torch.stack(updated_states)
            reservoir_states_all.append(reservoir_states_batch)
        
        # Convert list of reservoir states to tensor
        reservoir_states_tensor = torch.stack(reservoir_states_all, dim=1)
        
        # Output computation across batch
        output = torch.matmul(reservoir_states_tensor, self.Wout.T)
        
        return output
  

class PsiNN1(nn.Module):
    """
    A Reservoir of for Echo State Networks

    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation.
            Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix.
            Default: `'normal'`
        net_gain_and_bias: if ``True``, the network uses additional ``g`` (gain)
            and ``b`` (bias) parameters. Default: ``False``
    """
    def __init__(self, input_size = 25, reservoir_size = 512, output_size = 50, 
                 win_initializer= "uniform", res_initializer = "normal", 
                 alpha=0.5, spectral_radius=0.99, device = "cpu"):
        super(PsiNN1, self).__init__()
        
        # Parameters
        self.device = device
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.alpha = alpha

        # Input weights initialization
        self.Win = torch.Tensor(reservoir_size, input_size).to(self.device)
        if win_initializer == 'uniform':
            nn.init.uniform_(self.Win, -0.1, 0.1)
        elif win_initializer == 'normal':
            nn.init.normal_(self.Win, mean=0, std=0.1)
        elif win_initializer == 'xavier':
            nn.init.xavier_uniform_(self.Win)
        elif win_initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.Win, nonlinearity='relu')
        
        self.W = torch.Tensor(reservoir_size, reservoir_size).to(self.device)
        # Reservoir weights initialization
        if res_initializer == 'uniform':
            nn.init.uniform_(self.W, -0.1, 0.1)
        elif res_initializer == 'normal':
            nn.init.normal_(self.W, mean=0, std=0.1)
        elif res_initializer == 'xavier':
            nn.init.xavier_uniform_(self.W)
        elif res_initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.W, nonlinearity='relu')

        # Scaling the reservoir weights to have spectral radius near 1 (for echo state property)
        eigenvalues = torch.linalg.eigvals(self.W)
        spectral_radius_actual = torch.max(torch.abs(eigenvalues))
        self.W = self.W / spectral_radius_actual * spectral_radius
  
        # Output weights (only these will be trained)
        self.fc1 = nn.Linear(reservoir_size, 128)
        self.fc2 = nn.Linear(128, output_size)

        #self.Wout = nn.Parameter(torch.randn(output_size, reservoir_size) * 0.1)
        
    def forward(self, x):
        """
        x: Input tensor of shape (sequence_length, input_size)
        """
        batch_size, sequence_length, _ = x.size()
        
        # Initial reservoir state
        reservoir_state = torch.zeros(self.reservoir_size).to(self.device)
        
        # To store reservoir states for each time step
        reservoir_states = []
        
        for t in range(sequence_length):
            reservoir_state = (1 - self.alpha) * reservoir_state + self.alpha * torch.tanh(
                torch.mv(self.Win, x[t, :]) + torch.mv(self.W, reservoir_state)
            )
            reservoir_states.append(reservoir_state)
        
        # Convert reservoir states list to tensor
        reservoir_states_tensor = torch.stack(reservoir_states)
        
        # Output computation
        # Pass through the 2-layer MLP
        koopman_representation = self.fc1(reservoir_states_tensor)
        output = self.fc2(koopman_representation)
        #output = torch.mm(reservoir_states_tensor, self.Wout.T)
        
        return output
    
class KoopmanOperator0(nn.Module):
    def __init__(self, operator_dim, spectral_radius):
        super(KoopmanOperator0, self).__init__()
        self.K_operator = nn.Parameter(torch.randn(operator_dim, operator_dim) * 0.1)
        self.spectral_radius = spectral_radius

    def spectral_decomposition(self):
        eigenvalues = torch.linalg.eigvals(self.K_operator.data)
        spectral_radius_actual = torch.max(torch.abs(eigenvalues))
        return eigenvalues, spectral_radius_actual
    
    def forward(self, x, eigen = False):

        #eigenvalues = torch.linalg.eigvals(self.K_operator.data)
        #spectral_radius_actual = torch.max(torch.abs(eigenvalues))
        eigenvalues, spectral_radius_actual = self.spectral_decomposition()
        self.K_operator.data = self.K_operator.data / spectral_radius_actual * self.spectral_radius
        """
        upper_bound = self.K_operator.sum(dim=1).max().item()
        if upper_bound > 1 : 
            #self.weight /= upper_bound
            self.K_operator.data /= upper_bound
        """
        if eigen:
            return torch.mm(x, torch.diag(eigenvalues))
        
        return torch.mm(x, self.K_operator)

class KoopmanOperator(nn.Module):
    def __init__(self, operator_dim, spectral_radius):
        super(KoopmanOperator, self).__init__()
        self.K_operator = nn.Parameter(torch.randn(operator_dim, operator_dim) * 0.1)
        self.spectral_radius = spectral_radius

    def spectral_decomposition(self):
        eigenvalues = torch.linalg.eigvals(self.K_operator.data)
        spectral_radius_actual = torch.max(torch.abs(eigenvalues))
        return eigenvalues, spectral_radius_actual
    
    def forward(self, x):

        #eigenvalues = torch.linalg.eigvals(self.K_operator.data)
        #spectral_radius_actual = torch.max(torch.abs(eigenvalues))
        eigenvalues, spectral_radius_actual = self.spectral_decomposition()
        self.K_operator.data = self.K_operator.data / spectral_radius_actual * self.spectral_radius
        """
        upper_bound = self.K_operator.sum(dim=1).max().item()
        if upper_bound > 1 : 
            #self.weight /= upper_bound
            self.K_operator.data /= upper_bound
        """

        return torch.matmul(x, self.K_operator)

class KoopmanMode0(nn.Module):
    def __init__(self, operator_dim, state_dim):
        super(KoopmanMode, self).__init__()
        self.Xi_operator = nn.Parameter(torch.randn(state_dim, operator_dim) * 0.1)
        
        """
        torch.nn.Parameter(torch.randn(self.input_dim, self.output_size,
                                            dtype=torch.complex128, device = device)
        """
                                            
    def forward(self, x):
        #torch.mm(phi_x.to(torch.complex128),
        #                           torch.transpose(self.Xi, 0, 1)).real
        return torch.mm(x, torch.transpose(self.Xi_operator, 0, 1))
    
class KoopmanMode(nn.Module):
    def __init__(self, operator_dim, state_dim):
        super(KoopmanMode, self).__init__()
        self.Xi_operator = nn.Parameter(torch.randn(state_dim, operator_dim) * 0.1)
                                            
    def forward(self, x):
        return torch.matmul(x, torch.transpose(self.Xi_operator, 0, 1))


