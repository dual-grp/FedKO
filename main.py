import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.dataset_utils import *
from utils.data_utils import *
from utils.koopman_utils import *
from utils.model_utils import *
from utils.options import args_parser
from core.client import Client
from core.server import Server
import wandb

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    # parse args and set seed
    args = args_parser()
    print("> Settings: ", args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # set device
    if torch.cuda.is_available():
        n_devices = min(torch.cuda.device_count(), args.n_gpu)
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False
    os.environ["OMP_NUM_THREADS"] = "1"
    num_processes = torch.multiprocessing.cpu_count()  # Number of available CPU cores



    if args.dataset == "psm":
        num_users = 24
        full_train, test_data, target = read_data(args.dataset)
        sample = int(len(full_train)/num_users)
        train_subsets = []
        for i in range(0, len(full_train), sample):
            subset = full_train[i:i+sample]
            train_subsets.append(subset)

    elif args.dataset == "smd":
        num_users = 2
        full_train, test_data, target = read_data(args.dataset)

    # Assuming 'clients_list' is a list of Client objects
    clients_list = []
    for i in range(num_users):

        client_i = Client(id = i, train_data = train_subsets[i], test_data = test_data, label = target,
                          model = args.model, feature_dim = args.input_size, reservoir_dim = args.reservoir_size,
                          output_dim = args.output_size, alpha = args.leaky_rate, spectral_radius = args.spectral_radius,
                          batch_size = args.local_bs, train_rate = args.train_rate, 
                          inner_lr = args.inner_lr, outer_lr = args.outer_lr,
                          lr_decay_factor = args.lr_decay)
        clients_list.append(client_i)

    server = Server(clients_list, devices[args.gpu_id], dataset = args.dataset,
                    withpid = args.pid, subsampling = args.frac, glob_iters = args.epochs)

    wandb.init(
        project="fedko",  # set the wandb project where this run will be logged
        name= args.model + "_" + args.dataset \
                                    + "_" + str(args.n_clients) \
                                    + "_" + str(args.pid),
        config={
        "model": args.model,
        "dataset": args.dataset,
        "num_clients": args.n_clients,
        "withpid": args.pid
        }
        )
    
    train_losses, test_losses, test_acc = server.train()

    wandb.finish()

