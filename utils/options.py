import argparse
import copy
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()
    # federated learning arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--n_clients', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--inner_lr', type=float, default=1e-3, help="inner learning rate")
    parser.add_argument('--outer_lr', type=float, default=1e-3, help="outer learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.9, help="learning rate decay")
    parser.add_argument('--lr_decay_step_size', type=int, default=500, help="step size to decay learning rate")

    # model and dataset arguments
    parser.add_argument('--model', type=str, default='reservoir', help='model name')
    parser.add_argument('--dataset', type=str, default='psm', help="name of dataset")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    #parser.add_argument('--iid', action='store_true', help='whether i.i.d or notc (default: non-iid)')
    #parser.add_argument('--spc', action='store_true', help='whether spc or not (default: dirichlet)')
    #parser.add_argument('--beta', type=float, default=0.2, help="beta for Dirichlet distribution")
    #parser.add_argument('--n_classes', type=int, default=10, help="number of classes")
    #parser.add_argument('--n_channels', type=int, default=1, help="number of channels")
    #parser.add_argument('--pid', type=int, default=1, help="train with pid")

    # optimizing arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help="Optimizer (default: SGD)")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum (default: 0.0)")
    parser.add_argument('--fed_strategy', type=str, default='fedavg', help="optimization scheme e.g. fedavg")
    parser.add_argument('--alpha', type=float, default=1.0, help="alpha for feddyn")

    # misc
    parser.add_argument('--n_gpu', type=int, default=4, help="number of GPUs")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
    parser.add_argument('--n_procs', type=int, default=1, help="number of processes per processor")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--no_record', action='store_true', help='whether to record or not (default: record)')
    parser.add_argument('--load_checkpoint', action='store_true', help='whether to load model (default: do not load)')
    parser.add_argument('--no_checkpoint', action='store_true', help='whether to save best model (default: checkpoint)')

    args = parser.parse_args()
    return args
