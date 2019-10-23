import train#, frozen
import sys
# sys.path.insert(0, "../../")
import supervised_convnet
import pickle
from collections import defaultdict
import numpy as np
import time
from termcolor import colored


# mode = "run"
save_loops = 50
use_cuda = False
results = []
conv_params = {}
filename = ""
def save_progress(results, conv_params, filename, epoch = 500):
    new_results = {}
    new_results["best_val_acc_hist"] = results
    new_results["conv_params"] = conv_params
    new_results["annotation"] = "bias + weights histogram saved"
    new_results["epoch"] = epoch
    # new_results["first_epo    ch_validate_accuracy_list"] = first_epoch_validate_accuracy_list
    with open(filename, "wb") as handle:
        pickle.dump(new_results, handle, protocol = pickle.HIGHEST_PROTOCOL)

# GPU for use in colab
def model_to_cuda(model):
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
    return model

"""
Default activation function: sigmoid
"""

run_mode = sys.argv[1]
hidden_size = 10
out_channels = 1
num_hidden_layers = 1
conv_params = defaultdict(list)

if run_mode == "unfrozen_convolution_relu":
    """
    """
    out_channels = 1
    filename = "unfrozen_convolution_relu.pl"
    try:
        with open(filename, "rb") as handle:
            results = pickle.load(handle)
    except:
        results = []
    results = []
    first_epoch_validate_accuracy_list = []
    running_mean = 0.0
    for _ in range(500):
        model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
                hidden_size = hidden_size, out_channels = out_channels,
                first_activation = "tanh", activation_func = "relu",
                num_hidden_layers = num_hidden_layers, seed = time.time() + _)
        model = model_to_cuda(model)
        best_val_acc, param_dict = train.trainer(model = model, batch_size = 2825, betas0= 1-0.14837031829213393, betas1=1- 0.00044025104003604366,
                                                lr= 0.005938797909346845, n_epochs= 233, train_size= 5000, weight_decay= 0.000119, use_cuda = use_cuda)
        running_mean = running_mean + (best_val_acc - running_mean)/(_ + 1)
        print("Running mean is", colored(running_mean, "red"))
        results.append(best_val_acc)
        conv_params["weight"].append(param_dict["conv1.weight"])
        conv_params["bias"].append(param_dict["conv1.bias"])
        if (_ % save_loops) == (0):
            save_progress(results, conv_params, filename, _)
elif run_mode == "unfrozen_convolution_relu9x9":
    """
    """
    out_channels = 1
    filename = "unfrozen_convolution_relu9x9.pl"
    try:
        with open(filename, "rb") as handle:
            results = pickle.load(handle)
    except:
        results = []
    results = []
    first_epoch_validate_accuracy_list = []
    running_mean = 0.0
    for _ in range(500):
        model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
                hidden_size = hidden_size, out_channels = out_channels,
                first_activation = "tanh", activation_func = "relu",
                num_hidden_layers = num_hidden_layers, seed = time.time() + _)
        model = model_to_cuda(model)
        best_val_acc, param_dict = train.trainer(model = model, batch_size = 2109, betas0= 1-0.05202967494721497, betas1=1- 0.0005827377065455242,
                                                lr= 0.06397799777638523, n_epochs= 145, train_size= 3807, weight_decay= 0.00031858400422819333, use_cuda = use_cuda)
        running_mean = running_mean + (best_val_acc - running_mean)/(_ + 1)
        print("Running mean is", colored(running_mean, "red"))
        results.append(best_val_acc)
        conv_params["weight"].append(param_dict["conv1.weight"])
        conv_params["bias"].append(param_dict["conv1.bias"])
        if (_ % save_loops) == (0):
            save_progress(results, conv_params, filename, _)
elif run_mode == "frozen_convolution_no_center_relu":
    """
    """
    filename = "frozen_convolution_no_center_relu.pl"
    out_channels = 1
    try:
        with open(filename, "rb") as handle:
            results = pickle.load(handle)
    except:
        results = []
    results = []
    for _ in range(500):
        model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                hidden_size = hidden_size, out_channels = out_channels,
                center = "omit", first_activation = "tanh",
                activation_func = "relu", num_hidden_layers = num_hidden_layers)
        model = model_to_cuda(model)
        best_val_acc, param_dict = train.trainer(model = model, batch_size = 100, betas0=1- 0.15428681703555264, betas1=1-1e-5,
                                                lr= 0.021919796543990972, n_epochs= 250, train_size= 5000, weight_decay= 0.0008230830669560186, use_cuda = use_cuda)
        # conv_params["weight"].append(param_dict["conv1.weight"])
        conv_params["bias"].append(param_dict["conv1.bias"])
        results.append(best_val_acc)
        if (_ % save_loops) == (0):
            save_progress(results, conv_params, filename, _)
elif run_mode == "frozen_convolution_pretrained_relu":
    """
    """
    run_num = 1
    filename = f"frozen_convolution_pretrained_relu_{run_num}.pl"
    out_channels = 1
    try:
        with open(filename, "rb") as handle:
            results = pickle.load(handle)
    except:
        results = []
    results = []
    for _ in range(500):
        model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                hidden_size = hidden_size, out_channels = out_channels,
                center = "omit", first_activation = "tanh",
                activation_func = "relu", num_hidden_layers = num_hidden_layers)
        model = model_to_cuda(model)
        best_val_acc, param_dict = train.trainer(model = model, batch_size = 5000, betas0=1- 0.10000000000000004, betas1=1-1.0081196321129792e-5,
                                                lr= 0.004388469485690077    , n_epochs= 250, train_size= 5000, weight_decay= 6.21073259786956e-5, use_cuda = use_cuda)
        # conv_params["weight"].append(param_dict["conv1.weight"])
        # conv_params["bias"].append(param_dict["conv1.bias"])
        results.append(best_val_acc)
        if (_ % save_loops) == (0):
            save_progress(results, conv_params, filename, _)
elif run_mode == "unfrozen_convolution_2_channels_relu":
    """
    check to make sure that out_channels below is not 1
    """
    out_channels = 2
    filename = "unfrozen_convolution_{}_channels.pl".format(out_channels)
    print("filename", filename)
    try:
        with open(filename, "rb") as handle:
            results = pickle.load(handle)
    except:
        results = []
    results = []
    for _ in range(500):
        model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3,
                hidden_size = hidden_size, out_channels = out_channels,
                first_activation = "tanh",
                activation_func = "relu", num_hidden_layers = num_hidden_layers,
                seed = time.time() + _)
        # best_val_acc, param_dict = train.trainer(model = model, batch_size = 968, betas0= 1-0.13209249743733834, betas1= 1-9.581886527012395e-05,
        #                                         lr= 0.0070890201458810925, n_epochs= 250, train_size= 5000, weight_decay= 0.00019488234448160615, use_cuda = use_cuda)
        best_val_acc, param_dict = train.trainer(model = model, batch_size = 968, betas0= 1-0.13209249743733834, betas1=1- 9.581886527012395e-5,
                                                lr= 0.0070890201458810925, n_epochs= 250, train_size= 5000, weight_decay= 0.00019488234448160615, use_cuda = use_cuda)

        results.append(best_val_acc)
        conv_params["weight"].append(param_dict["conv1.weight"])
        conv_params["bias"].append(param_dict["conv1.bias"])
        if (_ % save_loops) == (0):
            save_progress(results, conv_params, filename, _)
elif run_mode == "unfrozen_convolution_3_channels_relu":
    """
    check to make sure that out_channels below is not 1
    """
    out_channels = 3
    filename = "unfrozen_convolution_{}_channels.pl".format(out_channels)
    print("filename", filename)
    try:
        with open(filename, "rb") as handle:
            results = pickle.load(handle)
    except:
        results = []
    results = []
    for _ in range(500):
        model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3,
                hidden_size = hidden_size, out_channels = out_channels,
                first_activation = "tanh",
                activation_func = "relu", num_hidden_layers = num_hidden_layers,
                seed = time.time() + _)
        # best_val_acc, param_dict = train.trainer(model = model, batch_size = 100, betas0= 1-0.03575422958529493, betas1= 1-0.20000000000000004,
        #                                         lr= 0.008603286258853008, n_epochs= 250, train_size= 5000, weight_decay= 1e-05, use_cuda = use_cuda)
        best_val_acc, param_dict = train.trainer(model = model, batch_size = 3521, betas0= 1-0.20000000000000004, betas1=1- 0.003211923010710316,
                                                lr= 0.008499899017388025, n_epochs= 193, train_size= 5000, weight_decay= 0, use_cuda = use_cuda)

        results.append(best_val_acc)
        conv_params["weight"].append(param_dict["conv1.weight"])
        conv_params["bias"].append(param_dict["conv1.bias"])
        if (_ % save_loops) == (0):
            save_progress(results, conv_params, filename, _)


save_progress(results, conv_params, filename, _)
