import train, frozen
import sys
sys.path.insert(0, "../../")
import supervised_convnet
import pickle
from collections import defaultdict
import numpy as np
import time

mode = "run"
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
if mode == "run":
    run_mode = sys.argv[2]
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
        for _ in range(500):
            model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels,
                    first_activation = "tanh", activation_func = "relu",
                    num_hidden_layers = num_hidden_layers, seed = time.time() + _)
            model = model_to_cuda(model)
            best_val_acc, param_dict = train.trainer(model = model, batch_size = 100, betas0= 1-0.0018179320494754046, betas1=1- 0.001354073715524798,
                                                    lr= 0.004388469485690077, n_epochs= 150, train_size= 5000, weight_decay= 0, use_cuda = use_cuda)
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

elif mode == "analysis":
    import matplotlib.pyplot as plt
    filenames = ["frozen_convolution_pretrained_relu.pl", "frozen_convolution_no_center_relu.pl",
            "unfrozen_convolution_relu.pl"]#, "frozen_convolution_with_center_relu.pl"]
    labels = ["frozen_convolution_pretrained_relu", "frozen_convolution_no_center_relu",
    "unfrozen_convolution_relu", "frozen_convolution_with_center_relu"]
    alphas = [0.8, 0.6, 0.4, 0.2]
    plt.figure(figsize=(15,10))
    for index, filename in enumerate(filenames[:3]):
        with open(filename, "rb") as handle:
            results = pickle.load(handle)
        if index <= 2:
            b = np.array(results["best_val_acc_hist"])
            b = (b[b > 0.6])
            plt.hist(b, alpha = alphas[index], label = labels[index], normed=True)
        else:
            plt.hist(results, alpha = alphas[index], label = labels[index], normed=True)
        # print("filename", filename)
        # print("mean", np.mean(results))
        # print("std", np.std(results))
    plt.legend()
    # filenames = ["frozen_convolution_no_center.pl", "frozen_convolution_with_center.pl",
    #         "frozen_convolution_no_center_relu.pl", "frozen_convolution_with_center_relu.pl"]
    # alphas = [0.3, 0.5, 0.7, 0.5]
    plt.figure(figsize=(15,10))
    # with open("unfrozen_convolution_relu.pl", "rb") as handle:
    #     results = pickle.load(handle)
    # results["best_val_acc_hist"] = np.array(results["best_val_acc_hist"])
    # results["best_val_acc_hist"] = results["best_val_acc_hist"][results["best_val_acc_hist"] > 0.6]
    # plt.hist(results["best_val_acc_hist"],20, alpha = 0.3, label = "unfrozen_convolution", normed=True)
    # np.mean(results["best_val_acc_hist"])
    # np.std(results["best_val_acc_hist"])
    with open("unfrozen_convolution.pl", "rb") as handle:
        results = pickle.load(handle)
    b = np.array(results["best_val_acc_hist"])
    b = (b[b > 0.6])
    # len(b)
    plt.hist(b, alpha = 0.5, label = "unfrozen_convolution_relu", normed=True)
    with open("frozen_convolution_no_center.pl", "rb") as handle:
        results = pickle.load(handle)
    len(results)
    np.mean(results)
    plt.hist(results, alpha = 0.3, label = "frozen_convolution_no_center", normed=True)
    with open("frozen_convolution_with_center.pl", "rb") as handle:
        results = pickle.load(handle)
    plt.hist(results, alpha = 0.2, label = "frozen_convolution_with_center", normed=True)
    plt.legend()
    plt.show()
    w = results["conv_params"]
    [[]] * 9
    a,b,c,d,e,f,g,h,i = [], [], [], [], [], [], [], [], []
    for par in w["weight"]:
        a.append(par[0, 0, 0, 0])
        b.append(par[0, 0, 0, 1])
        c.append(par[0, 0, 0, 2])
        d.append(par[0, 0, 1, 0])
        e.append(par[0, 0, 1, 1])
        f.append(par[0, 0, 1, 2])
        g.append(par[0, 0, 2, 0])
        h.append(par[0, 0, 2, 1])
        i.append(par[0, 0, 2, 2])
    a,b,c,d,e,f,g,h,i = [np.array(_)[np.abs(np.array(_)) < 0.4] for _ in [a,b,c,d,e,f,g,h,i]]
    fig, axs = plt.subplots(3,3)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axs[0, 0].hist(a, 50)
    axs[0, 0].axvline(1/8, color = "r")
    axs[0, 1].hist(b, 50)
    axs[0, 1].axvline(1/8, color = "r")
    axs[0, 2].hist(c, 50)
    axs[0, 2].axvline(1/8, color = "r")
    axs[1, 0].hist(d, 50)
    axs[1, 0].axvline(1/8, color = "r")
    axs[1, 1].hist(e, 50)
    axs[1, 1].axvline(0, color = "r")
    axs[1, 2].hist(f, 50)
    axs[1, 2].axvline(1/8, color = "r")
    axs[2, 0].hist(g, 50)
    axs[2, 0].axvline(1/8, color = "r")
    axs[2, 1].hist(h, 50)
    axs[2, 1].axvline(1/8, color = "r")
    axs[2, 2].hist(i, 50)
    axs[2, 2].axvline(1/8, color = "r")
