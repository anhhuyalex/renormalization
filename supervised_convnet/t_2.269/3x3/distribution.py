import train, frozen
import sys
sys.path.insert(0, "../../")
import supervised_convnet
import pickle
from collections import defaultdict
import numpy as np


mode = "analysis"

"""
Default activation function: sigmoid
"""
if mode == "run":
    run_mode = sys.argv[2]
    hidden_size = 10
    out_channels = 1
    num_hidden_layers = 1
    conv_params = defaultdict(list)

    if run_mode == "unfrozen_convolution":
        """
        """
        out_channels = 1
        filename = "unfrozen_convolution.pl"
        try:
            with open(filename, "rb") as handle:
                results = pickle.load(handle)
        except:
            results = []
        for _ in range(500):
            model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels,
                    activation_func = "sigmoid", num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            results.append(best_val_acc)
            conv_params["weight"].append(param_dict["conv1.weight"])
            conv_params["bias"].append(param_dict["conv1.bias"])
    elif run_mode == "unfrozen_convolution_relu":
        """
        """
        out_channels = 1
        filename = "unfrozen_convolution_relu.pl"
        try:
            with open(filename, "rb") as handle:
                results = pickle.load(handle)
        except:
            results = []
        first_epoch_validate_accuracy_list = []
        for _ in range(1000):
            model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels,
                    first_activation = "tanh", activation_func = "relu",
                    num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            results.append(best_val_acc)
            first_epoch_validate_accuracy_list.append(first_epoch_validate_accuracy)
            conv_params["weight"].append(param_dict["conv1.weight"])
            conv_params["bias"].append(param_dict["conv1.bias"])
    elif run_mode == "frozen_convolution_no_center":
        """
        """
        out_channels = 1
        filename = "frozen_convolution_no_center.pl"
        try:
            with open(filename, "rb") as handle:
                results = pickle.load(handle)
        except:
            results = []
        for _ in range(500):
            model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels,
                    center = "omit", activation_func = "sigmoid", num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            results.append(best_val_acc)
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
        for _ in range(500):
            model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels,
                    center = "omit", first_activation = "tanh",
                    activation_func = "relu", num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            conv_params["bias"].append(param_dict["conv1.bias"])
            results.append(best_val_acc)
    elif run_mode == "frozen_convolution_with_center":
        """
        """
        filename = "frozen_convolution_with_center.pl"
        out_channels = 1
        try:
            with open(filename, "rb") as handle:
                results = pickle.load(handle)
        except:
            results = []
        for _ in range(500):
            model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels,
                    center = "keep", activation_func = "sigmoid", num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            results.append(best_val_acc)
    elif run_mode == "frozen_convolution_with_center_relu":
        """
        """
        filename = "frozen_convolution_with_center_relu.pl"
        out_channels = 1
        try:
            with open(filename, "rb") as handle:
                results = pickle.load(handle)
        except:
            results = []
        for _ in range(500):
            model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels,
                    center = "keep", activation_func = "relu", num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            results.append(best_val_acc)
    elif run_mode == "frozen_convolution_pretrained_relu":
        """
        """
        filename = "frozen_convolution_pretrained_relu.pl"
        out_channels = 1
        try:
            with open(filename, "rb") as handle:
                results = pickle.load(handle)
        except:
            results = []
        first_epoch_validate_accuracy_list = []
        for _ in range(500):
            model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels,
                    center = "pre_trained", first_activation = "tanh",
                    activation_func = "relu", num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            first_epoch_validate_accuracy_list.append(first_epoch_validate_accuracy)
            results.append(best_val_acc)
    elif run_mode == "unfrozen_convolution_many_channels":
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
        for _ in range(500):
            model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3,
                    hidden_size = hidden_size, out_channels = out_channels,
                    activation_func = "sigmoid", num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            results.append(best_val_acc)
    elif run_mode == "unfrozen_convolution_many_channels_relu":
        """
        check to make sure that out_channels below is not 1
        """
        out_channels = 3
        filename = "unfrozen_convolution_{}_channels_relu.pl".format(out_channels)
        print("filename", filename)
        try:
            with open(filename, "rb") as handle:
                results = pickle.load(handle)
        except:
            results = []
        for _ in range(500):
            model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3,
                    hidden_size = hidden_size, out_channels = out_channels,
                    activation_func = "relu", num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            results.append(best_val_acc)
    elif run_mode == "unfrozen_convolution_many_channel_many_hidden":
        """
        check to make sure that out_channels, num_hidden_layers above is not 1
        """
        out_channels = 2
        num_hidden_layers = 2
        filename = "unfrozen_convolution_{}_channels_{}_hidden.pl".format(out_channels, num_hidden_layers)
        print("filename", filename)
        try:
            with open(filename, "rb") as handle:
                results = pickle.load(handle)
        except:
            results = []
        for _ in range(500):
            model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
                    hidden_size = hidden_size, out_channels = out_channels, num_hidden_layers = num_hidden_layers)
            best_val_acc, param_dict, first_epoch_validate_accuracy = train.trainer(model = model)
            results.append(best_val_acc)

    new_results = {}
    new_results["best_val_acc_hist"] = results
    # new_results["conv_params"] = conv_params
    # new_results["annotation"] = "bias + weights histogram saved"
    new_results["first_epoch_validate_accuracy_list"] = first_epoch_validate_accuracy_list
    with open(filename, "wb") as handle:
        pickle.dump(new_results, handle, protocol = pickle.HIGHEST_PROTOCOL)
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
