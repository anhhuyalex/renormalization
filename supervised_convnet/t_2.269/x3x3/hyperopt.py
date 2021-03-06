import torch
import numpy as np
import pickle
from collections import defaultdict
import ray

from ax import RangeParameter, ParameterType
from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
# from ax.utils.notebook.plotting import render, init_notebook_plotting

import sys
import time
sys.path.insert(0, "../../")
import supervised_convnet
import train, frozen

# Parameters
num_hidden_layers = 1
out_channels = 1
num_workers = 4
run_mode =  sys.argv[1]
lattice_size =  sys.argv[2]
filename = f"hyperparameters_{run_mode}_for_{lattice_size}_onelayer.pl"
print("Saving to", filename)
n_loops = 200
save_loop = min(n_loops, 2)

ray.init()

@ray.remote
def init_model_and_train(hidden_size, batch_size, train_size, n_epochs, lr, weight_decay,
            betas0, betas1, seed):
    # Parameters
    num_hidden_layers = 1
    out_channels = 1


    if run_mode == "unfrozen_convolution_relu":
        model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
                hidden_size = hidden_size, out_channels = out_channels,
                first_activation = "tanh", activation_func = "relu",
                num_hidden_layers = num_hidden_layers, seed = seed)
        results = train.trainer(model = model, batch_size = batch_size, train_size = train_size, n_epochs = n_epochs, lr = lr,
                    weight_decay = weight_decay,
                    betas0 = 1-betas0, betas1 = 1-betas1)
    elif run_mode == "frozen_convolution_no_center_relu":
        model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                hidden_size = hidden_size, out_channels = out_channels,
                center = "omit", first_activation = "tanh",
                activation_func = "relu", num_hidden_layers = num_hidden_layers)
        results = train.trainer(model = model, batch_size = batch_size, train_size = train_size, n_epochs = n_epochs, lr = lr,
                    weight_decay = weight_decay,
                    betas0 = 1-betas0, betas1 = 1-betas1)
    elif run_mode == "frozen_convolution_pretrained_relu":
        model = frozen.SupervisedConvNet(filter_size = 3, square_size = 3, \
                hidden_size = hidden_size, out_channels = out_channels,
                center = "pre_trained", first_activation = "tanh",
                activation_func = "relu", num_hidden_layers = num_hidden_layers)
        results = train.trainer(model = model, batch_size = batch_size, train_size = train_size, n_epochs = n_epochs, lr = lr,
                    weight_decay = weight_decay,
                    betas0 = 1-betas0, betas1 = 1-betas1)
    elif run_mode == "unfrozen_convolution_3_channels":
        out_channels = 3
        model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3,
                hidden_size = hidden_size, out_channels = out_channels,
                first_activation = "tanh", activation_func = "relu",
                num_hidden_layers = num_hidden_layers, seed = seed)
        results = train.trainer(model = model, batch_size = batch_size, train_size = train_size, n_epochs = n_epochs, lr = lr,
                    weight_decay = weight_decay,
                    betas0 = 1-betas0, betas1 = 1-betas1)
    return results



def train_evaluate(parameterization):
    # parameters
    batch_size = parameterization["batch_size"]
    train_size = parameterization["train_size"]
    n_epochs = parameterization["n_epochs"]
    lr = parameterization["lr"]
    weight_decay = parameterization["weight_decay"]
    betas0 = parameterization["betas0"]
    betas1 = parameterization["betas1"]
    hidden_size = 10

    results = []
    for seed in range(num_workers):
        results.append(init_model_and_train.remote(hidden_size, batch_size, train_size,
        n_epochs, lr, weight_decay, betas0, betas1,
        time.time() + seed))

    results = ray.get(results)  # [0, 1, 2, 3]

    accuracy_list, state_dict = list(zip(*results))
    # print ("results", results)
    mean = np.mean(accuracy_list)
    SEM = np.std(accuracy_list)/np.sqrt(num_workers)
    # pool.close() # no more tasks
    # pool.join()  # wrap up current tasks
    return {"objective": (mean, SEM), "accuracy_list": accuracy_list, "state_dict" : state_dict}

# Initialize client
ax_client = AxClient()
try:
    with open(filename, "rb") as handle:
        hyper = pickle.load(handle)
    v = hyper["axclient"].copy()
    ax_client = ax_client.from_json_snapshot(v)
    accuracy_list = hyper["best_val_acc_hist"]
    conv_params = hyper["conv_params"]
except:
    ax_client.create_experiment(
        parameters=[
            {
              "name": "batch_size",
              "type": "range",
              "bounds": [1, 5000],
              "value_type": "int"
            },
            {
              "name": "train_size",
              "type": "range",
              "bounds": [100, 5000],
              "value_type": "int"
            },
            {
              "name": "n_epochs",
              "type": "range",
              "bounds": [50, 250],
              "value_type": "int"
            },
            {
              "name": "lr",
              "type": "range",
              "bounds": [1e-4, 1],
              "value_type": "float",
              "log_scale": True
            },
            {
              "name": "weight_decay",
              "type": "range",
              "bounds": [1e-5, 2e-1],
              "value_type": "float",
              "log_scale": True
            },
            {
              "name": "betas0",
              "type": "range",
              "bounds": [1e-5, 2e-1],
              "value_type": "float",
              "log_scale": True
            },
            {
              "name": "betas1",
              "type": "range",
              "bounds": [1e-5, 2e-1],
              "value_type": "float",
              "log_scale": True
            }
        ],
        parameter_constraints=["0.02 * train_size + -1 * batch_size <= 0", "train_size >= batch_size"],
        minimize=False,
        objective_name="objective",
        outcome_constraints=None,
        name="Test"
    )
    accuracy_list = []
    conv_params = defaultdict(list)

# print ("axclient",ax_client.experiment.trials )
# for loop in range(n_loops):
for loop in range(n_loops):
    print(f"Running trial {loop}/{n_loops}...")
    parameters, trial_index = ax_client.get_next_trial()
    print("trial_index", trial_index)
    time.sleep(2)
    # parameters["n_epochs"] = 5
     # Local evaluation here can be replaced with deployment to external system.
    evaluated = train_evaluate(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluated["objective"])
    best_val_accs = evaluated["accuracy_list"]
    param_dicts = evaluated["state_dict"]
    accuracy_list.extend(best_val_accs)
    for param_dict in param_dicts:
        conv_params["weight"].append(param_dict["conv1.weight"])
        conv_params["bias"].append(param_dict["conv1.bias"])

    print("accuracy_list", accuracy_list)
    # print("param_dicts", conv_params)

    print("Best params", ax_client.get_best_parameters())
    # periodic save
    if loop % save_loop == (save_loop - 1):
        optim_result = ax_client.get_best_parameters()
        # print("best_parameters", optim_result)
        # print("was I saved?", ax._save_experiment_and_generation_strategy_if_possible())
        hyper = {}
        hyper["best_params"] = optim_result
        hyper["axclient"] = ax_client.to_json_snapshot()
        hyper["best_val_acc_hist"] = accuracy_list
        hyper["conv_params"] = conv_params
        # print("optim_result", optim_result)
        with open(filename, "wb") as handle:
            pickle.dump(hyper, handle, protocol = pickle.HIGHEST_PROTOCOL)

ray.shutdown()
