import torch
import numpy as np
import pickle
import multiprocessing as mp

from ax import RangeParameter, ParameterType
from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate

import sys
import time
sys.path.insert(0, "../../")
import supervised_convnet
import train, frozen

# Parameters
num_hidden_layers = 1
out_channels = 1
weight = 0.02

# Initialize client
ax = AxClient()

def init_model_and_train(hidden_size, batch_size, train_size, n_epochs, lr, weight_decay,
            betas0, betas1, seed):
    model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
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



    pool = mp.Pool(processes=4)
    # accuracy = train.trainer(model = model, batch_size = batch_size, train_size = train_size, n_epochs = n_epochs, lr = lr,
    #             weight_decay = weight_decay,
    #             betas0 = 1-betas0, betas1 = 0.99)[0]
    # results = [pool.apply(train.trainer, {"model" : model, "batch_size" : batch_size,
    #             "train_size" : train_size, "n_epochs" : n_epochs, "lr" : lr,
    #             "weight_decay" : weight_decay,
    #             "betas0" : 1-betas0, "betas1" : 0.99}) for x in range(5)]
    results = [pool.apply_async(init_model_and_train, args=(hidden_size, batch_size, train_size,
                n_epochs, lr, weight_decay, betas0, betas1, time.time() + seed)) for seed in range(5)]
    output = [p.get()[0] for p in results] # only 1st argument is last accuracy
    mean = np.mean(output)
    SEM = np.std(output)/np.sqrt(5)
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks
    return mean, SEM

ax.create_experiment(
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
    parameter_constraints=["0.02 * n_epochs + -1 * batch_size <= 0"],
    minimize=False,
    outcome_constraints=None,
    name="Test"
)

for i in range(30):
    print(f"Running trial {i+1}/30...")
    parameters, trial_index = ax.get_next_trial()
     # Local evaluation here can be replaced with deployment to external system.
    ax.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))

optim_result = ax.get_best_parameters()
print("best_parameters", optim_result)

hyper = {}
hyper["best_params"] = optim_result
hyper["report"] = ax.get_report()
with open("hyperparameters.pl", "wb") as handle:
    pickle.dump(hyper, handle, protocol = pickle.HIGHEST_PROTOCOL)
