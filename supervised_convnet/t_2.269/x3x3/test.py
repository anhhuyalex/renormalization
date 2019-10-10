import pickle
run_mode = "frozen_convolution_no_center_relu"
with open(f"hyperparameters_{run_mode}.pl", "rb") as handle:
    hyper = pickle.load(handle)


from ax import RangeParameter, ParameterType
from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate

# Initialize client
ax = AxClient()
ax = ax.from_json_snapshot(hyper["axclient"])
print(ax.get_trial_parameters(10))
