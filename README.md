# renormalization


* Requirements
	* numpy, pytorch
* For overview of code + some plots, check out `main.ipynb`
* Generate Ising model data:
	* run `python generate_data.py`
	* Place correlated .npy file in correct temperature directory
* Create uncorrelated samples in `supervised_convnet/generate_uncorrelated_data.py`
	* Set `data` variable to be path of the Ising model data (of each temperature)
	* Place uncorrelated .npy file in correct temperature directory
* Train neural network to distinguish between correlated/uncorrelated samples in temperature directory e.g. `supervised_convnet/t_1/train.py`
	* Make sure both correlated and uncorrelated .npy file in `supervised_convnet/t_1`
* Neural network architecture in `supervised_convnet/supervised_convnet.py`
	* Also contains `IsingDataset` class for pytorch data set loading