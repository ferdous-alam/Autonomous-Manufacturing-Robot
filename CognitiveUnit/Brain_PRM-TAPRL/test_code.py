import numpy as np
from algorithm.utils import get_GP_reward_model
from lib.gaussian_process_regression import GaussianProcess

data = np.load('dump/dataset_23.npy', allow_pickle=True).tolist()
r_model = get_GP_reward_model(data, iter_num=1, viz_model='smooth')
# #

