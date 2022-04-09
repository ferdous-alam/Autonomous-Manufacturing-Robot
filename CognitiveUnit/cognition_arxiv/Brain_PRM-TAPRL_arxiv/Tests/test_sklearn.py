import numpy as np
from make_gp_prediction import make_gp_prediction


seed = 645657
np.random.seed(seed)
num_of_samples = 100  # test gp prediction for limited samples from training data
make_gp_prediction(num_of_samples, seed)
