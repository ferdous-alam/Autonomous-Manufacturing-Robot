import csv
import numpy as np
from lib.get_reward_from_AMSPnC_data import extract_rewards
from sklearn.neighbors import KernelDensity
import scipy.stats as stats
import math
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from run_algo import *


for i in range(10):
    print(f'iteration: {i}')
    val = run_algo(i)
    with open('data/reward_history.csv', 'a') as csv_file:
        reward_writer = csv.writer(csv_file)
        reward_writer.writerow([np.random.rand()])
    run_update(i)

print('finished!')

