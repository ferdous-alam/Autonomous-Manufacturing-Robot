import numpy as np
import csv
from pathlib import Path
import os


def get_reward_from_AMSPnC_data(iter_num):
    # load real-time reward csv file
    reward_history = []
    with open('data/reward_history.csv', newline='') as csv_file:
        reward_reader = csv.reader(csv_file, delimiter=',')
        for row in reward_reader:
            reward_history.append(float(row[0]))

    # create new folder for this iteration and save rewards
    directory = f'data/iter{iter_num+1}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(f'data/iter{iter_num+1}/rewards.npy', reward_history)

    # get the reward value from AMSPnC
    # corresponding to the requested iteration number
    reward = reward_history[iter_num]
    return reward
