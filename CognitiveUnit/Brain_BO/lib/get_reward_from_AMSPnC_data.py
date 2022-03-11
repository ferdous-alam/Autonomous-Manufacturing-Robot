import numpy as np
import csv
from pathlib import Path
import os


def get_reward_from_AMSPnC_data(iter_num):
    # load real-time reward csv file
    reward_history = []
    with open('data/reward_history_blank.csv', newline='') as csv_file:
        reward_reader = csv.reader(csv_file, delimiter=',')
        for row in reward_reader:
            reward_history.append(float(row[0]))

    reward = reward_history[iter_num]
    return reward
