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


def extract_rewards():
    # load real-time reward csv file
    reward_history = []
    with open('data/reward_history.csv', newline='') as csv_file:
        reward_reader = csv.reader(csv_file, delimiter=',')
        for row in reward_reader:
            reward_history.append(float(row[0]))

    return reward_history


def dummy_reward(state):
    target_reward = np.load('data/target_reward.npy')
    target_reward = target_reward.T
    reward_val = target_reward[state[0], state[1]]
    with open('data/reward_history.csv', 'a', newline='') as csv_file:
        reward_writer = csv.writer(csv_file)
        reward_writer.writerow([reward_val])

    csv_file.close()

    return reward_val
