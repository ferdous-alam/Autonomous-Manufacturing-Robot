import numpy as np
import csv


def extract_rewards(trial_num):
    # load real-time reward csv file
    reward_history = []
    if trial_num == 1:
        filename = 'Test_policy/reward_history_E4_T1.csv'
    elif trial_num == 2:
        filename = 'Test_policy/reward_history_E4_T2.csv'
    else:
        filename = 'Test_policy/reward_history_E4_T3.csv'

    with open(filename, newline='') as csv_file:
        reward_reader = csv.reader(csv_file, delimiter=',')
        for row in reward_reader:
            reward_history.append(float(row[0]))

    return reward_history


def extract_states(trial_num):
    # load real-time reward csv file
    reward_history = []
    if trial_num == 1:
        states = np.load('Test_policy/stored_artifacts_E4_T1.npy')
    elif trial_num == 2:
        states = np.load('Test_policy/stored_artifacts_E4_T2.npy')
    else:
        states = np.load('Test_policy/stored_artifacts_E4_T3.npy')
    states = states.tolist()
    return states
