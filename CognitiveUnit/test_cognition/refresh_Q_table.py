from test_algos import CognitionAlgorithms
import numpy as np

def refresh_Q_table(reward_history, lxy, dia, sample_count):

    current_state = [lxy,dia]
    algo = CognitionAlgorithms()
    state = current_state
    Q_table, action, current_state, next_state, reward = algo.update_Q_table(reward_history,current_state)
    
    Q_hat = np.load('Q_hat.npy')
    
    # debug test: text file 
    f = open("agent_history.txt", "a")
    sample_count = int(sample_count)
    f.write(f"s_{sample_count}:{[lxy, dia]}, a_{sample_count}:{action}, s_{sample_count+1}:{next_state}, r_{sample_count}:{reward}\n")
    f.flush()
    return None

