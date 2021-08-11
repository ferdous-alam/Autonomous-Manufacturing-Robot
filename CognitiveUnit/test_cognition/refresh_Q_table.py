from test_algos import CognitionAlgorithms
import numpy as np

def refresh_Q_table(reward_history, lxy, dia, sample_count):

    current_state = [lxy,dia]
    algo = CognitionAlgorithms()
    Q_table, action, current_state, next_state, reward = algo.update_Q_table(reward_history,current_state)
    
    Qvalues = Q_table[39,53,:]
    Q_hat = np.load('Q_hat.npy')
    Qvalues_old = Q_hat[39, 53, :]
    
    print(f'Q_old: {Qvalues_old}')
    print(f'Q_new: {Qvalues}')
    # debug test 
    f = open("agent_history.txt", "a")
    sample_count = int(sample_count)
    f.write(f"s_{sample_count}:{current_state}, a_{sample_count}:{action}, s_{sample_count+1}:{next_state}, r_{sample_count}:{reward}\n")
    f.close()
    return None

    
    