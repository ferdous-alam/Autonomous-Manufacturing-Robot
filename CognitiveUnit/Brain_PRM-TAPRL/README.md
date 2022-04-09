## Brain for AMSPnC: PRM-TAPRL algorithm
Probabilistic reward modeling using temporal abstractions in reinforcement learning (PRM-TAPRL) is implemented on the AMSPnC machine. Each reward value is collected from a 3D printed PnC artifact.

## Details:
1. Run policy for 24 timesteps --> 24 3D printed artifacts 
2. Dump files are created with all the decision making details
3. Please delete the any pre-existing files on the dump folder except the 'temp.py' file
4. Please make sure there are only three files in the data folder: 'Q_trained_source.npy', 'reward_history.csv', 'source_reward.npy' 
5. **Please make sure 'reward_hisotry.csv' file is empty** (This is mandatory!!)
6. Make sure **sample_count = 1** in the LABVIEW file, measures have been taken to match the MATLAB vs Python indices
7. # ------ parameters ----------------
    num_of_options = 5   # number of options to be created
    H = 5      # length of each option
    epsilon = 0.75    # exploration for creating options

## Example dump file 
```
iteration number: 1 ########################## 
        Artifact printing step: -----------------> 
            current_state: [1, 2], 
            action_taken: [1, 1], 
            next_state: [2, 3], 
            artifact_to_be_printed_now: [400, 800] micro meters
            artifact_to_be_printed_next: [450, 850] micro meters


        Brain update step: -----------------> 
            reward: 9.5656
            trajectory:---> s_t: [400, 800], a_t: [1, 1], r: [450, 850], s_t+1: 9.5656 
------------------------------------------------------- 

iteration number: 2 ########################## 
    Artifact printing step: -----------------> 
        current_state: [2, 3], 
        action_taken: [1, 0], 
        next_state: [3, 3], 
        artifact_to_be_printed_now: [450, 850] micro meters
        artifact_to_be_printed_next: [500, 850] micro meters


    Brain update step: -----------------> 
        reward: 9.23265
        trajectory:---> s_t: [450, 850], a_t: [1, 0], r: [500, 850], s_t+1: 9.23265 
------------------------------------------------------- 

```