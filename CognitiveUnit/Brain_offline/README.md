## Brain for AMSPnC: Offline policy
This is the policy learned from the source task; 48 FEM simulations. During this implementation the policy is not updated with real data obtained from AMSPnC.

## Details:
1. Run policy for 24 timesteps --> 24 3D printed artifacts 
2. Dump files are created with all the decision making details
3. Please delete the any pre-existing files on the dump folder
4. Make sure **sample_count = 1** in the LABVIEW file, measures have been taken to match the MATLAB vs Python indices

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
            No update required: offline execution of source optimal policy 
            reward: 99.2612
    ------------------------------------------------------- 

    iteration number: 2 ########################## 
        Artifact printing step: -----------------> 
            current_state: [2, 3], 
            action_taken: [0, 1], 
            next_state: [2, 4], 
            artifact_to_be_printed_now: [450, 850] micro meters
            artifact_to_be_printed_next: [450, 900] micro meters


        Brain update step: -----------------> 
            No update required: offline execution of source optimal policy 
            reward: 99.24854564
    ------------------------------------------------------- 

    iteration number: 3 ########################## 
        Artifact printing step: -----------------> 
            current_state: [2, 4], 
            action_taken: [0, 0], 
            next_state: [2, 4], 
            artifact_to_be_printed_now: [450, 900] micro meters
            artifact_to_be_printed_next: [450, 900] micro meters


        Brain update step: -----------------> 
            No update required: offline execution of source optimal policy 
            reward: 1115564.0
    ------------------------------------------------------- 
```