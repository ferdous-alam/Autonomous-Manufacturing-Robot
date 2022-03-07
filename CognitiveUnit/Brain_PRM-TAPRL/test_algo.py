from algorithm.algo import run_PRM_TAPRL_feedback

for iter_num in range(10):
    x = run_PRM_TAPRL_feedback(iter_num)
    print(x)

