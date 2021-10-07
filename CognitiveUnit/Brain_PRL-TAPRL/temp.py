

# append to log file for experiment details
log_file = open("dump/experiment_no_{}_details.txt".format(exp_num), "a")
details = 'This is a test \n'
log_file.write(details)
log_file.close()
