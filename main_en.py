# import ElFarol_Arthur as E
import ElFarol_GaussianThreshold_en as EGT

# memories = [1,3,6,9,12]
# predictors = [1,3,6,9,12]
memories = [12]
predictors = [12]
num_agents = [100]
num_rounds = [100]
num_experiments = 100
threshold = .6
# std_thresholds = [0.1, 0.2, 0.4, 0.8]
# std_thresholds = [0.03, 0.05, 0.06, 0.07, 0.09]
std_thresholds = [0]

# E.run_sweep(memories, predictors, num_experiments, num_agents, threshold, num_rounds, mirrors=True, DEBUG=False)
# E.run_sweep(memories, predictors, num_experiments, num_agents, threshold, num_rounds, mirrors=False, DEBUG=False)
# print('Sweep finished!')

# EGT.simulation(num_agents=100, threshold=threshold, std_threshold=0.08, num_predictors=12, memory_length=12, num_rounds=100, DEBUG=False, to_file=True)
EGT.run_sweep(memories, predictors, num_experiments, num_agents, threshold, std_thresholds, num_rounds, mirrors=True, DEBUG=False)
print('Finished!')
