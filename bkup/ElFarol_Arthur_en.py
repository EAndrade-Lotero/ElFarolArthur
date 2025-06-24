print("Importing packages...")
from random import choice, sample, randint, uniform
import numpy as np
import pandas as pd
from os import remove
from itertools import product
print("Ready!")


def save_dataframe(dataFrame, filename, initial, mirrors=True, many=False):
    """Save the DataFrame to a CSV file, handling file paths and initial/append modes."""
    if not many:
        if mirrors:
            filename = "../Data_Farol/normal/data_all/" + filename
        else:
            filename = "../Data_Farol/normal/data_no_mirrors/" + filename
    else:
        if mirrors:
            filename = "../Data_Farol/data_all/" + filename
        else:
            filename = "../Data_Farol/data_no_mirrors/" + filename
    if initial:
        try:
            remove(filename)
        except:
            pass
        with open(filename, 'w') as f:
            dataFrame.to_csv(f, header=False, index=False)
    else:
        with open(filename, 'a') as f:
            dataFrame.to_csv(f, header=False, index=False)

def simulation(num_agents, threshold, memory_length, num_predictors, num_rounds, initial=True, identifier='', mirrors=True, DEBUG=False, to_file=True):
    """Run a full simulation and optionally save the results."""
    bar = Bar(num_agents, threshold, memory_length, num_predictors, identifier, mirrors)
    if DEBUG:
        print("**********************************")
        print("Initial agents:")
        for a in bar.agents:
            print(a)
        print("**********************************\n")
    for i in range(num_rounds):
        if DEBUG:
            print("Round", i)
            print("History:", bar.history)
            # for p in bar.predictors:
            #     print(f"Predictor: {str(p)} - Prediction: {p.prediction} - inaccuracy: {p.inaccuracy}")
            # print("****************************")
        bar.play_round(i + 1)
        if DEBUG:
            for a in bar.agents:
                print(a)
    data = bar.create_agents_dataframe()
    data['Num_rounds'] = num_rounds
    filename = f'simulation-{memory_length}-{num_predictors}-{num_agents}-{num_rounds}.csv'
    if to_file:
        if num_agents < 1000:
            save_dataframe(data, filename, initial, mirrors)
        else:
            save_dataframe(data, filename, initial, mirrors, many=True)
        if DEBUG:
            print('Data saved in', filename)
    return bar

def run_sweep(memories, predictors, num_experiments, num_agents, threshold, num_rounds, mirrors=True, DEBUG=False):
    """Run a sweep of simulations over different parameter combinations."""
    print('********************************')
    print('Running simulations...')
    print('********************************\n')
    identifier = 0
    for d in memories:
        for k in predictors:
            for N in num_agents:
                for T in num_rounds:
                    initial = True
                    print('Running experiments with parameters:')
                    print(f"Memory={d}; Predictors={k}; Number of agents={N}; Number of rounds={T}; Mirrors?:{mirrors}")
                    for i in range(num_experiments):
                        simulation(N, threshold, d, k, T, initial=initial, identifier=identifier, mirrors=mirrors, DEBUG=DEBUG)
                        identifier += 1
                        initial = False