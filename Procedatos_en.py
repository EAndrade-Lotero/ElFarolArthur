import pandas as pd
import numpy as np
from itertools import product

def read_data(memories, predictors, num_agents, num_rounds, std_thresholds=[None], mirrors=True, verbose=True, many=False, tail=False):
    """
    Reads simulation data from CSV files for different parameter sweeps.
    Handles both standard and Gaussian threshold models.
    """
    names_ = ['Memory', 'Num_predictors', 'Identifier', 'Round', 'Agent', 'State', 'Score', 'Policy', 'Prediction', 'Accuracy', 'Num_agents']
    df_list = []
    for d in memories:
        for k in predictors:
            for N in num_agents:
                for T in num_rounds:
                    for s in std_thresholds:
                        if s is None:
                            names = names_
                            if verbose:
                                print(f"Reading data sweep memory {d} predictors {k} number of agents {N} and number of rounds {T}")
                            if not many:
                                if mirrors:
                                    file = './data/mirrors/simulation-' + str(d) + "-" + str(k) + '-' + str(N) + '-' + str(T) + ".csv"
                                else:
                                    file = './data/simulation-' + str(d) + "-" + str(k) + '-' + str(N) + '-' + str(T) + ".csv"
                            else:
                                if mirrors:
                                    file = './data/mirrors/simulation-' + str(d) + "-" + str(k) + '-' + str(N) + '-' + str(T) + ".csv"
                                else:
                                    file = './data/simulation-' + str(d) + "-" + str(k) + '-' + '-' + str(N) + '-' + str(T) + ".csv"
                            if verbose:
                                print(f"Loading data from file {file}...")
                            try:
                                aux = pd.read_csv(file, names=names, header=None)
                                if 'Memory' in aux['Memory'].unique().tolist():
                                    aux = aux.iloc[1:]
                                aux['Num_rounds'] = T
                                if tail:
                                    aux = pd.DataFrame(aux[aux.Round > int(max(aux.Round) * .8)])
                                df_list.append(aux)
                                if verbose:
                                    print("Done")
                            except:
                                print(f"File {file} does not exist! Skipping to next option")
                        else:
                            names = names_[:2] + ['Std_threshold'] + names_[2:]
                            if verbose:
                                print(f"Reading data sweep memory {d} predictors {k} std threshold {s} number of agents {N} and number of rounds {T}")
                            if not many:
                                if mirrors:
                                    file = './Data_Farol_Gaussian_Threshold/normal/data_all/data_no_mirrors/simulation-' + str(d) + "-" + str(k) + "-" + str(s) + '-' + str(N) + '-' + str(T) + ".csv"
                                else:
                                    file = './Data_Farol_Gaussian_Threshold/normal/data_all/simulation-' + str(d) + "-" + str(k) + "-" + str(s) + '-' + str(N) + '-' + str(T) + ".csv"
                            else:
                                if mirrors:
                                    file = './Data_Farol_Gaussian_Threshold/data_all/data_no_mirrors/simulation-' + str(d) + "-" + str(k) + "-" + str(s) + '-' + str(N) + '-' + str(T) + ".csv"
                                else:
                                    file = './Data_Farol_Gaussian_Threshold/data_all/simulation-' + str(d) + "-" + str(k) + "-" + str(s) + '-' + '-' + str(N) + '-' + str(T) + ".csv"
                            if verbose:
                                print(f"Loading data from file {file}...")
                            try:
                                aux = pd.read_csv(file, names=names, header=None)
                                if 'Memory' in aux['Memory'].unique().tolist():
                                    aux = aux.iloc[1:]
                                aux['Num_rounds'] = T
                                if tail:
                                    aux = pd.DataFrame(aux[aux.Round > int(max(aux.Round) * .8)])
                                df_list.append(aux)
                                if verbose:
                                    print("Done")
                            except:
                                print(f"File {file} does not exist! Skipping to next option")

    if verbose:
        print("Preparing dataframe...")
    data = pd.concat(df_list)
    if verbose:
        print(data.head())
    try:
        # Convert columns to appropriate types
        data['Memory'] = data['Memory'].astype(int)
        data['Num_predictors'] = data['Num_predictors'].astype(int)
        data['Num_agents'] = data['Num_agents'].astype(int)
        data['Num_rounds'] = data['Num_rounds'].astype(int)
        data['Identifier'] = data['Identifier'].astype(int)
        data['Round'] = data['Round'].astype(int)
        data['Agent'] = data['Agent'].astype(int)
        data['State'] = data['State'].astype(int)
        data['Score'] = data['Score'].astype(int)
        data['Policy'] = data['Policy'].astype(str)
        data['Prediction'] = data['Prediction'].astype(int)
        data['Accuracy'] = data['Accuracy'].astype(float)
        if 'Std_threshold' in data.columns:
            data['Std_threshold'] = data['Std_threshold'].astype(float)
    except:
        data = data.iloc[1:]
        if verbose:
            print(data.head())
        data['Memory'] = data['Memory'].astype(int)
        data['Num_predictors'] = data['Num_predictors'].astype(int)
        data['Num_agents'] = data['Num_agents'].astype(int)
        data['Num_rounds'] = data['Num_rounds'].astype(int)
        data['Identifier'] = data['Identifier'].astype(int)
        data['Round'] = data['Round'].astype(int)
        data['Agent'] = data['Agent'].astype(int)
        data['State'] = data['State'].astype(int)
        data['Score'] = data['Score'].astype(int)
        data['Policy'] = data['Policy'].astype(str)
        data['Prediction'] = data['Prediction'].astype(int)
        data['Accuracy'] = data['Accuracy'].astype(float)
        if 'Std_threshold' in data.columns:
            data['Std_threshold'] = data['Std_threshold'].astype(float)
    # Set column order
    if 'Std_threshold' in data.columns:
        columns = ['Memory', 'Num_predictors', 'Std_threshold', 'Num_agents', 'Num_rounds', 'Identifier', 'Round', 'Agent', 'State', 'Score', 'Policy', 'Prediction', 'Accuracy']
    else:
        columns = ['Memory', 'Num_predictors', 'Num_agents', 'Num_rounds', 'Identifier', 'Round', 'Agent', 'State', 'Score', 'Policy', 'Prediction', 'Accuracy']
    data = data[columns]
    if verbose:
        print("Shape:", data.shape)
        print("Memory value counts:", data['Memory'].value_counts())
        print("Predictors value counts:", data['Num_predictors'].value_counts())
        if 'Std_threshold' in data.columns:
            print("Std_threshold value counts:", data['Std_threshold'].value_counts())
        print("Agents value counts:", data['Num_agents'].value_counts())
        print("Rounds value counts:", data['Num_rounds'].value_counts())
    return data

def merge_models(df):
    """
    Aggregates results by model and identifier, computing attendance, deviation, efficiency, and inaccuracy.
    """
    models = df.Model.unique().tolist()
    df['Attendance'] = df.groupby(['Model', 'Identifier', 'Round'])['State'].transform('mean')
    m_attendance = df.groupby(['Model', 'Identifier'])['Attendance'].mean().reset_index(name='Attendance')
    sd_attendance = df.groupby(['Model', 'Identifier'])['Attendance'].std().reset_index(name='Deviation')
    data_s = []
    try:
        a = df['Accuracy'].unique()
        for mod, grp in df.groupby('Model'):
            data_s.append(pd.DataFrame({
                'Efficiency': grp.groupby('Identifier')['Score'].mean().tolist(),
                'Inaccuracy': grp.groupby('Identifier')['Accuracy'].mean().tolist(),
                'Identifier': grp['Identifier'].unique().tolist(),
                'Model': mod
            }))
    except:
        for mod, grp in df.groupby('Model'):
            data_s.append(pd.DataFrame({
                'Efficiency': grp.groupby('Identifier')['Score'].mean().tolist(),
                'Identifier': grp['Identifier'].unique().tolist(),
                'Model': mod
            }))
    df2 = pd.concat(data_s)
    df2 = pd.merge(df2, m_attendance, on=['Model', 'Identifier'])
    df2 = pd.merge(df2, sd_attendance, on=['Model', 'Identifier'])
    return df2

def merge_parameters(df, parameters, variables):
    """
    Aggregates results by parameter combinations, computing attendance, deviation, efficiency, and inaccuracy.
    """
    assert(len(parameters) == 2)
    if ('Attendance' in variables) or ('Deviation' in variables):
        df['Attendance'] = df.groupby(parameters + ['Identifier', 'Round'])['State'].transform('mean')
        m_attendance = df.groupby(parameters + ['Identifier'])['Attendance'].mean().reset_index(name='Attendance')
        sd_attendance = df.groupby(parameters + ['Identifier'])['Attendance'].std().reset_index(name='Deviation')
    print("Attendance and Deviation ready!")
    data_s = []
    A = df.groupby(parameters)
    p1 = df[parameters[0]].unique().tolist()
    p2 = df[parameters[1]].unique().tolist()
    for m in product(p1, p2):
        grp = A.get_group(m)
        diccionario = {}
        diccionario[parameters[0]] = m[0]
        diccionario[parameters[1]] = m[1]
        diccionario['Identifier'] = grp['Identifier'].unique().tolist()
        diccionario['Efficiency'] = grp.groupby('Identifier')['Score'].mean().tolist()
        diccionario['Inaccuracy'] = grp.groupby('Identifier')['Accuracy'].mean().tolist()
        data_s.append(pd.DataFrame(diccionario))
    df_ = pd.concat(data_s)
    df_ = pd.merge(df_, m_attendance, on=parameters + ['Identifier'])
    df_ = pd.merge(df_, sd_attendance, on=parameters + ['Identifier'])
    print("Dataframe ready!")
    return df_

class PriorityList():
    """
    Implements a simple priority queue using a dictionary.
    """
    def __init__(self):
        self.dictionary = {}

    def __str__(self):
        string = '['
        initial = True
        for cost in self.dictionary:
            elements = self.dictionary[cost]
            for element in elements:
                if initial:
                    string += '(' + str(element) + ',' + str(cost) + ')'
                    initial = False
                else:
                    string += ', (' + str(element) + ',' + str(cost) + ')'
        return string + ']'

    def push(self, element, cost):
        """Add an element with a given cost to the priority list."""
        try:
            self.dictionary[cost].append(element)
        except:
            self.dictionary[cost] = [element]

    def pop(self):
        """Remove and return the element with the lowest cost."""
        min_cost = np.min(np.array(list(self.dictionary.keys())))
        candidates = self.dictionary[min_cost]
        element = candidates.pop()
        if len(candidates) == 0:
            del self.dictionary[min_cost]
        return element

    def is_empty(self):
        """Check if the priority list is empty."""
        return len(self.dictionary) == 0

def difference(x, y):
    """
    Returns the absolute difference between the two elements of x
    corresponding to the two largest values in y (used for analysis).
    """
    if len(x) == 0:
        return np.nan
    elif len(x) < 2:
        return 0
    else:
        y1 = PriorityList()
        for i, j in enumerate(y):
            y1.push(i, 100 - j)
        ind_max1 = y1.pop()
        ind_max2 = y1.pop()
        return np.abs(x[ind_max1] - x[ind_max2])
