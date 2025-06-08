print("Importing packages...")
from random import choice, sample, randint, uniform
import numpy as np
import pandas as pd
from os import remove
from itertools import product
print("Ready!")

def distance(x, y):
    """Returns the absolute distance between two values."""
    return abs(x - y)

class Predictor:
    def __init__(self, memory_length, mirrors):
        # Window size for memory, random if memory_length >= 1
        if memory_length < 1:
            self.window = 0
        else:
            self.window = randint(1, memory_length)
        # Whether the predictor is cyclic
        self.cyclic = choice([True, False])
        # Whether the predictor uses the "mirror" strategy
        if mirrors:
            self.mirror = choice([True, False])
        else:
            self.mirror = False
        self.inaccuracy = [np.nan]  # List to store inaccuracy over time
        self.prediction = []      # List to store predictions

    def predict(self, memory, num_agents, threshold):
        """Make a prediction based on memory, number of agents, and threshold."""
        memory_length = len(memory)
        cyclic = self.cyclic
        window = self.window
        mirror = self.mirror
        if cyclic:
            indices = list(range(memory_length - 1, -1, -window))
            values = [memory[x] for x in indices]
        else:
            values = memory[-window:]
        try:
            prediction = int(np.mean(values))
        except:
            prediction = memory[-1]
        if mirror:
            prediction = num_agents - prediction
        self.prediction.append(prediction)

    def __str__(self):
        window = str(self.window)
        cyclic = "-cyclic" if self.cyclic else "-window"
        mirror = "-mirror" if self.mirror else ""
        return window + cyclic + mirror

class Agent:
    def __init__(self, states, scores, predictors, active_predictor):
        self.state = states           # List of states (attendance decisions)
        self.score = scores           # List of scores
        self.predictors = predictors  # List of predictors
        self.active_predictor = active_predictor  # List of active predictors

    def __str__(self):
        return "S:{0}, Sc:{1}, P:{2}".format(self.state, self.score, str(self.active_predictor[-1]))

class Bar:
    def __init__(self, num_agents, threshold, memory_length, num_predictors, identifier, mirrors):
        self.num_agents = num_agents
        self.threshold = threshold
        self.memory_length = memory_length
        self.num_predictors = num_predictors
        self.identifier = identifier
        self.history = []         # Attendance history
        self.predictors = []      # All possible predictors

        # Create all possible predictors
        windows = list(range(1, memory_length + 1))
        cyclics = [True, False]
        if mirrors:
            mirrors_ = [True, False]
        else:
            mirrors_ = [False]
        tuples = product(windows, cyclics, mirrors_)
        for tpl in tuples:
            p = Predictor(self.memory_length, mirrors)
            p.window = tpl[0]
            p.cyclic = tpl[1]
            p.mirror = tpl[2]
            self.predictors.append(p)

        # Create agents, each with a random subset of predictors
        self.agents = []
        for i in range(self.num_agents):
            if self.num_predictors <= len(self.predictors):
                agent_predictors = sample(self.predictors, self.num_predictors)
            else:
                agent_predictors = self.predictors
            self.agents.append(Agent([randint(0, 1)], [], agent_predictors, [choice(agent_predictors)]))
        self.calculate_attendance()    # Initial attendance
        self.calculate_scores()        # Initial scores
        self.update_predictions()      # Initial predictions

    def calculate_states(self):
        """Update each agent's state (attendance decision) based on their predictor's prediction."""
        for a in self.agents:
            prediction = a.active_predictor[-1].prediction[-1] / self.num_agents
            if prediction <= self.threshold:
                a.state.append(1)
            else:
                a.state.append(0)

    def calculate_attendance(self):
        """Calculate total attendance for the current round."""
        attendance = np.sum([a.state[-1] for a in self.agents])
        self.history.append(attendance)

    def calculate_scores(self):
        """Update each agent's score based on attendance and threshold."""
        attendance = self.history[-1] / self.num_agents
        for a in self.agents:
            if a.state[-1] == 1:
                if attendance > self.threshold:
                    a.score.append(-1)
                else:
                    a.score.append(1)
            else:
                a.score.append(0)

    def update_predictions(self):
        """Update predictions for all predictors based on recent history."""
        history = self.history[-self.memory_length:]
        for p in self.predictors:
            p.predict(history, self.num_agents, self.threshold)

    def update_inaccuracy(self):
        """Update inaccuracy for all predictors."""
        history = self.history
        for p in self.predictors:
            if self.memory_length == 0:
                p.inaccuracy.append(1)
            else:
                predictions = p.prediction
                inaccuracy_history = np.mean([distance(history[i + 1], predictions[i]) for i in range(len(history) - 1)])
                p.inaccuracy.append(inaccuracy_history)

    def choose_predictor(self, DEBUG=False):
        """Each agent chooses the predictor with the best (lowest) recent inaccuracy."""
        for a in self.agents:
            inaccuracies = [p.inaccuracy[-1] for p in a.predictors]
            index_min = np.argmin(inaccuracies)
            if DEBUG:
                print("Accuracies are:")
                print([f"{str(p)} : {p.inaccuracy[-1]}" for p in a.predictors])
            a.active_predictor.append(a.predictors[index_min])

    def play_round(self, round_num):
        """Play a single round: update states, attendance, scores, inaccuracy, predictors, and predictions."""
        self.calculate_states()
        self.calculate_attendance()
        self.calculate_scores()
        self.update_inaccuracy()
        self.choose_predictor(DEBUG=False)
        self.update_predictions()

    def create_agents_dataframe(self):
        """Create a DataFrame with all agents' data for analysis or saving."""
        round_list = []
        agent_list = []
        state_list = []
        score_list = []
        policy_list = []
        prediction_list = []
        inaccuracy_list = []
        num_iterations = len(self.history) - 1
        for i in range(self.num_agents):
            for r in range(num_iterations):
                agent_list.append(i)
                round_list.append(r)
                a = self.agents[i]
                p = a.active_predictor[r]
                state_list.append(a.state[r])
                score_list.append(a.score[r])
                policy_list.append(str(p))
                prediction_list.append(p.prediction[r])
                inaccuracy_list.append(p.inaccuracy[r])
        data = pd.DataFrame.from_dict({
            'Round': round_list,
            'Agent': agent_list,
            'State': state_list,
            'Score': score_list,
            'Policy': policy_list,
            'inaccuracy': inaccuracy_list,
            'Prediction': prediction_list
        })

        id_ = self.identifier if self.identifier != '' else 'A'
        data['Identifier'] = id_
        data['Memory'] = self.memory_length
        data['Num_predictors'] = self.num_predictors
        data['Num_agents'] = self.num_agents
        data = data[['Memory', 'Num_predictors', 'Identifier', 'Round', 'Agent',
                     'State', 'Score', 'Policy', 'Prediction', 'inaccuracy']]
        return data

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