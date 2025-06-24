import pandas as pd

from os import remove
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional

from bar_classes import Bar

class InteractiveBar:

    @staticmethod
    def save_dataframe(
                dataFrame: pd.DataFrame, 
                filename: str, 
                initial: bool, 
                mirrors: Optional[bool]=True, 
                many: Optional[bool]=False
            ) -> None:
        """Save the DataFrame to a CSV file, handling file paths and initial/append modes."""
        if not many:
            if mirrors:
                filename = Path("../Data_Farol/normal/data_all/") / filename
            else:
                filename = Path("../Data_Farol/normal/data_no_mirrors/") / filename
        else:
            if mirrors:
                filename = Path("../Data_Farol/data_all/") / filename
            else:
                filename = Path("../Data_Farol/data_no_mirrors/") / filename
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

    @staticmethod
    def simulation(
                num_agents: int, 
                threshold: float, 
                memory_length: int, 
                num_predictors: int, 
                num_rounds: int, 
                initial: Optional[bool]=True, 
                identifier: Optional[str]='', 
                mirrors: Optional[bool]=True, 
                DEBUG: Optional[bool]=False, 
                to_file: Optional[bool]=False
            ) -> pd.DataFrame:
        """Run a full simulation and optionally save the results."""
        bar = Bar(
            num_agents, 
            threshold, 
            memory_length, 
            num_predictors, 
            identifier, 
            mirrors
        )
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
        filename = f'simulation-{memory_length}-{num_predictors}-{num_agents}-{num_rounds}.csv'
        if to_file:
            if num_agents < 1000:
                InteractiveBar.save_dataframe(data, filename, initial, mirrors)
            else:
                InteractiveBar.save_dataframe(data, filename, initial, mirrors, many=True)
            if DEBUG:
                print('Data saved in', filename)
        return data

    @staticmethod
    def run_sweep(
                memories: int, 
                predictors: int, 
                num_experiments: int, 
                num_agents: int, 
                threshold: float, 
                num_rounds: int, 
                mirrors: Optional[bool]=True,
                DEBUG: Optional[bool]=False
            ) -> pd.DataFrame:
        """Run a sweep of simulations over different parameter combinations."""
        if DEBUG:
            print('********************************')
            print('Running simulations...')
            print('********************************\n')
        identifier = 0
        df_list = []
        for d in tqdm(memories, desc="Running memory sweeps", leave=False):
            for k in tqdm(predictors, desc="Running predictor sweeps", leave=False):
                for N in tqdm(num_agents, desc="Running agent sweeps", leave=False):
                    for T in tqdm(num_rounds, desc="Running round sweeps", leave=False):
                        initial = True
                        if DEBUG:
                            print('Running experiments with parameters:')
                            print(f"Memory={d}; Predictors={k}; Number of agents={N}; Number of rounds={T}; Mirrors?:{mirrors}")
                        for i in tqdm(range(num_experiments), desc="Running experiments", leave=False):
                            df = InteractiveBar.simulation(
                                num_agents=N, 
                                threshold=threshold, 
                                memory_length=d, 
                                num_predictors=k, 
                                num_rounds=T, 
                                initial=initial, 
                                identifier=identifier, 
                                mirrors=mirrors, 
                                DEBUG=DEBUG
                            )
                            identifier += 1
                            initial = False
                            df_list.append(df)
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            return combined_df
        else:
            print("No data was generated during the simulations.")
            return pd.DataFrame()