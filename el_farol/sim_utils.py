import pandas as pd

from os import remove
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional

from bar_classes import Bar, GaussianBar

class InteractiveBar:

    @staticmethod
    def simulation(
                num_agents: int, 
                threshold: float, 
                memory_length: int, 
                num_predictors: int, 
                num_rounds: int, 
                identifier: Optional[str]='', 
                mirrors: Optional[bool]=True, 
                DEBUG: Optional[bool]=False, 
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
        
class InteractiveGaussianBar:

    @staticmethod
    def simulation(
                num_agents: int, 
                threshold: float,
                std_threshold: float, 
                seed: int,
                memory_length: int, 
                num_predictors: int, 
                num_rounds: int, 
                identifier: Optional[str]='', 
                mirrors: Optional[bool]=True, 
                DEBUG: Optional[bool]=False, 
            ) -> pd.DataFrame:
        """Run a full simulation and optionally save the results."""
        bar = GaussianBar(
            num_agents, 
            threshold, 
            std_threshold,
            seed,
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
        return data

    @staticmethod
    def run_sweep(
                memories: int, 
                predictors: int, 
                std_thresholds: int,
                seed: int,
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
        for sd in tqdm(std_thresholds, desc="Running std thresholds sweeps", leave=False):
            for d in tqdm(memories, desc="Running memory sweeps", leave=False):
                for k in tqdm(predictors, desc="Running predictor sweeps", leave=False):
                    for N in tqdm(num_agents, desc="Running agent sweeps", leave=False):
                        for T in tqdm(num_rounds, desc="Running round sweeps", leave=False):
                            if DEBUG:
                                print('Running experiments with parameters:')
                                print(f"Memory={d}; Predictors={k}; Number of agents={N}; Number of rounds={T}; Mirrors?:{mirrors}")
                            for i in tqdm(range(num_experiments), desc="Running experiments", leave=False):
                                df = InteractiveGaussianBar.simulation(
                                    num_agents=N, 
                                    threshold=threshold, 
                                    std_threshold=sd,
                                    seed=seed,
                                    memory_length=d, 
                                    num_predictors=k, 
                                    num_rounds=T, 
                                    identifier=identifier, 
                                    mirrors=mirrors, 
                                    DEBUG=DEBUG
                                )
                                identifier += 1
                                df_list.append(df)
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            return combined_df
        else:
            print("No data was generated during the simulations.")
            return pd.DataFrame()