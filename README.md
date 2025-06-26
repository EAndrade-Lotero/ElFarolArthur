# ElFarolArthur

**ElFarolArthur** is a Python implementation of Brian Arthur's method of predictors to solve [El Farol Bar problem](https://sites.santafe.edu/~wbarthur/elfarol.htm).

---

## Overview

The primary goal of ElFarolArthur is to allow us to understand the **method of predictors**, proposed by Brian Arthur as a solution to the problem. 

The following is our flow diagram that replicates the method:

![Flow diagram method of predictors](https://github.com/EAndrade-Lotero/ElFarolArthur/blob/master/LaTeX/FlowDiagram/diagram.pdf "Flow diagram method of predictors")

There are three main modules:

- **bar_classes**: Contains the class definition for predictors, agents and bar.
- **sim_utils**: Contains the methods to run the simulations and sweeps.
- **data_utils**: Contains the methods to obtain some of the measures used in our analyses.

This repository also contains the Jupyter notebooks used to create the figures for our paper:

[1] "A paradox in Brian Arthur's solution to his El Farol bar problem" *under review*.

---

## Usage example

See this [Jupiter notebook](https://github.com/EAndrade-Lotero/ElFarolArthur/blob/master/notebooks/Figure1.ipynb).

---

## Repository Structure

```

ElFarolArthur/
├── el_farol/                  # Core source code
│   ├── config                 # Paths for figures and data
│   ├── bar_classes.py         # Classes to replicate the method of predictors
│   ├── sim_utils.py           # Helper methods for simulations and sweeps
│   ├── main.py                # Helper functions to run simulations with fixed random seed
│   └── data_utils.py          # Helper methods for measurement
├── notebooks/                 # Jupyter notebooks to create figures used in [1]
├── images/                    # Main folder for figures used in [1]
│   ├── diagrams/           # Diagrams with the information flow in the method of predictors
│   ├── Figures for paper/     # Figures used in [1]
├── LaTeX/FlowDiagram          # LaTeX code to create overal diagram
└── README.md                  # Project documentation

````

---

## Getting Started

### Prerequisites

- Python 3.7 or higher

### Installation

Clone the repository:

```bash
git clone https://github.com/EAndrade-Lotero/ElFarolArthur.git
cd ElFarolArthur
````

Install the required packages:

```bash
pip install -r requirements.txt
```