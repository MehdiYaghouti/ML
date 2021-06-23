import pandas as pd
from ax import *

import torch
import numpy as np
from   scipy.stats import binom


from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.plot.exp_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

# Plotting imports and initialization
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
from ax.plot.pareto_utils import compute_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier

from typing import Optional
from torch  import Tensor
from botorch.test_functions.base import MultiObjectiveTestProblem

from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO


from ax.modelbridge.modelbridge_utils import observed_hypervolume
from botorch.test_functions.multi_objective import BraninCurrin

N =  [ 20,  18,  19,  17,  32,  20  ]
p =  [ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ]
w =  [ 1,   1,   1,  1,   1,  1   ]

def EggHolder(X):
    # Max: 1
    # Min: -0.525
    # (x0_max,x1_max): (12,10)
    # Range(x0): 0-20
    # Range(x1): 0-20
    x_0 = X[..., 0]
    x_1 = X[..., 1]
    x_0 = x_0*5+40
    x_1 = x_1*2+10


    return  (-(x_1 + 47) * np.sin(np.sqrt(abs(x_0/2 + (x_1 + 47)))) -x_0 * np.sin(np.sqrt(abs(x_0 - (x_1 + 47)))))   /173.78273


class EggHolderProductBitCost(MultiObjectiveTestProblem):
    """Two objective problem composed of the following discrete objectives:
            y         =    TT_k binom.pmf(b_k, N_k, p_k)
            Bit_Cost  =  - (Sum_k w_k * b_k)
    """

    dim = 6
    num_objectives = 2
    _bounds = [(0.0, 20.), (0.0, 20.0), (0.0, 20.0), (0.0, 20.0), (0.0, 20.0), (0.0, 20.0)]
    _ref_point = [0.5, -(w[0]+w[1]+w[2]+w[3]+w[4]+w[5])*20]
    _max_hv = 1e3

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        super().__init__(noise_std=noise_std, negate=negate)


    def EggHolderProduct(self,X: Tensor) -> Tensor:
        return EggHolder(torch.stack([X[..., 0], X[..., 1]], dim=-1)) *EggHolder(torch.stack([X[..., 2], X[..., 3]], dim=-1))  *EggHolder(torch.stack([X[..., 4], X[..., 5]], dim=-1))


    def BitCost(self,X: Tensor) -> Tensor:
        bitCost = torch.zeros(X.shape[0])
        for i in range(self.dim):
            bitCost += X[...,i]*w[i]
        return -bitCost

    def evaluate_true(self, X: Tensor) -> Tensor:
        y       = self.EggHolderProduct(X=X)
        bitCost = self.BitCost(X=X)

        return torch.stack([y, bitCost], dim=-1)



eggHolderProductBitcost = EggHolderProductBitCost(negate=False).to(dtype=torch.double, device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),)


bits = [ RangeParameter(name='b'+str(i),lower=0, upper=20, parameter_type=ParameterType.INT) for i in range(len(N))]


search_space = SearchSpace(parameters=bits,)

class MetricA(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return float(eggHolderProductBitcost(torch.tensor(x))[0])

class MetricB(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return float(eggHolderProductBitcost(torch.tensor(x))[1])

metric_a = MetricA("Accuracy", ["b"+str(i) for i in range(6)], noise_sd=0.0, lower_is_better=False)
metric_b = MetricB("BitCost", ["b"+str(i) for i in range(6)], noise_sd=0.0, lower_is_better=False)

mo = MultiObjective(metrics=[metric_a, metric_b],)

objective_thresholds = [ ObjectiveThreshold(metric=metric, bound=val, relative=False)
                         for metric, val in zip(mo.metrics, eggHolderProductBitcost.ref_point)   ]

optimization_config = MultiObjectiveOptimizationConfig(objective=mo,objective_thresholds=objective_thresholds,)


N_INIT = 10
N_BATCH = 80


EHVIexperiment = build_experiment()
EHVIdata = initialize_experiment(EHVIexperiment)

EHVIhvList = []
EHVImodel = None
for i in range(N_BATCH):
    EHVImodel = get_MOO_EHVI(experiment=EHVIexperiment,data=EHVIdata,)
    generatorRun = EHVImodel.gen(1)
    trial = EHVIexperiment.new_trial(generator_run=generatorRun)
    trial.run()
    EHVIdata = Data.from_multiple_data([EHVIdata, trial.fetch_data()])

    exp_df = exp_to_df(EHVIexperiment)
    outcomes = np.array(exp_df[['Accuracy', 'BitCost']], dtype=np.double)
    
    try:
        hv = observed_hypervolume(modelbridge=EHVImodel)
    except:
        hv = 0
        print("Failed to compute hv")
    EHVIhvList.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

EHVIoutcomes = np.array(exp_to_df(EHVIexperiment)[['Accuracy', 'BitCost']], dtype=np.double)

frontier = compute_pareto_frontier(experiment=EHVIexperiment,data=EHVIexperiment.fetch_data(),primary_objective=metric_a,secondary_objective=metric_b,absolute_metrics=["Accuracy","BitCost"],num_points=25,)


render(plot_pareto_frontier(frontier, CI_level=0.90))
