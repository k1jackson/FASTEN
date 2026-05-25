from __future__ import annotations
from .common import torch, pd
from torch.distributions import *
from typing import TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING: 
    from .config import ModelOutput


class ModelParam():
    def __init__(self, output: ModelOutput, label: str): 
        self.base: str = label.replace("0", "").replace("1", "")
        self.label: str = f"{output.label}_{label}"
        self.name: str = f"{output.name} {label.replace('_', ' ').title()}"
        self.priors = Constraint(output, self.base)
        self.constraints = deepcopy(self.priors)

    def load_constraints(self, support: Support, min_val: float, max_val: float):
        if support.upper == self.base: self.constraints.add_lower(max_val)
        if support.lower == self.base: self.constraints.add_upper(min_val)
        self.constraints.add_intervals()


class ModelDist(): 
    TYPES = {"Bernoulli", "Beta", "Binomial", "Cauchy", "Chi2", "ContinuousBernoulli", "Exponential", 
        "FisherSnedecor", "Gamma", "Geometric", "Gumbel", "HalfCauchy", "HalfNormal", "InverseGamma", 
        "Laplace", "LogNormal", "NegativeBinomial", "Normal", "Pareto", "Poisson", "Uniform"}  

    def __init__(self, name: str):
        self.name: str = name
        self.params: dict[str, ModelParam] = dict()
        self.support = Support(self.name)
        if hasattr(ModelDist, name): self.base = getattr(ModelDist, name)
        else: self.base = getattr(torch.distributions, name)
    
    def load_parameters(self, output: ModelOutput) -> dict[str, ModelParam]:
        base = getattr(torch.distributions, self.name)
        if self.name == "Uniform": labels = ["low", "high"]
        else: labels = list(base.arg_constraints.keys())
        for label in labels:
            if label == "probs": continue
            param = ModelParam(output, label)
            self.params[f"{output.label}_{label}"] = param
        return self.params

    @staticmethod
    def Bernoulli(logits: float | torch.Tensor) -> Bernoulli:
        return Bernoulli(logits = logits)
    @staticmethod
    def ContinuousBernoulli(logits: float | torch.Tensor) -> ContinuousBernoulli:
        return ContinuousBernoulli(logits = logits)
    @staticmethod
    def Geometric(logits: float | torch.Tensor) -> Geometric:
        return Geometric(logits = logits)
    @staticmethod
    def Binomial(total_count: float | torch.Tensor, logits: float | torch.Tensor) -> Binomial:
        if isinstance(total_count, torch.Tensor):
            rounded_total_count = total_count + (total_count.round() - total_count).detach()
            return Binomial(rounded_total_count, logits = logits)
        else: return Binomial(round(total_count), logits = logits)
    @staticmethod
    def NegativeBinomial(total_count: int | torch.Tensor, logits: float | torch.Tensor) -> NegativeBinomial:
        return NegativeBinomial(total_count, logits = logits)


class Constraint():
    POSITIVE = {"concentration", "total_count", "scale", "df", "rate", "alpha"}
    RULES = ["greater_than", "less_than", "between"]
    VALUES = ["lower", "upper", "interval"]

    def __init__(self, output: ModelOutput, base: str, eps: float = 1e-16):
        self.greater_than = self.less_than = self.between = False
        self.lower = self.upper = self.interval = 0.0
        if base in self.POSITIVE: self.greater_than, self.lower = True, eps
        if hasattr(output, f"{base}_min"): 
            self.add_lower(getattr(output, f"{base}_min"))
        if hasattr(output, f"{base}_max"): 
            self.add_upper(getattr(output, f"{base}_max"))
        self.add_intervals()

    def add_upper(self, upper): self.less_than, self.upper = True, upper
    def add_lower(self, lower): self.greater_than, self.lower = True, lower
    def add_intervals(self):
        if self.greater_than and self.less_than:
            self.greater_than = self.less_than = False
            self.between, self.interval = True, self.upper - self.lower
    
    def get_rule(self, rule: str) -> bool: return getattr(self, rule)
    def get_value(self, value: str) -> float: return getattr(self, value)


class Support:
    DISCRETE = {"Bernoulli", "Binomial", "Geometric", "NegativeBinomial", "Poisson"}
    POSITIVE = {"FisherSnedecor", "InverseGamma", "LogNormal"}
    NON_NEGATIVE = {"Binomial", "Poisson", "Geometric", "NegativeBinomial", "Chi2", "Exponential", "Gamma", "HalfCauchy", "HalfNormal"}
    PROBABILITY = {"Bernoulli", "Beta", "ContinuousBernoulli"}
    DEPENDENT = {"Uniform", "Binomial", "Pareto"}

    def __init__(self, name, eps = 1e-16):
        self.lower = self.upper = None
        self.discrete = (name in self.DISCRETE)
        if name in self.POSITIVE: self.lower = eps
        if name in self.NON_NEGATIVE: self.lower = 0.0
        if name in self.PROBABILITY: self.lower, self.upper = 0.0, 1.0
        self.dependent = (name in self.DEPENDENT)
        if name == "Uniform": self.lower, self.upper = "low", "high"
        if name == "Binomial": self.upper = "total_count"
        if name == "Pareto": self.lower = "scale"

    def validate(self, data: pd.Series) -> bool:
        if isinstance(self.lower, float) and (data < self.lower).any(): return False
        if isinstance(self.upper, float) and (data > self.upper).any(): return False
        return True

    def get_bound(self, attr: str, params: dict[str, ModelParam], stats: torch.Tensor, fit: Distribution, std_devs: int = 3) -> torch.Tensor:
        bound, sign = getattr(self, attr), -1 if attr == "lower" else 1
        if bound is None: return (fit.mean + sign * std_devs * torch.sqrt(fit.variance)).unsqueeze(-1)
        if isinstance(bound, str): return stats[:, [param.base == bound for param in params.values()]]
        else: return torch.tensor([bound]).repeat(stats.shape[0]).unsqueeze(-1)