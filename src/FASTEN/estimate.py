from .common import np, pd, torch, F, os, tqdm
from .param import ModelDist, Constraint
from .config import ModelOutput
from pandas.api.typing import DataFrameGroupBy


class Estimator(): 

    class Moments():
        @staticmethod
        def Exponential(data: pd.Series) -> torch.Tensor:
            mean = data.mean()
            if not mean: raise AssertionError()
            return torch.tensor([1 / mean])
        @staticmethod
        def Normal(data: pd.Series) -> torch.Tensor: 
            mean, var = data.mean(), data.var()
            if np.isnan(var) or not var: raise AssertionError()
            return torch.tensor([mean, np.sqrt(var)])
        @staticmethod
        def HalfNormal(data: pd.Series) -> torch.Tensor: 
            mean = np.sqrt(data.pow(2).mean())
            if not mean: raise AssertionError()
            return torch.tensor([mean])
        @staticmethod
        def LogNormal(data: pd.Series) -> torch.Tensor: 
            mean, var = np.log(data).mean(), np.sqrt(np.log(data).var())
            if np.isnan(var) or not var: raise AssertionError()
            return torch.tensor([mean, var])
        @staticmethod
        def Uniform(data: pd.Series) -> torch.Tensor: 
            if data.min() == data.max(): # raise AssertionError()
                return torch.tensor([data.min(), data.max()])
            return torch.tensor([data.min(), data.max()])
        @staticmethod
        def Geometric(data: pd.Series) -> torch.Tensor: 
            return torch.tensor([1 / (1 + data.mean())])
        @staticmethod
        def Poisson(data: pd.Series) -> torch.Tensor: 
            mean = data.mean()
            if not mean: raise AssertionError()
            return torch.tensor([mean])
        @staticmethod
        def Bernoulli(data: pd.Series) -> torch.Tensor: 
            return torch.tensor([data.mean()])
        @staticmethod
        def Laplace(data: pd.Series) -> torch.Tensor:
            mad = (data - data.median()).abs().mean() 
            if mad <= 0: raise AssertionError()
            return torch.tensor([data.median(), mad])
        @staticmethod
        def Pareto(data: pd.Series) -> torch.Tensor: 
            log_sum = np.log(data / data.min()).sum()
            if not log_sum: raise AssertionError()
            return torch.tensor([data.min(), data.shape[0] / log_sum])
        @staticmethod
        def Binomial(data: pd.Series) -> torch.Tensor: 
            mean, var = data.mean(), data.var()
            if not var or np.isnan(var): raise AssertionError()
            if not mean or var >= mean: raise AssertionError()
            probs = 1 - var / mean
            total_counts = max(mean / probs, data.max())
            return torch.tensor([total_counts, np.log(probs / (1 - probs))])
        @staticmethod
        def NegativeBinomial(data: pd.Series) -> torch.Tensor:
            mean, var = data.mean().item(), data.var().item()
            if not var or np.isnan(var): raise AssertionError()
            if not mean or var <= mean: raise AssertionError()
            total_counts, probs = mean**2 / (var - mean), 1 - mean / var
            return torch.tensor([total_counts, np.log(probs / (1 - probs))])
    
    def __init__(self, estimator: str, outputs: dict[str, ModelOutput]):
        self.estimator: str = estimator
        self.outputs: dict[str, ModelOutput] = outputs

    def execute(self, groups: DataFrameGroupBy) -> pd.DataFrame:
        torch.set_num_threads(os.cpu_count())
        params = [self.parallelize(groups, label) for label in self.outputs]
        return pd.concat(params, axis = 1)
    
    def parallelize(self, total_groups: DataFrameGroupBy, label: str) -> pd.DataFrame:
        output, groups, params = self.outputs[label], total_groups[label], dict()
        for group, data in tqdm(groups, desc = output.name):
            params[group] = self.estimate(output, data)
        return pd.DataFrame.from_dict(params, "index", None, output.dist.params).sort_index()
        
    def estimate(self, output: ModelOutput, data: pd.Series) -> np.ndarray: 
        if self.estimator == "MoM" and hasattr(self.Moments, output.dist.name):
            method = getattr(self.Moments, output.dist.name)
            try: return method(data).numpy()
            except AssertionError: pass
        return self.max_likelihood(output.dist, torch.from_numpy(data.values))

    def max_likelihood(self, dist: ModelDist, data: torch.Tensor) -> np.ndarray: 
        self.load_constraints(dist, data)
        if dist.support.discrete: data = data.to(int)
        weights = torch.randn(len(dist.params), requires_grad = True, dtype = float)
        optimizer = torch.optim.LBFGS([weights], max_iter = 200, line_search_fn = "strong_wolfe",
                                      tolerance_grad = 1e-12, tolerance_change = 1e-12)

        def closure():
            optimizer.zero_grad()
            params = self.apply_constraints(weights) 
            params.nan_to_num_(nan = 1e-16)
            fit = dist.base(*params)
            loss = -1 * fit.log_prob(data).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad(): 
            return self.apply_constraints(weights).numpy()

    def load_constraints(self, dist: ModelDist, data: pd.Series): 
        for rule in Constraint.RULES: setattr(self, rule, torch.zeros(len(dist.params), dtype = bool))
        for value in Constraint.VALUES: setattr(self, value, torch.zeros(len(dist.params), dtype = float))
        for i, param in enumerate(dist.params.values()):
            min_val, max_val = data.min().item(), data.max().item()
            param.load_constraints(dist.support, min_val, max_val)
            for rule in Constraint.RULES: getattr(self, rule)[i] = param.constraints.get_rule(rule)
            for value in Constraint.VALUES: getattr(self, value)[i] = param.constraints.get_value(value)

    def apply_constraints(self, weights: torch.Tensor) -> torch.Tensor:
        params = weights.clone()
        params[self.greater_than] = F.softplus(params[self.greater_than]) + self.lower[self.greater_than]
        params[self.less_than] = self.upper[self.less_than] - F.softplus(params[self.less_than])
        params[self.between] = F.sigmoid(params[self.between]) * self.interval[self.between] + self.lower[self.between]
        return params