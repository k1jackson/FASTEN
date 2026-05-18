from __future__ import annotations
from .common import pd, np, torch, os
from .data import Samples, Statistics, Dataset
from .learn import Loss
from .model import Model
from .plot import plot_predict


class Predictor():
    def __init__(self, model: Model, dataset: Dataset = Dataset()):
        self.model: Model = model
        self.true: Dataset = dataset
        self.pred: Dataset = Dataset()
        self.error: pd.DataFrame = None
        self.criterion: Loss = Loss(model)
    
    def load_inputs(self, inputs_file: str, num_runs: int): 
        self.true.stats.load_data(inputs_file, self.model.inputs, None)
        self.true.stats.scale_inputs(self.model.input_scaler)
        num_runs = max(num_runs, 1)
        index = np.arange(self.true.stats.inputs.shape[0]).repeat(num_runs)
        self.true.samples.inputs = self.true.stats.inputs.loc[index]
        self.true.samples.inputs = self.true.samples.inputs.reset_index(drop = True)
        self.true.samples.group_data()
        
    def dump_statistics(self, outputs_file: str): 
        self.pred.stats.dump_data(outputs_file)
    
    def dump_samples(self, outputs_file: str):
        self.pred.samples.dump_data(outputs_file)

    def execute(self):
        self.pred.stats = self.predict_stats()
        self.pred.stats.unscale_inputs(self.model.input_scaler)
        self.pred.stats.unscale_outputs(self.model.param_scaler)
        self.true.stats.unscale_inputs(self.model.input_scaler)
        if isinstance(self.true.stats.outputs, pd.DataFrame): 
            self.true.stats.unscale_outputs(self.model.param_scaler)
        self.pred.samples = self.predict_samples()

    def evaluate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dependent = [output.dist.support.dependent for output in self.model.outputs.values()]
        mse, kld, nll = self.criterion.evaluate(self.true, self.pred, any(dependent))
        mse = pd.DataFrame(mse.detach().cpu().numpy(), columns = self.model.params)
        kld = pd.DataFrame(kld.detach().cpu().numpy(), columns = self.model.outputs)
        if any(dependent): return mse, kld, None
        nll = pd.DataFrame(nll.detach().cpu().numpy(), columns = self.model.outputs)
        return mse, kld, nll

    def predict_stats(self) -> Statistics:
        self.model.network.eval()
        inputs = self.true.stats.inputs.values
        x = torch.from_numpy(inputs).to(self.model.args.device)
        with torch.no_grad(): y = self.model.network(x)
        outputs = y.detach().cpu().numpy()
        params = pd.DataFrame(outputs, columns = self.model.params)
        return Statistics(self.true.stats.inputs, params)
    
    def predict_samples(self) -> Samples:
        outputs = pd.DataFrame()
        index = self.true.samples.group.values
        for label, output in self.model.outputs.items():
            mask = [param in output.dist.params for param in self.model.params]
            stats = self.pred.stats.outputs.iloc[index, mask]
            params = torch.from_numpy(stats.values)
            fit = output.dist.base(*params.unbind(dim = 1))
            outputs[label] = fit.sample().numpy()
        return Samples(self.true.samples.inputs, outputs)