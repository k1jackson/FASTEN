from __future__ import annotations
from .common import pd, os, np
from .config import ModelOutput
from .utils import Scaler, Encoder
from .estimate import Estimator
from .model import Model
from sklearn.model_selection import train_test_split
from k_means_constrained import KMeansConstrained
from typing import Any
from abc import ABC


class Dataset():
    def __init__(self, samples: Samples = None, stats: Statistics = None):
        self.samples = samples if samples else Samples()
        self.stats = stats if stats else Statistics()

    def load_samples(self, data_file: str, model: Model):
        self.samples.load_data(data_file, model.inputs, model.outputs)
        self.samples.filter_data(model.outputs)
        for label, config in model.outputs.items(): 
            config.validate_data(self.samples.outputs, label)
        for label, config in model.inputs.items(): 
            config.validate_data(self.samples.inputs, label)
        self.samples.encode_inputs(model.encoder)
        self.samples.scale_inputs(model.input_scaler)
        self.samples.group_data()

    def load_stats(self, model: Model, estimator: str):
        self.estimate_stats(Estimator(estimator, model.outputs))
        self.stats.scale_outputs(model.param_scaler)
        model.network.load_scaler(model.param_scaler)

    def estimate_stats(self, estimator: Estimator):
        input_data = self.samples.inputs.groupby(self.samples.group)
        output_data = self.samples.outputs.groupby(self.samples.group)
        self.stats.inputs = input_data.first().reset_index(drop = True)
        self.stats.outputs = estimator.execute(output_data)

    def split(self, split_prop: float, rand_seed: int): 
        if not split_prop: return None
        groups, index = self.stats.cluster_data(split_prop), self.stats.inputs.index
        train_index, test_index = train_test_split(index, test_size = split_prop, stratify = groups, random_state = rand_seed)
        train_index, test_index = sorted(train_index), sorted(test_index)
        train_samples, test_samples = self.samples.group.isin(train_index), self.samples.group.isin(test_index)
        test_stats = Statistics(self.stats.inputs.loc[test_index].reset_index(drop = True), 
            None if self.stats.outputs is None else self.stats.outputs.loc[test_index].reset_index(drop = True))
        self.stats = Statistics(self.stats.inputs.loc[train_index].reset_index(drop = True), 
            None if self.stats.outputs is None else self.stats.outputs.loc[train_index].reset_index(drop = True))
        test_samples = Samples(self.samples.inputs.loc[test_samples].reset_index(drop = True), 
            self.samples.outputs.loc[test_samples].reset_index(drop = True))
        self.samples = Samples(self.samples.inputs.loc[train_samples].reset_index(drop = True), 
            self.samples.outputs.loc[train_samples].reset_index(drop = True))
        return Dataset(samples = test_samples, stats = test_stats)


class Data(ABC): 
    def __init__(self, inputs: pd.DataFrame = None, outputs: pd.DataFrame = None):
        self.inputs, self.outputs = inputs, outputs
    
    def _repr_html_(self): 
        return pd.concat([self.inputs, self.outputs], axis = 1)._repr_html_()

    def __str__(self): 
        return pd.concat([self.inputs, self.outputs], axis = 1).__str__()

    def dump_data(self, data_file: str):
        data = pd.concat([self.inputs, self.outputs], axis = 1)
        data.to_csv(data_file, sep = '\t', index = False)
    
    def load_data(self, data_file: str, inputs: dict[str, Any], outputs: dict[str, Any]):
        inputs, outputs = list(inputs), list(outputs) if outputs else None
        self.inputs = pd.read_csv(data_file, sep = "\t", usecols = inputs)[inputs]
        self.inputs = self.inputs.sort_values(by = inputs)
        if outputs: 
            self.outputs = pd.read_csv(data_file, sep = "\t", usecols = outputs)[outputs]
            self.outputs = self.outputs.loc[self.inputs.index].reset_index(drop = True)
        self.inputs = self.inputs.reset_index(drop = True)


class Samples(Data):
    def __init__(self, inputs: pd.DataFrame = None, outputs: pd.DataFrame = None):
        super().__init__(inputs, outputs)
        if isinstance(inputs, pd.DataFrame): self.group_data()
        else: self.group: pd.Series = None

    def filter_data(self, outputs: dict[str, ModelOutput]):
        for label, output in outputs.items():
            if output.min_thresh is not None:
                mask = (self.outputs[label] > output.min_thresh)
                self.outputs = self.outputs[mask].reset_index(drop = True)
                self.inputs = self.inputs[mask].reset_index(drop = True)
            if output.max_thresh is not None:
                mask = (self.outputs[label] < output.max_thresh)
                self.outputs = self.outputs[mask].reset_index(drop = True)
                self.inputs = self.inputs[mask].reset_index(drop = True)

    def group_data(self):
        matches = (self.inputs != self.inputs.shift())
        self.group = matches.any(axis = 1).cumsum() - 1

    def scale_inputs(self, scaler: Scaler): scaler.transform(self.inputs)
    def unscale_inputs(self, scaler: Scaler): scaler.inverse_transform(self.inputs)
    def encode_inputs(self, encoder: Encoder): encoder.transform(self.inputs)


class Statistics(Data): 
    def cluster_data(self, split_prop: float = 1, n_clusters: int | None = None) -> np.ndarray:
        if n_clusters is None:
            n_clusters = int(self.inputs.shape[0] * split_prop / 5)
        if n_clusters < 1: return None
        kmeans = KMeansConstrained(n_clusters = n_clusters, size_min = 5)
        return kmeans.fit_predict(self.inputs)
    
    def scale_outputs(self, scaler: Scaler): scaler.transform(self.outputs)
    def unscale_outputs(self, scaler: Scaler): scaler.inverse_transform(self.outputs)
    def scale_inputs(self, scaler: Scaler): scaler.transform(self.inputs)
    def unscale_inputs(self, scaler: Scaler): scaler.inverse_transform(self.inputs)
