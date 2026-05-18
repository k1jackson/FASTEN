from __future__ import annotations
from .common import np, pd, torch
from .config import ModelInput


class Scaler():
    def __init__(self):
        self.min = None
        self.max = None
        self.range = None
        self.fitted = False

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.min = np.min(data.values, axis = 0)
            self.max = np.max(data.values, axis = 0)
        if isinstance(data, torch.Tensor):
            self.min = data.min(dim = 0).values.numpy()
            self.max = data.max(dim = 0).values.numpy()
        self.range = self.max - self.min
        self.range[self.range == 0] = 1.0
        self.fitted = True

    def transform(self, data):
        if not self.fitted: self.fit(data)
        data -= self.min
        data /= self.range
        if isinstance(data, pd.DataFrame): data.clip(0, 1)
        if isinstance(data, torch.Tensor): torch.clamp(data, 0, 1)

    def inverse_transform(self, data):
        data *= self.range
        data += self.min


class Encoder():
    def __init__(self, inputs: dict[str, ModelInput]): 
        self.inputs = inputs
            
    def fit(self, data: pd.Series) -> bool:
        self.labels, self.names = [], []
        self.origins, self.strings = [], []
        for i, origin in enumerate(self.inputs.values()):
            if origin.type != "string": continue
            for string in data[origin.label].unique():
                self.labels.append(f"{origin.label}_{string}")
                self.names.append(f"{origin.name}: {string.capitalize()}")
                self.origins.append(i)
                self.strings.append(string)
        if len(self.strings) != len(set(self.strings)):
            raise AssertionError("Set of strings must be disjoint for all categorical inputs.")
        return len(self.strings) > 0
    
    def transform(self, data: pd.Series):
        if not self.fit(data): return
        col = {string: i for i, string in enumerate(self.strings)}
        values = np.zeros((data.shape[0], len(self.labels)))
        for origin in self.inputs.values():
            if origin.type != "string": continue
            for row, string in enumerate(data[origin.label]):
                values[row, col[string]] = 1
                label = self.labels[col[string]]
                name = self.names[col[string]]
                self.inputs[label] = ModelInput(label, name, "integer")
            data.drop(columns = origin.label, inplace = True)
            self.inputs.pop(origin.label)
        data[self.labels] = pd.DataFrame(values, dtype = float, index = data.index)