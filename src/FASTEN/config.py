from .common import pd, torch
from .param import ModelDist
from typing import Literal, Self
from warnings import warn
import pydantic as pdc


class ModelArgs(pdc.BaseModel):
    test_split: float = pdc.Field(default = 0.1, ge = 0.0, lt = 1.0)
    valid_split: float = pdc.Field(default = 0.1, ge = 0.0, lt = 1.0)
    estimator: Literal["MoM", "MLE"] = "MLE"
    rand_seed: int | None = None 

    architecture: Literal["rectangular", "pyramidal"] = "pyramidal"
    hidden_layers: int = pdc.Field(default = 2, ge = 0)
    hidden_size: int = pdc.Field(default = 64, gt = 0)

    device: Literal["cpu", "cuda"] = "cpu"    
    batch_size: int = pdc.Field(default = 32, gt = 0)
    num_epochs: int = pdc.Field(default = 1e5, gt = 0)

    early_stop: bool = True
    patience: int = pdc.Field(default = 20, ge = 0)
    min_delta: float = pdc.Field(default = 0.0)
    
    optimizer: Literal["SGD", "Adam", "AdamW"] = "AdamW"
    loss_func: Literal["MSE", "KLD", "NLL"] = "NLL"
    learn_rate: float = pdc.Field(default = 1e-3, gt = 0.0) 
    weight_decay: float = pdc.Field(default = 0.0, ge = 0.0)
    momentum: float = pdc.Field(default = 0.0, ge = 0.0)

    @pdc.field_validator("device", mode = "after")
    @classmethod
    def validate_device(cls, value):
        if value == "cuda" and not torch.cuda.is_available():
            warn("PyTorch cannot find a compatible GPU. Defaulting to CPU.")
            return torch.device("cpu")
        return torch.device(value)
    
    @pdc.field_validator("optimizer", mode = "after")
    @classmethod
    def validate_optimizer(cls, value: str):
        return getattr(torch.optim, value)
    
    @pdc.model_validator(mode = "after")
    def validate_early_stop(self) -> Self:
        if not self.valid_split and self.early_stop:
            raise ValueError("Non-empty validation set required for early stopping.")
        return self
    
    @pdc.model_validator(mode = "after")
    def validate_splits(self) -> Self:
        if self.valid_split + self.test_split >= 1:
            raise ValueError("Non-empty training set required. Decrease size of validation or testing set.")
        return self


class ModelInput(pdc.BaseModel):
    label: str
    name: str = pdc.Field(default_factory = lambda data: data['label'])
    type: Literal["float", "integer", "string"] = "float"

    def validate_data(self, data: pd.DataFrame, label: str):
        if data[label].isna().any():
            raise ValueError(f"Training data contains missing or undefined values: {self.name}.")
        if self.type == "string" and not pd.api.types.is_string_dtype(data[label]):
            raise ValueError(f"Training data has invalid values: {self.name}.")
        if self.type in ["integer", "float"]:
            if not pd.api.types.is_numeric_dtype(data[label]):
                raise ValueError(f"Training data has invalid values: {self.name}.")
            else: data[label] = data[label].astype(float)
        if self.type == "integer" and (data[label] % 1 != 0).any(): 
            warn(f"Integer type specified for non-integer training data: {self.name}. Rounding to nearest integer.")
            data[label] = data[label].round()


class ModelOutput(pdc.BaseModel): # validate priors
    model_config = pdc.ConfigDict(arbitrary_types_allowed = True, extra = "allow")

    label: str
    dist: str | ModelDist
    name: str = pdc.Field(default_factory = lambda data: data['label'])
    type: Literal["float", "integer"] = "float"
    min_thresh: float | None = None
    max_thresh: float | None = None

    @pdc.field_validator("dist", mode = "after")
    @classmethod
    def validate_distribution(cls, value: str) -> ModelDist:
        try: dist = ModelDist(value)
        except AttributeError: raise ValueError(f"Invalid distribution specified.")
        return dist
    
    @pdc.model_validator(mode = "after")
    def validate_discrete(self) -> Self:
        if self.dist.support.discrete and self.type != "integer":
            raise ValueError(f"Discrete distribution specified for non-integer training data: {self.name}.")
        return self
    
    def validate_data(self, data: pd.DataFrame, label: str):
        if data[label].isna().any():
            raise ValueError(f"Training data contains missing or undefined values: {self.name}.")
        if not pd.api.types.is_numeric_dtype(data[label]):
            raise ValueError(f"Training data has invalid values: {self.name}.")
        else: data[label] = data[label].astype(float)
        if self.type == "integer" and (data[label] % 1 != 0).any(): 
            warn(f"Integer type specified for non-integer training data: {self.name}. Rounding to nearest integer.")
            data[label] = data[label].round()
        if not self.dist.support.validate(data[label]): 
            raise AssertionError(f"Training data contains values outside domain of distribution: {self.name}")