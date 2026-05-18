from .common import np, torch, nn, F, os, json
from .config import ModelArgs, ModelInput, ModelOutput
from .param import ModelParam, Constraint
from .utils import Scaler, Encoder
from pydantic import ValidationError
from zipfile import ZipFile
import joblib, random


class Network(nn.Module):
    def __init__(self, args: ModelArgs, input_size: int, output_size: int):
        super(Network, self).__init__()
        self.layers, sizes = nn.ModuleList(), [input_size]
        if not args.hidden_layers: sizes.append(output_size)
        elif args.architecture == "pyramidal":
            step = (args.hidden_size - output_size) / args.hidden_layers
            for i in range(args.hidden_layers):
                current_size = int(args.hidden_size - (i * step))
                sizes.append(max(current_size, output_size) )
            sizes.append(output_size)
        elif args.architecture == "rectangular":
            sizes.extend([args.hidden_size] * args.hidden_layers)
            sizes.append(output_size)
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))

    def load_constraints(self, params: dict[str, ModelParam]):
        for rule in Constraint.RULES: 
            self.register_buffer(rule, torch.zeros(len(params), dtype = bool))
        for value in Constraint.VALUES: 
            self.register_buffer(value, torch.zeros(len(params), dtype = float))
        for i, param in enumerate(params.values()):
            for rule in Constraint.RULES: self.get_buffer(rule)[i] = param.priors.get_rule(rule)
            for value in Constraint.VALUES: self.get_buffer(value)[i] = param.priors.get_value(value)
        for label in ["low", "high"]:
            loop = (param.base == "label" for param in params.values())
            mask = np.fromiter(loop, dtype = bool, count = len(params))
            self.register_buffer(label, torch.from_numpy(mask))
        self.to(next(self.parameters()).device)

    def load_scaler(self, scaler: Scaler):
        if not scaler.fitted: return
        self.register_buffer("min", torch.from_numpy(scaler.min))
        self.register_buffer("range", torch.from_numpy(scaler.range))
        self.to(next(self.parameters()).device)
    
    def forward(self, x): 
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        x[:, self.greater_than] = F.softplus(x[:, self.greater_than]) + (self.lower[self.greater_than] - self.min[self.greater_than]) / self.range[self.greater_than]
        x[:, self.less_than] = (self.upper[self.less_than] - self.min[self.less_than]) / self.range[self.less_than] - F.softplus(x[:, self.less_than])
        x[:, self.between] = F.sigmoid(x[:, self.between]) * self.interval[self.between] / self.range[self.between] + (self.lower[self.between] - self.min[self.between]) / self.range[self.between]
        x[:, self.high] = x[:, self.low] + F.softplus(x[:, self.high] - x[:, self.low]) 
        return x


class Model():
    def __init__(self, config_file: str = None, model_file: str = None):
        torch.set_default_dtype(torch.float64)
        self.inputs: dict[str, ModelInput] = dict()
        self.outputs: dict[str, ModelOutput] = dict()
        self.params: dict[str, ModelParam] = dict()
        self.args: ModelArgs = None
        if config_file:
            with open(config_file, "r") as file: 
                self.config = json.load(file)
        else: self.config = {"train": {}}
        if not model_file: self.create()
        else: self.load(model_file)

    def validate_data(self, inputs: dict, outputs: dict):
        for label, config in inputs.items():
            try: self.inputs[label] = ModelInput(**config, label = label)
            except ValidationError as error: raise error
        for label, config in outputs.items():
            try: output = ModelOutput(**config, label = label)
            except ValidationError as error: raise error
            params = output.dist.load_parameters(output)
            self.params.update(params)
            self.outputs[label] = output

    def validate_args(self, model: dict, train: dict):
        model = {k: v for k, v in model.items() if not isinstance(v, list)}
        train = {k: v for k, v in train.items() if not isinstance(v, list)}
        try: self.args = ModelArgs(**{**model, **train})
        except ValidationError as error: raise error
        if self.args.rand_seed is not None: 
            self.set_seed(self.args.rand_seed)
        self.network = Network(self.args, len(self.inputs), len(self.params))
        self.network = self.network.to(self.args.device)
        self.network.load_constraints(self.params)
        self.network.load_scaler(self.param_scaler)

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
    
    def create(self):
        self.input_scaler, self.param_scaler = Scaler(), Scaler()
        self.validate_data(self.config["inputs"], self.config["outputs"])
        self.validate_args(self.config["model"], self.config["train"])
        self.encoder = Encoder(self.inputs)
        
    def load(self, model_file):
        with ZipFile(model_file, "r") as model:
            with model.open("input_scaler.pkl", "r") as file:
                self.input_scaler = joblib.load(file)
            with model.open("param_scaler.pkl", "r") as file:
                self.param_scaler = joblib.load(file)
            with model.open("encoder.pkl", "r") as file:
                self.encoder = joblib.load(file)  
            with model.open("config.json", "r") as file:
                model_config = json.load(file)
                self.config["inputs"] = model_config["inputs"]
                self.config["outputs"] = model_config["outputs"]
                self.config["model"] = model_config["model"]
            self.validate_data(self.config["inputs"], self.config["outputs"])
            self.validate_args(self.config["model"], self.config["train"])
            with model.open("model.pth", "r") as file:
                self.network.load_state_dict(torch.load(file))
                self.network.load_scaler(self.param_scaler)
            
    def dump(self, output):
        model_file = f"{output}/model.zip"
        with ZipFile(model_file, "w") as model:
            with model.open("input_scaler.pkl", "w") as file:
                joblib.dump(self.input_scaler, file)
            with model.open("param_scaler.pkl", "w") as file:
                joblib.dump(self.param_scaler, file)
            with model.open("encoder.pkl", "w") as file:
                joblib.dump(self.encoder, file)  
            with model.open("model.pth", "w") as file:
                torch.save(self.network.state_dict(), file) 
            config = json.dumps(self.config, indent = 4)
            model.writestr("config.json", config)

    # def extract_config(self) -> dict:
    #     train, model = self.args.model_dump_json(), dict()
    #     train["device"], train["optimizer"] = train["device"].type, train["optimizer"].__name__
    #     for key in ["architecture", "hidden_layers", "hidden_size"]: model[key] = train.pop(key)
    #     inputs = {label: value.model_dump_json() for label, value in self.inputs.items()}
    #     outputs = {label: value.model_dump_json() for label, value in self.outputs.items()}
    #     return {"inputs": inputs, "outputs": outputs, "train": train, "model": model}
        