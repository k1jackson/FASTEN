from .common import os, json
from .plot import plot_train, plot_predict
from .train import Trainer
from .predict import Predictor
from copy import deepcopy
import optuna


class Tuner:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.config = trainer.model.config
        self.study: optuna.Study = None

    def dump_trials(self, output_dir: str):
        trials_file = f"{output_dir}/trials.tsv" 
        trial_data = self.study.trials_dataframe().sort_values(by = "value")
        trial_data.to_csv(trials_file, sep = "\t", index = False)

        best_params = self.study.best_trial.params
        for arg, value in self.config["model"].items():
            if isinstance(value, list): self.config["model"][arg] = best_params[arg]
        for arg, value in self.config["train"].items():
            if isinstance(value, list): self.config["train"][arg] = best_params[arg]
        with open(f"{output_dir}/config.json", "w") as file:
            json.dump(self.config, file, indent = 4)

    def load_study(self, output_dir: str, study_name: str = "tune_data"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): 
            os.mkdir(output_dir)
            os.mkdir(f"{output_dir}/plots")
            os.mkdir(f"{output_dir}/plots/trials")
        storage_name = f"sqlite:///{output_dir}/{study_name}.db"
        sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(study_name = study_name, storage = storage_name, 
            load_if_exists = True, direction = "minimize", sampler = sampler)       
    
    def execute(self, n_trials: int):
        if not n_trials: return
        while len(self.study.get_trials(states = [optuna.trial.TrialState.COMPLETE])) < n_trials:
            self.study.optimize(Objective(self), n_trials = 1, catch = (ValueError, OverflowError))


class Objective:
    def __init__(self, tuner: Tuner):
        self.tuner = tuner
        self.trainer = tuner.trainer
        self.model = tuner.model
        self.trial_dir = f"{self.tuner.output_dir}/plots/trials"

    def __call__(self, trial) -> float: 
        trial_model, trial_train = dict(), dict()
        for arg, value in self.tuner.config["model"].items():
            if not isinstance(value, list): trial_model[arg] = value
            else: trial_model[arg] = trial.suggest_categorical(arg, value)
        for arg, value in self.tuner.config["train"].items():
            if not isinstance(value, list): trial_train[arg] = value
            else: trial_train[arg] = trial.suggest_categorical(arg, value)
        self.model.validate_args(trial_model, trial_train)
        try: self.trainer.execute()
        except ValueError: 
            message = "Training diverged: loss is NaN (possible exploding gradients)"
            raise optuna.exceptions.TrialPruned(message)
        
        plot_train(self.trainer, f"{self.trial_dir}/trial_{trial.number}")
        predictor = Predictor(self.model, deepcopy(self.trainer.test.dataset))
        mse, kld, nll = plot_predict(predictor, f"{self.trial_dir}/trial_{trial.number}")
        match self.model.args.loss_func:
            case "MSE": return mse.mean().mean()
            case "KLD": return kld.mean().mean()
            case _: return nll.mean().mean()