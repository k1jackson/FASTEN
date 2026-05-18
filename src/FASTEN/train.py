from .common import torch, os, tqdm, shutil
from .learn import *
from .data import Dataset
from .model import Model


class Trainer():
    def __init__(self, model: Model):
        self.model: Model = model
        self.dataset: Dataset = Dataset()
        self.train: Partition = None
        self.valid: Partition = None
        self.test: Partition = None

    def load_data(self, samples_file: str):
        self.dataset.load_samples(samples_file, self.model)
        self.dataset.load_stats(self.model, self.model.args.estimator)
        self.split_data()

    def split_data(self):
        self.train = Partition(self.dataset, self.model.args.loss_func)
        test = self.train.dataset.split(self.model.args.test_split, self.model.args.rand_seed) 
        split = self.model.args.valid_split / (1 - self.model.args.test_split)
        valid = self.train.dataset.split(split, self.model.args.rand_seed)
        if test: self.test = Partition(test, self.model.args.loss_func)
        if valid: self.valid = Partition(valid, self.model.args.loss_func)

    def load_args(self):
        torch.set_num_threads(os.cpu_count())
        for partition in [self.train, self.valid, self.test]:
            if partition: partition.load(self.model.args.batch_size, self.model.args.device)
        args = {"lr": self.model.args.learn_rate, "weight_decay": self.model.args.weight_decay}
        if self.model.args.optimizer.__name__ == "SGD": args["momentum"] = self.model.args.momentum
        self.optimizer = self.model.args.optimizer(self.model.network.parameters(), **args)
        patience, min_delta = self.model.args.patience, self.model.args.min_delta
        if self.model.args.early_stop: self.early_stop = EarlyStop(patience, min_delta)
        self.criterion = Loss(self.model).to(self.model.args.device)

    def execute(self): 
        self.load_args()
        for _ in tqdm(range(self.model.args.num_epochs)):
            if self.train: self.train_model()
            if self.valid: self.test_model(valid = True)
            if self.valid and self.model.args.early_stop:
                loss = self.valid.loss[-1]
                if self.early_stop(loss): break
        if self.test: self.test_model(valid = False)

    def train_model(self):
        self.model.network.train()
        train_loss = 0
        for x, y in self.train.dataloader:
            self.optimizer.zero_grad()
            y_pred = self.model.network(x)
            loss = self.criterion(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(self.train.dataloader)
        self.train.loss.append(train_loss)

    def test_model(self, valid: bool):
        if valid: partition = self.valid
        else: partition = self.test
        self.model.network.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in partition.dataloader:
                y_pred = self.model.network(x)
                loss = self.criterion(y_pred, y)
                test_loss += loss.item()
        test_loss /= len(partition.dataloader)
        partition.loss.append(test_loss)

    def dump_model(self, output):
        if os.path.exists(output): 
            shutil.rmtree(output)
        os.mkdir(output)
        os.mkdir(f"{output}/plots")
        self.model.dump(output)