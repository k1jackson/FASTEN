from .model import Model
from .train import Trainer
from .predict import Predictor
from .tune import Tuner
from .plot import plot_train, plot_predict, plot_tune
import argparse, tracemalloc, time

def parse_args():
    superparser = argparse.ArgumentParser(description = "A flexible software framework to approximate computationally \
                                                         intensive simulations using neural-network-based emulators",
                                          formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    subparsers = superparser.add_subparsers(dest = "command")
    train = subparsers.add_parser("train", description = "Trains emulator on simulation data", 
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    train.add_argument("-c", "--config", required = True, help = "JSON file defining configuration parameters")
    train.add_argument("-i", "--input", required = True, help = "TSV file with simulation data")
    train.add_argument("-o", "--output", default = "outputs", help = "Folder to output model and figures")
    train.add_argument("-m", "--model", default = None, help = "ZIP file containing initial model")
    tune = subparsers.add_parser("tune", description = "Tunes hyperparameters for emulator on simulation data", 
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    tune.add_argument("-c", "--config", required = True, help = "JSON file defining configuration parameters")
    tune.add_argument("-i", "--input", required = True, help = "TSV file with simulation data")
    tune.add_argument("-o", "--output", default = "outputs", help = "Folder to output optimal configs and figures")
    tune.add_argument("-n", "--trials", type = int, default = 50, help = "Number of optimization trials")
    predict = subparsers.add_parser("predict", description = "Predicts simulation data from inputs with emulator", 
                                    formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    predict.add_argument("-m", "--model", required = True, help = "ZIP file containing model")
    predict.add_argument("-i", "--input", required = True, help = "TSV file with simulation inputs")
    predict.add_argument("-o", "--output", default = "outputs.tsv", help = "TSV file to output predicted simulation data")
    predict.add_argument("-n", "--runs", default = 0, type = int, help = "Number of simulation runs per input")
    return superparser.parse_args()

def train(args):
    model = Model(config_file = args.config, 
                  model_file = args.model)
    trainer = Trainer(model)
    trainer.load_data(args.input)
    trainer.execute()
    trainer.dump_model(args.output)

    plot_train(trainer, f"{args.output}/plots/training")
    predictor = Predictor(model, trainer.train.dataset)
    plot_predict(predictor, f"{args.output}/plots/training")
    if not trainer.test: return
    predictor = Predictor(model, trainer.test.dataset)
    plot_predict(predictor, f"{args.output}/plots/testing")

def predict(args): 
    model = Model(model_file = args.model)
    predictor = Predictor(model)
    predictor.load_inputs(args.input, args.runs)
    predictor.execute()
    
    if args.runs: predictor.dump_samples(args.output)
    else: predictor.dump_statistics(args.output)

def tune(args): 
    model = Model(config_file = args.config)
    trainer = Trainer(model)
    if args.trials:
        trainer.load_data(args.input)

    tuner = Tuner(trainer)
    tuner.load_study(args.output)
    tuner.execute(args.trials)
    tuner.dump_trials(args.output)
    plot_tune(tuner, f"{args.output}/plots")

def main():
    args = parse_args()
    # tracemalloc.start()
    start = time.perf_counter()
    match args.command:
        case "train": train(args)
        case "predict": predict(args)
        case "tune": tune(args)
    runtime = time.perf_counter() - start
    # _, memory = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    if runtime < 60: print(f"Total Runtime: {runtime:.2f} s")
    elif runtime < 60 * 60: print(f"Total Runtime: {runtime / 60:.2f} m")
    else: print(f"Total Runtime: {runtime / (60 * 60):.2f} h")
    # print(f"Peak Memory: {(memory / (1024 * 1024)):.2f} MB")
        