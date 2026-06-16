# FASTEN

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](CHANGELOG.md)


<!-- Explain the *what* and *why* of your project in 2–4 sentences.
     Answer: What problem does it solve? Who is it for? Why does it exist? -->

FASTEN is a flexible and user-friendly framework for building PyTorch-based deep learning emulators for epidemic simulations with stochastic outputs. FASTEN provides three intuitive modules to (1) train deep neural networks on simulation data, (2) select optimal hyperparameters, and (3) generate predictions from unseen inputs. 


## Installation


FASTEN can be installed via ```pip```:

```bash
pip install fasten-cli
```

## Usage

There are two primary components to FASTEN: the [configuration file designer](https://k1jackson.github.io/FASTEN/) and the command line tool.

### Configuration File Designer
Before executing a FASTEN, the user must construct a workflow configuration file that outlines the simulation data format and training hyperparameters. The [configuration file designer](https://k1jackson.github.io/FASTEN/) guides users through this process with contextual instructions and validation checks.

### Command Line Tool
The FASTEN workflow decomposes the model emulation process into three phases: (1)training, (2) hyperparameter tuning, and (3) output prediction. Each phase is invoked through a dedicated command line module, with a shared configuration file governing the underlying behavior. The command line tool can used as follows:

**Training:**

```bash
usage: FASTEN train [-h] -c CONFIG -i INPUT [-o OUTPUT] [-m MODEL]

options:
  -h, --help           show this help message and exit
  -c, --config CONFIG  JSON file defining configuration parameters
  -i, --input INPUT    TSV file with simulation data
  -o, --output OUTPUT  Folder to output model and figures (default: outputs)
  -m, --model MODEL    ZIP file containing initial model (default: None)
```

**Hyperparameter Tuning:**

```bash
usage: FASTEN tune [-h] -c CONFIG -i INPUT [-o OUTPUT] [-n TRIALS] [--unique]

options:
  -h, --help           show this help message and exit
  -c, --config CONFIG  JSON file defining configuration parameters
  -i, --input INPUT    TSV file with simulation data
  -o, --output OUTPUT  Folder to output optimal configs and figures (default: outputs)
  -n, --trials TRIALS  Total number of optimation trials (default: 100)
  --unique             Prevents re-training with duplicate hyperparameter sets (default: False)
```

**Output Prediction:**

```bash
usage: FASTEN predict [-h] -m MODEL -i INPUT [-o OUTPUT] [-n RUNS]

options:
  -h, --help           show this help message and exit
  -m, --model MODEL    ZIP file containing model
  -i, --input INPUT    TSV file with simulation inputs
  -o, --output OUTPUT  TSV file to output predicted simulation data (default: outputs.tsv)
  -n, --runs RUNS      Number of simulation runs per input (default: 0)
```
