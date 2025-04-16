## Usage

First, create a dataset for training using `src/pipeline/make_dataset.py`. Currently, the dataset generation parameters need to be set in the script itself, instead of from the command line. This creates a `.h5` file that will be stored in the `data/dataset` directory.

To train a model, use one of the scripts in `scripts/model_training`. This are individual scripts for each model. Some of them are likely outdated compared to `train_NODE_MLP.py`, which is the model that shows the best performance for the original NV sensing regression problem. The other models are simple MLP and LSTM architectures, Atttention GRU, convolutional + MLP layers, and a vision transformer.

After trying to optimzie each model, I found that the neural ODE approach consistently worked best.

For logging, `wandb` is used. The package is free for individual users (maximum of two contributors for a given proejct). I would highly recommend making your own account to use the `wandb` logging the way it is currently set up.

## Project Structure
```bash
.
├── README.md
├── conda.yml
├── data
│   ├── datasets
│   ├── raw
│   └── utils.py
├── project_structure.txt
├── scripts
│   ├── model_testing
│   │   └── predict.py
│   └── model_training
│       ├── train_NODE_MLP.py
│       ├── ...
├── setup.cfg
├── setup.py
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── EnsembleNV_MWbroadband_addressing_time_domain.py
│   │   └── EnsembleNV_MWbroadband_addressing_time_domain_parallel.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ode
│   │   │   ├── __init__.py
│   │   │   ├── ode.py
│   │   │   └── ode_models.py
│   │   ├── submodels.py
│   │   └── utils.py
│   ├── modules
│   ├── pipeline
│   │   ├── __init__.py
│   │   ├── data_module.py
│   │   ├── make_data_files.py
│   │   ├── make_data_files_parallel.py
│   │   ├── make_dataset.py
│   │   ├── tests
│   │   └── transforms.py
│   ├── predict.py
│   ├── utils.py
│   └── visualization
│       ├── __init__.py
│       ├── fig_utils.py
│       └── visualize.py
└── tests
```
