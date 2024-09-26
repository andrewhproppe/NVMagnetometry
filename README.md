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
