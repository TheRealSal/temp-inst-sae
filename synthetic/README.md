# Code for Synthetic data

## Setup with `uv`
This project uses `uv` as the package management tool. To reproduce the environment, run:
```bash
uv sync
```
The virtual Python environment will be located in the `.venv` directory. To activate it, use:
```bash
source .venv/bin/activate
```

## Identifiability Verification
```shell
python complete-3.py
```
## Scaling to Large Language Model Activation Dimensions
```shell
bash linear_complexity.sh 44 
```
