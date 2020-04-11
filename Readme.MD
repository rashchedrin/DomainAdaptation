# Install requirements:
## For conda:
```
conda env create -f conda_environment.yml
```
Activate conda env:
```
conda activate domain_adaptation
```

## For pip:
```
pip install -r
```

# Download office-31 dataset:
```
python download_dataset.py
```

# Run tests:
```
python -m pytest tests
```
