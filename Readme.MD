Install requirements:

```
conda env create -f conda_environment.yml
```
Activate conda env:
```
conda activate domain_adaptation
```

Download office-31 dataset:
```
python download_dataset.py
```

Run tests:
```
python -m pytest tests
```
