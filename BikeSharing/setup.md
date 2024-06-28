# Bikesharing Regression model

## Package the model

- `mypy.ini` : eslint similar
- `pyprojects.toml` : Create a wheel file, wheel file is used to distribute it.
- `MANIFEST.in` : 
- `setup.py` : 

### Create build
```bash
pip install --upgrade build
python -m build
```

### Extract the tar file
```
tar -xf .\dist\bikeshare_model-0.0.1.tar.gz
```


### Extract the wheel file
```
pip install .\dist\bikeshare_model-0.0.1-py3-none-any.whl
```