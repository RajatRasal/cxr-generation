Only for Linux `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`

`export TRANSFORMERS_CACHE=/data/huggingface_cache/`

1. [Install poetry](https://python-poetry.org/docs/#installation)
1. Run `poetry install`
1. Run `./scripts/train.sh`


## DDPM

`pytest run test/ddpm/test_training.py -s`