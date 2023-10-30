On Linux
`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`

`poetry install`

`OPENAI_LOGDIR=/vol/biomedic3/rrr2417/.tmp poetry run image_train -- --data_dir /vol/biodata/data/chest_xray/mimic-cxr-jpg/files/ --log_interval 10 --save_interval 100 --batch_size 64`

