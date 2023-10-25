python_sources()

pex_binary(
    name="train",
    entry_point="./image_train.py",
    dependencies=["3rdparty:reqs#setuptools"],
    args=["--data_dir", "./.improved_diffusion_training"],
)
