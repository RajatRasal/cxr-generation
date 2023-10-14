python_requirements(
    name="reqs",
    module_mapping={
        "SimpleITK": ["SimpleITK"],
        "huggingface-hub": ["huggingface_hub"],
    },
)

python_requirements(
    name="reqs_test",
    module_mapping={
        # SimpleITK is needed here because the library is called simple_itk in pypi
        # and also referred to with that name in the default.lock .
        "SimpleITK": ["SimpleITK"],
        "huggingface-hub": ["huggingface_hub"],
        "importlib-metadata": ["importlib_metadata"],
    },
    resolve="pytest",
)

python_requirements(
    name="flake",
    source="flake_requirements.txt",
    resolve="flake8",
)

python_requirements(
    name="mypy",
    source="mypy_requirements.txt",
    resolve="mypy",
)

python_requirements(
    name="pytest",
    source="pytest_requirements.txt",
    resolve="pytest",
)
