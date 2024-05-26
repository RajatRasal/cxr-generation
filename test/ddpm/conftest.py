import os
from typing import Tuple

import pytest
from lightning.pytorch import seed_everything


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture(autouse=True)
def seed_everything_fixture(seed):
    seed_everything(seed)


@pytest.fixture
def model_folder_and_name_and_exp() -> Tuple[str, str]:
    # TODO: Make this work with relative path from project root
    folder = "/vol/biomedic3/rrr2417/cxr-generation/test/ddpm/model"
    os.makedirs(folder, exist_ok=True)
    return folder, "test_model", "test_exp"
