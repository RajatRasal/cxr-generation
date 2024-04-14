import os
import random

import pytest
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from semantic_editing.attention import AttentionStoreAccumulate, AttentionStoreTimestep, AttnProcessorWithAttentionStore
from semantic_editing.classifier_free_guidance import CFGWithDDIM
from semantic_editing.diffusion import StableDiffusionAdapter
from semantic_editing.dynamic_prompt_learning import DynamicPromptOptimisation
from semantic_editing.utils import seed_everything, device_availability


SEED = 0
DDIM_STEPS = 50
GUIDANCE_SCALE_CFG = 7.5
DPL_STEPS = 20
DPL_NTI_STEPS = 50
IMAGE_SIZE = 512
STABLE_DIFFUSION_VERSION = "runwayml/stable-diffusion-v1-5"
# STABLE_DIFFUSION_VERSION = "CompVis/stable-diffusion-v1-4"


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skipper = pytest.mark.skip(reason="Only run when --run-slow is given")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skipper)


@pytest.fixture
def fig_dir():
    dirname = "test/semantic_editing/figures" 
    os.makedirs(dirname, exist_ok=True)
    return dirname


@pytest.fixture
def weights_dir():
    dirname = "test/semantic_editing/weights" 
    os.makedirs(dirname, exist_ok=True)
    return dirname


@pytest.fixture
def more_editing_dir():
    dirname = "test/semantic_editing/more_editing_dir" 
    os.makedirs(dirname, exist_ok=True)
    return dirname


@pytest.fixture
def images_dir():
    return "test/semantic_editing/images"


@pytest.fixture
def seed():
    return SEED


@pytest.fixture
def image_size():
    return IMAGE_SIZE


@pytest.fixture
def background_map_hps():
    return {
        "background_threshold": 0.2,
        "algorithm": "kmeans",
        "n_clusters": 5,
    }


@pytest.fixture(autouse=True)
def initialise_random_seeds(seed):
    seed_everything(seed)


@pytest.fixture
def jet_cmap():
    plt.set_cmap("jet")


@pytest.fixture
def generator(seed):
    generator = torch.Generator(device_availability())
    generator.manual_seed(seed)
    return generator


def sd_model():
    model = StableDiffusionPipeline.from_pretrained(
        STABLE_DIFFUSION_VERSION,
	    torch_dtype=torch.float32,
	    safety_checker=None,
	).to(device_availability())
    # model.text_encoder.text_model.encoder.requires_grad_(False)
    # model.text_encoder.text_model.final_layer_norm.requires_grad_(False)
    # model.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    return model


def sd_adapter():
    model = sd_model()
    return StableDiffusionAdapter(model, ddim_steps=DDIM_STEPS)


@pytest.fixture
def sd_adapter_fixture():
    return sd_adapter()


@pytest.fixture
def image_prompt(images_dir):
    return Image.open(os.path.join(images_dir, "cat_mirror.jpeg")), "A cat sitting next to a mirror"


@pytest.fixture
def image_prompt_cat_and_dog(images_dir):
    return Image.open(os.path.join(images_dir, "catdog.jpg")), "a cat and a dog"


@pytest.fixture
def image_prompt_pear_and_apple(images_dir):
    return Image.open(os.path.join(images_dir, "pear_and_apple.jpg")), "a pear and an apple"


@pytest.fixture
def image_prompt_sports_equipment(images_dir):
    return Image.open(os.path.join(images_dir, "sports_equipment.jpg")), "a basketball, a football, and a tennis ball on a racket"


@pytest.fixture
def image_prompt_horse_and_sheep(images_dir):
    return Image.open(os.path.join(images_dir, "horse_and_sheep.jpg")), "a horse and a sheep"


@pytest.fixture
def image_prompt_book_clock_bottle(images_dir):
    return Image.open(os.path.join(images_dir, "book_clock_bottle.jpg")), "a clock on a pile of books next to a bottle"


@pytest.fixture
def image_prompt_football_on_bench(images_dir):
    return Image.open(os.path.join(images_dir, "football_on_bench.jpeg")), "a football on a bench in the park"


@pytest.fixture
def image_prompt_cake_on_plate(images_dir):
    return Image.open(os.path.join(images_dir, "brownie_cake_on_plate.jpg")), "a slice of chocolate cake on a plate"


@pytest.fixture
def image_prompt_cat_bird_stitching(images_dir):
    return Image.open(os.path.join(images_dir, "cat_bird_stitching.jpg")), "a cat and a bird on a cross stitching pattern"


@pytest.fixture
def image_prompt_cat_bird_painting(images_dir):
    return Image.open(os.path.join(images_dir, "cat_bird_painting.jpg")), "a painting of a cat and a bird on a green background"


@pytest.fixture
def image_prompt_cat_dog_watercolour(images_dir):
    return Image.open(os.path.join(images_dir, "cat_dog_watercolour.jpg")), "a dog and a cat are shown in this watercolour"


@pytest.fixture
def image_prompt_cat_dog_flowers(images_dir):
    return Image.open(os.path.join(images_dir, "cat_dog_flowers.png")), "a cat and a dog sitting in flowers"


@pytest.fixture
def dpl_1(sd_adapter_fixture, image_size):
    return DynamicPromptOptimisation(
        sd_adapter_fixture,
        image_size=image_size,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=DPL_STEPS,
        num_inner_steps_nti=DPL_NTI_STEPS,
        attention_balancing_coeff=1,
        attention_balancing_alpha=25,
        attention_balancing_beta=0.3,
        disjoint_object_coeff=0,
        background_leakage_coeff=0,
    )


@pytest.fixture
def dpl_2(sd_adapter_fixture, image_size):
    return DynamicPromptOptimisation(
        sd_adapter_fixture,
        image_size=image_size,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=DPL_STEPS,
        num_inner_steps_nti=DPL_NTI_STEPS,
        attention_balancing_coeff=1,
        attention_balancing_alpha=25,
        attention_balancing_beta=0.3,
        disjoint_object_coeff=0.05,
        disjoint_object_alpha=25,
        disjoint_object_beta=0.9,
        background_leakage_coeff=0,
    )


@pytest.fixture
def dpl_3(sd_adapter_fixture, image_size, seed, background_map_hps):
    return DynamicPromptOptimisation(
        sd_adapter_fixture,
        image_size=image_size,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=DPL_STEPS,
        num_inner_steps_nti=DPL_NTI_STEPS,
        attention_balancing_coeff=1.0,
        attention_balancing_alpha=25,
        attention_balancing_beta=0.3,
        background_leakage_coeff=0.05,
        background_leakage_alpha=50,
        background_leakage_beta=0.9,
        background_clusters_threshold=background_map_hps["background_threshold"],
        clustering_algorithm=background_map_hps["algorithm"],
        max_clusters=background_map_hps["n_clusters"],
        clustering_random_state=seed,
        disjoint_object_coeff=0.05,
        disjoint_object_alpha=25,
        disjoint_object_beta=0.9,
    )


@pytest.fixture
def nti(sd_adapter_fixture, image_size):
    return DynamicPromptOptimisation(
        sd_adapter_fixture,
        image_size=image_size,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=DPL_STEPS,
        num_inner_steps_nti=DPL_NTI_STEPS,
        attention_balancing_coeff=0,
        disjoint_object_coeff=0,
        background_leakage_coeff=0,
    )


@pytest.fixture
def cfg_ddim(sd_adapter_fixture, image_size):
    return CFGWithDDIM(
        sd_adapter_fixture,
        guidance_scale=1,
        image_size=image_size,
        attention_accumulate=True,
    )

