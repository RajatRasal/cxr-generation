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
from semantic_editing.utils import init_stable_diffusion, seed_everything, device_availability


GUIDANCE_SCALE_CFG = 7.5
STABLE_DIFFUSION_VERSIONS = {
    "1.4": "CompVis/stable-diffusion-v1-4",
    "1.5": "runwayml/stable-diffusion-v1-5",
    "2.0": "stabilityai/stable-diffusion-2",
    "2.1": "stabilityai/stable-diffusion-2-1",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run tests fast",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skipper = pytest.mark.skip(reason="Only run when --run-slow is given")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skipper)


@pytest.fixture
def steps(pytestconfig):
    fast = pytestconfig.getoption("fast")
    return {
        "ddim_steps": 50,
        "dpl_steps": 1 if fast else 21,
        "nti_steps": 1 if fast else 51,
        "cross_replace_steps": 0 if fast else 40,
        "self_replace_steps": 0 if fast else 40,
    }


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
    return 10000


@pytest.fixture
def image_size():
    return 512


@pytest.fixture
def background_map_hps():
    return {
        "background_threshold": 0.2,
        "algorithm": "kmeans",
        "n_clusters": 5,
        "kwargs": {"random_state": 1, "n_init": 10},
    }


@pytest.fixture
def jet_cmap():
    plt.set_cmap("jet")


def sd_adapter(model_name, ddim_steps, device="cpu"):
    return StableDiffusionAdapter(
        model_name,
        ddim_steps=ddim_steps,
        device=device,
    )


@pytest.fixture
def sd_adapter_fixture(steps):
    return sd_adapter(
        STABLE_DIFFUSION_VERSIONS["1.4"],
        steps["ddim_steps"],
        device=device_availability(),
    )


def _index_noun_pairs(tokens, nouns):
    return [(tokens.index(noun), noun) for noun in nouns]


@pytest.fixture
def image_prompt(images_dir):
    return Image.open(os.path.join(images_dir, "cat_mirror.jpeg")), "A cat sitting next to a mirror", None, None


@pytest.fixture
def image_prompt_cat_and_dog(images_dir):
    prompt = "a cat and a dog"
    tokens = prompt.split(" ")
    nouns = ["cat", "dog"]
    return Image.open(os.path.join(images_dir, "catdog.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_pear_and_apple(images_dir):
    prompt = "a pear and an apple"
    tokens = prompt.split(" ")
    nouns = ["pear", "apple"]
    return Image.open(os.path.join(images_dir, "pear_and_apple.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_sports_equipment(images_dir):
    prompt = "a basketball a football and a tennis ball on a racket"
    tokens = prompt.split(" ")
    nouns = ["basketball", "football"]
    return Image.open(os.path.join(images_dir, "sports_equipment.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_horse_and_sheep(images_dir):
    prompt = "a horse and a sheep"
    tokens = prompt.split(" ")
    nouns = ["horse", "sheep"]
    return Image.open(os.path.join(images_dir, "horse_and_sheep.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_book_clock_bottle(images_dir):
    prompt = "a clock on a pile of books next to a bottle"
    tokens = prompt.split(" ")
    nouns = ["clock", "books"]
    return Image.open(os.path.join(images_dir, "book_clock_bottle.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_football_on_bench(images_dir):
    prompt = "a football on a bench in the park"
    tokens = prompt.split(" ")
    nouns = ["football", "bench"]
    return Image.open(os.path.join(images_dir, "football_on_bench.jpeg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_cake_on_plate(images_dir):
    prompt = "a slice of chocolate cake on a plate"
    tokens = prompt.split(" ")
    nouns = ["cake", "plate"]
    return Image.open(os.path.join(images_dir, "brownie_cake_on_plate.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_cat_bird_stitching(images_dir):
    prompt = "a cat and a bird on a cross stitching pattern"
    tokens = prompt.split(" ")
    nouns = ["cat", "bird"]
    return Image.open(os.path.join(images_dir, "cat_bird_stitching.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_cat_bird_painting(images_dir):
    prompt = "a painting of a cat and a bird on a green background"
    tokens = prompt.split(" ")
    nouns = ["cat", "bird"]
    return Image.open(os.path.join(images_dir, "cat_bird_painting.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_cat_dog_watercolour(images_dir):
    prompt = "a dog and a cat are show in this watercolour"
    tokens = prompt.split(" ")
    nouns = ["dog", "cat"]
    return Image.open(os.path.join(images_dir, "cartoon_cat_and_dog.jpg")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def image_prompt_cat_dog_flowers(images_dir):
    prompt = "a cat and a dog sitting in flowers"
    tokens = prompt.split(" ")
    nouns = ["cat", "dog"]
    return Image.open(os.path.join(images_dir, "cat_dog_flowers.png")), prompt, tokens, _index_noun_pairs(tokens, nouns)


@pytest.fixture
def dpl_1(sd_adapter_fixture, steps, image_size):
    return DynamicPromptOptimisation(
        sd_adapter_fixture,
        image_size=image_size,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=steps["dpl_steps"],
        num_inner_steps_nti=steps["nti_steps"],
        attention_balancing_coeff=1.0,
        attention_balancing_alpha=25,
        attention_balancing_beta=0.3,
        disjoint_object_coeff=0,
        background_leakage_coeff=0,
    )


@pytest.fixture
def dpl_2(sd_adapter_fixture, steps, image_size):
    return DynamicPromptOptimisation(
        sd_adapter_fixture,
        image_size=image_size,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=steps["dpl_steps"],
        num_inner_steps_nti=steps["nti_steps"],
        attention_balancing_coeff=1,
        attention_balancing_alpha=25,
        attention_balancing_beta=0.3,
        disjoint_object_coeff=0.005,
        disjoint_object_alpha=25,
        disjoint_object_beta=0.9,
        background_leakage_coeff=0,
    )


def init_dpl_3(sd_adapter, steps, image_size, seed, background_map_hps):
    return DynamicPromptOptimisation(
        sd_adapter,
        image_size=image_size,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=steps["dpl_steps"],
        num_inner_steps_nti=steps["nti_steps"],
        attention_balancing_coeff=1.0,
        attention_balancing_alpha=25,
        attention_balancing_beta=0.3,
        disjoint_object_coeff=0.005,
        disjoint_object_alpha=25,
        disjoint_object_beta=0.9,
        background_leakage_coeff=0.05,
        background_leakage_alpha=50,
        background_leakage_beta=0.7,
        background_clusters_threshold=background_map_hps["background_threshold"],
        clustering_algorithm=background_map_hps["algorithm"],
        max_clusters=background_map_hps["n_clusters"],
        clustering_kwargs=background_map_hps["kwargs"],
    )


@pytest.fixture
def dpl_3(sd_adapter_fixture, steps, image_size, seed, background_map_hps):
    return init_dpl_3(sd_adapter_fixture, steps, image_size, seed, background_map_hps)


@pytest.fixture
def dpl_3_14(image_size, steps, seed, background_map_hps):
    adapter = sd_adapter(STABLE_DIFFUSION_VERSIONS["1.4"])
    return init_dpl_3(adapter, steps, image_size, seed, background_map_hps)


@pytest.fixture
def dpl_3_15(image_size, steps, seed, background_map_hps):
    adapter = sd_adapter(STABLE_DIFFUSION_VERSIONS["1.5"])
    return init_dpl_3(adapter, steps, image_size, seed, background_map_hps)


@pytest.fixture
def dpl_3_20(image_size, steps, seed, background_map_hps):
    adapter = sd_adapter(STABLE_DIFFUSION_VERSIONS["2.0"])
    return init_dpl_3(adapter, steps, image_size, seed, background_map_hps)


@pytest.fixture
def dpl_3_21(image_size, steps, seed, background_map_hps):
    adapter = sd_adapter(STABLE_DIFFUSION_VERSIONS["2.1"])
    return init_dpl_3(adapter, steps, image_size, seed, background_map_hps)


@pytest.fixture
def nti(sd_adapter_fixture, steps, image_size):
    return DynamicPromptOptimisation(
        sd_adapter_fixture,
        image_size=image_size,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=steps["dpl_steps"],
        num_inner_steps_nti=steps["nti_steps"],
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

