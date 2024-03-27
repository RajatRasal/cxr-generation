import random

import pytest
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from semantic_editing.attention import AttentionStoreAccumulate, AttentionStoreTimestep, AttendExciteCrossAttnProcessor
from semantic_editing.classifier_free_guidance import CFGWithDDIM
from semantic_editing.diffusion import StableDiffusionAdapter
from semantic_editing.null_text_inversion import NullTokenOptimisation
from semantic_editing.prompt_token_optimisation import PromptTokenOptimisation
from semantic_editing.dynamic_prompt_learning import DynamicPromptOptimisation
from semantic_editing.utils import seed_everything


SEED = 0
DDIM_STEPS = 50
GUIDANCE_SCALE_CFG = 7.5
DPL_STEPS = 20
DPL_NTI_STEPS = 50
# STABLE_DIFFUSION_VERSION = "runwayml/stable-diffusion-v1-5"
STABLE_DIFFUSION_VERSION = "CompVis/stable-diffusion-v1-4"

@pytest.fixture(autouse=True)
def initialise_random_seeds():
    seed_everything(SEED)


@pytest.fixture
def jet_cmap():
    plt.set_cmap("jet")


@pytest.fixture
def generator():
    # TODO: Set cuda as a variable
    generator = torch.Generator("cuda")
    generator.manual_seed(SEED)
    return generator


def sd_model():
    # TODO: Set cuda as a variable
    model = StableDiffusionPipeline.from_pretrained(
        STABLE_DIFFUSION_VERSION,
	    torch_dtype=torch.float32,
	    safety_checker=None,
	).to("cuda")
    model.text_encoder.text_model.encoder.requires_grad_(False)
    model.text_encoder.text_model.final_layer_norm.requires_grad_(False)
    model.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    return model


def sd_adapter():
    model = sd_model()
    return StableDiffusionAdapter(model, ddim_steps=DDIM_STEPS)


@pytest.fixture
def sd_adapter_fixture():
    return sd_adapter()


@pytest.fixture
def attention_store_accumulate():
    return AttentionStoreAccumulate()


@pytest.fixture
def attention_store_timestep():
    return AttentionStoreTimestep()


@pytest.fixture
def sd_adapter_with_attn_accumulate(attention_store_accumulate):
    adapter = sd_adapter()
    adapter.register_attention_control(
        attention_store_accumulate,
        AttendExciteCrossAttnProcessor,
    )
    return adapter


@pytest.fixture
def sd_adapter_with_attn_timestep(attention_store_timestep):
    adapter = sd_adapter()
    adapter.register_attention_control(
        attention_store_timestep,
        AttendExciteCrossAttnProcessor,
    )
    return adapter


@pytest.fixture
def image_prompt():
    # TODO: remove absolute path to cat mirror and change to relative path
    return Image.open("/vol/biomedic3/rrr2417/cxr-generation/test/semantic_editing/cat_mirror.jpeg"), "A cat sitting next to a mirror"


@pytest.fixture
def image_prompt_cat_and_dog():
    return Image.open("/vol/biomedic3/rrr2417/cxr-generation/test/semantic_editing/catdog.jpg"), "a cat and a dog"


@pytest.fixture
def image_prompt_girl_and_boy_trampoline():
    return Image.open("/vol/biomedic3/rrr2417/cxr-generation/test/semantic_editing/catdog.jpg"), "a cat and a dog"


@pytest.fixture
def dpl_1(sd_adapter_with_attn_timestep, sd_adapter_with_attn_accumulate):
    return DynamicPromptOptimisation(
        sd_adapter_with_attn_timestep,
        sd_adapter_with_attn_accumulate,
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
def dpl_2(sd_adapter_with_attn_timestep, sd_adapter_with_attn_accumulate):
    return DynamicPromptOptimisation(
        sd_adapter_with_attn_timestep,
        sd_adapter_with_attn_accumulate,
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
def dpl_3(sd_adapter_with_attn_timestep, sd_adapter_with_attn_accumulate):
    return DynamicPromptOptimisation(
        sd_adapter_with_attn_timestep,
        sd_adapter_with_attn_accumulate,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=DPL_STEPS,
        num_inner_steps_nti=DPL_NTI_STEPS,
        attention_balancing_coeff=1.0,
        attention_balancing_alpha=25,
        attention_balancing_beta=0.3,
        background_leakage_coeff=0.05,
        background_leakage_alpha=50,
        background_leakage_beta=0.9,
        disjoint_object_coeff=0.05,
        disjoint_object_alpha=25,
        disjoint_object_beta=0.9,
        max_clusters=5,
        clustering_algorithm="kmeans",
    )


@pytest.fixture
def dpl_nti(sd_adapter_with_attn_timestep, sd_adapter_with_attn_accumulate):
    return DynamicPromptOptimisation(
        sd_adapter_with_attn_timestep,
        sd_adapter_with_attn_accumulate,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps_dpl=DPL_STEPS,
        num_inner_steps_nti=DPL_NTI_STEPS,
        attention_balancing_coeff=0,
        disjoint_object_coeff=0,
        background_leakage_coeff=0,
    )


@pytest.fixture
def nti(sd_adapter_with_attn_timestep):
    return NullTokenOptimisation(
        sd_adapter_with_attn_timestep,
        guidance_scale=GUIDANCE_SCALE_CFG,
        num_inner_steps=DPL_NTI_STEPS,
    )


# @pytest.fixture
# def pti(sd_adapter):
#     return PromptTokenOptimisation(sd_adapter, guidance_scale=7.5, num_inner_steps=20)


@pytest.fixture
def cfg_ddim(sd_adapter_with_attn_accumulate):
    return CFGWithDDIM(sd_adapter_with_attn_accumulate, guidance_scale=1, image_size=512)

