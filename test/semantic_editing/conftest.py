import random

import pytest
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from semantic_editing.attention import AttentionStore, AttendExciteCrossAttnProcessor
from semantic_editing.classifier_free_guidance import CFGWithDDIM
from semantic_editing.diffusion import StableDiffusionAdapter
from semantic_editing.null_text_inversion import NullTokenOptimisation
from semantic_editing.prompt_token_optimisation import PromptTokenOptimisation
from semantic_editing.dynamic_prompt_learning import DynamicPromptOptimisation
from semantic_editing.utils import seed_everything


SEED = 8888
# STABLE_DIFFUSION_VERSION = "runwayml/stable-diffusion-v1-5"
STABLE_DIFFUSION_VERSION = "CompVis/stable-diffusion-v1-4"

@pytest.fixture(autouse=True)
def initialise_random_seeds():
    seed_everything(SEED)


@pytest.fixture
def generator():
    generator = torch.Generator("cuda")
    generator.manual_seed(SEED)
    return generator


@pytest.fixture
def sd_model():
    return StableDiffusionPipeline.from_pretrained(
        STABLE_DIFFUSION_VERSION,
	    torch_dtype=torch.float32,
	    safety_checker=None,
	).to("cuda")


@pytest.fixture
def sd_adapter(sd_model):
    return StableDiffusionAdapter(sd_model, ddim_steps=50)


@pytest.fixture
def attention_store():
    return AttentionStore()


@pytest.fixture
def sd_adapter_with_attn_excite(sd_model, attention_store):
    adapter = StableDiffusionAdapter(sd_model, ddim_steps=50)
    adapter.register_attention_control(
        attention_store,
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
def dpl(sd_adapter_with_attn_excite):
    return DynamicPromptOptimisation(
        sd_adapter_with_attn_excite,
        guidance_scale=7.5,
        num_inner_steps_dpl=2,
        disjoint_object_coeff=0,
        background_leakage_coeff=0,
    )


@pytest.fixture
def nti(sd_adapter):
    return NullTokenOptimisation(sd_adapter, guidance_scale=7.5, num_inner_steps=20)


@pytest.fixture
def pti(sd_adapter):
    return PromptTokenOptimisation(sd_adapter, guidance_scale=7.5, num_inner_steps=20)


@pytest.fixture
def cfg_ddim(sd_adapter_with_attn_excite):
    return CFGWithDDIM(sd_adapter_with_attn_excite, guidance_scale=1, image_size=512)

