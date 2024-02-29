from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention


PLACE_IN_UNET = Literal["up", "down", "mid"]


class AttentionStore:

    def __init__(self, attention_resolution: int = 16):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion process.
        """
        self._num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attention_resolution = attention_resolution

    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    ### NOTE: huggingface guys modify this code to only save the attention maps with 256
    def __call__(self, attn, is_cross: bool, place_in_unet: PLACE_IN_UNET):
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == self.attention_resolution ** 2:
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self._num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    # def get_average_attention(self):
    #     average_attention = self.attention_store
    #     return average_attention

    def aggregate_attention(self, from_where: List[PLACE_IN_UNET]) -> torch.FloatTensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.attention_store  # self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(
                    -1,
                    self.attention_resolution,
                    self.attention_resolution,
                    item.shape[-1],
                )
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def register_attention_processor(self):
        self._num_att_layers += 1


class AttnProcessorWithAttentionStore(ABC):

    def __init__(self, attention_store: AttentionStore, place_in_unet: PLACE_IN_UNET):
        self.attention_store = attention_store
        self.attention_store.register_attention_processor()
        self.place_in_unet = place_in_unet

    @abstractmethod
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class AttendExciteCrossAttnProcessor(AttnProcessorWithAttentionStore):

    def __init__(self, attention_store: AttentionStore, place_in_unet: PLACE_IN_UNET):
        super().__init__(attention_store, place_in_unet)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        self.attention_store(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
 

def prepare_unet(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            module.requires_grad_(True)
        else:
            module.requires_grad_(False)
    return unet

