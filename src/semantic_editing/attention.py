import math
from abc import ABC, abstractmethod
from itertools import product
from typing import List, Literal, Optional, get_args

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention


PLACE_IN_UNET = Literal["up", "down", "mid"]
ATTENTION_COMPONENT = Literal["attn", "feat", "query", "key", "value"]


class AttentionStore(ABC):

    def __init__(self):
        self._num_att_layers = 0
        self.reset()

    @staticmethod
    def _attention_store_key_string(place_in_unet, is_cross, attn_component, res):
        attn_type = "cross" if is_cross else "self"
        return f"{place_in_unet}_{attn_type}_{attn_component}_{str(res)}"

    @staticmethod
    def get_empty_store():
        step_store = {}
        places_in_unet = ["down", "mid", "up"]
        is_cross = [True, False]
        attn_components = list(get_args(ATTENTION_COMPONENT)) 
        resolutions = [8, 16, 32, 64]
        for place_in_unet, _is_cross, attn_component, res in product(places_in_unet, is_cross, attn_components, resolutions):
            dict_key = AttentionStore._attention_store_key_string(place_in_unet, _is_cross, attn_component, res)
            step_store[dict_key] = None
        return step_store

    @abstractmethod
    def __call__(
        self,
        attn: torch.FloatTensor,
        is_cross: bool,
        place_in_unet: str,
        hidden_states: torch.FloatTensor,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def between_steps(self):
        raise NotImplementedError

    @abstractmethod
    def get_average_attention(self):
        raise NotImplementedError

    def aggregate_attention(
        self,
        places_in_unet: List[str],
        res: int,
        is_cross: bool,
        element_name: ATTENTION_COMPONENT,
    ) -> torch.FloatTensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        assert self._cur_att_layer == 0, "Aggregate attention should only be called between diffusion steps"

        out = []
        num_pixels = res ** 2
        attention_maps = self.get_average_attention()
        for place_in_unet in places_in_unet:
            dict_key = self._attention_store_key_string(place_in_unet, is_cross, element_name, res)
            item = attention_maps[dict_key]
            if item is None:
                continue
            # TODO: Remove this nasty hack. Push this down into each child class.
            if isinstance(item, list):
                for x in item:
                    out.append(x.reshape(-1, res, res, x.shape[-1]))
            else:
                out.append(item.reshape(-1, res, res, item.shape[-1]))
        if len(out) > 0:
            out = torch.cat(out, dim=0)
            out = out.sum(0) / out.shape[0]
        else:
            raise ValueError("No attentions maps found matching the criteria")
        return out

    def reset(self):
        self._cur_step_index = 0
        self._cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = self.get_empty_store()

    def register_attention_processor(self):
        self._num_att_layers += 1


class AttentionStoreTimestep(AttentionStore):

    def __call__(
        self,
        attn: torch.FloatTensor,
        is_cross: bool,
        place_in_unet: str,
        hidden_states: torch.FloatTensor,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
    ):
        res = int(math.sqrt(attn.shape[1]))
        dict_key = self._attention_store_key_string(place_in_unet, is_cross, "attn", res)
        if self.step_store[dict_key] is None:
            self.step_store[dict_key] = [attn]
        else:
            self.step_store[dict_key].append(attn)

        self._cur_att_layer += 1
        if self._cur_att_layer == self._num_att_layers:
            self._cur_att_layer = 0
            self._cur_step_index += 1
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()
    
    def get_average_attention(self):
        return self.attention_store
    

class AttentionStoreAccumulate(AttentionStore):

    def __call__(
        self,
        attn: torch.FloatTensor,
        is_cross: bool,
        place_in_unet: str,
        hidden_states: torch.FloatTensor,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
    ):
        res = int(math.sqrt(attn.shape[1]))
        attn_components = list(get_args(ATTENTION_COMPONENT))
        for attn_component, tensor in zip(attn_components, [attn, hidden_states, query, key, value]):
            dict_key = self._attention_store_key_string(place_in_unet, is_cross, attn_component, res)
            self.step_store[dict_key] = tensor

        self._cur_att_layer += 1
        if self._cur_att_layer == self._num_att_layers:
            self._cur_att_layer = 0
            self._cur_step_index += 1
            self.between_steps()

    def between_steps(self):
        for key, value in self.step_store.items():
            if value is None:
                continue
            if self.attention_store[key] is None:
                self.attention_store[key] = value
            else:
                self.attention_store[key] += value
        self.step_store = self.get_empty_store()
    
    def get_average_attention(self):
        # TODO: A tensor is not added here at every step. So we need to ensure that we know the true
        # length and don't assume that cur_step = 50
        cur_step = 50.0
        average_attention = {
            key: None if item is None else item / cur_step
            for key, item in self.attention_store.items()
        }
        return average_attention
    

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
        # TODO: Only call this when grad is True
        # if attention_probs.requires_grad:
        self.attention_store(
            attention_probs,
            is_cross,
            self.place_in_unet,
            hidden_states,
            query,
            key,
            value,
        )

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

