import math
from abc import ABC, abstractmethod
from itertools import product
from typing import List, Literal, Optional, Union, get_args

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention

from semantic_editing.validate import _validate_attn_map


PLACE_IN_UNET = Literal["up", "down", "mid"]
ATTENTION_COMPONENT = Literal["attn", "feat", "query", "key", "value"]


def _unflatten_attn_map(item: torch.FloatTensor, res: int) -> torch.FloatTensor:
    return item.reshape(-1, res, res, item.shape[-1])


class AttentionStore(ABC):

    def __init__(self):
        self._num_att_layers = -1
        self.reset()

    def _attention_store_key_string(
        self,
        place_in_unet: PLACE_IN_UNET,
        is_cross: bool,
        attn_component: ATTENTION_COMPONENT,
        res: int,
    ):
        attn_type = "cross" if is_cross else "self"
        return f"{place_in_unet}_{attn_type}_{attn_component}_{str(res)}"

    def _get_empty_store(self):
        step_store = {}
        places_in_unet = ["down", "mid", "up"]
        is_cross = [True, False]
        attn_components = list(get_args(ATTENTION_COMPONENT)) 
        resolutions = [8, 16, 32, 64]
        for place_in_unet, _is_cross, attn_component, res in product(places_in_unet, is_cross, attn_components, resolutions):
            dict_key = self._attention_store_key_string(place_in_unet, _is_cross, attn_component, res)
            step_store[dict_key] = None
        return step_store

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
        self._step_store(
            attn,
            is_cross,
            place_in_unet,
            hidden_states,
            query,
            key,
            value,
        )

        self._cur_att_layer += 1
        if self._cur_att_layer == self._num_att_layers:
            self._cur_att_layer = 0
            self._between_steps()

    @abstractmethod
    def _step_store(
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
    def _between_steps(self):
        raise NotImplementedError

    @abstractmethod
    def _accumulate(
        self,
        item: Union[torch.FloatTensor, List[torch.FloatTensor]],
        res: int,
    ) -> List[torch.FloatTensor]:
        raise NotImplementedError

    @abstractmethod
    def _get_average_attention(self):
        raise NotImplementedError

    def aggregate_attention(
        self,
        places_in_unet: List[PLACE_IN_UNET],
        res: int,
        is_cross: bool,
        element_name: ATTENTION_COMPONENT,
    ) -> torch.FloatTensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self._get_average_attention()
        for place_in_unet in places_in_unet:
            dict_key = self._attention_store_key_string(place_in_unet, is_cross, element_name, res)
            item = attention_maps[dict_key]
            if item is None:
                continue
            out.extend(self._accumulate(item, res))

        if len(out) > 0:
            out = torch.cat(out, dim=0)
            out = out.sum(0) / out.shape[0]
        else:
            raise ValueError("No attentions maps found matching the criteria")

        return out

    def reset(self):
        self._cur_att_layer = 0
        self._cur_step_index = 0
        self.step_store = self._get_empty_store()
        self.attention_store = self._get_empty_store()

    def register_attention_processor(self):
        self._num_att_layers += 1


class AttentionStoreTimestep(AttentionStore):

    def _accumulate(self, item: List[torch.FloatTensor], res: int) -> List[torch.FloatTensor]:
        return [_unflatten_attn_map(_item, res) for _item in item]

    def _step_store(
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
        # TODO: As a general class, this should store feature, query, key and value
        # vectors also.
        dict_key = self._attention_store_key_string(place_in_unet, is_cross, "attn", res)
        if self.step_store[dict_key] is None:
            self.step_store[dict_key] = [attn]
        else:
            self.step_store[dict_key].append(attn)

    def _between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self._get_empty_store()
    
    def _get_average_attention(self):
        return self.attention_store
    

class AttentionStoreAccumulate(AttentionStore):

    def _accumulate(self, item: torch.FloatTensor, res: int) -> List[torch.FloatTensor]:
        return [_unflatten_attn_map(item, res)]

    def _step_store(
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
        attn_components = ["attn", "feat", "query", "key", "value"]
        attn_values = [attn, hidden_states, query, key, value]
        for attn_component, tensor in zip(attn_components, attn_values):
            dict_key = self._attention_store_key_string(place_in_unet, is_cross, attn_component, res)
            self.step_store[dict_key] = tensor

    def _between_steps(self):
        for key, value in self.step_store.items():
            if value is None:
                continue
            if self.attention_store[key] is None:
                self.attention_store[key] = value
            else:
                self.attention_store[key] += value
        self.step_store = self._get_empty_store()
    
    def _get_average_attention(self):
        # TODO: A tensor is not added here at every step. So we need to ensure that we know the true
        # length and don't assume that cur_step = 50
        cur_step = 50.0
        average_attention = {
            key: None if item is None else item / cur_step
            for key, item in self.attention_store.items()
        }
        return average_attention


class AttentionStoreEdit(AttentionStore, ABC):

    def __init__(self, ddim_steps: int, cross_replace_steps: int, self_replace_steps: int, indices_to_edit: List[int]):
        super().__init__()
        self.ddim_steps = ddim_steps
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.indices_to_edit = indices_to_edit
        self._cur_step_index = 0

    @abstractmethod
    def _cross_attention_edit_step(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def _self_attention_edit_step(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

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
    ) -> torch.FloatTensor:
        super().__call__(
            attn,
            is_cross,
            place_in_unet,
            hidden_states,
            query,
            key,
            value,
        )
        return self._edit(attn, is_cross)

    def _edit(
        self,
        attn: torch.FloatTensor,
        is_cross: bool,
    ) -> torch.FloatTensor:
        if is_cross and self._cur_step_index < self.cross_replace_steps:
            attn = self._cross_attention_edit_step(attn)
        elif (not is_cross) and self._cur_step_index < self.self_replace_steps:
            attn = self._self_attention_edit_step(attn)

        if self._cur_step_index == self.ddim_steps:
            self._cur_step_index = 0

        return attn

    def _accumulate(self, item: torch.FloatTensor, res: int) -> List[torch.FloatTensor]:
        return [_unflatten_attn_map(_item, res) for _item in item]

    def _between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self._get_empty_store()
        self._cur_step_index += 1

    def _get_average_attention(self):
        return self.attention_store

    def aggregate_attention(
        self,
        places_in_unet: List[PLACE_IN_UNET],
        res: int,
        is_cross: bool,
        element_name: ATTENTION_COMPONENT,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class AttentionStoreReplace(AttentionStoreEdit):

    def _cross_attention_edit_step(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def _self_attention_edit_step(self, attn: torch.FloatTensor):
        raise NotImplementedError


class AttentionStoreReweight(AttentionStoreEdit):

    def _cross_attention_edit_step(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError
        # C, H, W = attn.shape
        # attn_ = attn_.reshape(4, C // 4, H, W)
        # attn_replace = attn_[2].clone()
        # for i in self.indices_to_edit:
        #     attn_replace[:, :, i] *= 2
        # attn_[3] = attn_replace
        # return attn_.reshape(C, H, W)

    def _self_attention_edit_step(self, attn: torch.FloatTensor):
        raise NotImplementedError


class AttnProcessorWithAttentionStore:

    def __init__(self, attention_store: AttentionStore, place_in_unet: PLACE_IN_UNET):
        self.attention_store = attention_store
        self.attention_store.register_attention_processor()
        self.place_in_unet = place_in_unet

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

        # Only need to store attention maps during the Attend and Excite process
        # TODO: Refactor this so that we aren't checking for the None condition
        res = self.attention_store(
            attention_probs,
            is_cross,
            self.place_in_unet,
            hidden_states,
            query,
            key,
            value,
        )
        # TODO: This is only None at prediction time. So could refactor this better
        # so we're not always checking the conditons below.
        if res is not None:
            attention_probs = res

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

