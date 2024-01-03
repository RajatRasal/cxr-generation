import abc
from typing import Dict, Union, Optional, Tuple, List

import torch
from transformers import PreTrainedTokenizer

from .local_blend import LocalBlend
from .ptp_utils import get_time_words_attention_alpha, get_word_inds
from .seq_aligner import get_replacement_mapper, get_refinement_mapper


class AttentionController(abc.ABC):

    def __init__(self, low_resource: bool = False):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.low_resource = low_resource

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0
    
    @abc.abstractmethod
    def forward(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class EmptyController(AttentionController):
    
    def forward(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        return attn
    
    
class AttentionStore(AttentionController):

    def __init__(self, *args, **kwargs):
        super(AttentionStore, self).__init__(*args, **kwargs)
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": []
        }

    def forward(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControllerEdit(AttentionStore, abc.ABC):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        device: torch.device = None,
        local_blend: Optional[LocalBlend] = None,
        *args,
        **kwargs,
    ):
        super(AttentionControllerEdit, self).__init__(*args, **kwargs)
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        super(AttentionControllerEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace
                attn[1:] = attn_replace_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    

class AttentionReplace(AttentionControllerEdit):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        device: torch.device = None,
        local_blend: Optional[LocalBlend] = None,
        *args,
        **kwargs,
    ):
        super(AttentionReplace, self).__init__(
            tokenizer,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            device,
            local_blend,
            *args,
            **kwargs,
        )
        self.mapper = get_replacement_mapper(prompts, tokenizer).to(device)

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum("hpw, bwn->bhpn", attn_base, self.mapper)


class AttentionRefine(AttentionControllerEdit):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        device: torch.device = None,
        local_blend: Optional[LocalBlend] = None,
        *args,
        **kwargs,
    ):
        super(AttentionRefine, self).__init__(
            tokenizer,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            device,
            local_blend,
            *args,
            **kwargs,
        )
        mapper, alphas = get_refinement_mapper(prompts, tokenizer)
        self.mapper = mapper.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1]).to(device)

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace


class AttentionReweight(AttentionControllerEdit):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        equalizer: torch.Tensor,
        prev_controller: Optional[AttentionControllerEdit] = None,
        device: torch.device = None,
        local_blend: Optional[LocalBlend] = None,
        *args,
        **kwargs,
    ):
        super(AttentionReweight, self).__init__(
            tokenizer,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            device,
            local_blend,
            *args,
            **kwargs,
        )
        self.equalizer = equalizer.to(device)
        self.prev_controller = prev_controller

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace


def get_equalizer(
    tokenizer: PreTrainedTokenizer,
    text: str,
    word_select: Union[int, Tuple[int, ...]],
    values: Union[List[float], Tuple[float, ...]]
) -> torch.Tensor:
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def aggregate_attention(
    attention_store: AttentionStore,
    prompts: List[str],
    res: int,
    from_where: List[str],
    is_cross: bool,
    select: int,
) -> torch.Tensor:
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()
