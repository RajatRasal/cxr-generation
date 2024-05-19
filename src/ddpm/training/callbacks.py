import abc
import random
from abc import abstractmethod
from typing import List, Tuple

import torch


class DiffusionCallback(abc.ABC):

    @abstractmethod
    def __call__(self, x: torch.FloatTensor, timestep: torch.LongTensor, eps: torch.FloatTensor):
        raise NotImplementedError()

    
class TrajectoryCallback(DiffusionCallback):

    def __init__(self):
        self.timesteps = []

    def __call__(self, x: torch.FloatTensor, timestep: torch.LongTensor, eps: torch.FloatTensor):
        self.timesteps.append(x.cpu())

    def sample(self, n: int, dim: int, seed: int = 0) -> Tuple[List[int], List[List[float]]]:
        """
        Plot n randomly chosen trajectories
        """
        trajectories = []
        batch_size = len(self.timesteps[0])
        random.seed(seed)
        for i in random.sample(list(range(batch_size)), k=n):
            trajectories.append([
                timestep[i, :, dim].item()
                for timestep in self.timesteps
            ])
        return list(range(len(trajectories[0]))), trajectories
