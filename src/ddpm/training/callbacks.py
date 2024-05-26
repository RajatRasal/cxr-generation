import abc
from abc import abstractmethod
from typing import List, Optional, Tuple

import torch


class DiffusionCallback(abc.ABC):

    @abstractmethod
    def __call__(
        self,
        x: Optional[torch.FloatTensor],
        timestep: Optional[torch.LongTensor],
        eps: Optional[torch.FloatTensor],
    ):
        raise NotImplementedError()

    
class TrajectoryCallback(DiffusionCallback):

    def __init__(self):
        self.timesteps = []

    def __call__(
        self,
        x: Optional[torch.FloatTensor],
        timestep: Optional[torch.LongTensor],
        eps: Optional[torch.FloatTensor],
    ):
        self.timesteps.append(x.cpu())

    def sample(self, n: int, dim: int) -> List[torch.FloatTensor]:
        """
        Plot the first n trajectories
        """
        trajectories = []
        batch_size = len(self.timesteps[0])
        for i in range(n):
            trajectory = [timestep[i, :, dim] for timestep in self.timesteps]
            trajectories.append(torch.cat(trajectory))
        return trajectories
