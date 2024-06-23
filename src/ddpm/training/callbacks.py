import abc
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

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
        self.data = []
        self.timesteps = []

    def __call__(
        self,
        x: torch.FloatTensor,
        timestep: int,
        eps: Optional[torch.FloatTensor],
    ):
        self.data.append(x.cpu())

    def sample(self, n: int, dim: int) -> List[torch.FloatTensor]:
        """
        Plot the first n trajectories
        """
        trajectories = []
        batch_size = len(self.data[0])
        for i in range(n):
            trajectory = [timestep[i, :, dim] for timestep in self.data]
            trajectories.append(torch.cat(trajectory))
        return trajectories


class TrajectoryCallback2D(DiffusionCallback):

    def __init__(self):
        self.data = []
        self.timesteps = []

    def __call__(
        self,
        x: Optional[torch.FloatTensor],
        timestep: Optional[torch.LongTensor],
        eps: Optional[torch.FloatTensor],
    ):
        self.data.append(x.cpu())

    def sample(self, n: int) -> List[List[torch.FloatTensor]]:
        """
        Plot the first n trajectories
        """
        trajectories = []
        for i in range(n):
            trajectory = [d[i] for d in self.data]
            trajectories.append(trajectory)
        return trajectories
