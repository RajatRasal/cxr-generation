import abc
import random
from abc import abstractmethod

import torch
from matplotlib.axes import Axes


class DiffusionCallback(abc.ABC):

    @abstractmethod
    def __call__(self, x: torch.FloatTensor, timestep: torch.LongTensor, eps: torch.FloatTensor):
        raise NotImplementedError()

    
class TrajectoryCallback(DiffusionCallback):

    def __init__(self):
        self.timesteps = []

    def __call__(self, x: torch.FloatTensor, timestep: torch.LongTensor, eps: torch.FloatTensor):
        self.timesteps.append(x)

    def plot(self, n: int, iter_chunks: int, ax: Axes, plot_legend: bool = False):
        trajectories = []
        for i in random.sample(list(range(len(self.timesteps[0]))), k=n):
            trajectory = []
            for timestep in self.timesteps:
                trajectory.append(timestep[i].squeeze(0).tolist())
            trajectories.append(trajectory)

        T = len(self.timesteps) 
        # Include the T = 0 generation
        ts = list(range(0, T, iter_chunks)) + [T - 1]
        for trajectory in trajectories:
            xs = [trajectory[i][0] for i in ts]
            ys = [trajectory[i][1] for i in ts]
            ax.plot(xs, ys)
            # noise
            ax.plot(xs[0], ys[0], "x", color="black")
            # generated data
            ax.plot(xs[-1], ys[-1], "o", color="black")
