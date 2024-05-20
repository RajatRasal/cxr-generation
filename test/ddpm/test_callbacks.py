import torch

from diffusers.schedulers import DDPMScheduler

from ddpm.training.callbacks import TrajectoryCallback
from ddpm.training.plots import trajectory_plot


def test_trajectory_callback():
    # build test dataset as a forward diffusion process
    mean = 5.
    T = 1000
    N = 500
    scheduler = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2")
    trajectories = []
    for i in range(N):
        if i < N // 2:
            x0 = torch.normal(torch.tensor([-mean, -mean]), torch.tensor([1., 1.]))
        else:
            x0 = torch.normal(torch.tensor([mean, mean]), torch.tensor([1., 1.]))
        x0 = x0.unsqueeze(0)
        xs = [x0]
        x = x0
        for t in range(1, T):
            x = scheduler.add_noise(x, torch.randn_like(x), torch.tensor(t).long())
            xs.append(x)
        trajectories.append(torch.cat(xs).unsqueeze(0))
    trajectories = torch.cat(trajectories)

    # write data to callback
    S = 10  # 00
    callback = TrajectoryCallback()
    for i in range(T):
        x_t = trajectories[:, i, :].unsqueeze(1)
        callback(x_t, torch.tensor(i).long(), torch.randn_like(x_t))
    timesteps_0, trajectories_0 = callback.sample(n=S, dim=0)
    timesteps_1, trajectories_1 = callback.sample(n=S, dim=1)

    assert len(callback.timesteps) == T
    assert timesteps_0 == timesteps_1
    assert len(trajectories_0) == S
    assert len(trajectories_1) == S
    assert all(len(traj) == T for traj in trajectories_0)
    assert all(len(traj) == T for traj in trajectories_1)
    
    trajectory_plot(timesteps_0, trajectories_0, T, 15, 4, "test/ddpm/figures/trajectory.png")