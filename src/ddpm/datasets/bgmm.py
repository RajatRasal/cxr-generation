from typing import Tuple, List

import torch
from torch.distributions import Categorical, Dirichlet, Gamma, MultivariateNormal, Uniform
from tqdm import tqdm


def gmm_cluster_prior(min_clusters: int, max_clusters: int) -> int:
    n_clusters = max_clusters - min_clusters
    sample = Categorical(torch.tensor([1 / n_clusters] * n_clusters)).sample().item()
    return sample + min_clusters


def gmm_means_prior(
    n_clusters: int,
    dimensions: int = 2,
    center_box: Tuple[float, float] = (-10., 10.),
) -> torch.FloatTensor:
    means_distribution = Uniform(
        torch.tensor([center_box[0]] * dimensions),
        torch.tensor([center_box[1]] * dimensions),
    )
    means = [means_distribution.sample() for _ in range(n_clusters)]
    return means


def gmm_stddev_prior(
    n_clusters: int,
    stddev_concentration: float = 6.,
    stddev_rate: float = 4.,
) -> torch.FloatTensor:
    stddev_distribution = Gamma(torch.tensor(stddev_concentration), torch.tensor(stddev_rate))
    stddevs = [stddev_distribution.sample() for _ in range(n_clusters)]
    return stddevs


def gmm_mixture_probs_prior(
    categories_concentration: float,
    n_clusters: int,
) -> torch.FloatTensor:
    dirichlet = Dirichlet(categories_concentration * torch.ones(n_clusters))
    categorical_probs = dirichlet.sample()
    return categorical_probs

# TODO: Instead of returning tuples, we should return data classes.
class GMM:

    def __init__(
        self,
        means: List[torch.FloatTensor],
        covs: List[torch.FloatTensor],
        mixture_probs: torch.FloatTensor,
    ):
        assert len(means) == len(covs) == mixture_probs.shape[0]
        assert mixture_probs.sum() == 1

        self.means = means
        self.covs = covs
        self.mixture_probs = mixture_probs

        self.cat = Categorical(self.mixture_probs)
        self.mvns = [MultivariateNormal(mean, cov) for mean, cov in zip(self.means, self.covs)]

    def sample(self) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        # (x, z) ~ p(x, z)
        mixture = self.cat.sample()
        return self.mvns[mixture].sample(), mixture

    def get_mixture_parameters(self, mixture: torch.IntTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.means[mixture.item()], self.covs[mixture.item()]

    def samples(self, N: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # [(x_1, z_1), (x_2, z_2), ..., (x_N, z_N)]
        data = []
        mixtures = []
        for _ in tqdm(range(N)):
            _data, _mixture = self.sample()
            data.append(_data.unsqueeze(0))
            mixtures.append(_mixture.unsqueeze(0))
        data = torch.cat(data)
        mixtures = torch.cat(mixtures)
        return data, mixtures

    def log_likelihood(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        return 
