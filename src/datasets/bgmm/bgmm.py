from dataclasses import dataclass
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
    means = torch.cat([means_distribution.sample() for _ in range(n_clusters)])
    return means


def gmm_stddev_prior(
    n_clusters: int,
    stddev_concentration: float = 6.,
    stddev_rate: float = 4.,
) -> torch.FloatTensor:
    stddev_distribution = Gamma(torch.tensor(stddev_concentration), torch.tensor(stddev_rate))
    stddevs = torch.cat([stddev_distribution.sample() for _ in range(n_clusters)])
    return stddevs


def gmm_mixture_probs_prior(
    categories_concentration: float,
    n_clusters: int,
) -> torch.FloatTensor:
    dirichlet = Dirichlet(categories_concentration * torch.ones(n_clusters))
    categorical_probs = dirichlet.sample()
    return categorical_probs

# TODO: Instead of returning tuples, we should return data classes.
@dataclass
class GMMSample:
    samples: torch.FloatTensor
    mixtures: torch.IntTensor


@dataclass
class GMMMixtureParameters:
    mean: torch.FloatTensor
    cov: torch.FloatTensor


class GMM:

    def __init__(
        self,
        means: torch.FloatTensor,
        covs: torch.FloatTensor,
        mixture_probs: torch.FloatTensor,
    ):
        assert means.shape[0] == covs.shape[0] == mixture_probs.shape[0]

        self.means = means
        self.covs = covs
        self.cat = Categorical(mixture_probs)
        self.mvns = [MultivariateNormal(mean, cov) for mean, cov in zip(self.means, self.covs)]

    @property
    def mixture_probs(self) -> torch.FloatTensor:
        return self.cat.logits

    def sample(self) -> GMMSample:
        # (x, z) ~ p(x, z)
        mixture = self.cat.sample()
        return GMMSample(
            self.mvns[mixture].sample().unsqueeze(0),
            mixture.unsqueeze(0),
        )

    def get_mixture_parameters(self, mixture: torch.IntTensor) -> GMMMixtureParameters:
        return GMMMixtureParameters(self.means[mixture.item()], self.covs[mixture.item()])

    def samples(self, N: int) -> GMMSample:
        # [(x_1, z_1), (x_2, z_2), ..., (x_N, z_N)]
        data = []
        mixtures = []
        for _ in tqdm(range(N)):
            sample = self.sample()
            data.append(sample.samples)
            mixtures.append(sample.mixtures)
        data = torch.cat(data)
        mixtures = torch.cat(mixtures)
        return GMMSample(data, mixtures)

    def _log_likelihood(self, samples: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        https://github.com/pytorch/pytorch/blob/35ea5c6b2289cadcc54da8d065f195a6841250f9/torch/distributions/mixture_same_family.py#L159
        """
        # (batch_size, channels, components)
        log_prob_x = torch.cat([mvn.log_prob(samples).unsqueeze(-1) for mvn in self.mvns], dim=-1)
        # (channels, components)
        log_prob_mixtures = torch.log_softmax(self.mixture_probs, dim=-1)
        # (batch_size, channels)
        log_lik = torch.logsumexp(log_prob_x + log_prob_mixtures, dim=-1)
        return log_prob_x, log_prob_mixtures, log_lik

    def log_likelihood(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        return self._log_likelihood(samples)[-1]

    def predict(self, samples: torch.FloatTensor) -> torch.LongTensor: 
        log_prob_x, log_prob_mixtures, log_lik = self._log_likelihood(samples)
        # (batch_size, channels, components)
        log_logits = log_prob_x + log_prob_mixtures
        # (batch_size, channels, components)
        responsibilities = (log_logits - log_lik[:, :, None]).exp()
        # (batch_size, channels)
        _classes = torch.argmax(responsibilities, dim=-1)
        return _classes
