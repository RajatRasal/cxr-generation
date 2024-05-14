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


def sample_gmm(
    means: List[torch.FloatTensor],
    stddevs: List[torch.FloatTensor],
    categories_concentration: float = 0.01,
    # n_samples: int = 1000,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    A GMM with Dirichlet prior over the categories, so that we can control the number
    of datapoints sampled from each category, i.e. if concentration is small each sample
    will be primarily centered around one of the categories.

    We place a uniform prior over the means, such that each category's mean can fall
    uniformly anywhere within the box between (center_box[0], center_box[0]) and 
    (center_box[1], center_box[1]).

    A gamma prior is placed over the stddevs, such most blobs have a stddev focussed
    around the mode of the gamma distribution.
    """
    assert len(means) == len(stddevs)
    assert categories_concentration < 0.1

    # stddev for mvn
    dim = len(means[0])
    scales_tril = [torch.diag(stddev * torch.ones((dim,))) for stddev in stddevs]

    # prior over categories
    dirichlet = Dirichlet(categories_concentration * torch.ones(len(means)))
    categorical_probs = dirichlet.sample()

    # gmm
    cat = Categorical(categorical_probs)
    mvns = [MultivariateNormal(means[comp], scale_tril=scales_tril[comp]) for comp in range(len(means))]

    # 1 data point
    _class = torch.argmax(categorical_probs)
    return mvns[cat.sample()].sample(), means[_class]


def gmm_dataset(N: int = 1000, **gmm_kwargs) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # TODO: Add option to return true means as conditions
    data = []
    cond = []
    for _ in tqdm(range(N)):
        _data, _cond = sample_gmm(**gmm_kwargs)
        data.append(_data.unsqueeze(0))
        cond.append(_cond.unsqueeze(0))
    data = torch.cat(data)
    cond = torch.cat(cond)
    return data, cond    


# def gmm_dataset(N: int = 1000, cond_estimate: bool = True, **gmm_kwargs) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
#     dataset = []
#     conds = []
#     for _ in tqdm(range(N)):
#         data_point, mean = gmm(**gmm_kwargs)
#         data_point = data_point.unsqueeze(0).transpose(1, 2)
#         dataset.append(data_point)
#         if cond_estimate:
#             # cond = empirical mean
#             cond = data_point.squeeze(0).mean(axis=1).unsqueeze(0)
#         else:
#             # cond = a gmm mode = mean of the gmm component
#             cond = mean
#         print(cond.shape)
#         print(data_point.shape)
#         conds.append(cond)
#     conds = torch.cat(conds)
#     dataset = torch.cat(dataset)
#     return dataset, conds
