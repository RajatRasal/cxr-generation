import torch

from datasets.mnist.datamodules import MNISTDataModule


def test_image_normalisation_mnist_datamodule():
    split_ratio = (0.9, 0.1)
    dm = MNISTDataModule(batch_size=32, split_ratio=split_ratio)

    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    data, labels = [], []
    for x in train_dataloader:
        data.append(x["image"])
        labels.append(x["label"])
    data = torch.cat(data)
    labels = torch.cat(labels)
    assert data.shape == (60000 * split_ratio[0], 1, 28, 28)
    assert torch.isclose(data.mean(), torch.tensor(0.), atol=5e-5) and torch.isclose(data.std(), torch.tensor(1.), atol=1e-3)
    assert labels.shape == (60000 * split_ratio[0], 10)

    val_dataloader = dm.val_dataloader()
    data, labels = [], []
    for x in val_dataloader:
        data.append(x["image"])
        labels.append(x["label"])
    data = torch.cat(data)
    labels = torch.cat(labels)
    assert data.shape == (60000 * split_ratio[1], 1, 28, 28)
    assert torch.isclose(data.mean(), torch.tensor(0.), atol=5e-3) and torch.isclose(data.std(), torch.tensor(1.), atol=1e-3)
    assert labels.shape == (60000 * split_ratio[1], 10)

    # TODO: Unittest for test and predict options
    # TODO: Unittest for ColourMNIST with imagenet normalisation?
    # TODO: Property to get out the normalisation values
