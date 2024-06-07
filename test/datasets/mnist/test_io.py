import os

import numpy as np

from datasets.mnist.io import load_idx, save_idx


def test_io(mnist_file):
    TEST_FILE = "test"

    data = load_idx(mnist_file)
    save_idx(data, TEST_FILE)
    data_ = load_idx(test_file)

    os.remove(TEST_FILE)
    assert np.alltrue(data == data_)
