import math
import os

import pytest
import torch

from tests import _PATH_DATA

# skip test if data files are not found
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")

class TestDataset:
    # check the length of the training dataset
    def test_load_traindata(self):
        self.train_test_data = torch.load(f"{_PATH_DATA}/processed/train_test_processed.pt")
        self.images = self.train_test_data["train_data"]
        self.labels = self.train_test_data["train_labels"]
        assert len(self.images) == 18000 and len(self.labels) == 18000

    # check the length of the test dataset
    def test_load_testdata(self):
        self.train_test_data = torch.load(f"{_PATH_DATA}/processed/train_test_processed.pt")
        self.images = self.train_test_data["test_data"]
        self.labels = self.train_test_data["test_labels"]
        assert len(self.images) == 3600 and len(self.labels) == 3600

    # check the shape of the training dataset
    def test_shape_traindata(self):
        self.train_test_data = torch.load(f"{_PATH_DATA}/processed/train_test_processed.pt")
        for (images, labels) in zip(self.train_test_data["train_data"], self.train_test_data["train_labels"]):
            assert (1, 128, 128) == images.shape

    # check the shape of the test dataset
    def test_shape_testdata(self):
        self.train_test_data = torch.load(f"{_PATH_DATA}/processed/train_test_processed.pt")
        for (images, labels) in zip(self.train_test_data["test_data"], self.train_test_data["test_labels"]):
            assert (1, 128, 128) == images.shape

    # check if the labels are not None
    def test_labels_traindata(self):
        self.train_test_data = torch.load(f"{_PATH_DATA}/processed/train_test_processed.pt")
        for (images, labels) in zip(self.train_test_data["test_data"], self.train_test_data["test_labels"]):
            assert labels is not None
