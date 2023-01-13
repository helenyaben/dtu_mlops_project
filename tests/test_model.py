from src.models.model import MyAwesomeModel
import torch
import pytest
from torch import nn
import numpy as np

class TestModel:
    # mark.parametrize enables parametrization of arguments for a test function
    # checks if certain input has the correct shape
    # @pytest.mark.parametrize(
    #     "test_input,expected",
    #     [
    #         (torch.rand((128, 128)), RuntimeError),
    #     ],
    # )
    # def test_model_input_shape(self, test_input, expected):
    #     model = MyAwesomeModel()
    #     # if the input has the wrong shape, the model should raise a ValueError
    #     with pytest.raises(
    #         expected, match="Expected each x sample to have shape 1,128,128"
    #     ):
    #         model.forward(test_input)

    # checks if the output shape is correct
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (torch.rand((64, 1, 128, 128)), torch.Size([64, 12])),
            (torch.rand((1,1, 128, 128)), torch.Size([1,12])),
            (torch.rand((100,1, 128, 128)), torch.Size([100,12])),
        ],
    )
    def test_model_output(self, test_input, expected):
        model = MyAwesomeModel()
        output = model(test_input)
        assert output.shape == expected, f"Expected output shape {expected}, got {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__])
