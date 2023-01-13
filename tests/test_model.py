import pytest
import torch

from src.models.model import MyAwesomeModel


class TestModel:
    # mark.parametrize enables parametrization of arguments for a test function
    # checks if certain input has the correct shape

    # checks if the output shape is correct
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (torch.rand((64, 1, 128, 128)), torch.Size([64, 12])),
            (torch.rand((10, 1, 128, 128)), torch.Size([10, 12])),
            (torch.rand((100, 1, 128, 128)), torch.Size([100, 12])),
        ],
    )
    def test_model_output(self, test_input, expected):
        model = MyAwesomeModel()
        output = model(test_input)
        assert output.shape == expected, f"Model output shape not as expected (Expected: {torch.Size([64, 10])}, Current:{output.shape})"

