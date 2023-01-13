import pytest
import torch

from src.models.model import MyAwesomeModel


class TestModel:
    # mark.parametrize enables parametrization of arguments for a test function
    # checks if certain input has the correct shape
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (torch.randn(1, 2, 28), ValueError),
            (torch.randn( 1, 28, 2), ValueError),
            (torch.randn(100, 3, 28, 28), ValueError),
        ],
    )
    def test_model_input_shape(self, test_input, expected):
        model = MyAwesomeModel()
        state_dict = torch.load("src/models/my_trained_model.pt")
        model.load_state_dict(state_dict)

        # if the input has the wrong shape, the model should raise a ValueError
        with pytest.raises(
            expected, match="Expected each x sample to have shape 1,28,28"
        ):
            model.forward(test_input)

    # checks if the output shape is correct
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (torch.randn(128, 128), (1)),
            (torch.randn(64, 128, 128), (64, 1)),
            (torch.randn(100, 1, 128, 128), (100, 1)),
        ],
    )
    def test_model_output(self, test_input, expected):
        model = MyAwesomeModel()
        state_dict = torch.load("src/models/my_trained_model.pt")
        model.load_state_dict(state_dict)
        
        assert output.shape == expected
