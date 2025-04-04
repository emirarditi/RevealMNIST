# Reveal MNIST

A gym environment for the CS 445&545 Project. This environment gradually reveals parts of an MNIST image based on agents actions'. Images are broken down to sub patches of size 4x4, so the agent moves in a 7x7 grid.

## Action Space
The action space is as follows:
- Left
- Right
- Up
- Down
- Predict 

## State Space
The state space has a size of 788:
- first 784: the input image
- Agent X
- Agent Y
- Number of consequent predictions
- Image reveal percentage 

## Installation

To install the package, run:

```bash
pip install -e .
```

## Predictor Model
The predictor model is a simple CNN model that takes the MNIST image as input and outputs the predicted digit. The model is trained on the MNIST dataset and can be used to predict the digit in the image.
It is already implemented and the weights are provided to you. For your project, you SHOULD NOT update any part of the predictor model.
The weights of the model is located in the **mnist_predictor_masked.pt** file.

## Usage

Import the package and create a gym environment:

```python
import gymnasium as gym
import reveal_mnist

# Create the environment
env = gym.make('RevealMNIST-v0', 
               classifier_model_weights_loc="LOCATION_OF_PROVIDED_WEIGHTS",
               device='DEVICE_TO_USE', visualize=True) # the device should be "cpu", "gpu" or "mps" based on your config
```

## Running the Random Agent
```bash
python3 tests/random_agent.py
```

## License

[MIT License](LICENSE)