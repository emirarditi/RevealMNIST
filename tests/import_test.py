import gymnasium as gym
import reveal_mnist

env = gym.make('RevealMNIST-v0',
               classifier_model_weights_loc="./mnist_predictor_masked.pt",
               device='DEVICE_TO_USE', visualize=True) # the device should be "cpu", "gpu" or "mps" based on your config
