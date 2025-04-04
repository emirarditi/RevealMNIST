import gymnasium as gym
import reveal_mnist

env = gym.make('RevealMNIST-v0', classifier_model_weights_loc="/Users/emirarditi/reveal_mnist/utils/mnist_predictor_masked.pt", device='mps')