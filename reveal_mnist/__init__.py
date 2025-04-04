from gymnasium.envs.registration import register

register(
    id='RevealMNIST-v0',
    entry_point="reveal_mnist.envs.reveal_mnist:RevealMNISTEnv",
)