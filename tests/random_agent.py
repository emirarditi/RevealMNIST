import gymnasium as gym
import reveal_mnist

env = gym.make('RevealMNIST-v0', classifier_model_weights_loc="mnist_predictor_masked.pt", device='mps')

NUM_EPISODES = 1000

for i in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    length = 0
    cumulative_reward = 0
    final_reward = 0
    while not done:
        length += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        done = terminated or truncated
        if done:
            final_reward = reward
        #env.render() # Uncomment to visualize the environment
    print(f"Episode {i + 1}: Length: {length}, Cumulative Reward: {cumulative_reward}")
