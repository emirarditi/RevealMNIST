import gymnasium as gym
from gymnasium import spaces
import torch
from torchvision import datasets, transforms
import numpy as np
import random
from .MNISTPredictor import MNISTPredictor
import matplotlib.pyplot as plt

class RevealMNISTEnv(gym.Env):
    def __init__(self, classifier_model_weights_loc, device="cpu", visualize=False, stochastic=False):
        super().__init__()

        self.device = device
        classifier = MNISTPredictor()
        self.classifier = MNISTPredictor()
        if classifier_model_weights_loc is None:
            raise ValueError("Classifier model weights location must be provided.")
        self.classifier.load_state_dict(torch.load(classifier_model_weights_loc, map_location=device))
        self.classifier.to(device)
        self.classifier.eval()
        self.move_cost = -1
        self.predict_cost = -2
        self.max_episode_steps = 200
        self.step_count = 0
        self.visualize = visualize
        self.stochastic = stochastic
        # Load MNIST
        self.mnist = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        self.image_size = 28
        self.patch_size = 4
        self.grid_size = self.image_size // self.patch_size  # 4x4 patches
        self.num_patches = self.grid_size ** 2  # 16 patches

        # Observation: 784 (image) + 2 (position) + 1 (failed predicts) + 1 (revealed patch ratio)
        self.observation_space = spaces.Box(low=0, high=1, shape=(788,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # left, right, up, down, predict

        self.max_failed_predicts = 3
        self.reset()

        if visualize:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
            self.fig.show()  # Show the figure once

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.image_idx = random.randint(0, len(self.mnist) - 1)
        self.full_image, self.label = self.mnist[self.image_idx]
        self.full_image = self.full_image.squeeze(0)  # shape: 28x28

        self.revealed_mask = torch.zeros_like(self.full_image, dtype=torch.bool)
        self.revealed_patches = set()

        self.agent_x = random.randint(0, self.grid_size - 1)
        self.agent_y = random.randint(0, self.grid_size - 1)
        self.consecutive_failed_predicts = 0

        self._reveal_current_patch()

        return self._get_obs(), {}

    def _reveal_current_patch(self):
        x_start = self.agent_x * self.patch_size
        y_start = self.agent_y * self.patch_size

        self.revealed_mask[y_start:y_start + self.patch_size, x_start:x_start + self.patch_size] = True
        self.revealed_patches.add((self.agent_x, self.agent_y))

    def _get_obs(self):
        revealed_image = self.full_image.clone()
        revealed_image[~self.revealed_mask] = 0
        flat_image = revealed_image.flatten()

        position = torch.tensor([self.agent_x, self.agent_y], dtype=torch.float32)
        predicts = torch.tensor([self.consecutive_failed_predicts], dtype=torch.float32)
        revealed_ratio = torch.tensor([len(self.revealed_patches) / self.num_patches], dtype=torch.float32)

        return torch.cat([flat_image, position, predicts, revealed_ratio]).numpy()

    def step(self, action):
        if self.stochastic:
            if random.random() < 0.2:
                action = random.randint(0, 4)
        terminated = False
        reward = 0
        self.step_count += 1
        if action == 4:  # predict
            revealed_image = self.full_image.clone()
            revealed_image[~self.revealed_mask] = 0
            input_tensor = revealed_image.unsqueeze(0).unsqueeze(0).to(self.device)  # shape: 1x1x28x28

            with torch.no_grad():
                pred = self.classifier(input_tensor).argmax(dim=1).item()

            if pred == self.label:
                revealed_ratio = len(self.revealed_patches) / self.num_patches
                reward = 100 * (1 - revealed_ratio)
                terminated = True
                self.consecutive_failed_predicts = 0
            else:
                reward = self.predict_cost
                self.consecutive_failed_predicts += 1
                if self.consecutive_failed_predicts >= self.max_failed_predicts:
                    terminated = True
                    reward = -100

        else:
            # Movement penalty always applied for non-predict actions
            reward += self.move_cost
            self.consecutive_failed_predicts = 0

            if action == 0:  # left
                self.agent_x = max(self.agent_x - 1, 0)
            elif action == 1:  # right
                self.agent_x = min(self.agent_x + 1, self.grid_size - 1)
            elif action == 2:  # up
                self.agent_y = max(self.agent_y - 1, 0)
            elif action == 3:  # down
                self.agent_y = min(self.agent_y + 1, self.grid_size - 1)

        self._reveal_current_patch()

        if len(self.revealed_patches) == self.num_patches:
            terminated = True

        if self.step_count >= self.max_episode_steps and not terminated:
            reward = -100
            terminated = True

        obs = self._get_obs()

        return obs, reward, terminated, False, {}

    def render(self):
        if not self.visualize:
            raise EnvironmentError("Visualization is not enabled. Set visualize=True to enable rendering.")
        # Clear previous images without causing flicker
        self.ax1.clear()
        self.ax2.clear()

        # Full image (unmodified)
        self.ax1.imshow(self.full_image.numpy(), cmap='gray')
        self.ax1.set_title("Full Image")
        self.ax1.axis('off')

        # Partially revealed image (only the revealed parts)
        revealed_image = self.full_image.clone()
        revealed_image[~self.revealed_mask] = 0  # Set un-revealed parts to 0
        self.ax2.imshow(revealed_image.numpy(), cmap='gray')
        self.ax2.set_title(f"Revealed Image\nAgent: ({self.agent_x}, {self.agent_y})")
        self.ax2.axis('off')

        # Draw the updated figure smoothly without flicker
        self.fig.canvas.draw_idle()  # This efficiently redraws the canvas without flicker
        plt.pause(0.001)
