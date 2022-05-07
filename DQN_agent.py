from datetime import datetime
import gym
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
from Modules import ReplayBuffer, simple_Transition
import numpy as np
import random
import copy
import math
from Modules import MULTIDISCRETE_RESNET
import gym_pick_and_place
from helpers import save_csv


N_EPISODES = 1000
STEPS_PER_EPISODE = 50
HEIGHT = 200
WIDTH = 200
MEMORY_SIZE = 2000
BATCH_SIZE = 12
LEARNING_RATE = 0.001
EPS_STEADY = 0.0
EPS_START = 1.0
EPS_END = 0.2
EPS_DECAY = 8000

SAVE_TRAINING_DATA = False

N_TEST_EPISODES = 1
N_TEST_STEPS_PER_EPISODE = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN_Agent:
    def __init__(
        self,
        learning_rate=LEARNING_RATE,
        load_weights_path=None,
        train=True,
        model_type="binary"
    ):

        torch.manual_seed(20)
        torch.cuda.manual_seed(20)
        np.random.seed(20)
        random.seed(20)
        self.model_type = model_type
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.TRAINING_DATA_PATH = "training_data/{}.csv".format(model_type)
        self.env = gym.make(
            "Pick_and_Place-v0", model_type=model_type, train=train
        )
        self.n_actions_1, self.n_actions_2 = (
            self.env.action_space.nvec[0],
            self.env.action_space.nvec[1],
        )
        self.action_space = self.n_actions_1 * self.n_actions_2
        self.policy_net = MULTIDISCRETE_RESNET(
            number_actions_dim_2=self.n_actions_2).to(device)

        checkpoint = None

        if load_weights_path:
            checkpoint = torch.load(
                load_weights_path, map_location=torch.device('cpu'))
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded weights from " + load_weights_path)

        self.normal_rgb = T.Compose(
            [
                T.ToPILImage(),
                T.ColorJitter(brightness=0.5, contrast=0.5,
                              saturation=0.5, hue=0.5),
                T.ToTensor(),
            ]
        )
        self.depth_threshold = np.round(
            self.env.model.cam_pos0[self.env.model.camera_name2id(
                "top_down")][2]
            - self.env.environment.TABLE_HEIGHT
            + 0.01,
            decimals=3,
        )

        if train:
            self.set_up_train_data(
                learning_rate, load_weights_path, checkpoint)

    def set_up_train_data(self, learning_rate, load_weights_path, checkpoint):
        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=0.00002
        )

        if load_weights_path is not None:
            self.load_weights(checkpoint, load_weights_path)
        else:
            self.steps_done = 0
            self.eps_threshold = EPS_START

            now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

            self.WEIGHT_PATH = "model_weights/{}_{}_weights.pt".format(self.model_type, now)

    def load_weights(self, checkpoint, load_weights_path):
        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["step"] if "step" in checkpoint.keys(
        ) else 0
        self.eps_threshold = (
            checkpoint["epsilon"] if "epsilon" in checkpoint.keys(
            ) else EPS_STEADY
        )
        self.WEIGHT_PATH = load_weights_path

    def epsilon_greedy(self, state):
        sample = random.random()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )

        self.steps_done += 1

        if sample > self.eps_threshold:
            return self.greedy(state)
        else:
            return self.random_action()

    def random_action(self):
        while True:
            # make sure coordinate in not below the table to speed up training
            action = random.randrange(self.action_space)
            action_1 = action % self.n_actions_1
            x = action_1 % self.env.IMAGE_WIDTH
            y = action_1 // self.env.IMAGE_WIDTH
            depth = self.env.current_observation["depth"][y][x]
            coordinates = self.env.camera.convert_pixel_to_world_coordinates(
                pixel_x=x,
                pixel_y=y,
                depth=depth,
                height=self.env.IMAGE_HEIGHT,
                width=self.env.IMAGE_WIDTH,
            )
            if coordinates[2] >= (self.env.environment.TABLE_HEIGHT - 0.01):
                break

        return torch.tensor([[action]], dtype=torch.long)

    def greedy(self, state):
        with torch.no_grad():
            max_idx = self.policy_net(state.to(device)).view(-1).max(0)[1]
            max_idx = max_idx.view(1)
            return max_idx.unsqueeze_(0).cpu()

    def transform_observation(self, observation):
        depth = copy.deepcopy(observation["depth"])
        depth[np.where(depth > self.depth_threshold)] = self.depth_threshold

        rgb = copy.deepcopy(observation["rgb"])

        depth += np.random.normal(loc=0, scale=0.001, size=depth.shape)
        depth *= -1
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

        depth = np.expand_dims(depth, 0)
        rgb_tensor = self.normal_rgb(rgb).float()

        depth_tensor = torch.tensor(depth).float()
        obs_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0)
        obs_tensor.unsqueeze_(0)

        del rgb, depth, rgb_tensor, depth_tensor

        return obs_tensor

    def transform_action(self, action):
        action_value = action.item()
        action_1 = action_value % self.n_actions_1
        action_2 = action_value // self.n_actions_1

        return np.array([action_1, action_2])

    def learn(self):
        if len(self.memory) < 2 * BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = simple_Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state[:BATCH_SIZE]).to(device)
        action_batch = torch.cat(
            batch.action[:BATCH_SIZE]).to(device)

        reward_batch = torch.cat(
            batch.reward[:BATCH_SIZE]).to(device)

        q_pred = (
            self.policy_net(state_batch).view(
                BATCH_SIZE, -1).gather(1, action_batch)
        )

        q_expected = reward_batch.float()

        if self.model_type == "binary":
            loss = F.binary_cross_entropy(
                q_pred, q_expected)

        elif self.model_type == "manhattan":
            loss = F.binary_cross_entropy(
                q_pred, q_expected)

        elif self.model_type == "categorical":
            q_expected = torch.max(q_expected, 1)[1]
            loss = F.cross_entropy(
                q_pred, q_expected)

        elif self.model_type == "euclidean":
            q_pred = q_pred.reshape(-1)
            q_expected = q_expected.reshape(-1)
            loss = F.mse_loss(q_pred, q_expected)

        loss.backward()

        self.optimizer.step()

        self.optimizer.zero_grad()

    def train(self):
        self.optimizer.zero_grad()
        for episode in range(N_EPISODES):
            total_grasps = 0
            total_rewards = 0
            state = self.env.reset()
            state = self.transform_observation(state)
            for _ in range(STEPS_PER_EPISODE):
                action = self.epsilon_greedy(state)
                env_action = self.transform_action(action)
                next_state, reward, done, action_info = self.env.step(
                    env_action
                )
                success = action_info["success"]
                if success:
                    total_grasps += 1
                total_rewards += reward

                reward = torch.tensor([[reward]])

                next_state = self.transform_observation(next_state)

                self.memory.push(state, action, reward)

                state = next_state

                if done:
                    break

                self.learn()

            if SAVE_TRAINING_DATA:
                save_csv(self.TRAINING_DATA_PATH, episode + 1,
                         total_grasps, total_rewards)

                torch.save(
                    {
                        "step": self.steps_done,
                        "model_state_dict": self.policy_net.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epsilon": self.eps_threshold,
                    },
                    self.WEIGHT_PATH,
                )

    def test(self):
        for _ in range(N_TEST_EPISODES):
            number_grasp = 0
            total_rewards = 0
            state = self.env.reset()
            state = self.transform_observation(state)
            for _ in range(N_TEST_STEPS_PER_EPISODE):
                action = self.greedy(state)
                env_action = self.transform_action(action)
                next_state, reward, done, action_info = self.env.step(
                    env_action
                )
                success = action_info["success"]
                if success:
                    number_grasp += 1
                total_rewards += reward
                state = self.transform_observation(next_state)
                if done:
                    break
