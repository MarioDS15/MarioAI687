import torch
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data import TensorDictReplayBuffer, ListStorage
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
class Agent:
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 lr=0.00025, 
                 gamma=0.9, 
                 epsilon=1.0, 
                 eps_decay=0.99999975, 
                 eps_min=0.1, 
                 replay_buffer_capacity=40_000, 
                 batch_size=32, 
                 sync_network_rate=10000,
                 use_enemy_channel=False):

        self.num_actions = num_actions
        self.learn_step_counter = 0
        self.use_enemy_channel = use_enemy_channel

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()
        storage = ListStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        if self.use_enemy_channel:
            state, enemy_visible = observation
            enemy_channel = np.full((1, state.shape[1], state.shape[2]), enemy_visible, dtype=np.float32)
            extended = np.concatenate([np.array(state), enemy_channel], axis=0)
        else:
            extended = np.array(observation)

        extended_tensor = torch.tensor(extended, dtype=torch.float32).unsqueeze(0).to(self.online_network.device)

        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return self.online_network(extended_tensor).argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        if self.use_enemy_channel:
            s, vis1 = state
            ns, vis2 = next_state
            s_extended = np.concatenate([np.array(s), np.full_like(s[0:1], vis1)], axis=0)
            ns_extended = np.concatenate([np.array(ns), np.full_like(ns[0:1], vis2)], axis=0)
        else:
            s_extended = np.array(state)
            ns_extended = np.array(next_state)

        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(s_extended, dtype=torch.float32).detach(),
            "action": torch.tensor(action).detach(),
            "reward": torch.tensor(reward).detach(),
            "next_state": torch.tensor(ns_extended, dtype=torch.float32).detach(),
            "done": torch.tensor(done).detach()
        }, batch_size=[]))

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)
        keys = ("state", "action", "reward", "next_state", "done")
        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        self.decay_epsilon()
