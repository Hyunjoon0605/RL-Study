import torch

from model.policy.utils import *
from model.policy.DDPG_model import *


class DDPG(nn.Module):

    def __init__(self, config, blue):
        super(DDPG, self).__init__()

        self.n_dim = config.model_config.node_state_dim
        self.e_dim = config.model_config.edge_state_dim
        self.a_dim = config.model_config.action_dim
        self.h_dim = config.model_config.hidden_dim

        self.num_agent = config.num_blue if blue else config.num_red
        self.num_env = config.num_env

        self.blue = blue
        self.team_name = 'Blue' if self.blue else 'Red'
        self.agent_idx = 0 if self.blue else 1

        self.gamma = 0.99
        self.mse = nn.MSELoss()
        self.lr_critic = 0.001
        self.lr_actor = 0.0003
        self.batch_size = 32
        self.tau = 0.003
        self.memory = ReplayMemoryGraph(100000)
        self.memory_prev_obs = None

        self.noise = [[OUNoise(np.zeros(self.a_dim)) for _ in range(self.num_agent)] for _ in range(self.num_env)]
        self.noise_weight = 1.0
        self.noise_weight_min = 0.1
        self.noise_weight_decay = 0.000005

        edge_read_out = [[0, 1, 2], [0]] if self.blue else [[3, 4, 5], [4]]
        node_read_out = 0 if self.blue else 1
        if self.blue:
            con_range_actor = config.env_config.blue_con_range
        else:
            con_range_actor = config.env_config.red_con_range

        self.actor = Actor(node_input_dim=self.n_dim, edge_input_dim=self.e_dim,
                           hidden_dim=self.h_dim, action_dim=self.a_dim,
                           edge_type=edge_read_out, node_type=node_read_out,
                           aggregator='min', con_range=con_range_actor)
        self.actor_target = Actor(node_input_dim=self.n_dim, edge_input_dim=self.e_dim,
                                  hidden_dim=self.h_dim, action_dim=self.a_dim,
                                  edge_type=edge_read_out, node_type=node_read_out,
                                  aggregator='min', con_range=con_range_actor)

        edge_read_out = [[0, 1, 2], [0]] if self.blue else [[3, 4, 5], [4]]
        node_read_out = 0 if self.blue else 1
        if self.blue:
            con_range_critic = [{0: None, 1: None, 2: None}, {0: None}]
        else:
            con_range_critic = [{3: None, 4: None, 5: None}, {4: None}]

        self.critic = Critic(node_input_dim=self.n_dim, edge_input_dim=self.e_dim,
                             hidden_dim=self.h_dim, action_dim=self.a_dim,
                             edge_type=edge_read_out, node_type=node_read_out,
                             aggregator='sum', con_range=con_range_critic)
        self.critic_target = Critic(node_input_dim=self.n_dim, edge_input_dim=self.e_dim,
                                    hidden_dim=self.h_dim, action_dim=self.a_dim,
                                    edge_type=edge_read_out, node_type=node_read_out,
                                    aggregator='sum', con_range=con_range_critic)

        self.update_model(self.actor, self.actor_target, 1.0)
        self.update_model(self.critic, self.critic_target, 1.0)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=1e-2)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)

    @staticmethod
    def update_model(source, target, tau):
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, model_name):
        torch.save(self.state_dict(), model_name)

    def load_model(self, model_name):
        self.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

    def insert(self, transition):
        self.memory.push((self.memory_prev_obs, *transition))

    def save_replay_buffer(self, file_name):
        torch.save(self.memory, file_name)

    @torch.no_grad()
    def get_action(self, obs, deterministic=False):

        action = self.actor(obs).detach().cpu().numpy().reshape(self.num_env, self.num_agent)
        self.memory_prev_obs = obs

        if not deterministic:
            noise = np.array([self.noise[j][i]() for i in range(self.num_agent) for j in range(self.num_env)]).reshape(*action.shape) * self.noise_weight
            action += noise
            if self.noise_weight > self.noise_weight_min:
                self.noise_weight -= self.noise_weight_decay * self.num_env

        return action

    def train_ok(self):
        return self.batch_size <= len(self.memory)

    def train_model(self, train_iter):

        ### 0. Check if it satisfy the training condition
        if not self.train_ok():
            return dict()

        ### 1. Main Training Loop (for 'train_iter' times)
        loss_value_all = loss_policy_all = 0.0
        for _ in range(train_iter):

            ###### 1.1. Sample transitions from the replay buffer
            state, next_state, action, reward, terminal, valid = self.memory.sample(self.batch_size)

            ###### 1.2. Compute critic loss & update the critic network (policy evaluation)
            ######### 1.2.1. Compute target q value: q_target = r + γ * (1 - d) * Q'(s', π'(s))
            with torch.no_grad():
                next_q_val = self.critic_target(next_state, self.actor_target(next_state))
                target_q = (reward + (1 - terminal) * self.gamma * next_q_val).detach()

            ######### 1.2.2. Compute pred q value: q_pred = Q(s, a)
            q = self.critic(state, action)

            ######### 1.2.3. Update critic network to minimize the Bellman error
            loss_value = self.mse(q[valid], target_q[valid])
            self.optimizer_critic.zero_grad()
            loss_value.backward()
            self.optimizer_critic.step()

            ###### 1.3. Compute actor loss & update the actor network (policy improvement)
            ######### 1.3.1. Update actor network to maximize Q value given state.
            loss_policy = -self.critic(state, self.actor(state))[valid].mean()
            self.optimizer_actor.zero_grad()
            loss_policy.backward()
            self.optimizer_actor.step()

            ###### 1.4. Update target network (soft update)
            self.update_model(self.actor, self.actor_target, self.tau)
            self.update_model(self.critic, self.critic_target, self.tau)

            ###### 1.5. Save the loss value
            loss_value_all += loss_value.item()
            loss_policy_all += loss_policy.item()

        return {f'{self.team_name} Value Loss': loss_value_all / train_iter,
                f'{self.team_name} Policy Loss': loss_policy_all / train_iter}
