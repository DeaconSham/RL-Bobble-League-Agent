import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

def neural_network(sizes, activation = nn.Tanh):
    """
    Docstring for neural_network

    Helper function to build a neural network
    
    :param sizes: List of integers defining the number of neurons in each layer
    :param activation: Activation function used between layers
    :return: A neural network
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

class actor_critic_neural_network(nn.Module):
    """
        Docstring for actor_critic_neural_network

        Class contains two different neural networks:
        Actor (pi_net): Decides what to do (returns action means)
        Critic (v_net): Estimates how good the situation is (returns value) 
        
        :param obs_dim: Size of the observation vector
        :param act_dim: Size of the action vector
        :param hidden: List defining hidden layers in neural network
        """
    def __init__(self, obs_dim=14, act_dim=6, hidden=[256, 256]):
        super().__init__()

        self.pi_net = neural_network([obs_dim] + hidden + [act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        self.v_net = neural_network([obs_dim] + hidden + [1])
    
    def forward(self, obs):
        """
        Docstring for forward

        Forward propagation through both neural networks
        
        :param obs: Observation tensor
        :return dist: Normal distribution over actions
        :return value: Value estimate
        """

        mu = self.pi_net(obs)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        dist = Independent(dist, 1)
        value = self.v_net(obs).squeeze(-1)
        return dist, value
    
    def step(self, obs):
        """
        Docstring for step
        
        Training method: returns action, value, log_prob as numpy array.
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)

            dist, value = self.forward(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu().numpy(), value.cpu().item(), log_prob.cpu().item()
    
    def act(self, obs, deterministic=False):
        """
        Docstring for act
        
        Inference method: returns action as numpy array.
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)

            mu = self.pi_net(obs_tensor)
            if deterministic:
                action = mu
            else:
                std = self.log_std.exp()
                dist = Independent(Normal(mu, std), 1)
                action = dist.sample()

            return action.cpu().numpy()