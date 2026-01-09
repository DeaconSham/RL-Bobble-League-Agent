import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

def mlp(sizes, activation = nn.Tanh):
    """
    Docstring for mlp

    Helper function to build a multilayer perceptron neural network
    
    :param sizes: List of integers defining the number of neurons in each layer
    :param activation: Activation function used between layers
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

class actor_string_neural_network(nn.Module):
    """
        Docstring for actor_critic_neural_network

        Class contains two different neural networks:
        Actor (pi_net): Decides what to do (returns action means)
        Critic (v_net): Estimates how good the situation is (returns value) 
        
        :param obs_dim: Size of the observation vector (TODO)
        :param act_dim: Size of the action vector
        :param hidden: List defining hidden layers in neural network
        """
    def __init__(self, obs_dim, act_dim=6, hidden=[256, 256]):
        super().__init__()

        # policy network(which outputs action mean)
        self.pi_net = mlp([obs_dim] + hidden + [act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

        # value network
        self.v_net = mlp([obs_dim] + hidden + [1])
    
    def policy(self, obs):
        """
        Docstring for policy
        
        Forward propagation for actor neural network
        Returns a normal probability distribution 

        """
        mu = self.pi_net(obs)
        std = self.log_std.exp()
        return Normal(mu, std)
    
    def value(self, obs):
        """
        Docstring for value
        
        Forward propagation for critic neural network
        Returns the estimated value of the current state

        """
        return self.v_net(obs).squeeze(-1)