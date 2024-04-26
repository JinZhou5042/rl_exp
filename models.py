import torch 
import torch.nn as nn
import numpy as np

def fanin_init(size, fanin=None):
    """Initialize weights for actor and critic networks"""
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)

HID_LAYER1 = 40
HID_LAYER2 = 40
WFINAL = 0.003

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.norm0 = nn.BatchNorm1d(self.state_dim)
                                    
        self.fc1 = nn.Linear(self.state_dim, HID_LAYER1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())            
        self.bn1 = nn.BatchNorm1d(HID_LAYER1)
                                    
        self.fc2 = nn.Linear(HID_LAYER1, HID_LAYER2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
                                    
        self.bn2 = nn.BatchNorm1d(HID_LAYER2)
                                    
        self.fc3 = nn.Linear(HID_LAYER2, self.action_dim)
        self.fc3.weight.data.uniform_(-WFINAL, WFINAL)
        
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.Softmax = nn.Softmax(dim=1)
            
    def forward(self, input):
        input_norm = self.norm0(input)                            
        h1 = self.ReLU(self.fc1(input_norm))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(h1_norm))
        h2_norm = self.bn2(h2)
        # action = self.Tanh((self.fc3(h2_norm)))
        action = self.Softmax(self.fc3(h2_norm))
        return action
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(self.state_dim, HID_LAYER1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.bn1 = nn.BatchNorm1d(HID_LAYER1)
        self.fc2 = nn.Linear(HID_LAYER1 + self.action_dim, HID_LAYER2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        
        self.fc3 = nn.Linear(HID_LAYER2, 1)
        self.fc3.weight.data.uniform_(-WFINAL, WFINAL)
        
        self.ReLU = nn.ReLU()
        
    def forward(self, input, action):
        h1 = self.ReLU(self.fc1(input))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(torch.cat([h1_norm, action], dim=1)))
        Q_val = self.fc3(h2)
        return Q_val
