import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self, ncells):
        super(Qnet, self).__init__()

        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = ncells
        self.fc1 = nn.Linear(ncells, 2*ncells)
        self.fc2 = nn.Linear(2*ncells, 2*ncells)
        self.fc_a = nn.Linear(2*ncells, ncells)
        self.fc_v = nn.Linear(2*ncells, 1)
        self.m = nn.LeakyReLU(0.1)
        
        init_params(self)

    def forward(self, x):
        x = self.m(self.fc1(x))
        x = self.m(self.fc2(x))
        a = self.fc_a(x)
        a = a - a.max(-1, keepdim=True)[0].detach()
        v = self.fc_v(x)
        q = a + v
        return q

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, self.ncells))
        #print(obs.shape)
        out = self.forward(obs)
        coin = random.random() #0<coin<1
        if coin < epsilon:
            return np.random.randint(0, self.ncells)
        else:
            return out.argmax().item()

def init_params(net, val=np.sqrt(2)):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, val)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, val)
            if module.bias is not None:
                module.bias.data.zero_()
                
                
def merge_network_weights(q_target_state_dict, q_state_dict, tau):
    dict_dest = dict(q_target_state_dict)
    for name, param in q_state_dict:
        if name in dict_dest:
            dict_dest[name].data.copy_((1 - tau) * dict_dest[name].data
                                       + tau * param)


def train_network(q, q_target, memory, optimizer, train_number, batch_size, gamma, double):

    for i in range(int(train_number)):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        if double:
            max_a_prime = q(s_prime).argmax(1, keepdim=True)
            with torch.no_grad():
                max_q_prime = q_target(s_prime).gather(1, max_a_prime)
        else:
            with torch.no_grad():
                max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = done_mask * (r + gamma * max_q_prime) + (1 - done_mask) * 1 / (1-gamma) * r
        loss = F.smooth_l1_loss(q_a, target)  #huber loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss #for logging
