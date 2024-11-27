
import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor_Net(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
    

        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6,512)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
     
        x = self.linear(x.view(x.size(0), -1))
        x = self.actor_linear(x)
        return x
    

class Double_Q_Net(nn.Module):
    def __init__(self, num_inputs,num_actions):
        super().__init__()


        self.backbone1 = nn.Sequential(nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ReLU())

        self.linear1 = nn.Linear(32 * 6 * 6,512)
        self.critic_linear1 = nn.Linear(512, num_actions)
        self._initialize_weights()


        self.backbone2 = nn.Sequential(nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ReLU())
        self.linear2 = nn.Linear(32 * 6 * 6,512)
        self.critic_linear2 = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)


    def forward(self, x):
        y1 = self.backbone1(x)
        y1 = self.linear1(y1.view(y1.size(0), -1))
        y1 = self.critic_linear1(y1)

        y2 = self.backbone1(x)
        y2 = self.linear2(y2.view(y2.size(0), -1))
        y2 = self.critic_linear2(y2)


        return y1,y2


if __name__ == '__main__':
    x = torch.randn(size=(5,4,84,84))
    
    policy = Actor_Net(num_inputs=4,num_actions=7)
    action = policy(x)
    print(action)

    ctritic = Double_Q_Net(num_inputs=4,num_actions=7) 
    q1,q2 = ctritic(x)
    print(q1,q2)


    


