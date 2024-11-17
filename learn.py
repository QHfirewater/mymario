import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical

class Multistep_learn:
    def __init__(self,global_model,lr=1e-4,num_per_epochs = 10,device = 'cpu',epsilon = 0.2, beta = 0.01, batch_size = 256) -> None:
        
        self.model = global_model
        self.lr =lr
        self.num_per_epochs = num_per_epochs
        self.device = device
        self.epsilon = epsilon
        self.beta = beta
        self.batch_size = batch_size
        #表示启用的模型
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.lr)


    def __call__(self, timesteps):
        states,actions,old_log_policies,advantages,R = timesteps
        log_info = {}
        actor_loss_list = []
        critic_loss_list = []
        entropy_loss_list = []
        value_mean_list = []
        advantages_list = []
        
        data_length = states.shape[0]
        epoch = data_length//self.batch_size
        self.device = 'cpu'


        self.model.train()
        for i in range(self.num_per_epochs): #表示每组数据运行的次数
            indice = torch.randperm(data_length)
            for j in range(epoch):
                batch_indices = indice[int(j * self.batch_size): int((j + 1) * self.batch_size)]
                if self.device:
                    self.model = self.model.to(self.device)
                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    old_log_policies = old_log_policies.to(self.device)
                    advantages = advantages.to(self.device)
                    R = R.to(self.device)

                
        
                logits,value = self.model(states[batch_indices])
                logits = F.softmax(logits, dim=1)
                new_m = Categorical(logits)
                
                new_log_policy = new_m.log_prob(actions[batch_indices])
            

                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages[batch_indices]))
    
                
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - self.beta * entropy_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()



                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                entropy_loss_list.append(entropy_loss.item())
                value_mean_list.append(np.mean(value.cpu().squeeze().detach().numpy()))
                advantages_list.append(np.mean(advantages[batch_indices].cpu().squeeze().detach().numpy()))
            

        log_info['actor_loss'] = np.mean(actor_loss_list)
        log_info['critic_loss'] = np.mean(critic_loss_list)
        log_info['entropy_loss'] = np.mean(entropy_loss_list)
        log_info['value_mean'] = np.mean(value_mean_list)
        log_info['advantages'] = np.mean(advantages_list)

        
        return  log_info

        # return self.model.state_dict()