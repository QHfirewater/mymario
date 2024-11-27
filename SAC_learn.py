import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical
from models.SAC_model import Actor_Net,Double_Q_Net
import wandb

       

class Learner:
    def __init__(self,actor_target,state_dim,action_dim,lr=1e-4, batch_size = 256,gamma = 0.2,alpha = 0.2,adaptive_alpha = True,device = 'cpu') -> None:
        self.actor_target = actor_target
        self.actor = Actor_Net(state_dim, action_dim).to(device)
        self.q_critic = Double_Q_Net(state_dim,action_dim).to(device)
        self.q_critic_target = Double_Q_Net(state_dim,action_dim).to(device)
        self.lr = lr
        self.gamma = gamma
        self.adaptive_alpha = adaptive_alpha
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.tau = 0.005
        self.H_mean = 0
        self.count = 0

        # self.record = Wandb_log(project_name='mario44',model=self.actor_target)

        self.actor.load_state_dict(self.actor_target.state_dict())

       
		

        if self.adaptive_alpha:
			# We use 0.6 because the recommended 0.98 will cause alpha explosion.
            self.target_entropy = 0.5 * (-np.log(1 / action_dim))  # H(discrete)>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
            self.alpha_optimer = torch.optim.Adam([self.log_alpha], lr=self.lr)

		
        #表示启用的模型
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr= self.lr)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr= self.lr)
		

    def __call__(self,database):

        print('进入训练程序')
        wandb.init(project='SAC_mario',reinit=True)
        wandb.watch(self.actor_target, log="all", log_freq=10, log_graph=True)
        wandb.watch(self.q_critic, log="all", log_freq=10, log_graph=True)
        # 生成监视器
        # record = Wandb_log(project_name='mario44',model=self.actor_target)
        record = {}
        while True:
            if database.size > 2 * self.batch_size:
                print('******************模型训练次数*************',self.count)
                self.count += 1

                s, a, r, s_next, dw = database.sample(self.batch_size)

                #------------------------------------------ Train Critic ----------------------------------------#
                '''Compute the target soft Q value'''
                with torch.no_grad():
                    next_probs = self.actor(s_next) #[b,a_dim]
                    next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]

                    next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
                    min_next_q_all = torch.min(next_q1_all, next_q2_all)
                    v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
                    target_Q = r + (~dw) * self.gamma * v_next

                '''Update soft Q net'''
                q1_all, q2_all = self.q_critic(s) #[b,a_dim]
                q1 = q1_all.gather(1, a) #[b,1]
                q2  = q2_all.gather(1, a)
                q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
                self.q_critic_optimizer.zero_grad()
                q_loss.backward()
                self.q_critic_optimizer.step()

                #------------------------------------------ Train Actor ----------------------------------------#
                for params in self.q_critic.parameters():
                    #Freeze Q net, so you don't waste time on computing its gradient while updating Actor.
                    params.requires_grad = 	False

                probs = self.actor(s) #[b,a_dim]
                log_probs = torch.log(probs + 1e-8) #[b,a_dim]
                with torch.no_grad():
                    q1_all, q2_all = self.q_critic(s)  #[b,a_dim]
                min_q_all = torch.min(q1_all, q2_all)

                a_loss = torch.sum(probs * (self.alpha*log_probs - min_q_all), dim=1, keepdim=True) #[b,1]

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                self.actor_optimizer.step()

                for params in self.q_critic.parameters():
                    params.requires_grad = 	True

                #------------------------------------------ Train Alpha ----------------------------------------#
                if self.adaptive_alpha:
                    with torch.no_grad():
                        self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
                    alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

                    self.alpha_optimer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimer.step()

                    self.alpha = self.log_alpha.exp().item()

                #------------------------------------------ Update Target Net ----------------------------------#
                for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                self.actor_target.load_state_dict(self.actor.state_dict())


                #记录训练过程
                record['q_loss'] = q_loss.item()
                record['a_loss']  = a_loss.mean().item()
                record['entropy'] = self.H_mean.item()
                record['train_iter'] = self.count

                wandb.log(data=record, step=record['train_iter'])

                print('q_loss:',q_loss,"a_loss",a_loss.mean(),'entropy',self.H_mean)


if __name__ =='__main__':
    actor_target = Actor_Net(4,7)
    state_dim = 4
    action_shape = 7
    lr = 1e-5
    batch_size = 512
    gamma = 0.99
    alpha = 0.2
    adaptive_alpha = True
    device = 'cuda'
    learner = Learner(actor_target,state_dim,action_shape,lr, batch_size,gamma,alpha,adaptive_alpha,device)
    








