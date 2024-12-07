import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import Double_Q_Net, Policy_Net, ReplayBuffer
import time


class Learn:
	def __init__(self, state_dim,action_dim,lr,gamma,alpha,batch_size,hide_shape=[200,200],adaptive_alpha = True,device = 'cpu'):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hide_shape = hide_shape
		self.tau = 0.005
		self.H_mean = 0
		self.device = device
		self.lr = lr
		self.adaptive_alpha = adaptive_alpha
		self.gamma = gamma
		self.batch_size = batch_size
		self.alpha = alpha
	

		self.actor = Policy_Net(self.state_dim, self.action_dim, self.hide_shape).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

		self.q_critic = Double_Q_Net(self.state_dim, self.action_dim, self.hide_shape).to(self.device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		for p in self.q_critic_target.parameters(): 
			p.requires_grad = False

		if self.adaptive_alpha:
			# We use 0.6 because the recommended 0.98 will cause alpha explosion.
			self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  # H(discrete)>0
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)


	def __call__(self, database,actor_target):
		print('训练器启动成功')
		count = 0
	# def train(self,database):
		while True:
			'''update if its time'''
			# train 50 times every 50 steps rather than 1 training per step. Better!
			
			if database.size >= 1e4 :
				count = count +  1
				if count % 100 == 0:
					print(f'*************************{count}*****************')

				for j in range(50):
					s, a, r, s_next, dw = database.sample(self.batch_size)
					s = s.to(self.device)
					a = a.to(self.device)
					r = r.to(self.device)
					s_next = s_next.to(self.device)
					dw = dw.to(self.device)
				
					

					#------------------------------------------ Train Critic ----------------------------------------#
					'''Compute the target soft Q value'''
					with torch.no_grad():
						next_probs = self.actor(s_next) #[b,a_dim]
						next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]
						next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
						min_next_q_all = torch.min(next_q1_all, next_q2_all)
						v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
						target_Q = r + (1-dw) * self.gamma * v_next

					'''Update soft Q net'''
					q1_all, q2_all = self.q_critic(s) #[b,a_dim]
					q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a) #[b,1]
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

						self.alpha_optim.zero_grad()
						alpha_loss.backward()
						self.alpha_optim.step()

						self.alpha = self.log_alpha.exp().item()

					#------------------------------------------ Update Target Net ----------------------------------#
					for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
						target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
					
				actor_target.load_state_dict(self.actor.state_dict())
			else:
				time.sleep(2)

	def save(self, timestep, EnvName):
		torch.save(self.actor.state_dict(), f"./model/sacd_actor_{timestep}_{EnvName}.pth")
		torch.save(self.q_critic.state_dict(), f"./model/sacd_critic_{timestep}_{EnvName}.pth")


	def load(self, timestep, EnvName):
		self.actor.load_state_dict(torch.load(f"./model/sacd_actor_{timestep}_{EnvName}.pth"))
		self.q_critic.load_state_dict(torch.load(f"./model/sacd_critic_{timestep}_{EnvName}.pth"))
