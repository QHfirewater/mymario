import torch
from torch import nn

class ReplayBuffer(nn.Module):
	def __init__(self, state_dim, dvc, max_size=int(3e4)):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0
		
		

		self.s = torch.zeros(size=(max_size,*state_dim),dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.dvc)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros(size=(max_size,*state_dim),dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = s.to(self.dvc)
		self.a[self.ptr] = a
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw



		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
	



if __name__ =='__main__':
	database = ReplayBuffer(state_dim = (4,84,84),max_size=int(1e5),dvc='cpu')
	database.share_memory()
	print(database.s)
	print(database.s_next)