import torch
from torch import nn


class ReplayBuffer(nn.Module):
	def __init__(self, state_dim, max_size=int(1e4)):
		super().__init__()
		self.max_size = max_size
		self.ptr = torch.tensor(0)
		self.size = torch.tensor(0)
		
		

		self.s = torch.zeros(size=(max_size,*state_dim),dtype=torch.float)
		self.a = torch.zeros((max_size, 1),dtype=torch.long)
		self.r = torch.zeros((max_size, 1),dtype=torch.float)
		self.s_next = torch.zeros(size=(max_size,*state_dim),dtype=torch.float)
		self.dw = torch.zeros((max_size, 1),dtype=torch.int)
		

	def add(self, s, a, r, s_next, dw):

		#为了保证数据是原位更改
		self.s[self.ptr] -= self.s[self.ptr]
		self.a[self.ptr] -= self.a[self.ptr]
		self.r[self.ptr] -= self.r[self.ptr]
		self.s_next[self.ptr] -= self.s_next[self.ptr]
		self.dw[self.ptr] -= self.dw[self.ptr]
	


		self.s[self.ptr] += s
		self.a[self.ptr] += a
		self.r[self.ptr] += r
		self.s_next[self.ptr] += torch.from_numpy(s_next)
		self.dw[self.ptr] += dw

		
		ptr = (self.ptr + 1) % self.max_size
		self.ptr -= self.ptr
		self.ptr += ptr

		if self.size < self.max_size:
			self.size += 1

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))
		return self.s[ind].clone(), self.a[ind].clone(), self.r[ind].clone(), self.s_next[ind].clone(), self.dw[ind].clone()


if __name__ =='__main__':
	database = ReplayBuffer(state_dim = (4,84,84),max_size=int(3e4),dvc='cuda')
	database.share_memory()
	print(database.s.shape)
	print(database.s_next.shape)
	print(database.s.data.data_ptr())
	s = torch.rand(size=(4,84,84))
	a = 1
	r = 3
	s_next = s.numpy()
	dw = 1
	database.add(s,a,r,s_next,dw)
	print(database.s.shape)
	print(database.s.is_shared())
	print(database.s.data.data_ptr())
