import torch
from torch import nn


# class ReplayBuffer(nn.Module):
# 	def __init__(self, state_dim, max_size=int(1e6)):
# 		super().__init__()
# 		self.max_size = max_size
# 		self.ptr = torch.tensor(0)
# 		self.size = torch.tensor(0)
		
		

# 		self.s = torch.zeros(size=(max_size,*state_dim),dtype=torch.float)
# 		self.a = torch.zeros((max_size, 1),dtype=torch.long)
# 		self.r = torch.zeros((max_size, 1),dtype=torch.float)
# 		self.s_next = torch.zeros(size=(max_size,*state_dim),dtype=torch.float)
# 		self.dw = torch.zeros((max_size, 1),dtype=torch.int)
		

# 	def add(self, s, a, r, s_next, dw):

# 		# #为了保证数据是原位更改
# 		# self.s[self.ptr] -= self.s[self.ptr]
# 		# self.a[self.ptr] -= self.a[self.ptr]
# 		# self.r[self.ptr] -= self.r[self.ptr]
# 		# self.s_next[self.ptr] -= self.s_next[self.ptr]
# 		# self.dw[self.ptr] -= self.dw[self.ptr]
	


# 		# self.s[self.ptr] += s
# 		# self.a[self.ptr] += a
# 		# self.r[self.ptr] += r
# 		# self.s_next[self.ptr] += torch.from_numpy(s_next)
# 		# self.dw[self.ptr] += dw

		
# 		# ptr = (self.ptr + 1) % self.max_size
# 		# self.ptr -= self.ptr
# 		# self.ptr += ptr

# 		self.s[self.ptr] = torch.from_numpy(s)
# 		self.a[self.ptr] = a
# 		self.r[self.ptr] = r
# 		self.s_next[self.ptr] = torch.from_numpy(s_next)
# 		self.dw[self.ptr] = dw

# 		self.ptr = (self.ptr + 1) % self.max_size
# 		self.size = min(self.size + 1, self.max_size)



# 		if self.size < self.max_size:
# 			self.size += 1

# 	def sample(self, batch_size):
# 		ind = torch.randint(0, self.size, size=(batch_size,))
# 		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
	

class ReplayBuffer(nn.Module):
	def __init__(self, state_dim, max_size=int(1e6)):
		super().__init__()
		self.max_size = torch.tensor(max_size)
		self.ptr = torch.tensor(0)
		self.size = torch.tensor(0)

		self.s = torch.zeros((max_size, state_dim),dtype=torch.float,requires_grad=False)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,requires_grad=False)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,requires_grad=False)
		self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,requires_grad=False)
		self.dw = torch.zeros((max_size, 1),dtype=torch.float,requires_grad=False)

	def add(self, s, a, r, s_next, dw):


		#为了保证数据是原位更改
		self.s[self.ptr] -= self.s[self.ptr]
		self.a[self.ptr] -= self.a[self.ptr]
		self.r[self.ptr] -= self.r[self.ptr]
		self.s_next[self.ptr] -= self.s_next[self.ptr]
		self.dw[self.ptr] -= self.dw[self.ptr]
	

		# print('cyffs',s[self.ptr].shape,s.shape)
		# print('cyffs',s[self.ptr],s[0])

		self.s[self.ptr] += torch.from_numpy(s)[0]
		self.a[self.ptr] += a
		self.r[self.ptr] += r
		self.s_next[self.ptr] += torch.from_numpy(s_next)[0]
		self.dw[self.ptr] += dw

		
		ptr = (self.ptr + 1) % self.max_size
		self.ptr -= self.ptr
		self.ptr += ptr


		if self.size < self.max_size:
			self.size += 1



		# self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		# self.a[self.ptr] = a
		# self.r[self.ptr] = r
		# self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		# self.dw[self.ptr] = dw

		# self.ptr = (self.ptr + 1) % self.max_size
		# self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


if __name__ =='__main__':
	database = ReplayBuffer(state_dim = (8,),max_size=int(3e4))
	database.share_memory()
	print(database.s.shape)
	print(database.s_next.shape)
	print(database.s.data.data_ptr())
	s = torch.rand(size=(8,))
	a = 1
	r = 3
	s_next = s.numpy()
	dw = 1
	database.add(s,a,r,s_next,dw)
	print(database.s.shape)
	print(database.s.is_shared())
	print(database.s.data.data_ptr())
