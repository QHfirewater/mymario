from utils import evaluate_policy, str2bool
from datetime import datetime
from SACD import SACD_agent
import  gym
import os, shutil
import argparse
import torch
from models.SAC_model import Actor_Net,Double_Q_Net
from  torch import multiprocessing as _mp
from SAC_evaluate import model_evalutor
from envs.build_env import get_instance_env
from buffer import  ReplayBuffer
from torch.distributions import Categorical
from learn import Learn
from collect import collector
import time

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=1, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default  = False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=4e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=10, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hid_shape', type=list, default=[200,200], help='Hidden net shape')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
opt = parser.parse_args()
print(opt)

def main():

	env = get_instance_env(env_id='LunarLander-v2',world=1,stage=1)
	eval_env = get_instance_env(env_id='LunarLander-v2',world=1,stage=1)


	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.n
	

	# Seed Everything
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	print('Algorithm: SACD', 'state_dim:',opt.state_dim, 'action_dim:',opt.action_dim,'Random Seed:',opt.seed,  '\n')

	#初始化数据存储器
	database = ReplayBuffer(env.observation_space.shape[0])
	database.share_memory()

	#Build model
	if not os.path.exists('model'): 
		os.mkdir('model')

	# if opt.Loadmodel: agent.load(opt.ModelIdex, BriefEnvName[opt.EnvIdex])


	actor_target = Actor_Net(opt.state_dim, opt.action_dim)
	actor_target.share_memory()

	# #数据采集
	mp = _mp.get_context("spawn") 

	process1 = mp.Process(target=collector, args=(actor_target,database))
	process1.start()


	#模型训练
	
	learner = Learn(opt.state_dim, opt.action_dim, opt.lr, opt.gamma,opt.alpha, opt.batch_size)
	process2 = mp.Process(target=learner, args=(database,actor_target))
	process2.start()


	# #模型评估
	process3 = mp.Process(target=model_evalutor, args=(actor_target,'LunarLander-v2',1,1))
	process3.start()

	
	total_steps = 0
	while True:

		s = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
		done = False
		'''Interact & trian'''
		while not done:
			#e-greedy exploration
			# if total_steps < opt.random_steps:
			# 	a = env.action_space.sample()
			# else:
			# 	state = torch.from_numpy(s) #from (s_dim,) to (1, s_dim)
			# 	probs = actor_target(state)
			# 	a = Categorical(probs).sample().item()

			# s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
			# done = (dw or tr)
			# database.add(s, a, r, s_next, dw)
			# s = s_next

			

			'''record & log'''
			if total_steps % 10 == 0:
				score = evaluate_policy(eval_env, actor_target, turns=3)
				print('seed:', opt.seed,'steps: {}'.format(int(total_steps)/10), 'score:', int(score))
			time.sleep(1)
			total_steps += 1

				
	# env.close()
	# eval_env.close()


if __name__ == '__main__':
	main()

