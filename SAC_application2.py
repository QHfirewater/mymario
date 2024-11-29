import torch
from envs.build_env import get_instance_env
from models.SAC_model import Actor_Net,Double_Q_Net
from SAC_collect import collector
from SAC_learn import Learner
from logs.mylog import logger,Wandb_log
from  torch import multiprocessing as _mp
from torch.nn import functional as F
from SAC_evaluate import model_evalutor
import warnings
warnings.filterwarnings('ignore')
import time
from buffer import ReplayBuffer

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical
from envs.env_manage import MultiEnvManager
from models.SAC_model import Actor_Net

import time



torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

class SAC:

    def __init__(self,env_id= None,world  =1,stage = 1,model_state_dict= None) -> None:


        
        #处理程序的配置问题
        assert env_id is not None , "Please specify env_id"
        self.env_id = env_id

        

        #生成训练的环境
        self.world = world
        self.stage = stage
        self.env = get_instance_env(env_id,self.world,self.stage,)
        logger.info('环境启动成功!')

        
        #读取环境中动作空间形状
        self.action_shape = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape
        self.state_dim = self.obs_shape[0]


        #生成神经网络模型
        self.actor_target = Actor_Net(self.state_dim, self.action_shape)
        self.actor_target.share_memory()
        

        logger.info('神经网络启动成功')


        if model_state_dict is not None:
            self.actor_target.load_state_dict(torch.load(model_state_dict))
        print(self.actor_target)


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


        


    
    def train(self):

        

        #数据库
        device = 'cuda:0'
        self.database = ReplayBuffer(state_dim =self.obs_shape)
        self.database.share_memory()

        #数据采集
        envs = MultiEnvManager(self.env,collect_env_num)
        obs_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.n
        model = self.actor_target
    

        curr_states = []
        if  len(curr_states)==0:
            for i in range(collect_env_num):
                curr_state,_,_,_ = envs.pipe_parents[i].recv()
                curr_states.append(curr_state)

            curr_states = torch.from_numpy(np.concatenate(curr_states, 0))



        with torch.no_grad():
            model.eval()
            model = model
            logits = model(curr_states)
            logits = F.softmax(logits, dim=1)
            old_m = Categorical(logits)  #这里是采用采样方法做的
            action = old_m.sample()
        
            next_state = []
            for i in range(collect_env_num):
                # print('环境：',i)
                envs.pipe_parents[i].send(action[i].item())
                _state, _reward, _done, info = envs.pipe_parents[i].recv()
                if _done:
                    _done = 1
                else:
                    _done = 0
                

            
                self.database.add(curr_states[i],action[i],_reward,_state[0],_done)
                next_state.append(_state)



            next_state = torch.from_numpy(np.concatenate(next_state, 0))
            curr_states = next_state
        

        if self.database.size > 1.2 * self.batch_size:
                print('******************模型训练次数*************',self.count)
                self.count += 1

                s, a, r, s_next, dw = self.database.sample(self.batch_size)

                s = s.to(self.device)
                a = a.to(self.device)
                r = r.to(self.device)
                s_next = s_next.to(self.device)
                dw = dw .to(self.device)
                # print(s)


                #------------------------------------------ Train Critic ----------------------------------------#
                '''Compute the target soft Q value'''
                with torch.no_grad():
                    next_probs = self.actor(s_next) #[b,a_dim]
                    next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]

                    next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
                    min_next_q_all = torch.min(next_q1_all, next_q2_all)
                    v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
                    target_Q = r + (1 - dw) * self.gamma * v_next

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

            


      











        mp = _mp.get_context("spawn")
        collect_env_num = 10
        process1 = mp.Process(target=collector, args=(
                                                    self.env, 
                                                    self.actor_target,
                                                    collect_env_num,
                                                    self.database,
                                                    device))
        process1.start()
        logger.info('初始化数据采集')

     

        #模型训练
        lr=1e-5
        batch_size = 512
        gamma = 0.99
        alpha = 0.2
        adaptive_alpha = True
        device = 'cuda'

        learner = Learner(self.actor_target,self.state_dim,self.action_shape,lr, batch_size,gamma,alpha,adaptive_alpha,device)
        process2 = mp.Process(target=learner, args=(self.database,))
        process2.start()


        logger.info('初始化模型训练')


        #模型评估
        process3 = mp.Process(target=model_evalutor, args=(self.actor_target,self.env_id,self.world,self.stage))
        process3.start()
        logger.info('初始化评估器')




        count  = 0
        while True:
            # count += 1
            # print(self.database.s[count])
            time.sleep(1)
            print('主程序中：')
            print(self.database.s.is_shared())
            print(self.database.s.data.data_ptr())
            # log_info['train_iter'] =curr_episode
            

            # if curr_episode % save_model_freq == 0:
            #     ckpt_path = r'D:\reinforce\mario\my_mario\ckpt\{}_iteration.pth.tar'.format(curr_episode)
            #     torch.save(self.model.state_dict(),ckpt_path)


            

if __name__ == '__main__':
    agent = SAC(env_id = "mario")
    agent.train()
    