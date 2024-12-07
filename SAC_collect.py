import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical
from envs.env_manage import MultiEnvManager
from models.SAC_model import Actor_Net

import time




def collector(global_env, global_model, collect_env_num, database, device = 'cpu'):
    envs = MultiEnvManager(global_env,collect_env_num)
    obs_shape = global_env.observation_space.shape
    action_shape = global_env.action_space.n
    model = Actor_Net(obs_shape[0],action_shape)
    model.load_state_dict(global_model.state_dict())


    curr_states = []
    if  len(curr_states)==0:
        for i in range(collect_env_num):
            curr_state,_,_,_ = envs.pipe_parents[i].recv()
            curr_states.append(curr_state)

        curr_states = torch.from_numpy(np.concatenate(curr_states, 0))



    with torch.no_grad():
        model.eval()
        while True:
            model = model
            logits = model(curr_states)
            logits = F.softmax(logits, dim=1)
            old_m = Categorical(logits)  #这里是采用采样方法做的
            actions = old_m.sample()
            
        
            next_state = []
            for i in range(collect_env_num):
                # print('环境：',i)
                if database.size < 10000:
                    action =  global_env.action_space.sample()
                else:
                    action = actions[i].item()


                envs.pipe_parents[i].send(action)  
                

                _state, _reward, _done, info = envs.pipe_parents[i].recv()
                if _done:
                    _done = 1
                else:
                    _done = 0
                

            
                database.add(curr_states[i],action,_reward,_state[0],_done)
                next_state.append(_state)



            next_state = torch.from_numpy(np.concatenate(next_state, 0))
            curr_states = next_state

            
            model.load_state_dict(global_model.state_dict())

            time.sleep(0.01)
            # print('子程序中')
            # print(database.s.is_shared())
            # print(database.s.data.data_ptr())
            
            
        


