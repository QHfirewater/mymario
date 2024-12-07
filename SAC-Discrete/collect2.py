import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical
from envs.env_manage import MultiEnvManager
from models.model import Model


def collector(collect_child_pipe,global_env,global_model,collect_env_num,collect_steps =512,gamma =0.9,tau=1, device = 'cpu'):
    envs = MultiEnvManager(global_env,collect_env_num)
    obs_shape = global_env.observation_space.shape[0]
    action_shape = global_env.action_space.n
    model = Model(obs_shape,action_shape)

    curr_states = []
    if  len(curr_states)==0:
        for i in range(collect_env_num):
            curr_state,_,_,_ = envs.pipe_parents[i].recv()
            curr_states.append(curr_state)

        curr_states = torch.from_numpy(np.concatenate(curr_states, 0)).to(device)
        curr_states = curr_states.to(device)


    old_log_policies = []
    actions = []
    values = []
    states = []
    rewards = []
    dones = []
    # print('opt.collect_steps',opt.collect_steps)
    while True:
        model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            model.eval()
            old_log_policies = []
            actions = []
            values = []
            states = []
            rewards = []
            dones = []
            for _ in range(collect_steps):  #这里其实是定义了每个轮回走多少步
                states.append(curr_states)
                next_state = []
                reward = []
                done = []

                
                model = model.to(device)

                logits,value = model(curr_states)
                logits = F.softmax(logits, dim=1)
                old_m = Categorical(logits)  #这里是采用采样方法做的
                action = old_m.sample()
                old_log_policy = old_m.log_prob(action)

                values.append(value.squeeze())
                actions.append(action)
                old_log_policies.append(old_log_policy)


                for i in range(collect_env_num):
                    envs.pipe_parents[i].send(action[i].item())
                    _state, _reward, _done, info = envs.pipe_parents[i].recv()
                    next_state.append(_state)
                    reward.append(_reward)
                    done.append(_done)

                next_state = torch.from_numpy(np.concatenate(next_state, 0)).to(device)
                reward = torch.FloatTensor(reward).to(device)
                done = torch.FloatTensor(done).to(device)
                rewards.append(reward)
                dones.append(done)
                curr_states = next_state

            curr_states = curr_states.to(device)
            _, next_value = model(curr_states)
            next_value = next_value.squeeze()
            old_log_policies = torch.cat(old_log_policies).detach()
            actions = torch.cat(actions).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)

            gae = 0
            R = []
            for value, reward, done in list(zip(values, rewards, dones))[::-1]:
                gae = gae * gamma * tau
                gae = gae + reward + gamma * next_value.detach() * (1 - done) - value.detach()
                next_value = value
                R.append(gae + value)
            R = R[::-1]
            R = torch.cat(R).detach()
            advantages = R - values
            timesteps = (states,actions,old_log_policies,advantages,R)

    
        collect_child_pipe.send(timesteps)
        # model_state = collect_child_pipe.recv()
        

