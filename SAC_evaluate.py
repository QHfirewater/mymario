
import torch
import numpy as np
import pandas as pd
from envs.build_env import get_instance_env
from models.SAC_model import Actor_Net
from torch.nn import functional as  F
import time


def model_evalutor(global_model,env_id,world,stage):
    


    env = get_instance_env(env_id,world,stage)
    action_shape = env.action_space.n
    obs_shape = env.observation_space.shape[0]
    model = Actor_Net(obs_shape,action_shape)


    obs = env.reset()
    eval_reward = 0
    eval_step = 0
    num = 0
    with torch.no_grad():
        model.eval()
        while True:
            logits = model(torch.tensor(obs))
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()

            obs,reward,done,info = env.step(action)
            env.render()
            time.sleep(0.1)

            eval_reward =+ reward
                
            eval_step += 1
            if done:
                obs = env.reset()
                eval_position = info['x_pos']
                print('eval_step:',eval_step,'eval_position:',eval_position)
                res = (eval_reward,eval_step,eval_position)
                # child.send(res)
                # train_state_dict = child.recv()
                model.load_state_dict(global_model.state_dict())
                eval_reward = 0 
                eval_step = 0


                    
            # Uncomment following lines if you want to save model whenever level is completed
            if info["flag_get"]:
                print("have the flag,  Finished!!!")
                num +=1
                torch.save(model.state_dict(),r"D:\reinforce\mario\my_mario\ckpt\{}_get_flag_{}_{}.pth.tar".format(num,world,stage))
            
            
            

        



