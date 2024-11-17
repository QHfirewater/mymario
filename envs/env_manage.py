import sys
sys.path.append(r'.')
import torch.multiprocessing as mp
import numpy as np


from envs.build_env import get_instance_env




mp = mp.get_context("spawn")


class MultiEnvManager:
    """s
    Overview:
        Create an AsyncSubprocessEnvManager to manage multiple environments.
        Each Environment is run by a respective subprocess.
    Interfaces:
        seed, launch, ready_obs, step, reset, active_env

    """

    @staticmethod
    def worker_fn_robust(env_id,child_pipe,world,stage) -> None:
        """
        Overview:
            A more robust version of subprocess's target function to run. Used by default.
        """

        env = get_instance_env(env_id,world,stage)
        obs = env.reset()
        reward = None
        done = None
        info = None
        while True:
            timestep = (obs, reward,  done, info)
            child_pipe.send(timestep)
            action = child_pipe.recv()
            obs,reward,done,info = env.step(action)
            if done:
                obs = env.reset()    #如果结束，就重启环境

    def __init__(self,global_env,env_num) -> None:
        
        self.env_num = env_num
        self.env_id = global_env.env_id
        self.world = global_env.world
        self.stage = global_env.stage
        self.pipe_parents, self.pipe_children = {}, {}
        self._subprocesses = {}


        for index in range(self.env_num):
            self.pipe_parents[index], self.pipe_children[index] = mp.Pipe()
            self._subprocesses[index] = mp.Process(target=self.worker_fn_robust,
                                                    args=(self.env_id,self.pipe_children[index],self.world,self.stage),
                                                    daemon=True)  
            self._subprocesses[index].start()
            print('环境{}启动成功！'.format(index))



if __name__  == '__main__':


    env = get_instance_env('mario',1,1,None)

    obs = env.reset()
    

    envs = MultiEnvManager(global_env=env,env_num=3)
    print(envs.env_id)
    print(envs.stage)
    for i in range(3):
        _state, _reward, _done, info = envs.pipe_parents[i].recv()
        print('_state',_state.shape)
        print('_reward',_reward)
        print('_done',_done)
    
    

