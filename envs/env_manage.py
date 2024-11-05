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
                obs = env.reset() 

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
    

    envs = MultiEnvManager(env_id='mario',env_num=3,world=1,stage=1)
    obs_list = []
    action_list = []
    old_log_policy_list = []
    reward_list = []
    value_list = []
    done_list = []
    info_list = []
    next_obs_list = []

    # current_obs, action, old_log_policy ,reward, value, done, info , obs

    for x in range(50):
        print('第{}波'.format(x))
        for i in range(3):
            obs,reward,done,info = envs.pipe_parents[i].recv()
            action = np.random.randint(0,7)
            envs.pipe_parents[i].send(action)

            obs_list.append(obs)
            action_list.append(action)
        
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

    obs_list = np.concatenate(obs_list,axis=0)
    print(obs_list.shape)


