import sys
sys.path.append(r'.')
from envs.env_wrapper import CustomReward,CustomSkipFrame,state_transform
import warnings
warnings.filterwarnings('ignore')
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,RIGHT_ONLY



def get_instance_env(env_id,world,stage,video_path=None,train = False):
    
    if env_id == 'mario':
        
        env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world,stage))
        env = JoypadSpace(env,SIMPLE_MOVEMENT)
        # if video_path:
        #     env = gym.wrappers.RecordVideo(env, video_folder=video_path,step_trigger=lambda step: True,name_prefix='rl-video')
        env = CustomReward(env,world,stage,output_path = video_path)
        env = CustomSkipFrame(env)
        env.env_id = env_id
        env.world = world
        env.stage = stage

    
    
    if env_id =='LunarLander-v2':
        if video_path:
            env = gym.make('LunarLander-v2',render_mode="human")
        else:
            env = gym.make('LunarLander-v2')

        env = state_transform(env)
        env.env_id = 'LunarLander-v2'
        env.world = 1
        env.stage = 1

    return env






    
if __name__ == '__main__':
    
    # env = get_instance_env('mario',1,1,None)
    env = get_instance_env('LunarLander-v2',1,1,None)
  
    # print(env.__dict__)
    print(env.observation_space.shape)
    print(env.action_space.n)
    obs = env.reset()
 
    print(obs)
    print(obs.shape)
    print(env.step(env.action_space.sample()))