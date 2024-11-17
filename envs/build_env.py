import sys
sys.path.append(r'.')
from envs.env_wrapper import CustomReward,CustomSkipFrame
import warnings
warnings.filterwarnings('ignore')
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,RIGHT_ONLY



def get_instance_env(env_id,world,stage,video_path=None):
    
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

        return env



    
if __name__ == '__main__':
    
    env = get_instance_env('mario',1,1,None)
    print(env.spec)
    print(env.world)
    print(env.__dict__)
    print(type(env.observation_space.shape))
    print(env.observation_space.shape)
    # obs = env.reset()
    # print('obs',obs)
    # env.step(env.action_space.sample())