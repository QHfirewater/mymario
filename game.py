from ppo_application import PPO

def mario():
    env_id = "mario"
    #1-3
    #2-1
    #3-3？
    #3-4？
    #4-3
    #4-4
    algo = PPO(env_id=env_id,world=1,stage=3)
    algo.train()
    # video_path = r'D:\reinforce\mario\my_mario\movie\1_3.mp4'
    # model_stat = r'D:\reinforce\mario\my_mario\ckpt\80_iteration.pth.tar'
    algo.deploy(model_stat,video_path)


if __name__ == '__main__': 
    mario()  