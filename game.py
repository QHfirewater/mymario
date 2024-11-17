from ppo_application import PPO

def mario():
    env_id = "mario"
    #1-3
    #2-1
    #3-3？
    #3-4？
    #4-3
    #4-4
    #5-3
    #6-3
    #7-2 关卡重复
    #7-3 关卡重复
    #7-4 关卡重复
    algo = PPO(env_id=env_id,world=8,stage=2)
    algo.train()
    # video_path = r'D:\reinforce\mario\my_mario\movie\7_1.mp4'
    # model_stat = r'D:\reinforce\mario\my_mario\movie\get_flag_7_1.pth.tar'
    # algo.deploy(model_stat,video_path)


if __name__ == '__main__': 
    mario()  