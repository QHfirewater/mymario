import gym
import numpy as np

#自定义玛丽奥wrap
from gym.spaces import Box
import cv2
import subprocess as sp
import traceback



class Monitor:
    def __init__(self, width, height, saved_path):
    
        # self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
        #                 "-pix_fmt", "rgb24", "-r", "40", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "30", "-i", "-","-b:v","32M", "-vcodec", "mpeg4", saved_path]
        try:


            self.pipe = sp.Popen(self.command, stdin=sp.PIPE,shell= True)

        except Exception:
            print(traceback.print_exc())

    def record(self, image_array):
        # print(type(image_array))
        # print(image_array.shape)
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))




def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        # frame = cv2.resize(frame, (84, 84))[None, :, :]
        return frame
    else:
        return np.zeros((1, 84, 84))

class CustomReward(gym.Wrapper):
    def __init__(self, env=None, world=None, stage=None,output_path = None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage

        if output_path:
            self.monitor = Monitor(256, 240, output_path)
        else:
            self.monitor = None
        
    def step(self, action):
        state, reward, done, truncate,info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)

        if truncate:
            done = True
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.    #相当于是加入优势在里面了
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        
        if self.world == 1 and self.stage == 3:
            if (info["x_pos"] <= 750 and info["x_pos"] >= 630  and info["y_pos"] <= 200):
                reward = -50
                done = True

        if self.world == 2 and self.stage == 1:
            if (info["x_pos"] <= 3030 and info["x_pos"] >= 3023  and info["y_pos"] <= 150):
                reward = -50
                done = True


        if self.world == 4 and self.stage == 2:
            if (info["x_pos"] <= 1234 and info["x_pos"] > 1200 and info["y_pos"] < 100):
                reward = -50
                done = True
        
        if self.world == 5 and self.stage == 3:
            if (info["x_pos"] <= 783 and info["x_pos"] > 640 and info["y_pos"] < 200):
                reward = -50
                done = True


        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.current_x - 500:
                reward -= 50
                done = True
                
        # if self.world == 4 and self.stage == 4:
        #     if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
        #             1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
        #         reward = -50
        #         done = True

        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
    
        return process_frame(self.env.reset())


class CustomSkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


