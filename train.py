# train.py
import pandas as pd, gymnasium as gym
from stable_baselines3 import PPO
from envs.greenhouse_env import GreenhouseEnv
from data_loader import fetch_thingspeak

df = fetch_thingspeak(channel_id=2968337, api_key="READ-KEY")   # 👉 READ key
env = gym.wrappers.TimeLimit(GreenhouseEnv(df), max_episode_steps=288) # one day

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb")
model.learn(total_timesteps=100_000)
model.save("models/latest_actor")

# Convert → ONNX → TFLite-Micro int8
import torch, onnx
obs = env.reset()[0]
torch.onnx.export(model.policy, torch.tensor(obs).unsqueeze(0),
                  "models/latest_actor.onnx", opset_version=17)

# …follow TensorFlow Lite converter steps, then xxd -i > model.h
