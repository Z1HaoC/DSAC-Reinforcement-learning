from torch import nn
import numpy as np
import datetime
import torch
import cv2
import gym
import os
from time import sleep
import pathlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
class HumanPref(nn.Module):
    def __init__(self, input_dim):
        super(HumanPref, self).__init__()
        self.input_dim = input_dim

        self.dense1 = nn.Linear(self.input_dim, 500)
        self.dense2 = nn.Linear(500, 300)
        self.dense3 = nn.Linear(300, 1)
        self.tanh = nn.Tanh()
        # self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, s1, s2=None):

        s1 = torch.nn.functional.relu(self.dense1(s1))
        s1 = torch.nn.functional.relu(self.dense2(s1))
        s1 = torch.sigmoid(self.dense3(s1))
        # r1 = self.batch_norm(s1)
        r1 = s1
        if s2 is None:
            return r1
        else:
            s2 = torch.nn.functional.relu(self.dense1(s2))
            s2 = torch.nn.functional.relu(self.dense2(s2))
            s2 = torch.sigmoid(self.dense3(s2))
            # r2 = self.batch_norm(s2)
            r2 = s2
            output = torch.cat([r1, r2], dim=1)
            print(output)

            return output


class HumanPreference(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.directory = 'models'
        self.learning_rate = 0.01
        self.create_model()
        self.load(self.directory)

    def create_model(self):
        self.model = HumanPref(self.input_dim).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self,
              preferences_buffer,
              batch_size):

        # sample a batch from the replay buffer
        (
            batch_state1,
            batch_state2,
            batch_preferences
        ) = preferences_buffer.sample_batch(batch_size)
        state1 = torch.Tensor(batch_state1).to(device)
        state2 = torch.Tensor(batch_state2).to(device)
        preferences = torch.Tensor(batch_preferences).to(device)

        y = preferences
        y_hat = self.model(state1, state2)

        loss = self.criterion(y_hat, y)
        # print(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.save(self.directory) 
        
    def save(self, directory):
        torch.save(self.model.state_dict(), "%s/preference_model.pth" % (directory))
        

    def load(self, directory):
        self.model.load_state_dict(
            torch.load("%s/preference_model.pth" % (directory))
        )


    def predict(self, obs):
        self.model.eval()
        # 将输入数据转换为numpy数组再转为张量，并指定设备
        obs_np = np.array(obs, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = self.model(obs_tensor)
        # 将结果移回CPU并转换为numpy数组
        return pred.cpu().detach().numpy()



