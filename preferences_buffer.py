"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque
import pickle
import numpy as np


class PreferencesBuffer(object):
    def __init__(self, buffer_size, filename='buffer_data.pkl', random_seed=0):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.filename = filename
        random.seed(random_seed)

    def add(self, s1, s2, pre):
        data = (s1, s2, pre)
        if self.count < self.buffer_size:
            self.buffer.append(data)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(data)
        # Save the buffer to a file
        self.save_to_file()
        
    def save_to_file(self):
        with open(self.filename, 'wb') as file:
            pickle.dump(list(self.buffer), file)
            
    def load_from_file(self):
        try:
            with open(self.filename, 'rb') as file:
                self.buffer = deque(pickle.load(file))
                self.count = len(self.buffer)
                print('load '+ str(self.count)+ ' datas')
        except FileNotFoundError:
            # Handle the case when the file doesn't exist yet
            pass

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s1_batch = np.array([_[0] for _ in batch])
        s2_batch = np.array([_[1] for _ in batch])
        pre_batch = np.array([_[2] for _ in batch])

        return s1_batch, s2_batch, pre_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
