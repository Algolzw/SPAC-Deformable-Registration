import numpy as np
import torch

from collections import deque

class ReplayMemory:
    def __init__(self, max_size=300, look_forward_steps=5, gamma=0.9):
        self.buffer = deque(maxlen=max_size)
        self.look_forward_steps = look_forward_steps
        self.gamma = gamma

    def __len__(self):
        return len(self.buffer)

    def store(self, memory):
        self.buffer.append(memory)

    def sample(self, batch_size):
        sample_indices = np.random.choice(range(len(self.buffer)), batch_size, replace=False)

        ss, aa, rr, ss_, dd = [], [], [], [], []
        hh, hh_ = [], []
        max_index = len(self.buffer)
        for i in sample_indices:
            s = self.buffer[i].s
            a = self.buffer[i].a
            r = self.buffer[i].r
            s_ = self.buffer[i].s_
            d = self.buffer[i].d

            for step in range(self.look_forward_steps):
                if max_index > i + step:
                    r += self.gamma * self.buffer[i + step].r
                else:
                    break

            ss.append(s)
            aa.append(a)
            rr.append(r)
            ss_.append(s_)
            dd.append(d)

        hiddens = None
        hiddens_ = None

        return (np.concatenate(ss, axis=0),
                np.concatenate(aa, axis=0),
                np.array(rr),
                np.concatenate(ss_, axis=0),
                np.array(dd))




