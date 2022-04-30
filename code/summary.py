import numpy as np
from tensorboardX import SummaryWriter

class Summary():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.episode_step_count = []
        self.episode_mean_value = []
        self.episode_rewards = []
        self.writer = SummaryWriter(logdir=self.log_dir)

    def iter_steps(self):
        return len(self.episode_step_count)

    def add_info(self,
                 episode_step_count,
                 episode_value,
                 episode_rewards):
        self.episode_step_count.append(episode_step_count)
        self.episode_mean_value.append(np.sum(episode_value)/float(episode_step_count))
        self.episode_rewards.append(episode_rewards)
        if len(self.episode_step_count) >= 10:
            self.write_info()

    def write_info(self):

        mean_step = np.mean(self.episode_step_count[-5:])
        mean_value = np.mean(self.episode_mean_value[-5:])
        mean_reward = np.mean(self.episode_rewards[-5:])

        self.writer.add_scalar(tag='performance_reward', scalar_value=float(mean_reward), global_step=self.iter_steps())
        self.writer.add_scalar(tag='performance_length', scalar_value=float(mean_step), global_step=self.iter_steps())
        self.writer.add_scalar(tag='performance_value', scalar_value=float(mean_value), global_step=self.iter_steps())

        # self.writer.close()

    def close(self):
        self.writer.close()




