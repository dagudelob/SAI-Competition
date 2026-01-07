import torch.nn.functional as F
import numpy as np

from sai_rl import SAIClient

from ddpg import DDPG_FF
from training import training_loop

## Initialize the SAI client
sai = SAIClient(
    comp_id="booster-soccer-showdown",
    renderer="mjviewer",
)

## Make the environment
env = sai.make_env()

## Define the Preprocessor class
class Preprocessor():
    def __init__(self):
        self.num_shared_features = 33
        self.features_per_task = [45, 54, 39]
        self.extra_features = [x - self.num_shared_features for x in self.features_per_task]
        self.total_extra_features = sum(self.extra_features)
        
        i = 0
        self.feature_ranges = []
        for x in self.extra_features:
            self.feature_ranges.append(np.arange(i, i + x))
            i += x

    def get_task_onehot(self, info):
        if "task_index" in info: 
            return info["task_index"]
        else: 
            return np.array([])

    def get_task_features(self, obs, task_onehot):
        features = np.zeros((obs.shape[0], self.total_extra_features))
        if task_onehot.shape[0] == 0:
            return features

        task_idx = int(np.argmax(task_onehot[0]))
        # print(f"task_idx: {task_idx}")
        L = self.extra_features[task_idx]

        specific_features = obs[:, -L:]
        features[:,self.feature_ranges[task_idx]] = specific_features
        return features

    def modify_state(self, obs, info):
        if len(obs.shape) == 1: 
            obs = np.expand_dims(obs, axis=0) 
        
        shared_obs = obs[:, :self.num_shared_features] 
        
        task_onehot = self.get_task_onehot(info) 
        if len(task_onehot.shape) == 1: 
            task_onehot = np.expand_dims(task_onehot, axis=0) 

        task_features = self.get_task_features(obs, task_onehot)
        return np.hstack((shared_obs, task_features, task_onehot))


s, info = env.reset()
p = Preprocessor()
full_s = p.modify_state(s, info)

## Create the model
model = DDPG_FF(
    n_features=full_s.shape[1],
    action_space=env.action_space,
    neurons=[128, 64, 64, 32, 16],
    activation_function=F.elu,
    learning_rate=0.0001,
)


## Define an action function
def action_function(policy):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (
        expected_bounds[1] - expected_bounds[0]
    )
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return (
        env.action_space.low
        + (env.action_space.high - env.action_space.low) * bounded_percent
    )


## Train the model
training_loop(env, model, action_function)

## Watch
sai.watch(model, action_function, Preprocessor)

## Benchmark the model locally
sai.benchmark(model, action_function)
