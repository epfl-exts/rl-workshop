#!/usr/bin/env python

import torch
import tempfile
import numpy as np
import tqdm

from agents.dqn import DQNAgent, DenseQNetworkFactory
from env.env import DeliveryDrones, WindowedGridView
from rl_helpers.rl_helpers import set_seed






################################################
################################################
# Evaluation State Variables
################################################
EPISODE_SEEDS = [845, 99, 65, 96, 85, 39, 51, 17, 52, 35]
TOTAL_EPISODE_STEPS = 1000
participating_agents = {
    "baseline-1" : "baseline_models/random-agent-4.pt",
    "baseline-2" : "baseline_models/random-agent-1.pt",
    "baseline-3" : "baseline_models/random-agent-2.pt",
    "baseline-4" : "baseline_models/random-agent-3.pt",
    "baseline-5" : "baseline_models/random-agent-3.pt",
    "baseline-6" : "baseline_models/random-agent-3.pt",
    "baseline-7" : "baseline_models/random-agent-3.pt",
    "baseline-8" : "baseline_models/random-agent-3.pt",
    "baseline-9" : "baseline_models/random-agent-3.pt",
}

################################################
# Helper Functions
################################################
def agent_id(agent_name):
    """
    Returns a unique numeric id for an agent_name
    """
    global participating_agents
    agent_names = sorted(participating_agents.keys())
    return agent_names.index(agent_name)

def agent_name_from_id(agent_id):
    """
    Returns the unique agent name from an agent_id
    """
    global participating_agents
    agent_names = sorted(participating_agents.keys())
    return agent_names[agent_id]

################################################
################################################
# Load Baseline models
################################################
loaded_agent_models = {}
for _item in participating_agents.keys():
    loaded_agent_models[_item] = torch.load(participating_agents[_item])
# Baseline Models loaded !! Yayy !!

################################################
################################################
# Load submission model
################################################
submission_path = "baseline_models/random-agent-0.pt" # To be received directly
model = torch.load(submission_path)
participating_agents["PARTICIPANT_MODEL"] = model

overall_scores = []

for _episode_idx, episode_seed in enumerate(EPISODE_SEEDS):
    ################################################
    ################################################
    # Run Episode
    ################################################
    episode_scores = np.zeros(len(participating_agents.keys()))
    
    ################################################
    ################################################
    # Env Instantiation
    ################################################
    env_params = {
        'charge': 20,
        'charge_reward': -0.1,
        'crash_reward': -1,
        'delivery_reward': 1,
        'discharge': 10,
        'drone_density': 0.05,
        'dropzones_factor': 2,
        'n_drones': 3,
        'packets_factor': 3,
        'pickup_reward': 0,
        'rgb_render_rescale': 1.0,
        'skyscrapers_factor': 3,
        'stations_factor': 2
    }
    env_params["n_drones"] = len(participating_agents.keys())
    env_params["rgb_render_rescale"] = 2.0 # large video - florian's request

    env = WindowedGridView(DeliveryDrones(env_params), radius=3)
    set_seed(env, episode_seed) # Seed 

    # Gather First Obeservation (state)
    state = env.reset()

    # Episode step loop
    for _step in tqdm.tqdm(range(TOTAL_EPISODE_STEPS)):
        _action_dictionary = {}

        ################################################
        ################################################
        # Act on the Env (all agents, one after the other)
        ################################################
        for _idx, _agent_name in enumerate(sorted(participating_agents.keys())):
            agent = participating_agents[_agent_name]

            ################################################
            ################################################
            # Gather observation 
            ################################################
            state_agent = state[_idx] 

            ################################################
            ################################################
            # Decide action of the participating agent
            ################################################
            q_values = model([state_agent])[0]
            action = q_values.argmax().item()

            _action_dictionary[_idx] = action
            ################################################
            ################################################
            # Collect frames for the first episode to generate video
            ################################################
            if _episode_idx == 0:
                # Record videos with env.render 
                # Do it in a tempfile
                # Compile frames into a video (from flatland)
                pass

        # Perform action (on all agents)
        state, rewards, done, info = env.step(_action_dictionary)

        # Gather rewards for all agents (inside episode_score)
        _step_score = np.array(list(rewards.values())) # Check with florian about ordering 

        episode_scores += _step_score

    # Store the current episode scores
    overall_scores.append(episode_scores)

    
# Aggregate all scores into an overall score
# TODO : Add aggregation function (lets start with simple mean + std)

overall_scores = np.array(overall_scores)
# Compute participant means and stds across episodes
_score = overall_scores.mean(axis=0)
_score_secondary = overall_scores.std(axis=0)

_idx_of_submitted_agent = agent_id("PARTICIPANT_MODEL")
score = _score[_idx_of_submitted_agent]
score_secondary = _score_secondary[_idx_of_submitted_agent]

# Post process videos

print("Scores : ", score, score_secondary)
print(overall_scores)


