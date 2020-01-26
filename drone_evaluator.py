import os.path

import numpy as np
import torch
import tqdm

from env.env import DeliveryDrones, WindowedGridView
from rl_helpers.rl_helpers import set_seed

class DroneRacerEvaluator:
  def __init__(self, answer_folder_path=".", round=1):
    """
    `round` : Holds the round for which the evaluation is being done.
    can be 1, 2...upto the number of rounds the challenge has.
    Different rounds will mostly have different ground truth files.
    """
    self.answer_folder_path = answer_folder_path
    self.round = round

    ################################################
    ################################################
    # Evaluation State Variables
    ################################################
    self.EPISODE_SEEDS = [845, 99, 65, 96, 85, 39, 51, 17, 52, 35]
    self.TOTAL_EPISODE_STEPS = 1000
    self.participating_agents = {
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
    ################################################
    # Load Baseline models
    ################################################
    self.loaded_agent_models = {}
    for _item in self.participating_agents.keys():
        agent_path = os.path.join(answer_folder_path, self.participating_agents[_item])
        self.loaded_agent_models[_item] = torch.load(agent_path)
    # Baseline Models loaded !! Yayy !!


    ################################################
    # Helper Functions
    ################################################
  def agent_id(self, agent_name):
    """
    Returns a unique numeric id for an agent_name
    """
    agent_names = sorted(self.participating_agents.keys())
    return agent_names.index(agent_name)

  def agent_name_from_id(self, agent_id):
    """
    Returns the unique agent name from an agent_id
    """
    agent_names = sorted(self.participating_agents.keys())
    return agent_names[agent_id]



  def _evaluate(self, client_payload, _context={}):
    """
    `client_payload` will be a dict with (atleast) the following keys :
      - submission_file_path : local file path of the submitted file
      - aicrowd_submission_id : A unique id representing the submission
      - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
    """
    submission_file_path = client_payload["submission_file_path"]
    aicrowd_submission_id = client_payload["aicrowd_submission_id"]
    aicrowd_participant_uid = client_payload["aicrowd_participant_id"]


    ################################################
    ################################################
    # Load submission model
    ################################################
    # submission_path = "baseline_models/random-agent-0.pt" # To be received directly

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(submission_file_path, map_location=device)
    self.participating_agents["PARTICIPANT_MODEL"] = model

    self.overall_scores = []

    for _episode_idx, episode_seed in enumerate(self.EPISODE_SEEDS):
        ################################################
        ################################################
        # Run Episode
        ################################################
        episode_scores = np.zeros(len(self.participating_agents.keys()))

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
        env_params["n_drones"] = len(self.participating_agents.keys())
        env_params["rgb_render_rescale"] = 2.0 # large video - florian's request

        env = WindowedGridView(DeliveryDrones(env_params), radius=3)
        set_seed(env, episode_seed) # Seed

        # Gather First Obeservation (state)
        state = env.reset()

        # Episode step loop
        for _step in tqdm.tqdm(range(self.TOTAL_EPISODE_STEPS)):
            _action_dictionary = {}

            ################################################
            ################################################
            # Act on the Env (all agents, one after the other)
            ################################################
            for _idx, _agent_name in enumerate(sorted(self.participating_agents.keys())):
                agent = self.participating_agents[_agent_name]

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
        self.overall_scores.append(episode_scores)


    # Aggregate all scores into an overall score
    # TODO : Add aggregation function (lets start with simple mean + std)

    self.overall_scores = np.array(self.overall_scores)
    # Compute participant means and stds across episodes
    _score = self.overall_scores.mean(axis=0)
    _score_secondary = self.overall_scores.std(axis=0)

    _idx_of_submitted_agent = self.agent_id("PARTICIPANT_MODEL")
    score = _score[_idx_of_submitted_agent]
    score_secondary = _score_secondary[_idx_of_submitted_agent]

    # Post process videos

    print("Scores : ", score, score_secondary)
    print(self.overall_scores)

    _result_object = {
        "score" : score,
        "score_secondary" : score_secondary
    }

    return _result_object

if __name__ == "__main__":
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    answer_file_path = "."
    _client_payload = {}
    _client_payload["submission_file_path"] = "baseline_models/random-agent-3.pt"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = DroneRacerEvaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
