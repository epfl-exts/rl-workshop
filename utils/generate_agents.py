from agents.dqn import DQNAgent, DenseQNetworkFactory
from env.env import DeliveryDrones
from env.wrappers import WindowedGridView

import tqdm

"""
Generates random (non trained) agents to be used as baselines.
"""

NB_AGENTS = 10

for _idx in tqdm.tqdm(range(NB_AGENTS)):
    env_params = {
        'charge': 20,
        'charge_reward': -0.1,
        'crash_reward': -1,
        'delivery_reward': 1,
        'discharge': 10,
        'drone_density': 0.05,
        'dropzones_factor': 2,
        'packets_factor': 3,
        'pickup_reward': 0,
        'skyscrapers_factor': 3,
        'stations_factor': 2,
        "n_drones": NB_AGENTS,
        "rgb_render_rescale": 2.0
    }

    env = WindowedGridView(DeliveryDrones(env_params), radius=3)

    agent = DQNAgent(env,
                     DenseQNetworkFactory(env, hidden_layers=[256, 256]),
                     gamma=0.95,
                     epsilon_start=1.0,
                     epsilon_decay=0.999,
                     epsilon_end=0.01,
                     memory_size=10000,
                     batch_size=64,
                     target_update_interval=5
                     )

    agent.reset()
    agent.save("baseline_models/random-agent-{}.pt".format(_idx))
