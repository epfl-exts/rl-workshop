from collections import OrderedDict

import gym.spaces as spaces
import numpy as np
from gym import ObservationWrapper
from env.env import Drone, Packet, Dropzone, DeliveryDrones, Skyscraper, Station


class CompassQTable(ObservationWrapper):
    """
    Observation wrapper for Q-table based algorithms
    The state gives compass direction (W/SW/S/SE/E/NE/N/NW)
    to nearest packet or dropzone
    """

    def __init__(self, env):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.observation_space = spaces.Discrete(8)
        self.cardinals = ['←', '↙', '↓', '↘', '→', '↗', '↑', '↖']

    def observation(self, _):
        # Return state for each drone
        return {
            drone.index: self.get_drone_state(drone, *position)
            for drone, position in self.env.air.get_objects(Drone, zip_results=True)
        }

    def get_drone_state(self, drone, drone_y, drone_x):
        # Target direction: nearest packet or dropzone
        targets, positions = self.env.ground.get_objects(Packet if drone.packet is None else Dropzone)
        l1_distances = np.abs(positions[0] - drone_y) + np.abs(positions[1] - drone_x)
        target_idx = l1_distances.argmin()

        # Compass direction to it
        target_y, target_x = positions[0][target_idx], positions[1][target_idx]
        return self.compass_direction(drone_y, drone_x, target_y, target_x)

    def compass_direction(self, drone_y, drone_x, target_y, target_x):
        west, south = (drone_x - target_x), (target_y - drone_y)
        return np.argmax([
            (west > 0) and (south == 0),
            (west > 0) and (south > 0),
            (west == 0) and (south > 0),
            (west < 0) and (south > 0),
            (west < 0) and (south == 0),
            (west < 0) and (south < 0),
            (west == 0) and (south < 0),
            (west > 0) and (south < 0)
        ])

    def format_state(self, s):
        return self.cardinals[s]


class CompassChargeQTable(CompassQTable):
    """
    Observation wrapper for Q-table based algorithms
    Adding compass direction (W/SW/S/SE/E/NE/N/NW/X*)
    to nearest charge station and current charge level (quartile-index)
    *X means "on charging station"
    """

    def __init__(self, env):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.observation_space = spaces.Dict([
            ('target_dir', spaces.Discrete(8)),
            ('station_dir', spaces.Discrete(9)),
            ('charge_level', spaces.Discrete(4))
        ])
        self.cardinals.append('X')

    def get_drone_state(self, drone, drone_y, drone_x):
        # Get compass direction to nearest packet/dropzone
        target_dir = super().get_drone_state(drone, drone_y, drone_x)

        # Get direction to nearest charging station
        stations, positions = self.env.ground.get_objects(Station)
        l1_distances = np.abs(positions[0] - drone_y) + np.abs(positions[1] - drone_x)
        if l1_distances.min() == 0:
            station_dir = self.cardinals.index('X')
        else:
            station_idx = l1_distances.argmin()  # Index of the nearest station
            station_y, station_x = positions[0][station_idx], positions[1][station_idx]
            station_dir = self.compass_direction(drone_y, drone_x, station_y, station_x)

        # Get charge level
        quartiles = np.array([25, 50, 75])
        larger_quartiles = np.nonzero(drone.charge < quartiles)[0]
        charge_level = len(quartiles) if len(larger_quartiles) == 0 else larger_quartiles[0]

        return OrderedDict([
            ('target_dir', target_dir),
            ('station_dir', station_dir),
            ('charge_level', charge_level)
        ])

    def format_state(self, s):
        return 'target: {}, station: {}, charge: {}'.format(self.cardinals[s['target_dir']], self.cardinals[s['station_dir']], s['charge_level'])


class LidarCompassQTable(CompassQTable):
    """
    Observation wrapper for Q-table based algorithms
    The states indicate campass direction and lidar information
    """

    def __init__(self, env):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.observation_space = spaces.Dict([
            ('target_dir', spaces.Discrete(8)),
            ('lidar', spaces.MultiBinary(8))
        ])
        self.lidar_positions = {
            '←': [(0, -1)],
            '↙': [(1, -1)],
            '↓': [(1, 0)],
            '↘': [(1, 1)],
            '→': [(0, 1)],
            '↗': [(-1, 1)],
            '↑': [(-1, 0)],
            '↖': [(-1, -1)]
        }

    def get_drone_state(self, drone, drone_y, drone_x):
        # Get target and lidar directions
        target_dir = super().get_drone_state(drone, drone_y, drone_x)
        lidar = [
            self.sense_obstacles(self.lidar_positions[c], drone_y, drone_x)
            for c in self.cardinals if c in self.lidar_positions.keys()]

        # Use the same ordering as obs. space to avoid any issues
        return OrderedDict([('target_dir', target_dir), ('lidar', lidar)])

    # Lidar information
    def sense_obstacles(self, positions, drone_y=0, drone_x=0):
        for y, x in positions:
            y, x = (y + drone_y), (x + drone_x)
            if not self.env.air.is_inside([y, x]):
                return 1
            if isinstance(self.env.air[y, x], Drone):
                return 1
            if isinstance(self.env.ground[y, x], Skyscraper):
                return 1
        return 0

    def format_state(self, state):
        # Find directions with positive lidar signal
        positive_lidar_signals = np.nonzero(state['lidar'])[0]
        lidar_cardinals = np.take(self.cardinals, positive_lidar_signals)

        return 'target: {}, lidar: {}'.format(
            self.cardinals[state['target_dir']], ', '.join(lidar_cardinals))


class LidarCompassChargeQTable(LidarCompassQTable):
    """
    Observation wrapper for Q-table based algorithms
    Adding campass direction (W/SW/S/SE/E/NE/N/NW/X*)
    to nearest charge station and current charge level (quartile-index)
    *X means "on charging station"
    """

    def __init__(self, env):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.observation_space = spaces.Dict([
            ('target_dir', spaces.Discrete(8)),
            ('station_dir', spaces.Discrete(9)),
            ('charge_level', spaces.Discrete(4)),
            ('lidar', spaces.MultiBinary(8))
        ])
        self.cardinals.append('X')

    def get_drone_state(self, drone, drone_y, drone_x):
        # Get target and lidar direction
        lidar_target = super().get_drone_state(drone, drone_y, drone_x)

        # Get direction to nearest charging station
        stations, positions = self.env.ground.get_objects(Station)
        l1_distances = np.abs(positions[0] - drone_y) + np.abs(positions[1] - drone_x)
        if l1_distances.min() == 0:
            station_dir = self.cardinals.index('X')
        else:
            station_idx = l1_distances.argmin()  # Index of the nearest station
            station_y, station_x = positions[0][station_idx], positions[1][station_idx]
            station_dir = self.compass_direction(drone_y, drone_x, station_y, station_x)

        # Get charge level
        quartiles = np.array([25, 50, 75])
        larger_quartiles = np.nonzero(drone.charge < quartiles)[0]
        charge_level = len(quartiles) if len(larger_quartiles) == 0 else larger_quartiles[0]

        # Use the same ordering as obs. space to avoid any issues
        return OrderedDict([
            ('target_dir', lidar_target['target_dir']),
            ('station_dir', station_dir),
            ('charge_level', charge_level),
            ('lidar', lidar_target['lidar'])
        ])

    def format_state(self, s):
        lidar_target_info = super().format_state(s)
        charge_info = 'station: {}, charge: {}'.format(self.cardinals[s['station_dir']], s['charge_level'])
        return '{}, {}'.format(lidar_target_info, charge_info)


class WindowedGridView(ObservationWrapper):
    """
    Observation wrapper: (N, N, 6) numerical arrays with location of
    (0) drones         marked with                   1 / 0 otherwise
    (1) packets        marked with                   1 / 0 otherwise
    (2) dropzones      marked with                   1 / 0 otherwise
    (3) stations       marked with                   1 / 0 otherwise
    (4) drones charge  marked with   charge level 0..1 / 0 otherwise
    (5) obstacles      marked with                   1 / 0 otherwise
    Where N is the size of the window, i the number of drones
    """

    def __init__(self, env, radius):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.radius = radius
        assert radius > 0, "radius should be strictly positive"

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.radius * 2 + 1, self.radius * 2 + 1, 6), dtype=np.float)

    def observation(self, _):
        # Create grid
        grid = np.zeros(shape=self.env.shape + (6,))
        states = {}

        # Drones (and their packets) + charge
        for drone, (y, x) in self.env.air.get_objects(Drone, zip_results=True):
            grid[y, x, 0] = 1
            if drone.packet is not None:
                grid[y, x, 1] = 1
            grid[y, x, 4] = drone.charge / 100

        # Packets
        for packet, (y, x) in self.env.ground.get_objects(Packet, zip_results=True):
            grid[y, x, 1] = 1

        # Dropzones
        for dropzone, (y, x) in self.env.ground.get_objects(Dropzone, zip_results=True):
            grid[y, x, 2] = 1

        # Stations
        grid[self.env.ground.get_objects(Station)[1] + (3,)] = 1

        # Obstacles
        for skyscraper, (y, x) in self.env.ground.get_objects(Skyscraper, zip_results=True):
            grid[y, x, 5] = 1

        # Pad
        padded_shape = (self.env.shape[0] + 2 * self.radius, self.env.shape[1] + 2 * self.radius, 6)
        padded_grid = np.zeros(padded_shape)
        padded_grid[:, :, 5] = 1  # walls
        padded_grid[self.radius:-self.radius, self.radius:-self.radius, :] = grid

        # Return windowed state for each drone
        for drone, (y, x) in self.env.air.get_objects(Drone, zip_results=True):
            states[drone.index] = padded_grid[y:y + 2 * self.radius + 1, x:x + 2 * self.radius + 1, :].copy()
        return states

    def format_state(self, s):
        return 'Numpy array {}'.format(s.shape)
