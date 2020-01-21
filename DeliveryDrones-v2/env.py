import numpy as np
from enum import Enum
from collections import defaultdict, OrderedDict, Counter
import gym.spaces as spaces
from gym import Env, ObservationWrapper, RewardWrapper
from gym.utils import seeding
import os
import string

class Action(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    STAY = 4
    
class Skyscraper():
    def __repr__(self):
        return '#'
    
class Station():
    def __repr__(self):
        return '@'

class Dropzone():
    def __repr__(self):
        return 'Z'

class Packet():
    def __repr__(self):
        return 'P'

class Drone():
    def __init__(self, index):
        self.index = index
        self.packet = None
        self.charge = 100
        
    def __repr__(self):
        return 'D{}'.format(self.index)
        
class Grid():
    def __init__(self, shape):
        self.shape = shape
        self.grid = np.full(shape, fill_value=None, dtype=np.object)
    
    def __getitem__(self, key):
        return self.grid[key]
    
    def __setitem__(self, key, value):
        self.grid[key] = value
    
    def get_objects_positions(self, objs):
        results = []
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                for obj in objs:
                    if self.grid[y, x] == obj:
                        results.append((obj, (y, x)))
        return results
    
    def get_objects(self, object_type, positions=None, zip_results=False):
        """Filter objects matching criteria"""
        objects_mask = np.vectorize(lambda tile: isinstance(tile, object_type))(self.grid)

        if positions is not None:
            position_mask = np.full(shape=self.shape, fill_value=False)
            for x, y in filter(self.is_inside, positions):
                position_mask[x, y] = True
            objects_mask = np.logical_and(objects_mask, position_mask)
        
        if zip_results:
            # Make things much easier in for loops ".. for obj, pos in get_objects(..)"
            return zip(self[objects_mask], zip(*np.nonzero(objects_mask)))
        else:
            # Numpy like format: objects, (pos_x, pos_y)
            return self[objects_mask], np.nonzero(objects_mask)
    
    def spawn(self, objects, exclude_positions=None):
        """Spawn objects on empty tiles. Return positions."""
        positions_mask = (self.grid == None)
        
        if exclude_positions is not None:
            except_mask = np.full(shape=positions_mask.shape, fill_value=True)
            except_mask[exclude_positions] = False
            positions_mask = np.logical_and(positions_mask, except_mask)
            
        flat_idxs = np.random.choice(np.flatnonzero(positions_mask), size=len(objects), replace=False)
        idxs = np.unravel_index(flat_idxs, self.shape)
        self.grid[idxs] = objects
        return idxs
    
    def is_inside(self, position):
        try:
            np.ravel_multi_index(multi_index=position, dims=self.shape, mode='raise')
            return True
        except ValueError:
            return False
        
class DeliveryDrones(Env):
    # OpenAI Gym environment fields
    metadata = {'render.modes': ['ainsi'], 'drone_density': 0.05}
    
    def __init__(self, n, rewards_settings={'pickup': 1, 'delivery': 1, 'charge': -0.1, 'crash': -1}, discharge=10, station_charge=20):
        # Define size of the environment
        self.n_drones = n
        self.rewards_settings = rewards_settings
        self.discharge = discharge
        self.station_charge = station_charge
        self.side_size = int(np.ceil(np.sqrt(self.n_drones/self.metadata['drone_density'])))
        self.shape = (self.side_size, self.side_size)
        
        # Define spaces
        self.action_space = spaces.Discrete(len(Action))
        
    def step(self, actions):
        # By default, drones get a reward of zero
        rewards = {index: 0 for index in actions.keys()}
        
        # Do some air navigation for drones based on actions
        new_positions = defaultdict(list)
        air_respawns = []
        ground_respawns = []

        for drone, position in self.air.get_objects(Drone, zip_results=True):
            # Drone actually teleports temporarily remove it from the air
            self.air[position] = None
            
            # Get action and drone position
            action = Action.STAY if drone.index not in actions else Action(actions[drone.index])
            if action is Action.LEFT:
                new_position = position[0], position[1]-1
            elif action is Action.DOWN:
                new_position = position[0]+1, position[1]
            elif action is Action.RIGHT:
                new_position = position[0], position[1]+1
            elif action is Action.UP:
                new_position = position[0]-1, position[1]
            else:
                new_position = position
            
            # Is the drone planning to move outside the grid?
            if self.air.is_inside(new_position):
                # Is the drone going into a skyscraper?
                if isinstance(self.ground[new_position], Skyscraper):
                    air_respawns.append(drone)
                else:
                    new_positions[new_position].append(drone)  
            else:
                air_respawns.append(drone)
                    
        # Further air navigation for drones that didn't went outside the grid
        for position, drones in new_positions.items():
            # Is there a collision?
            if len(drones) > 1:
                air_respawns.extend(drones)
                continue
                
            # Get drone
            drone = drones[0]
            
            # Drone discharges after each step, except if on station
            if isinstance(self.ground[position], Station):
                drone.charge = min(100, drone.charge+self.station_charge) # charge
                rewards[drone.index] = self.rewards_settings['charge'] # cost of charging
            else:
                drone.charge -= self.discharge # discharge
                # Without charge left, drone crashes
                if drone.charge <= 0:
                    air_respawns.append(drone)
                    continue
                    
            # Move the drone and check what's on the ground     
            self.air[position] = drone

            # Take packet if any
            if (drone.packet is None) and isinstance(self.ground[position], Packet):
                rewards[drone.index] = self.rewards_settings['pickup']
                drone.packet = self.ground[position]
                self.ground[position] = None
                
            # Did we just deliver a packet?
            elif (drone.packet is not None) and isinstance(self.ground[position], Dropzone):
                # Pay the drone for the delivery
                rewards[drone.index] = self.rewards_settings['delivery']
                
                # Create new delivery
                ground_respawns.extend([drone.packet, self.ground[position]])
                self.ground[position] = None
                drone.packet = None
                        
        # Handle drone crashes
        for drone in air_respawns:
            # Drone restarts fully charged
            drone.charge = 100
            
            # Packet is destroyed
            rewards[drone.index] = self.rewards_settings['crash']
            if drone.packet is not None:
                ground_respawns.append(drone.packet)
                drone.packet = None
                        
        # Respawn objects
        self.ground.spawn(ground_respawns)
        skyscrapers, skyscrapers_positions = self.ground.get_objects(Skyscraper)
        self._pick_packets_after_respawn(self.air.spawn(
            air_respawns, exclude_positions=skyscrapers_positions))
        
        # Episode ends when drone respawns
        dones = {index: False for index in actions.keys()}
        for drone in air_respawns:
            dones[drone.index] = True
        
        # Return new states, rewards, done and other infos
        info = {'air_respawns': air_respawns, 'ground_respawns': ground_respawns}
        return self._get_grids(), rewards, dones, info
        
    def reset(self):
        # Create grids
        self.air = Grid(shape=self.shape)
        self.ground = Grid(shape=self.shape)
        
        # Create
        # Note: use 1-indexing to simplify state encoding where 0 denotes "absence"
        self.packets = [Packet() for _ in range(3*self.n_drones)]
        self.dropzones = [Dropzone() for _ in range(2*self.n_drones)]
        self.stations = [Station() for _ in range(2*self.n_drones)]
        self.drones = [Drone(index) for index in range(1, self.n_drones+1)]
        self.skyscrapers = [Skyscraper() for _ in range(3*self.n_drones)]
        
        # Spawn objects
        self.ground.spawn(self.packets)
        self.ground.spawn(self.dropzones)
        self.ground.spawn(self.stations)
        skyscrapers_position = self.ground.spawn(self.skyscrapers)
        self._pick_packets_after_respawn(self.air.spawn(
            self.drones, exclude_positions=skyscrapers_position
        ))
        
        return self._get_grids()
        
    def render(self, mode='ainsi'):
        if mode == 'ainsi':
            return self.__str__()
        else:
            super().render(mode=mode)
    
    def _get_grids(self):
        return {'ground': self.ground, 'air': self.air}
        
    def _pick_packets_after_respawn(self, positions):
        for y, x in zip(*positions):
            if isinstance(self.ground[y, x], Packet):
                self.air[y, x].packet = self.ground[y, x]
                self.ground[y, x] = None
    
    def __str__(self):
        # Convert air/ground tiles to text
        def tiles_to_char(ground_tile, air_tile):
            # Air is empty
            if air_tile is None:
                if ground_tile is None:
                    return ''
                elif isinstance(ground_tile, Packet):
                    return 'x'
                elif isinstance(ground_tile, Dropzone):
                    return '[ ]'
                elif isinstance(ground_tile, Station):
                    return '@'
                elif isinstance(ground_tile, Skyscraper):
                    return '#'
                
            # Air has a drone
            elif isinstance(air_tile, Drone):
                if isinstance(ground_tile, Skyscraper):
                    return '!!!' # TODO: remove
                
                if air_tile.packet is None:
                    if isinstance(ground_tile, Station):
                        return '{}@'.format(air_tile.index)
                    elif isinstance(ground_tile, Dropzone):
                        return '[{}]'.format(air_tile.index)
                    else:
                        return '{}'.format(air_tile.index)
                else:
                    if isinstance(ground_tile, Station):
                        return '{}@x'.format(air_tile.index)
                    else:
                        return '{}<x'.format(air_tile.index)
            return '?'
                
        grid_char = np.vectorize(tiles_to_char)(self.ground.grid, self.air.grid)
        
        # Assemble tiles into a grid
        tile_size = max(3, max([len(c) for c in grid_char.flatten()]))
        row_sep = ('+' + '-' * tile_size) * self.shape[1] + '+'
        lines = [row_sep]
        for i, row in enumerate(grid_char):
            line_str = '|'
            for j, tile_str in enumerate(row):
                tile_str = ' ' * ((tile_size-len(tile_str)) // 2) + tile_str
                tile_str = tile_str.ljust(tile_size, ' ')
                line_str += tile_str + '|'
            lines.append(line_str)
            lines.append(row_sep)
            
        return '\n'.join(lines)
    
    def format_action(self, i):
        return ['←', '↓', '→', '↑', 'X'][i]
        
class CompassQTable(ObservationWrapper):
    """
    Observation wrapper for Q-table based algorithms
    The state gives campass direction (W/SW/S/SE/E/NE/N/NW)
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
            for drone, position in self.env.air.get_objects(Drone, zip_results=True)}
        
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
            (west >  0) and (south == 0),
            (west >  0) and (south >  0),
            (west == 0) and (south >  0),
            (west <  0) and (south >  0),
            (west <  0) and (south == 0),
            (west <  0) and (south <  0),
            (west == 0) and (south <  0),
            (west >  0) and (south <  0)
        ])
    
    def format_state(self, s):
        return self.cardinals[s]
    
class CompassChargeQTable(CompassQTable):
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
            station_idx = l1_distances.argmin() # Index of the nearest station
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
            '←' : [(0, -1), (0, -2)],
            '↙': [(1, -1)],
            '↓' : [(1, 0), (2, 0)],
            '↘': [(1, 1)],
            '→' : [(0, 1), (0, 2)],
            '↗': [(-1, 1)],
            '↑' : [(-1, 0), (-2, 0)],
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
            station_idx = l1_distances.argmin() # Index of the nearest station
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

class GlobalGridView(ObservationWrapper):
    """
    Observation wrapper: (N, N, 5) numerical arrays with location of
    (1) drones         marked with    drone index 1..i / 0 otherwise
    (2) packets        marked with                   1 / 0 otherwise
    (3) dropzones      marked with                   1 / 0 otherwise
    (4) stations       marked with                   1 / 0 otherwise
    (5) drones charge  marked with   charge level 0..1 / 0 otherwise
    Where N is the size of the environment grid, i the number of drones
    """
    def __init__(self, env):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=self.n_drones, shape=self.env.shape+(5,), dtype=np.float)
        
    def observation(self, _):
        gridview = self.gen_gridview()
        return {index: gridview.copy() for index in range(1, self.env.n_drones+1)}
    
    def gen_gridview(self):
        # Create grid and get objects
        grid = np.zeros(shape=self.env.shape + (5,))

        # Drones (and their packets) + charge
        for drone, (y, x) in self.env.air.get_objects(Drone, zip_results=True):
            grid[y, x, 0] = drone.index
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
        
        return grid
    
class PlayerGridView(GlobalGridView):
    """
    Observation wrapper: (N, N, 6) arrays
    Similar to GlobalGridView for channels 1-5 with first "drones index" one made binary
    Channel 6 marks the position of the player's drone with a 1
    """
    def __init__(self, env):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.env.shape+(6,), dtype=np.float)
        
    def observation(self, _):
        states = {}

        # Get global gridview
        global_gridview = self.gen_gridview()
        
        for drone, (y, x) in self.env.air.get_objects(Drone, zip_results=True):
            # Channels 1-5
            gridview = np.zeros(shape=self.env.shape+(6,))
            gridview[:, :, :5] = global_gridview[:, :, :5]
            gridview[np.nonzero(global_gridview[:, :, 0])+(0,)] = 1
            
            # Drone position
            gridview[y, x, 5] = 1
            
            states[drone.index] = gridview
            
        return states

class WindowedGridView(ObservationWrapper):
    """
    Observation wrapper: (N, N, 6) numerical arrays with location of
    (1) drones         marked with                   1 / 0 otherwise
    (2) packets        marked with                   1 / 0 otherwise
    (3) dropzones      marked with                   1 / 0 otherwise
    (4) stations       marked with                   1 / 0 otherwise
    (5) drones charge  marked with   charge level 0..1 / 0 otherwise
    (6) obstacles      marked with                   1 / 0 otherwise
    Where N is the size of the environment grid, i the number of drones
    """
    def __init__(self, env, window):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.window = window
        assert window > 0, "Window size should be strictly positive"

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.window*2+1, self.window*2+1, 6), dtype=np.float)
        
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
        padded_shape = (self.env.shape[0]+2*self.window, self.env.shape[1]+2*self.window, 6)
        padded_grid = np.zeros(padded_shape)
        padded_grid[:, :, 5] = 1 # walls
        padded_grid[self.window:-self.window, self.window:-self.window, :] = grid

        # Return windowed state for each drone
        states = {}
        for drone, (y, x) in self.env.air.get_objects(Drone, zip_results=True):
            states[drone.index] = padded_grid[y:y+2*self.window+1, x:x+2*self.window+1, :].copy()
        return states