import numpy as np
from enum import Enum
from collections import defaultdict, OrderedDict
import gym.spaces as spaces
from gym import Env, ObservationWrapper, RewardWrapper
from gym.utils import seeding
import os

class Action(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    STAY = 4

class Dropzone():
    def __init__(self, name):
        self.name = name
        
    def __repr__(self):
        return 'D{}'.format(self.name)

class Packet():
    def __init__(self, name):
        self.name = name
        
    def __repr__(self):
        return 'P{}'.format(self.name)

class Drone():
    def __init__(self, name):
        self.name = name
        self.packet = None
        
    def __repr__(self):
        return str(self.name)
        
class Grid():
    def __init__(self, shape):
        self.shape = shape
        self.grid = np.full(shape, fill_value=None, dtype=np.object)
    
    def __getitem__(self, key):
        return self.grid[key]
    
    def __setitem__(self, key, value):
        self.grid[key] = value
    
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
    
    def spawn(self, objects):
        """Spawn objects on empty tiles. Return positions."""
        flat_idxs = np.random.choice(np.flatnonzero(self.grid == None), size=len(objects), replace=False)
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
    metadata = {
        'render.modes': ['ainsi'],
        'drone_density': 0.05
    }
    
    def __init__(self, drones_names):
        # Define size of the environment
        self.drones_names = drones_names
        self.n_drones = len(self.drones_names)
        self.side_size = int(np.ceil(np.sqrt(self.n_drones/self.metadata['drone_density'])))
        self.shape = (self.side_size, self.side_size)
        
        # Define spaces
        self.action_space = spaces.Discrete(len(Action))
        # TODO: implement 6-layers full state representation
        
    def step(self, actions):
        # By default, drones get a reward of zero
        rewards = {name: 0 for name in actions.keys()}
        
        # Do some air navigation for drones based on actions
        new_positions = defaultdict(list)
        air_respawns = []
        ground_respawns = []

        for drone, position in self.air.get_objects(Drone, zip_results=True):
            # Drones actually teleports, temporarily remove them from the air
            self.air[position] = None
            
            # Get action and drone position
            action = Action.STAY if drone.name not in actions else Action(actions[drone.name])
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
            
            # Drone plans to move to a valid place, save it
            if(self.air.is_inside(new_position)):
                new_positions[new_position].append(drone)
                  
            # Drone plans to move outside the grid -> crash
            else:
                # Drone gets a negative reward and respawns
                rewards[drone.name] = -1
                air_respawns.append(drone)
                
                # Delivery content is lost with drone crash (!)
                if drone.packet is not None:
                    ground_respawns.append(drone.packet)
                    drone.packet = None
                    
        # Fruther air navigation for drones that didn't went outside the grid
        for position, drones in new_positions.items():
            # Is there a collision?
            if len(drones) > 1:
                # Drones get a negative reward and have to respawn
                rewards.update({drone.name: -1 for drone in drones})
                air_respawns.extend(drones)
                
                # Delivery content is lost with drones crash (!)
                ground_respawns.extend([drone.packet for drone in drones if drone.packet is not None])
                for drone in drones:
                    drone.packet = None
                continue
                
            # If not, move the drone and check what's on the ground
            drone = drones[0]
            self.air[position] = drone

            # Is there a packet?
            if isinstance(self.ground[position], Packet):
                # Does the drone already have a packet?
                if drone.packet is not None:
                    # Then switch the packets
                    drone.packet, self.ground[position] = (self.ground[position], drone.packet)
                    
                # Otherwise, just take the packet
                else:
                    drone.packet = self.ground[position]
                    self.ground[position] = None
                    
            # A drop zone?
            elif isinstance(self.ground[position], Dropzone):
                dropzone = self.ground[position]
                
                # Did we just delivered a packet?
                if drone.packet is not None:
                    if drone.packet.name == dropzone.name:
                        # Pay the drone for the delivery
                        rewards[drone.name] = 1
                        
                        # Create new delivery
                        ground_respawns.extend([drone.packet, dropzone])
                        self.ground[position] = None
                        drone.packet = None
                        
        # Respawn objects
        self.ground.spawn(ground_respawns)
        self._pick_packets_after_respawn(self.air.spawn(air_respawns))
        
        # Episode ends when drone respawns
        dones = {name: False for name in actions.keys()}
        for drone in air_respawns:
            dones[drone.name] = True
        
        # Return new states, rewards, done and other infos
        info = {'air_respawns': air_respawns, 'ground_respawns': ground_respawns}
        return self._get_grids(), rewards, dones, info
        
    def reset(self):
        # Create grids
        self.air = Grid(shape=self.shape)
        self.ground = Grid(shape=self.shape)
        
        # Spawn objects
        self.ground.spawn([Packet(i) for i in range(self.n_drones)])
        self.ground.spawn([Dropzone(i) for i in range(self.n_drones)])
        self._pick_packets_after_respawn(
            self.air.spawn([Drone(name) for name in self.drones_names]))
        
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
                    return '[{:.1}]'.format(str(ground_tile.name))
                elif isinstance(ground_tile, Dropzone):
                    return '({:.1})'.format(str(ground_tile.name))
                
            # Air has a drone
            elif isinstance(air_tile, Drone):
                if air_tile.packet is None:
                    return '{:.1}'.format(str(air_tile.name))
                else:
                    return '{:.1}{:.1}{:.1}'.format(
                        str(air_tile.name),
                        str(air_tile.packet.name),
                        str(air_tile.name))
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
        
class TargetQTable(ObservationWrapper):
    """
    Observation wrapper for Q-table based algorithms with only
    the direction of the target as information (left/down/right/up states)
    Note: Target can be the direction of the nearest packet or associated dropzone
    """
    def __init__(self, env):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.observation_space = spaces.Discrete(4)
        
    def observation(self, _):
        # Return state for each drone
        return {
            drone.name: self.get_drone_state(drone, *position)
            for drone, position in self.env.air.get_objects(Drone, zip_results=True)}
        
    def get_drone_state(self, drone, drone_y, drone_x):        
        # Target direction: nearest packet or associated dropzone
        targets, positions = self.env.ground.get_objects(Packet if drone.packet is None else Dropzone)
        if drone.packet is None:
            l1_distances = np.abs(positions[0] - drone_y) + np.abs(positions[1] - drone_x)
            target_idx = l1_distances.argmin() # Index of the nearest packet
        else:
            target_idx = [d.name for d in targets].index(drone.packet.name)
            
         # Direction to go to reduce distance to the packet
        target_y, target_x = positions[0][target_idx], positions[1][target_idx]
        direction = np.argmax([drone_x-target_x, target_y-drone_y, target_x-drone_x, drone_y-target_y])
        # Arbitrarily use argmax(): direction to any positive value above would work
        
        return direction
    
class TargetCollisionQTable(TargetQTable):
    def __init__(self, env):
        # Initialize wrapper with observation space
        super().__init__(env)
        self.observation_space = spaces.Dict([
            ('target', spaces.Discrete(4)),
            ('collision', spaces.MultiBinary(4))
        ])
        
    def get_drone_state(self, drone, drone_y, drone_x):
        # Get target direction
        target = super().get_drone_state(drone, drone_y, drone_y)
        
        # TODO: Add collision info
        collision = [0, 0, 0, 0]
        
        # Use the same ordering as obs. space to avoid any issues
        return OrderedDict([('target', target), ('collision', collision)])
        
    
"""TODO: more complex observation wrapper with collision information
        # Helper function to convert relative positions to absolute ones
        def to_absolute(positions):
            to_abs = lambda p: (drone_y + p[1], drone_x + p[0])
            return list(map(to_abs, positions))

        # Encode wall information
        is_wall_below = not self.env.is_inside(position=(position[0]+1, position[1]))
        is_wall_above = not self.env.is_inside(position=(position[0]-1, position[1]))
        state.append(2 if is_wall_above else (1 if is_wall_below else 0))
        
        is_wall_right = not self.env.is_inside(position=(position[0], position[1]+1))
        is_wall_left = not self.env.is_inside(position=(position[0], position[1]-1))
        state.append(2 if is_wall_right else (1 if is_wall_left else 0))
        
        # Encode if moving below/above/left/right or "stay" is "safe"
        # TODO: merge this with wall information
        left_positions = to_absolute([(-1, -1), (0, -1), (1, -1), (0, -2)])
        below_positions = to_absolute([(1, -1), (1, 0), (1, 1), (2, 0)])
        right_positions = to_absolute([(-1, 1), (0, 1), (1, 1), (0, 2)])
        above_positions = to_absolute([(-1, -1), (-1, 0), (-1, 1), (-2, 0)])
        stay_positions = to_absolute([(-1, 0), (0, -1), (1, 0), (0, 1)])
        state.append(int(self.env.air.get_objects(Drone, left_positions)[0].size > 0))
        state.append(int(self.env.air.get_objects(Drone, below_positions)[0].size > 0))
        state.append(int(self.env.air.get_objects(Drone, right_positions)[0].size > 0))
        state.append(int(self.env.air.get_objects(Drone, above_positions)[0].size > 0))
        state.append(int(self.env.air.get_objects(Drone, stay_positions)[0].size > 0))
"""

"""
    # TODO: Drones layers
    my_drone_layer = np.zeros(shape=self.shape)
    other_drones_layer = np.zeros(shape=self.shape)

    # TODO: Packets layers
    my_packet_layer = np.zeros(shape=self.shape)
    other_packets_layer = np.zeros(shape=self.shape)

    # TODO: Dropzones layers
    my_dropzone_layer = np.zeros(shape=self.shape)
    other_dropzones_layer = np.zeros(shape=self.shape)

    return np.stack([
        my_drone_layer, other_drones_layer,
        my_packet_layer, other_packets_layer,
        my_dropzone_layer, other_dropzones_layer
    ], axis=-1)
"""