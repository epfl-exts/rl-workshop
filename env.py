import numpy as np
from enum import Enum
from collections import defaultdict

class Action(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    STAY = 4

class Dropzone():
    def __init__(self, name):
        self.name = name

class Packet():
    def __init__(self, name):
        self.name = name

class Drone():
    def __init__(self, name):
        self.name = name
        self.packet = None
        
class Grid():
    def __init__(self, shape):
        self.shape = shape
        self.grid = np.full(shape, fill_value=None, dtype=np.object)
        
    def __getitem__(self, key):
        return self.grid[key]
    
    def __setitem__(self, key, value):
        self.grid[key] = value
    
    def get_objects(self, object_type):
        objects_mask = np.vectorize(
            lambda tile: isinstance(tile, object_type)
        )(self.grid)
        objects_idxs = (
            np.indices(self.shape)[0][objects_mask],
            np.indices(self.shape)[1][objects_mask])
        return list(zip(self[objects_mask], list(zip(*objects_idxs))))
    
    def spawn(self, objects):
        # Pick empty tiles
        idxs = np.arange(self.grid.size).reshape(self.shape)
        available_idxs = idxs[self.grid == None]
        selected_idxs = np.random.choice(
            available_idxs, size=len(objects), replace=False)
                
        # Spawn objects
        np.put(self.grid, selected_idxs, objects)
        
class CityDrones():
    drone_density = 0.05
    
    def __init__(self, drones_names):
        # Create grids
        sides_size = int(np.ceil(np.sqrt(
            len(drones_names) / CityDrones.drone_density)))
        self.shape = (sides_size, sides_size)
        self.air = Grid(shape=self.shape)
        self.ground = Grid(shape=self.shape)
        
        # Spawn objects
        self.air.spawn([Drone(name) for name in drones_names])
        self.ground.spawn([Packet(i) for i in range(len(drones_names))])
        self.ground.spawn([Dropzone(i) for i in range(len(drones_names))])
        
    def step(self, actions):
        # Check how each drone plans to move
        new_positions = defaultdict(list)
        air_respawns = []
        ground_respawns = []
        rewards = {}
        for drone, position in self.air.get_objects(Drone):
            # Drones actually teleports, temporarily remove them from the air
            self.air[position] = None
            
            # Get action and drone position
            action = Action.STAY if drone.name not in actions else actions[drone.name]
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
            
            # Check where that new position is
            try:
                # Raises an exception if the drone plans to fly outside the grid
                np.ravel_multi_index(new_position, self.shape)
                
                # Drone plans to move to a valid place, save it
                new_positions[new_position].append(drone)
                  
            # Drone plans to move outside the grid!
            except ValueError:
                # Drone gets a negative reward and has to respawn
                rewards[drone] = -1
                air_respawns.append(drone)
                
                # Drone drops its packet if any
                if drone.packet is not None:
                    self.ground[position] = drone.packet
                    drone.packet = None
                    
        # Move drones that didn't went outisde the grid
        for position, drones in new_positions.items():
            # Is there a collision?
            if len(drones) > 1:
                # Drones get a negative reward and have to respawn
                rewards.update({drone: -1 for drone in drones})
                air_respawns.extend(drones)
                continue
                
            # If not, move the drone and check what's on the ground
            drone = drones[0]
            self.air[position] = drone

            # Is there a packet?
            if isinstance(self.ground[position], Packet):
                # Does the drone already have a packet?
                if drone.packet is not None:
                    # Then switch the packets
                    drone.packet, self.ground[position] = (
                        self.ground[position], drone.packet)
                    
                # Otherwise, just take the packet
                else:
                    drone.packet = self.ground[position]
                    self.ground[position] = None
                    
            # A drop zone?
            elif isinstance(self.ground[position], Dropzone):
                # Did we just delivered a packet?
                if drone.packet is not None:
                    if drone.packet.name == self.ground[position].name:
                        # Drone gets a positive reward
                        reward[drone] = 1
                        
                        # Respawn packet and dropzone
                        ground_respawns.extend([drone.packet, self.ground[position]])
                        self.ground[position] = None
                        
        # Respawn objects
        self.air.spawn(air_respawns)
        self.ground.spawn(ground_respawns)
    
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