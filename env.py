import numpy as np
from enum import Enum

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
    
    def get_objects(self, object_type):
        idxs = np.arange(self.grid.size).reshape(self.shape)
        objects_idxs = np.vectorize(
            lambda tile: isinstance(tile, object_type)
        )(self.grid)
        
        return list(zip(self.grid[objects_idxs], idxs[objects_idxs]))
    
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
        self.floor = Grid(shape=self.shape)
        
        # Spawn objects
        self.air.spawn([Drone(name) for name in drones_names])
        self.floor.spawn([Packet(i) for i in range(len(drones_names))])
        self.floor.spawn([Dropzone(i) for i in range(len(drones_names))])
        
    def step(self, actions):
        pass
    
    def __str__(self):
        # Convert air/floor tiles to text
        def tiles_to_char(floor_tile, air_tile):
            # Air is empty
            if air_tile is None:
                if floor_tile is None:
                    return ''
                elif isinstance(floor_tile, Packet):
                    return '[{:.1}]'.format(str(floor_tile.name))
                elif isinstance(floor_tile, Dropzone):
                    return '({:.1})'.format(str(floor_tile.name))
                
            # Air has a drone
            elif isinstance(air_tile, Drone):
                if air_tile.packet is None:
                    return '*'
                else:
                    return '*{:.1}*'.format(str(drone.packet.name))
            return '?'
                
        grid_char = np.vectorize(tiles_to_char)(self.floor.grid, self.air.grid)
        
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