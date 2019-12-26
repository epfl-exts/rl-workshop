import numpy as np

class Dropzone():
    def __init__(self, name):
        self.name = name

class Packet():
    def __init__(self, name):
        self.name = name

class Drone():
    def __init__(self, name):
        self.name = name
        
class Arena():
    drone_density = 0.05
    
    def __init__(self, drones_names):
        # Compute arena size
        arena_side_size = int(np.ceil(np.sqrt(
            len(drones_names) / Arena.drone_density)))
        self.shape = (arena_side_size, arena_side_size)
        
        # Create arena grid
        self.grid = np.full(self.shape, fill_value=None, dtype=np.object)
        
        # Spawn objects
        drones = [Drone(name) for name in drones_names]
        packets = [Packet(i) for i in range(len(drones_names))]
        dronzones = [Dropzone(i) for i in range(len(drones_names))]
        self.spawn(drones+packets+dronzones)
    
    def spawn(self, objects):
        # Pick empty tiles
        idxs = np.arange(self.grid.size).reshape(self.grid.shape)
        available_idxs = idxs[self.grid == None]
        selected_idxs = np.random.choice(
            available_idxs, size=len(objects), replace=False)
        
        # Spawn objects
        np.put(self.grid, selected_idxs, objects)
    
    def __str__(self):
        # Convert grid tiles to text
        def tile_to_char(tile):
            if tile is None:
                return ' '
            if isinstance(tile, Drone):
                return '§'
            if isinstance(tile, Packet):
                return 'x'
            if isinstance(tile, Dropzone):
                return 'o'
            return '?'
        grid_char = np.vectorize(tile_to_char)(self.grid)
        
        # Assemble tiles into a grid
        lines = ['+---'*self.shape[1]+'+']
        for i, row in enumerate(grid_char):
            line_str = '| '
            for j, tile_str in enumerate(row):
                line_str += tile_str + ' | '
            lines.append(line_str)
            lines.append('+---'*self.shape[1]+'+')
            
        return '\n'.join(lines)