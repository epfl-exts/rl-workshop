import numpy as np
from enum import Enum

class Tile(Enum):
    EMPTY = 0
    DRONE = 1
    PACKET = 2
    DROPZONE = 3
    DRONEWPAQUET = 4
        
class Arena():
    drone_density = 0.05
    
    def __init__(self, drone_names):
        # Compute arena size
        arena_side_size = int(np.ceil(np.sqrt(
            len(drone_names) / Arena.drone_density)))
        self.shape = (arena_side_size, arena_side_size)
        
        # Create arena grid
        self.grid = np.full(self.shape, fill_value=Tile.EMPTY)
        
        # Span drones
        np.put(
            self.grid, 
            self.pick_freetiles(len(drone_names), flat=True),
            Tile.DRONE)
        
        # Span packets
        np.put(
            self.grid,
            self.pick_freetiles(len(drone_names) *2, flat=True),
            Tile.PACKET
        )
    
    def pick_freetiles(self, n, flat=False):
        # Pick empty tiles
        idxs = np.arange(self.grid.size).reshape(self.grid.shape)
        available_idxs = idxs[self.grid == Tile.EMPTY]
        selected_idxs_flat = np.random.choice(
            available_idxs, size=n, replace=False)
        
        # Return (non) flat indices
        if flat:
            return selected_idxs_flat
        else:
            return list(zip(*np.unravel_index(
                selected_idxs_flat, shape=self.grid.shape
            )))
    
    def __str__(self):
        # Convert grid tiles to text
        def tile_to_char(tile):
            if tile is Tile.EMPTY:
                return ' '
            if tile is Tile.DRONE:
                return '§'
            if tile is Tile.PACKET:
                return 'x'
            if tile is Tile.DROPZONE:
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