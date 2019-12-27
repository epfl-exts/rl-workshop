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
        
    def __repr__(self):
        return self.name

class Packet():
    def __init__(self, name):
        self.name = name
        
    def __repr__(self):
        return self.name

class Drone():
    def __init__(self, name):
        self.name = name
        self.packet = None
        
    def __repr__(self):
        return self.name
        
class Grid():
    def __init__(self, shape):
        self.shape = shape
        self.grid = np.full(shape, fill_value=None, dtype=np.object)
    
    def __getitem__(self, key):
        return self.grid[key]
    
    def __setitem__(self, key, value):
        self.grid[key] = value
    
    def get_objects(self, object_type):
        'Return objects from that type with their position'
        objects_mask = np.vectorize(lambda tile: isinstance(tile, object_type))(self.grid)
        return self[objects_mask], np.nonzero(objects_mask)
    
    def spawn(self, objects):
        'Spawn objects on empty tiles. Return positions.'
        flat_idxs = np.random.choice(np.flatnonzero(self.grid == None), size=len(objects), replace=False)
        idxs = np.unravel_index(flat_idxs, self.shape)
        self.grid[idxs] = objects
        return idxs
        
class DeliveryDrones():
    drone_density = 0.05
    
    def __init__(self, drones_names):
        # Create grids
        sides_size = int(np.ceil(np.sqrt(
            len(drones_names) / self.drone_density)))
        self.shape = (sides_size, sides_size)
        self.air = Grid(shape=self.shape)
        self.ground = Grid(shape=self.shape)
        
        # Spawn objects
        self.ground.spawn([Packet(i) for i in range(len(drones_names))])
        self.ground.spawn([Dropzone(i) for i in range(len(drones_names))])
        self._pick_packets_after_respawn(
            self.air.spawn([Drone(name) for name in drones_names]))
        
    def sample(self):
        return np.random.choice(Action).value
        
    def step(self, actions):
        # Check how each drone plans to move
        new_positions = defaultdict(list)
        air_respawns = []
        ground_respawns = []
        rewards = {}
        drones, drones_idxs = self.air.get_objects(Drone)
        for drone, position in zip(drones, zip(*drones_idxs)):
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
                
                # Packet is destroyed and has to respawn
                if drone.packet is not None:
                    ground_respawns.append(drone.packet)
                    drone.packet = None
                    
        # Move drones that didn't went outisde the grid
        for position, drones in new_positions.items():
            # Is there a collision?
            if len(drones) > 1:
                # Drones get a negative reward and have to respawn
                rewards.update({drone: -1 for drone in drones})
                air_respawns.extend(drones)
                
                # Packets are destroyed and have to respawn
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
                        rewards[drone] = 1
                        
                        # Respawn packet and dropzone
                        ground_respawns.extend([drone.packet, self.ground[position]])
                        self.ground[position] = None
                        drone.packet = None
                        
        # Respawn objects
        self.ground.spawn(ground_respawns)
        self._pick_packets_after_respawn(self.air.spawn(air_respawns))
        
        # Return new states, rewards, done and other infos
        drones, _ = self.air.get_objects(Drone)
        new_states = {drone.name: self._get_state(drone) for drone in drones if drone.name in actions}
        return new_states, rewards, False, {}
    
    def _get_state(self, drone):
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
        
    def _pick_packets_after_respawn(self, positions):
        for x, y in zip(*positions):
            if isinstance(self.ground[x, y], Packet):
                self.air[x, y].packet = self.ground[x, y]
                self.ground[x, y] = None
        
    def render(self):
        print(self.__str__())
    
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