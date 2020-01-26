import numpy as np
from enum import Enum
from collections import defaultdict, OrderedDict, Counter
import gym.spaces as spaces
from gym import Env, ObservationWrapper, RewardWrapper
from gym.utils import seeding
import os
import string
from PIL import Image, ImageDraw, ImageFont
import itertools

dir_path = os.path.dirname(os.path.realpath(__file__))

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
    metadata = {'render.modes': ['ainsi']}
    
    def __init__(self, env_params={}):
        # Set environment parameters
        self.env_params = {
            'drone_density': 0.05, 'n_drones': 3,
            'pickup_reward': 0, 'delivery_reward': 1, 'crash_reward': -1, 'charge_reward': -0.1,
            'discharge': 10, 'charge': 20,
            'packets_factor': 3, 'dropzones_factor': 2, 'stations_factor': 2, 'skyscrapers_factor': 3,
            'rgb_render_rescale': 1.0
        }
        self.env_params.update(env_params)
        
        # Define spaces
        self.action_space = spaces.Discrete(len(Action))
        
    def __init_rgb_rendering(self):
        # Load RGBA image
        sprites_img = Image.open(os.path.join(dir_path, '16ShipCollection.png'))
        sprites_img_array = np.array(sprites_img)

        # Make black background transparent
        black_pixels = (sprites_img_array[:, :, 0] + sprites_img_array[:, :, 1] + sprites_img_array[:, :, 2]) == 0
        sprites_img_array[np.nonzero(black_pixels) + (3,)] = 0

        # Create tiles with the standard objects
        def get_ships_tile(row, col):
            tiles_size, small_padding, big_padding = 16, 4, 10
            top_corner = (42, 28)

            i = top_corner[0] + row*(tiles_size+small_padding)
            j = top_corner[1] + (col%5)*(tiles_size+small_padding) + (col//5) * (5*(tiles_size+small_padding) + big_padding)
            return Image.fromarray(sprites_img_array[i:i+tiles_size, j:j+tiles_size])

        self.tiles = {
            'packet': get_ships_tile(11, 9),
            'dropzone': get_ships_tile(11, 8),
            'station': get_ships_tile(18, 15),
            'skyscraper': get_ships_tile(18, 12)
        }

        # Define list of ships and colors
        ship_types = [(1, 2), (6, 3), (8, 0), (9, 3), (9, 4), (10, 2), (17, 3)]
        ship_colors = [0, 5, 10, 15, 20]

        # Shuffle them
        shuffled_ships_by_color = []
        for color_col in ship_colors:
            # Make sure we alternate between ships
            idx = np.arange(len(ship_types))
            np.random.shuffle(idx)
            shuffled = np.array(ship_types)[idx]
            shuffled[:, 1] += color_col
            shuffled_ships_by_color.append(shuffled.tolist())

        shuffled_ships = []
        for ships in zip(*shuffled_ships_by_color):
            # Make sure we alternate between colors
            shuffled_ships.extend(ships)

        # Create iterator
        ships_iter = itertools.cycle(iter(shuffled_ships))
        
        # Create drone tiles
        def overlay(img_a, img_b):
            overlay = Image.new('RGBA', [img_a.size[0]+12, img_a.size[1]])
            overlay.paste(img_a, (8, 0), img_a)
            overlay.paste(img_b, (0, 0), img_b)
            return overlay
        
        for index in range(self.env_params['n_drones']):
            label = 'drone_{}'.format(index)
            i, j = next(ships_iter)
            self.tiles[label] = get_ships_tile(i, j)
            self.tiles[label + '_packet'] = overlay(self.tiles['packet'], get_ships_tile(i, j))
            self.tiles[label + '_charging'] = overlay(self.tiles['station'], get_ships_tile(i, j))
            self.tiles[label + '_dropzone'] = overlay(self.tiles['dropzone'], get_ships_tile(i, j))

        # Create empty frame
        self.render_padding, self.tiles_size = 8, 16
        frames_size = self.tiles_size * self.shape[0] + self.render_padding * (self.shape[0] + 1)
        self.empty_frame = np.full(shape=(frames_size, frames_size, 4), fill_value=0, dtype=np.uint8)
        self.empty_frame[:, :, 3] = 255 # Remove background transparency
        
        # Side panel
        background_color = (20, 200, 200)
        self.panel = Image.new('RGBA', (120, self.empty_frame.shape[1]), color=background_color)
        annotations_draw = ImageDraw.Draw(self.panel, mode='RGBA')
        font = ImageFont.truetype(os.path.join(dir_path, 'Inconsolata-Bold.ttf'), 16)

        for i, drone in enumerate(self.drones):
            # Print sprite
            drone_sprite = self.tiles['drone_{}'.format(drone.index)]
            sprite_x = self.render_padding
            sprite_y = i * self.tiles_size + (i+1) * self.render_padding
            self.panel.paste(drone_sprite, (sprite_x, sprite_y), drone_sprite)

            # Print text
            text_x = sprite_x + self.tiles_size + self.render_padding
            text_y = sprite_y - 1
            text_color = (0, 0, 0)
            annotations_draw.text((text_x, text_y), 'Player {:>2}'.format(drone.index), fill=text_color, font=font)
        
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
                drone.charge = min(100, drone.charge+self.env_params['charge']) # charge
                rewards[drone.index] = self.env_params['charge_reward'] # cost of charging
            else:
                drone.charge -= self.env_params['discharge'] # discharge
                # Without charge left, drone crashes
                if drone.charge <= 0:
                    air_respawns.append(drone)
                    continue
                    
            # Move the drone and check what's on the ground     
            self.air[position] = drone

            # Take packet if any
            if (drone.packet is None) and isinstance(self.ground[position], Packet):
                rewards[drone.index] = self.env_params['pickup_reward']
                drone.packet = self.ground[position]
                self.ground[position] = None
                
            # Did we just deliver a packet?
            elif (drone.packet is not None) and isinstance(self.ground[position], Dropzone):
                # Pay the drone for the delivery
                rewards[drone.index] = self.env_params['delivery_reward']
                
                # Create new delivery
                ground_respawns.extend([drone.packet, self.ground[position]])
                self.ground[position] = None
                drone.packet = None
                        
        # Handle drone crashes
        for drone in air_respawns:
            # Drone restarts fully charged
            drone.charge = 100
            
            # Packet is destroyed
            rewards[drone.index] = self.env_params['crash_reward']
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
        # Define size of the environment
        self.side_size = int(np.ceil(np.sqrt(self.env_params['n_drones']/self.env_params['drone_density'])))
        self.shape = (self.side_size, self.side_size)
        
        # Create grids
        self.air = Grid(shape=self.shape)
        self.ground = Grid(shape=self.shape)
        
        # Create elements of the grid
        self.packets = [Packet() for _ in range(self.env_params['packets_factor']*self.env_params['n_drones'])]
        self.dropzones = [Dropzone() for _ in range(self.env_params['dropzones_factor']*self.env_params['n_drones'])]
        self.stations = [Station() for _ in range(self.env_params['stations_factor']*self.env_params['n_drones'])]
        self.skyscrapers = [Skyscraper() for _ in range(self.env_params['skyscrapers_factor']*self.env_params['n_drones'])]
        self.drones = [Drone(index) for index in range(self.env_params['n_drones'])]
        
        # Spawn objects
        self.ground.spawn(self.packets)
        self.ground.spawn(self.dropzones)
        self.ground.spawn(self.stations)
        skyscrapers_position = self.ground.spawn(self.skyscrapers)
        self._pick_packets_after_respawn(self.air.spawn(
            self.drones, exclude_positions=skyscrapers_position
        ))
        
        # Initialize elements required for RGB rendering
        self.__init_rgb_rendering()
        
        return self._get_grids()
        
    def render(self, mode='ainsi'):
        if mode == 'ainsi':
            return self.__str__()
        elif mode == 'rgb_array':
            return self.render_rgb()
        else:
            super().render(mode=mode)
            
    def render_rgb(self):
        # Render frame
        frame = Image.fromarray(self.empty_frame.copy())
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Check tile
                ground = self.ground[i, j]
                air = self.air[i, j]

                if (air is None) and (ground is None):
                    continue # Nothing to draw

                if air is None:
                    if isinstance(ground, Packet):
                        tile = self.tiles['packet']
                    elif isinstance(ground, Dropzone):
                        tile = self.tiles['dropzone']
                    elif isinstance(ground, Station):
                        tile = self.tiles['station']
                    elif isinstance(ground, Skyscraper):
                        tile = self.tiles['skyscraper']
                else:
                    # If air is not None, then it's a drone
                    drone = air

                    if drone.packet is None:
                        if ground == None:
                            tile = self.tiles['drone_{}'.format(drone.index)]
                        elif isinstance(ground, Station):
                            tile = self.tiles['drone_{}_charging'.format(drone.index)]
                        elif isinstance(ground, Dropzone):
                            tile = self.tiles['drone_{}_dropzone'.format(drone.index)]
                    else:
                        tile = self.tiles['drone_{}_packet'.format(drone.index)]
                        
                    # Encode charge in drone's transparency
                    tile_array = np.array(tile)
                    nontransparent = np.nonzero(tile_array[:, :, 3])
                    tile_array[nontransparent + (3,)] = int(drone.charge*255/100)
                    tile = Image.fromarray(tile_array)

                # Paste tile on frame
                tile_x = j*self.tiles_size + (j+1)*self.render_padding
                tile_y = i*self.tiles_size + (i+1)*self.render_padding
                frame.paste(tile, (tile_x, tile_y), mask=tile)
            
        frame = Image.fromarray(np.hstack([frame, self.panel]))
            
        # Rescale frame
        rescale = lambda old_size: int(old_size * self.env_params['rgb_render_rescale'])
        frame = frame.resize(size=(rescale(frame.size[0]), rescale(frame.size[1])), resample=Image.NEAREST)
        
        return np.array(frame)[:, :, :3] # RGB
    
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
            '←' : [(0, -1)],
            '↙': [(1, -1)],
            '↓' : [(1, 0)],
            '↘': [(1, 1)],
            '→' : [(0, 1)],
            '↗': [(-1, 1)],
            '↑' : [(-1, 0)],
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
            low=0, high=1, shape=(self.radius*2+1, self.radius*2+1, 6), dtype=np.float)
        
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
        padded_shape = (self.env.shape[0]+2*self.radius, self.env.shape[1]+2*self.radius, 6)
        padded_grid = np.zeros(padded_shape)
        padded_grid[:, :, 5] = 1 # walls
        padded_grid[self.radius:-self.radius, self.radius:-self.radius, :] = grid

        # Return windowed state for each drone
        for drone, (y, x) in self.env.air.get_objects(Drone, zip_results=True):
            states[drone.index] = padded_grid[y:y+2*self.radius+1, x:x+2*self.radius+1, :].copy()
        return states
    
    def format_state(self, s):
        return 'Numpy array {}'.format(s.shape)
