import itertools
import os
from collections import defaultdict
from enum import Enum

import gym.spaces as spaces
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gym import Env

assets_path = os.path.dirname(os.path.realpath(__file__))
font_path = os.path.join(assets_path, "assets", "font", "Inconsolata-Bold.ttf")
sprite_path = os.path.join(assets_path, "assets", "16ShipCollection.png")


class Action(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    STAY = 4


class Skyscraper:
    def __repr__(self):
        return '#'


class Station:
    def __repr__(self):
        return '@'


class Dropzone:
    def __repr__(self):
        return 'Z'


class Packet:
    def __repr__(self):
        return 'P'


class Drone():
    def __init__(self, index):
        self.index = index
        self.packet = None
        self.charge = 100

    def __repr__(self):
        return 'D{}'.format(self.index)


class Grid:
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
    metadata = {'render.modes': ['ansi']}

    def __init__(self, env_params={}):
        # Set environment parameters
        self.env_params = {
            'drone_density': 0.05,
            'n_drones': 3,
            'pickup_reward': 0,
            'delivery_reward': 1,
            'crash_reward': -1,
            'charge_reward': -0.1,
            'discharge': 10,
            'charge': 20,
            'packets_factor': 3,
            'dropzones_factor': 2,
            'stations_factor': 2,
            'skyscrapers_factor': 3,
            'rgb_render_rescale': 1.0
        }
        self.env_params.update(env_params)

        # Define spaces
        self.action_space = spaces.Discrete(len(Action))

    def __init_rgb_rendering(self):
        # Load RGBA image
        sprites_img = Image.open(sprite_path)
        sprites_img_array = np.array(sprites_img)

        # Make black background transparent
        black_pixels = (sprites_img_array[:, :, 0] + sprites_img_array[:, :, 1] + sprites_img_array[:, :, 2]) == 0
        sprites_img_array[np.nonzero(black_pixels) + (3,)] = 0

        # Create tiles with the standard objects
        def get_ships_tile(row, col):
            tiles_size, small_padding, big_padding = 16, 4, 10
            top_corner = (42, 28)

            i = top_corner[0] + row * (tiles_size + small_padding)
            j = top_corner[1] + (col % 5) * (tiles_size + small_padding) + (col // 5) * (5 * (tiles_size + small_padding) + big_padding)
            return Image.fromarray(sprites_img_array[i:i + tiles_size, j:j + tiles_size])

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
            shuffled_ships.extend(ships)

        # Create iterator
        ships_iter = itertools.cycle(iter(shuffled_ships))

        # Create drone tiles
        def overlay(img_a, img_b):
            overlay = Image.new('RGBA', [img_a.size[0] + 12, img_a.size[1]])
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
        self.empty_frame[:, :, 3] = 255  # Remove background transparency

        # Side panel
        background_color = (20, 200, 200)
        self.panel = Image.new('RGBA', (120, self.empty_frame.shape[1]), color=background_color)
        draw_handle = ImageDraw.Draw(self.panel, mode='RGBA')
        font = ImageFont.truetype(font_path, 16)

        for i, drone in enumerate(self.drones):
            # Print sprite
            drone_sprite = self.tiles['drone_{}'.format(drone.index)]
            sprite_x = self.render_padding
            sprite_y = i * self.tiles_size + (i + 1) * self.render_padding
            self.panel.paste(drone_sprite, (sprite_x, sprite_y), drone_sprite)

            player_name = 'Player {:>2}'.format(drone.index)
            if "player_name_mappings" in self.env_params.keys():
                """
                    Optional Setting for rendering videos on evaluator
                    -- Non-breaking change
                """
                player_name = self.env_params["player_name_mappings"][drone.index]

            # # Print text
            text_x = sprite_x + self.tiles_size + self.render_padding
            text_y = sprite_y - 1
            text_color = (0, 0, 0)
            draw_handle.text((text_x, text_y), player_name, fill=text_color, font=font)

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
                new_position = position[0], position[1] - 1
            elif action is Action.DOWN:
                new_position = position[0] + 1, position[1]
            elif action is Action.RIGHT:
                new_position = position[0], position[1] + 1
            elif action is Action.UP:
                new_position = position[0] - 1, position[1]
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
                drone.charge = min(100, drone.charge + self.env_params['charge'])  # charge
                rewards[drone.index] = self.env_params['charge_reward']  # cost of charging
            else:
                drone.charge -= self.env_params['discharge']  # discharge
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
        self.side_size = int(np.ceil(np.sqrt(self.env_params['n_drones'] / self.env_params['drone_density'])))
        self.shape = (self.side_size, self.side_size)

        # Create grids
        self.air = Grid(shape=self.shape)
        self.ground = Grid(shape=self.shape)

        # Create elements of the grid
        self.packets = [Packet() for _ in range(self.env_params['packets_factor'] * self.env_params['n_drones'])]
        self.dropzones = [Dropzone() for _ in range(self.env_params['dropzones_factor'] * self.env_params['n_drones'])]
        self.stations = [Station() for _ in range(self.env_params['stations_factor'] * self.env_params['n_drones'])]
        self.skyscrapers = [Skyscraper() for _ in range(self.env_params['skyscrapers_factor'] * self.env_params['n_drones'])]
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

    def render(self, mode='ansi'):
        if mode == 'ansi':
            return self.__str__()
        elif mode == 'rgb_array':
            return self._render_rgb()
        else:
            super().render(mode=mode)

    def _render_rgb(self):
        # Render frame
        frame = Image.fromarray(self.empty_frame.copy())
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Check tile
                ground = self.ground[i, j]
                air = self.air[i, j]

                if (air is None) and (ground is None):
                    continue  # Nothing to draw

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
                    tile_array[nontransparent + (3,)] = int(drone.charge * 255 / 100)
                    tile = Image.fromarray(tile_array)

                # Paste tile on frame
                tile_x = j * self.tiles_size + (j + 1) * self.render_padding
                tile_y = i * self.tiles_size + (i + 1) * self.render_padding
                frame.paste(tile, (tile_x, tile_y), mask=tile)

        frame = Image.fromarray(np.hstack([frame, self.panel]))

        # Rescale frame
        rescale = lambda old_size: int(old_size * self.env_params['rgb_render_rescale'])
        frame = frame.resize(size=(rescale(frame.size[0]), rescale(frame.size[1])), resample=Image.NEAREST)

        return np.array(frame)[:, :, :3]  # RGB

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
                tile_str = ' ' * ((tile_size - len(tile_str)) // 2) + tile_str
                tile_str = tile_str.ljust(tile_size, ' ')
                line_str += tile_str + '|'
            lines.append(line_str)
            lines.append(row_sep)

        return '\n'.join(lines)

    def format_action(i):
        return ['←', '↓', '→', '↑', 'X'][i]
