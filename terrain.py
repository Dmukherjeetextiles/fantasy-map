import random
import matplotlib.pyplot as plt
import noise
import numpy as np

def initialize_level(width, height):
    """Initialize the level with walls represented by '#'."""
    def level_row():
        return ['#'] * width
    return [level_row() for _ in range(height)]

def initialize_character(width, height, padding, wall_countdown):
    """Initialize the character with the given parameters."""
    return {
        'wallCountdown': wall_countdown,
        'padding': padding,
        'x': int(width / 2),
        'y': int(height / 2)
    }

def dig(level, character):
    """Simulate the digging process."""
    while character['wallCountdown'] > 0:
        x = character['x']
        y = character['y']

        if level[y][x] == '#':
            level[y][x] = ' '
            character['wallCountdown'] -= 1
        
        traverse = random.randint(1, 4)

        if traverse == 1 and x > character['padding']:
            character['x'] -= 1
        elif traverse == 2 and x < len(level[0]) - 1 - character['padding']:
            character['x'] += 1
        elif traverse == 3 and y > character['padding']:
            character['y'] -= 1
        elif traverse == 4 and y < len(level) - 1 - character['padding']:
            character['y'] += 1

def generate_noise_map(width, height, scale, octaves, persistence, lacunarity):
    """Generates a 2D Perlin noise map with a random offset."""
    offset_x = random.uniform(0, 10000)
    offset_y = random.uniform(0, 10000)

    noise_map = [[0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            noise_map[y][x] = noise.pnoise2(
                (x / scale) + offset_x, 
                (y / scale) + offset_y, 
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0
            )
    return noise_map

def apply_terrain_features(level, noise_map, sea_level, mountain_level, forest_threshold):
    """Applies terrain features based on noise values."""
    height = len(level)
    width = len(level[0])

    for y in range(height):
        for x in range(width):
            noise_value = noise_map[y][x]

            if noise_value < sea_level:
                level[y][x] = 'W'  # Water
            elif noise_value < mountain_level:
                if noise_value > forest_threshold:
                    level[y][x] = 'F'  # Forest
                else:
                    level[y][x] = 'P'  # Plains
            else:
                level[y][x] = 'M'  # Mountain

    return level

def smooth_terrain(level, iterations):
    """Smooths the terrain by averaging neighbor values."""
    height = len(level)
    width = len(level[0])

    for _ in range(iterations):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighbor_values = [
                    level[y - 1][x - 1], level[y - 1][x], level[y - 1][x + 1],
                    level[y][x - 1],                level[y][x + 1],
                    level[y + 1][x - 1], level[y + 1][x], level[y + 1][x + 1]
                ]
                if level[y][x] != '#' and level[y][x] != 'W':
                    if neighbor_values.count('M') > 4:
                        level[y][x] = 'M'
                    elif neighbor_values.count('F') > 5:
                        level[y][x] = 'F'
                    elif neighbor_values.count('P') > 6:
                        level[y][x] = 'P'

    return level

def plot_level(level, noise_map):
    """Plots the level with gradient colors and contour lines."""
    height = len(level)
    width = len(level[0])

    color_map = {
        '#': (0.2, 0.2, 0.2),
        ' ': (1.0, 1.0, 1.0),
        'W': (0.2, 0.5, 1.0), 
        'M': (0.5, 0.5, 0.5), 
        'F': (0.2, 0.7, 0.2),
        'P': (0.9, 0.8, 0.6)
    }

    rgb_values = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            terrain_type = level[y][x]
            base_color = color_map[terrain_type]

            noise_value = noise_map[y][x]
            shade_factor = (noise_value + 1) / 2
            shaded_color = tuple(c * shade_factor for c in base_color)

            rgb_values[y, x, :] = shaded_color

    fig, ax = plt.subplots()
    im = ax.imshow(rgb_values)

    contour_levels = 10
    contour_colors = 'k'
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    CS = ax.contour(X, Y, noise_map, levels=contour_levels, colors=contour_colors, linewidths=0.5)

    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    width, height = 150, 100
    wall_countdown = 8000
    padding = 2

    scale = 50
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    sea_level = -0.3
    mountain_level = 0.5
    forest_threshold = 0.1

    level = initialize_level(width, height)
    character = initialize_character(width, height, padding, wall_countdown)
    dig(level, character)

    noise_map = generate_noise_map(width, height, scale, octaves, persistence, lacunarity)
    level = apply_terrain_features(level, noise_map, sea_level, mountain_level, forest_threshold)
    level = smooth_terrain(level, 5) 

    plot_level(level, noise_map) 

if __name__ == "__main__":
    main()