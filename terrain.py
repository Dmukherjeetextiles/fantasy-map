import random
import matplotlib.pyplot as plt
import noise
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.neighbors import KDTree
from scipy.interpolate import interp1d

def initialize_level(width, height):
    return [['#'] * width for _ in range(height)]

def initialize_character(width, height, padding, wall_countdown):
    return {
        'wallCountdown': wall_countdown,
        'padding': padding,
        'x': int(width / 2),
        'y': int(height / 2)
    }

def dig(level, character):
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
    offset_x = random.uniform(0, 10000)
    offset_y = random.uniform(0, 10000)

    noise_map = np.zeros((height, width)) # Create noise_map as a NumPy array
    
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
    height = len(level)
    width = len(level[0])

    for y in range(height):
        for x in range(width):
            noise_value = noise_map[y][x]

            if noise_value < sea_level:
                level[y][x] = 'W'
            elif noise_value < mountain_level:
                if noise_value > forest_threshold:
                    level[y][x] = 'F'
                else:
                    level[y][x] = 'P'
            else:
                level[y][x] = 'M'
    return level, noise_map

def smooth_terrain(level, iterations):
    height = len(level)
    width = len(level[0])

    for _ in range(iterations):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighbor_values = [
                    level[y - 1][x - 1], level[y - 1][x], level[y - 1][x + 1],
                    level[y][x - 1], level[y][x + 1],
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

def generate_settlement_name():
    prefixes = ["North", "South", "East", "West", "New", "Old", "Bright", "Dark", "Green", "Silver", "Golden", 
                "Whispering", "Silent", "Hidden", "Sunlit", "Starlit", "Crimson", "Azure", "Iron", "Stone", 
                "Frost", "Ember"]
    suffixes = ["wood", "haven", "gate", "ville", "burgh", "ton", "ford", "brook", "glen", "hollow", "vale",
                "Reach", "Hold", "Keep", "Crest", "Crag", "Falls", "Ridge", "Spire", "Moor", "Fells",
                "Run", "Pass", "Wold", "Downs"]
    middles = ["High", "Low", "Deep", "Red", "White", "Black", "Gray", "Green", "Silver", "Golden", 
               "Blue", "Swift", "Cold", "Burning", "Silent", "Ancient"]
    
    name_parts = [random.choice(prefixes)]
    if random.random() < 0.5: 
        name_parts.append(random.choice(middles))
    name_parts.append(random.choice(suffixes))
    return ' '.join(name_parts)

def generate_settlements(level, noise_map, settlement_density, min_radius, max_radius):
    height = len(level)
    width = len(level[0])

    settlements = []
    for y in range(height):
        for x in range(width):
            if level[y][x] in ('P', 'F'):
                if random.random() < settlement_density:
                    is_overlapping = False
                    for existing_settlement in settlements:
                        distance = ((x - existing_settlement['x'])**2 + (y - existing_settlement['y'])**2)**0.5
                        if distance < existing_settlement['radius'] + max_radius:
                            is_overlapping = True
                            break

                    if not is_overlapping:
                        radius = random.uniform(min_radius, max_radius)
                        name = generate_settlement_name()
                        settlements.append({
                            'x': x,
                            'y': y,
                            'radius': radius,
                            'name': name,
                            'connectivity': int(radius * 5)
                        })
    return settlements

def find_enclosed_points(contour_data, level_value, elevation_map):
    """Finds points enclosed within a contour line without plotting."""
    # Use allsegs instead of collections
    paths = contour_data.allsegs[0]  # Get paths from the first (and only) level
    enclosed_points = []
    for path in paths:
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            if (elevation_map[y1, x1] > level_value) != (elevation_map[y2, x2] > level_value):
                enclosed_points.append((x1, y1))
    return enclosed_points

def a_star_search(level, start, goal, elevation_map, high_points, high_point_penalty=5):
    """A* search with high point avoidance."""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    closed_set = set()
    came_from = {}

    start_node = (start, 0, heuristic(start, goal))
    open_set.append(start_node)

    while open_set:
        current = min(open_set, key=lambda x: x[2])
        current_pos, current_cost, current_heuristic = current

        if current_pos == goal:
            return reconstruct_path(current_pos, came_from)

        open_set.remove(current)
        closed_set.add(current_pos)

        for neighbor in get_neighbors(level, current_pos):
            if neighbor in closed_set:
                continue

            # Calculate cost, penalizing water and high points
            base_cost = 2 if level[neighbor[1]][neighbor[0]] == 'W' else 1
            high_point_cost = high_point_penalty if neighbor in high_points else 0
            tentative_cost = current_cost + base_cost + high_point_cost

            if neighbor in [n[0] for n in open_set]:
                if tentative_cost < next(n[1] for n in open_set if n[0] == neighbor):
                    for i, node in enumerate(open_set):
                        if node[0] == neighbor:
                            open_set[i] = (neighbor, tentative_cost, tentative_cost + heuristic(neighbor, goal))
                            came_from[neighbor] = current_pos
            else:
                open_set.append((neighbor, tentative_cost, tentative_cost + heuristic(neighbor, goal)))
                came_from[neighbor] = current_pos

    return None

def reconstruct_path(current, came_from):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def get_neighbors(level, pos):
    x, y = pos
    neighbors = [
        (x - 1, y), (x + 1, y),
        (x, y - 1), (x, y + 1)
    ]
    valid_neighbors = []
    for neighbor in neighbors:
        nx, ny = neighbor
        if 0 <= nx < len(level[0]) and 0 <= ny < len(level):
            if level[ny][nx] != '#' and level[ny][nx] != 'M':
                valid_neighbors.append(neighbor)
    return valid_neighbors

def generate_roads(settlements, level, elevation_map):
    graph = nx.Graph()
    for settlement in settlements:
        graph.add_node(settlement['name'], pos=(settlement['x'], settlement['y']))

    # 1. Connect Settlements using Minimum Spanning Tree (MST)
    positions = np.array([(s['x'], s['y']) for s in settlements])
    distances = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    mst = nx.minimum_spanning_tree(nx.Graph(distances))

    for i, j in mst.edges():
        settlement1 = settlements[i]
        settlement2 = settlements[j]
        path = a_star_search(level, (settlement1['x'], settlement1['y']),
                            (settlement2['x'], settlement2['y']),
                            elevation_map, high_points=[]) # No high point avoidance for MST
        if path is not None:
            road_type = 'land'
            for px, py in path:
                if level[py][px] == 'W':
                    road_type = 'water'
                    break
            graph.add_edge(settlement1['name'], settlement2['name'], type=road_type, path=path)

    # 2. Identify High Points (without generating a contour plot)
    high_points = []
    for level_value in np.linspace(elevation_map.min(), elevation_map.max(), num=10):  # Check 10 elevation levels
        # Use contourf to find enclosed regions without plotting
        contour_data = plt.contourf(elevation_map, levels=[level_value - 0.01, level_value + 0.01])
        enclosed_points = find_enclosed_points(contour_data, level_value, elevation_map)
        high_points.extend(enclosed_points)
        plt.clf()  # Clear the figure to avoid creating a plot

        # Close the figure to suppress display
        plt.close(plt.gcf())

    # 3. Add Additional Connections (Avoiding High Points)
    settlement_positions = np.array([(s['x'], s['y']) for s in settlements])
    kdtree = KDTree(settlement_positions)

    for i, settlement1 in enumerate(settlements):
        neighbor_indices = kdtree.query_radius([(settlement1['x'], settlement1['y'])], r=40)[0]
        for j in neighbor_indices:
            settlement2 = settlements[j]
            if i != j and not graph.has_edge(settlement1['name'], settlement2['name']):
                distance = ((settlement1['x'] - settlement2['x']) ** 2 +
                            (settlement1['y'] - settlement2['y']) ** 2) ** 0.5

                # Increased connection probability, reduced distance penalty
                connection_probability = (settlement1['connectivity'] * settlement2['connectivity']) / (distance * 5)

                if random.random() < connection_probability:
                    # A* search with high point avoidance
                    path = a_star_search(level, (settlement1['x'], settlement1['y']),
                                        (settlement2['x'], settlement2['y']),
                                        elevation_map, high_points)
                    if path is not None:
                        road_type = 'land'
                        for px, py in path:
                            if level[py][px] == 'W':
                                road_type = 'water'
                                break
                        graph.add_edge(settlement1['name'], settlement2['name'], type=road_type, path=path)

    return graph




def apply_curves_to_path(path, elevation_map, num_control_points=5, smoothing_factor=0.5):
    if len(path) <= 2:
        return path

    # Calculate cumulative distances along the path
    x = np.array([p[0] for p in path])
    y = np.array([p[1] for p in path])
    distances = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    distances = np.insert(distances, 0, 0)  # Insert 0 at the beginning

    # Create control points based on distances
    total_distance = distances[-1]
    control_point_distances = np.linspace(0, total_distance, num=num_control_points)

    # Interpolate x and y coordinates for control points
    control_points_x = interp1d(distances, x, kind='linear')(control_point_distances)
    control_points_y = interp1d(distances, y, kind='linear')(control_point_distances)

    # Adjust control points based on elevation gradient
    for i in range(1, num_control_points - 1):
        cx, cy = int(round(control_points_x[i])), int(round(control_points_y[i]))
        gradient_x = (elevation_map[cy][cx + 1] - elevation_map[cy][cx - 1]) / 2
        gradient_y = (elevation_map[cy + 1][cx] - elevation_map[cy - 1][cx]) / 2
        control_points_x[i] += smoothing_factor * gradient_x
        control_points_y[i] += smoothing_factor * gradient_y

    # Create a cubic spline interpolation function
    f_x = interp1d(control_point_distances, control_points_x, kind='cubic')
    f_y = interp1d(control_point_distances, control_points_y, kind='cubic')

    # Generate the new curved path
    new_path_distances = np.linspace(0, total_distance, num=len(path))
    new_x = f_x(new_path_distances)
    new_y = f_y(new_path_distances)
    curved_path = [(int(round(x)), int(round(y))) for x, y in zip(new_x, new_y)]
    return curved_path


def plot_level(level, noise_map, settlements, roads_graph, elevation_map):
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

    # Plot settlements
    existing_texts = []
    for settlement in settlements:
        x = settlement['x']
        y = settlement['y']
        radius = settlement['radius']

        circle = plt.Circle((x, y), radius, facecolor='white', edgecolor='black', linewidth=1)
        ax.add_patch(circle)

        font_size = int(radius * 10)

        possible_positions = [
            (x, y + 2),
            (x, y - 2),
            (x + 2, y),
            (x - 2, y),
            (x + 2, y + 2),
            (x + 2, y - 2),
            (x - 2, y + 2),
            (x - 2, y - 2)
        ]

        possible_positions.sort(key=lambda pos: ((pos[0] - x)**2 + (pos[1] - y)**2)**0.5)

        text_x, text_y = None, None
        for pos in possible_positions:
            if ((pos[0] - x)**2 + (pos[1] - y)**2)**0.5 > radius and 0 <= pos[0] < width and 0 <= pos[1] < height:
                is_overlapping = False
                for existing_text_x, existing_text_y in existing_texts:
                    if abs(pos[0] - existing_text_x) < font_size and abs(pos[1] - existing_text_y) < font_size:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    text_x, text_y = pos
                    break

        if text_x is not None and text_y is not None:
            ax.text(text_x, text_y, settlement['name'], color='white', fontsize=font_size,
                    bbox={'facecolor': 'black', 'edgecolor': 'black', 'alpha': 1, 'pad': 0.5}, ha='center', va='center')
            existing_texts.append((text_x, text_y))

        else:
            vertical_text_y = y - font_size * len(settlement['name']) / 2
            is_overlapping = False
            for existing_text_x, existing_text_y in existing_texts:
                if abs(x - existing_text_x) < font_size and abs(vertical_text_y - existing_text_y) < font_size * len(
                        settlement['name']):
                    is_overlapping = True
                    break

            if not is_overlapping:
                ax.text(x, vertical_text_y, settlement['name'], color='white', fontsize=font_size, rotation=90,
                        bbox={'facecolor': 'black', 'edgecolor': 'black', 'alpha': 1, 'pad': 0.5}, ha='center',
                        va='center')
                existing_texts.append((x, vertical_text_y))

    # Plot roads with dotted lines for water sections
    for u, v, data in roads_graph.edges(data=True):
        if 'path' in data:  # Check if the path exists
            path = data['path']
            x_coords, y_coords = zip(*path)
            if data['type'] == 'water':
                ax.plot(x_coords, y_coords, color='gray', linestyle='dotted', linewidth=2)
            else:
                curved_path = apply_curves_to_path(path, elevation_map)
                ax.plot(*zip(*curved_path), color='gray', linewidth=2)

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

    sea_level = 0.03
    mountain_level = 0.5
    forest_threshold = 0.1

    level = initialize_level(width, height)
    character = initialize_character(width, height, padding, wall_countdown)
    dig(level, character)

    settlement_density = 0.002
    min_radius = 0.5
    max_radius = 1.0

    with tqdm(total=6) as pbar:
        pbar.set_description("Generating World")

        noise_map = generate_noise_map(width, height, scale, octaves, persistence, lacunarity)
        pbar.update(1)

        level, elevation_map = apply_terrain_features(level, noise_map, sea_level, mountain_level, forest_threshold)
        pbar.update(1)

        level = smooth_terrain(level, 5)
        pbar.update(1)

        settlements = generate_settlements(level, noise_map, settlement_density, min_radius, max_radius)
        pbar.update(1)

        roads_graph = generate_roads(settlements, level, elevation_map)
        pbar.update(1)

        plot_level(level, noise_map, settlements, roads_graph, elevation_map)
        pbar.update(1)


if __name__ == "__main__":
    main()
