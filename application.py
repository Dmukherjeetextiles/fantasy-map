import streamlit as st
import matplotlib.pyplot as plt
from terrain import initialize_level, initialize_character, dig, generate_noise_map, apply_terrain_features, smooth_terrain, generate_settlements, generate_roads
from scipy.interpolate import interp1d
import numpy as np

def apply_curves_to_path(path, elevation_map, is_water_path=False, num_control_points=5, smoothing_factor=0.5):
    """Applies curves to a path based on elevation.

    Args:
        path (list): List of (x, y) tuples representing the path.
        elevation_map (list): 2D array representing elevation.
        is_water_path (bool): If True, curve follows water contours.
        num_control_points (int): Number of control points for the curve.
        smoothing_factor (float): Strength of the curve.
    """
    if len(path) <= 2:
        return path

    x = np.array([p[0] for p in path])
    y = np.array([p[1] for p in path])
    distances = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    distances = np.insert(distances, 0, 0)

    total_distance = distances[-1]
    control_point_distances = np.linspace(0, total_distance, num=num_control_points)

    control_points_x = interp1d(distances, x, kind='linear')(control_point_distances)
    control_points_y = interp1d(distances, y, kind='linear')(control_point_distances)

    for i in range(1, num_control_points - 1):
        cx, cy = int(round(control_points_x[i])), int(round(control_points_y[i]))

        left_x = max(0, cx - 1)
        right_x = min(len(elevation_map[0]) - 1, cx + 1)
        top_y = max(0, cy - 1)
        bottom_y = min(len(elevation_map) - 1, cy + 1)

        if is_water_path:
            # For water paths, follow water contours
            gradient_x = (elevation_map[top_y][cx] - elevation_map[bottom_y][cx]) / 2
            gradient_y = (elevation_map[cy][right_x] - elevation_map[cy][left_x]) / 2
        else:
            # For land paths, use regular elevation-based smoothing
            gradient_x = (elevation_map[cy][right_x] - elevation_map[cy][left_x]) / 2
            gradient_y = (elevation_map[bottom_y][cx] - elevation_map[top_y][cx]) / 2

        elevation_diff = abs(
            elevation_map[cy][cx] - elevation_map[int(control_points_y[i + 1])][int(control_points_x[i + 1])]
        )
        adjusted_smoothing_factor = smoothing_factor * (1 + elevation_diff)

        control_points_x[i] += adjusted_smoothing_factor * gradient_x
        control_points_y[i] += adjusted_smoothing_factor * gradient_y

    f_x = interp1d(control_point_distances, control_points_x, kind='cubic')
    f_y = interp1d(control_point_distances, control_points_y, kind='cubic')

    new_path_distances = np.linspace(0, total_distance, num=len(path))
    new_x = f_x(new_path_distances)
    new_y = f_y(new_path_distances)
    curved_path = [(int(round(x)), int(round(y))) for x, y in zip(new_x, new_y)]
    return curved_path

def is_coastal(level, x, y, max_distance=2):
    """Checks if a point is within a certain distance from water."""
    height = len(level)
    width = len(level[0])

    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and level[ny][nx] == 'W':
                return True
    return False

import streamlit as st
import matplotlib.pyplot as plt
from terrain import initialize_level, initialize_character, dig, generate_noise_map, apply_terrain_features, smooth_terrain, generate_settlements, generate_roads
from scipy.interpolate import interp1d
import numpy as np

def apply_curves_to_path(path, elevation_map, is_water_path=False, num_control_points=5, smoothing_factor=0.5):
    """Applies curves to a path based on elevation.

    Args:
        path (list): List of (x, y) tuples representing the path.
        elevation_map (list): 2D array representing elevation.
        is_water_path (bool): If True, curve follows water contours.
        num_control_points (int): Number of control points for the curve.
        smoothing_factor (float): Strength of the curve.
    """
    if len(path) <= 2:
        return path

    x = np.array([p[0] for p in path])
    y = np.array([p[1] for p in path])
    distances = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    distances = np.insert(distances, 0, 0)

    total_distance = distances[-1]
    control_point_distances = np.linspace(0, total_distance, num=num_control_points)

    control_points_x = interp1d(distances, x, kind='linear')(control_point_distances)
    control_points_y = interp1d(distances, y, kind='linear')(control_point_distances)

    for i in range(1, num_control_points - 1):
        cx, cy = int(round(control_points_x[i])), int(round(control_points_y[i]))

        left_x = max(0, cx - 1)
        right_x = min(len(elevation_map[0]) - 1, cx + 1)
        top_y = max(0, cy - 1)
        bottom_y = min(len(elevation_map) - 1, cy + 1)

        if is_water_path:
            # For water paths, follow water contours
            gradient_x = (elevation_map[top_y][cx] - elevation_map[bottom_y][cx]) / 2
            gradient_y = (elevation_map[cy][right_x] - elevation_map[cy][left_x]) / 2
        else:
            # For land paths, use regular elevation-based smoothing
            gradient_x = (elevation_map[cy][right_x] - elevation_map[cy][left_x]) / 2
            gradient_y = (elevation_map[bottom_y][cx] - elevation_map[top_y][cx]) / 2

        elevation_diff = abs(
            elevation_map[cy][cx] - elevation_map[int(control_points_y[i + 1])][int(control_points_x[i + 1])]
        )
        adjusted_smoothing_factor = smoothing_factor * (1 + elevation_diff)

        control_points_x[i] += adjusted_smoothing_factor * gradient_x
        control_points_y[i] += adjusted_smoothing_factor * gradient_y

    f_x = interp1d(control_point_distances, control_points_x, kind='cubic')
    f_y = interp1d(control_point_distances, control_points_y, kind='cubic')

    new_path_distances = np.linspace(0, total_distance, num=len(path))
    new_x = f_x(new_path_distances)
    new_y = f_y(new_path_distances)
    curved_path = [(int(round(x)), int(round(y))) for x, y in zip(new_x, new_y)]
    return curved_path

def is_coastal(level, x, y, max_distance=2):
    """Checks if a point is within a certain distance from water."""
    height = len(level)
    width = len(level[0])

    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and level[ny][nx] == 'W':
                return True
    return False

def plot_level(level, noise_map, settlements, roads_graph, elevation_map):
    """Plots the level with gradient colors, contour lines, settlements, and roads,
    ensuring water is always behind land.
    """
    height = len(level)
    width = len(level[0])

    color_map = {
        'W': (0.2, 0.5, 1.0),  # Water
        'M': (0.5, 0.5, 0.5),  # Mountain
        'F': (0.2, 0.7, 0.2),  # Forest
        'P': (0.9, 0.8, 0.6)   # Path
    }

    rgb_values = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            terrain_type = level[y][x]
            base_color = color_map.get(terrain_type, (1.0, 1.0, 1.0))  # Default to white if type not in map
            noise_value = noise_map[y][x]
            shade_factor = (noise_value + 1) / 2
            shaded_color = tuple(c * shade_factor for c in base_color)
            rgb_values[y, x, :] = shaded_color

    fig, ax = plt.subplots(facecolor='white')

    # 1. Plot Water (zorder=0)
    water_mask = np.array([[cell == 'W' for cell in row] for row in level])
    x_coords = np.arange(width)
    for y in range(height):
        ax.fill_between(x_coords, y, y + 1, where=water_mask[y, :], color=(0.2, 0.5, 1.0), alpha=0.5, zorder=0)

    # 2. Plot Land (zorder=1)
    im = ax.imshow(rgb_values, zorder=1)

    # Optional: Contour lines
    contour_levels = 10
    contour_colors = 'k'
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    CS = ax.contour(X, Y, noise_map, levels=contour_levels, colors=contour_colors, linewidths=0.5)

    coastal_settlements = []
    for settlement in settlements:
        x, y = int(settlement['x']), int(settlement['y'])
        if is_coastal(level, x, y):
            coastal_settlements.append(settlement)

    # 3. Plot Roads (zorder=2)
    for u, v, data in roads_graph.edges(data=True):
        if 'path' in data:
            path = data['path']
            settlement1 = next(s for s in settlements if s['name'] == u)
            settlement2 = next(s for s in settlements if s['name'] == v)
            crosses_water = any(level[y][x] == 'W' for x, y in path)

            if crosses_water:
                curved_path = apply_curves_to_path(path, elevation_map, is_water_path=True)
            else:
                curved_path = apply_curves_to_path(path, elevation_map)

            x_coords, y_coords = zip(*curved_path)
            line_style = 'dotted' if crosses_water else '-'
            ax.plot(x_coords, y_coords, color='gray', linestyle=line_style, linewidth=2, zorder=2)

    # 4. Plot Settlement Circles (zorder=3)
    for settlement in settlements:
        x = settlement['x']
        y = settlement['y']
        radius = settlement['radius']
        circle = plt.Circle((x, y), radius, facecolor='white', edgecolor='black', linewidth=1, zorder=3)
        ax.add_patch(circle)

    ax.set_aspect('auto')
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout(pad=0)
    plt.close(plt.gcf())
    return fig


def main():
    st.title("Fantasy Map Maker")
    st.caption("Click the button below to Generate a New Map or Play with the Parameters in the Sidebar")

    st.sidebar.header("Parameters")
    width = st.sidebar.slider("Width", 50, 250, 150)
    height = st.sidebar.slider("Height", 50, 200, 100)
    wall_countdown = st.sidebar.slider("Digging Amount", 1000, 10000, 8000)
    scale = st.sidebar.slider("Noise Scale", 10, 100, 50)
    octaves = st.sidebar.slider("Noise Octaves", 1, 10, 6)
    persistence = st.sidebar.slider("Noise Persistence", 0.1, 1.0, 0.5)
    lacunarity = st.sidebar.slider("Noise Lacunarity", 1.0, 4.0, 2.0)
    sea_level = st.sidebar.slider("Sea Level", -1.0, -0.01, -0.3)
    mountain_level = st.sidebar.slider("Mountain Level", 0.0, 1.0, 0.5)
    forest_threshold = st.sidebar.slider("Forest Threshold", -0.5, 0.5, 0.1)
    smoothing_iterations = st.sidebar.slider("Smoothing Iterations", 0, 10, 5)

    settlement_density = st.sidebar.slider("Settlement Density", 0.001, 0.01, 0.002)
    min_radius = st.sidebar.slider("Min Settlement Radius", 0.1, 2.0, 0.5)
    max_radius = st.sidebar.slider("Max Settlement Radius", 0.1, 2.0, 1.0)

    if st.button("Generate Map") or 'generated' not in st.session_state:
        st.session_state.generated = True
        level = initialize_level(width, height)
        character = initialize_character(width, height, padding=2, wall_countdown=wall_countdown)
        dig(level, character)

        noise_map = generate_noise_map(width, height, scale, octaves, persistence, lacunarity)
        level, elevation_map = apply_terrain_features(level, noise_map, sea_level, mountain_level, forest_threshold)
        level = smooth_terrain(level, smoothing_iterations)

        settlements = generate_settlements(level, noise_map, settlement_density, min_radius, max_radius)
        roads_graph = generate_roads(settlements, level, elevation_map)

        fig = plot_level(level, noise_map, settlements, roads_graph, elevation_map)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
