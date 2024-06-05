import streamlit as st
import matplotlib.pyplot as plt
from terrain import initialize_level, initialize_character, dig, generate_noise_map, apply_terrain_features, smooth_terrain
import numpy as np

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

    return fig 

def main():
    st.title("Map Terrain Generator")
    st.caption("Press R to Generate a New Map or Play with the Parameters in the Sidebar")

    # Sidebar for parameter controls
    st.sidebar.header("Parameters")
    width = st.sidebar.slider("Width", 50, 250, 150)
    height = st.sidebar.slider("Height", 50, 200, 100)
    wall_countdown = st.sidebar.slider("Digging Amount", 1000, 10000, 8000)
    scale = st.sidebar.slider("Noise Scale", 10, 100, 50)
    octaves = st.sidebar.slider("Noise Octaves", 1, 10, 6)
    persistence = st.sidebar.slider("Noise Persistence", 0.1, 1.0, 0.5)
    lacunarity = st.sidebar.slider("Noise Lacunarity", 1.0, 4.0, 2.0)
    sea_level = st.sidebar.slider("Sea Level", -1.0, 0.0, -0.3)
    mountain_level = st.sidebar.slider("Mountain Level", 0.0, 1.0, 0.5)
    forest_threshold = st.sidebar.slider("Forest Threshold", -0.5, 0.5, 0.1)
    smoothing_iterations = st.sidebar.slider("Smoothing Iterations", 0, 10, 5)

    # Generate the terrain
    level = initialize_level(width, height)
    character = initialize_character(width, height, padding=2, wall_countdown=wall_countdown)
    dig(level, character)

    noise_map = generate_noise_map(width, height, scale, octaves, persistence, lacunarity)
    level = apply_terrain_features(level, noise_map, sea_level, mountain_level, forest_threshold)
    level = smooth_terrain(level, smoothing_iterations)

    # Plot the terrain using matplotlib - get the figure back
    fig = plot_level(level, noise_map)  

    # Display the plot in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    main() 