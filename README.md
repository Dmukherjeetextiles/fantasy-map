Cellular automata using Random and Noise

https://fantasy-map.streamlit.app/


# Fantasy Map Maker

This repository contains a Streamlit application for generating fantasy maps using Perlin noise. 

## Features

* **Procedural Map Generation:** Generate unique and diverse fantasy maps using Perlin noise.
* **Terrain Control:** Customize terrain features like mountains, forests, plains, and water levels.
* **Digging Simulation:** Simulate a character digging through the map, creating paths and tunnels.
* **Smoothing:** Smooth the terrain to create more realistic and aesthetically pleasing maps.
* **Interactive Controls:** Adjust parameters like noise scale, octaves, persistence, and lacunarity to fine-tune your map.

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Application:**
   ```bash
   streamlit run application.py
   ```

## Usage

Once the application is running, you can interact with the controls on the sidebar to modify the map generation parameters.

* **Width and Height:** Control the dimensions of the map.
* **Digging Amount:** Adjust the number of blocks the character will dig.
* **Noise Scale, Octaves, Persistence, and Lacunarity:** These parameters influence the appearance of the Perlin noise and the resulting terrain.
* **Sea Level, Mountain Level, and Forest Threshold:** Define the height ranges for different terrain types.
* **Smoothing Iterations:** Determine the degree of smoothing applied to the terrain.

**Press the "R" key on your keyboard to generate a new map with random parameters.**

## Deployment

The application is deployed on Streamlit Cloud: [https://fantasy-map.streamlit.app/](https://fantasy-map.streamlit.app/)

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
