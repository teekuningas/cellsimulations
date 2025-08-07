# Python Cellular Automata and Grid-Based Simulations

This repository is a collection of classic cellular automata and other grid-based simulations, refactored into a clean, reusable Python library.

## Project Structure

The repository is organized into a core library and several examples:

-   `automaton/`: This directory contains the core library.
    -   `common.py`: Defines the basic data structures like `State` and `Cell`.
    -   `rules.py`: Contains the logic for different simulation rules (e.g., `Rule1D`, `RuleGameOfLife`, `RuleBlock`, `RuleDiffusion`).
    -   `visualization.py`: Provides functions for plotting and animating the simulations.

-   `classic/`: Contains examples of classic cellular automata.
    -   `cellautomaton_1d.py`: A 1D elementary cellular automaton. You can change the `rule_number` to see different patterns.
    -   `cellautomaton_2d.py`: A 2D cellular automaton implementing Conway's Game of Life.
    -   `cellautomaton_2d_block.py`: A 2D block automaton implementing the HPP gas model.

-   `diff/`: An example of a continuous automaton that simulates diffusion.

-   `wind/`: A more complex application that simulates a simplified 2D atmosphere.

-   `simd_block/`: Contains a C implementation of the block automaton as a performance experiment. See the `README` in that directory for instructions on how to compile and use it.

## How to Run the Examples

First, ensure you have the necessary dependencies installed:

```bash
pip install numpy matplotlib scipy
```

Then, you can run any of the examples from the root directory of the project. For example:

```bash
# Run the 1D automaton (Rule 110)
python classic/cellautomaton_1d.py

# Run Conway's Game of Life
python classic/cellautomaton_2d.py

# Run the HPP gas model
python classic/cellautomaton_2d_block.py

# Run the diffusion simulation
python diff/diff_main.py

# Run the wind simulation
python wind/wind_main.py
```

Each script will generate an animation of the simulation. The 1D automaton will also save a PDF of its history to the `output/` directory.
