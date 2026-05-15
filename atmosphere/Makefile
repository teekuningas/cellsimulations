# Atmosphere Simulation
#
# Prerequisites: uv, nix (for ffmpeg)
#
# Quick test (snapshots + console output):
#   make test-ring
#   make test-grid
#   make test-sphere
#
# Full animations (MP4 output in output_animations/):
#   make ring
#   make grid
#   make sphere
#   make all

.PHONY: all ring grid sphere test-ring test-grid test-sphere clean

all: ring grid sphere

# --- Ring (altitude×angle, pressure + gravity, convection cells) ---

test-ring:
	uv run run_ring.py

ring:
	uv run animate_ring.py
	@echo "Created output_animations/ring.mp4"

# --- Grid (lat×lon, Coriolis, jet streams) ---

test-grid:
	uv run run_grid.py

grid:
	uv run animate_grid.py
	@echo "Created output_animations/grid.mp4"

# --- Sphere (3D combined: lat×lon×alt, Coriolis + gravity, wind reversal) ---

test-sphere:
	uv run run_sphere.py

sphere:
	uv run animate_sphere.py
	@echo "Created output_animations/sphere.mp4"

# --- Cleanup ---

clean:
	rm -rf output_animations __pycache__
