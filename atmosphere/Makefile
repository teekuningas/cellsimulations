# Atmosphere Simulation
#
# Prerequisites: uv, nix (for ffmpeg)
#
# Quick test (snapshots only):
#   make test-grid
#   make test-ring
#
# Full animation (frames + MP4):
#   make grid
#   make ring
#   make all

.PHONY: all grid ring test-grid test-ring clean

all: grid ring

# --- Grid (lat×lon, Coriolis, trade winds) ---

test-grid:
	uv run run_grid.py

grid: frames_grid
	nix shell nixpkgs#ffmpeg --command ffmpeg -y -framerate 30 \
		-i frames_grid/f_%04d.png \
		-c:v libx264 -pix_fmt yuv420p grid.mp4
	@echo "Created grid.mp4"

frames_grid:
	uv run animate_grid.py

# --- Ring (altitude×angle, buoyancy, convection) ---

test-ring:
	uv run run_ring.py

ring: frames_ring
	nix shell nixpkgs#ffmpeg --command ffmpeg -y -framerate 30 \
		-i frames_ring/f_%04d.png \
		-c:v libx264 -pix_fmt yuv420p ring.mp4
	@echo "Created ring.mp4"

frames_ring:
	uv run animate_ring.py

# --- Cleanup ---

clean:
	rm -rf frames_grid frames_ring output_grid output_ring *.mp4
