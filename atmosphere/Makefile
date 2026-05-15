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
#
# Sensitivity analysis:
#   make analyse

.PHONY: all grid ring test-grid test-ring clean
.PHONY: analyse analyse-diffusion analyse-friction analyse-gravity analyse-diffusion-role

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

# --- Ring (altitude×angle, pressure + gravity, convection) ---

test-ring:
	uv run run_ring.py

ring: frames_ring
	nix shell nixpkgs#ffmpeg --command ffmpeg -y -framerate 30 \
		-i frames_ring/f_%04d.png \
		-c:v libx264 -pix_fmt yuv420p ring.mp4
	@echo "Created ring.mp4"

frames_ring:
	uv run animate_ring.py

# --- Sensitivity analysis ---

analyse: analyse-diffusion analyse-friction analyse-gravity analyse-diffusion-role

analyse-diffusion:
	uv run analyse_diffusion.py

analyse-friction:
	uv run analyse_friction.py

analyse-gravity:
	uv run analyse_gravity.py

analyse-diffusion-role:
	uv run analyse_diffusion_role.py

# --- Cleanup ---

clean:
	rm -rf frames_grid frames_ring output_grid output_ring output_analysis *.mp4
