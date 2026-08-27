"""chad — a local MLX-backed, Claude-Code-style coding agent.

A flat collection of cooperating modules behind one console script (``chad``):
the inference engine, the tool layer, the agent loop, and the terminal UI.
"""

__version__ = "2.0.2"

# chad sets no MLX_* runtime vars. MLX_METAL_FAST_SYNCH, MLX_MAX_OPS_PER_BUFFER
# and MLX_MAX_MB_PER_BUFFER were each measured end-to-end on the 35B and every
# setting was slower than mlx's own defaults (1-4% at both 8k and 32k), so the
# mechanism that used to apply them at package import is gone. Any future entry
# has to be set before mlx's backend initializes — package import is the last
# hook early enough, since chad imports mlx lazily inside functions.
