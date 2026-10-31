# GD AI

An experimental agent that learns Geometry Dash by trial and error.
A **C++ Geode mod** observes the level and applies inputs; a **Python / PyTorch
PPO brain** learns when to press and release. The project includes a standalone
simulator and tests that run without the game.

Developed by **takto**, with assistance from AI coding tools.

## What it does

- Observes three 16 x 11 grids: nearby objects for player 1, nearby objects for
  player 2 in dual mode, and a coarser view further ahead. Each view has separate
  channels for hazards, solids, pads, orbs and portals.
- Uses player state and exact measurements of the five nearest objects as
  additional inputs.
- Chooses a press or release at one of four physics ticks per decision. The
  current bridge uses 240 physics ticks and 60 decisions per game second.
- Learns with PPO and rewards for progress, new ground and completion.
- Uses a practice coach to train difficult sections from checkpoints.
- Saves checkpoints, per-level statistics and replays. Optional recurrent memory,
  replay experiments and old-level reviews are available.

This is a research hobby project. Learning can be slow and inconsistent; there is
no guarantee of completing a particular level or transferring skills between
levels. Game runs controlled by this project are AI-assisted runs.

## Supported environment

| Component | Version used |
| --- | --- |
| OS | Windows x64 |
| Geometry Dash | 2.2081 |
| Geode SDK / loader | 5.11.2 |
| Python | 3.12 |
| PyTorch | 2.14.1, CPU build |
| NumPy | 2.5.4 |

The simulator can run without Geometry Dash or the SDK. The game bridge uses
WinSock and currently supports Windows only. Separate-button two-player levels
are not supported. Older NEAT prototypes are not included.

## Try the Python tests first (no game needed)

From the repository root, using a Windows command prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe tests\test_brain.py
.venv\Scripts\python.exe tests\test_ppo.py
.venv\Scripts\python.exe tests\test_sim_session.py 60
```

The integration check needs at least three completed training rounds. On a slow
machine, increase `60` to `120`. Tests use temporary brain folders and do not
need your trained model. CPU training is the supported default; a GPU is not required.

## Build and install the game bridge

You need your own copy of Geometry Dash, [Geode](https://geode-sdk.org/),
[Geode CLI and SDK](https://github.com/geode-sdk/cli), an x64 Visual Studio C++
developer environment, CMake 3.21 or newer, and Ninja. Use the 5.11.2 SDK and
install its matching prebuilt loader binaries using the Geode tools.

Open an **x64 Visual Studio Developer Command Prompt**, then:

```bat
set GEODE_SDK=C:\path\to\geode-sdk
mod\build.bat
```

The Geode CLI must be on `PATH`. If you maintain a separate bindings checkout,
set `GEODE_BINDINGS_REPO_PATH` to its path before building.
The build script compiles Codegen locally to match the selected SDK and bindings;
using a newer downloaded Codegen with this older SDK can generate incompatible headers.
Compilation uses two parallel jobs to limit peak memory usage.

The build produces `mod/build/takto.gd-ai.geode`; it does not install it into
your game automatically. Close GD, copy that file into the game's `geode/mods/`
directory, and restart. Game files, SDK sources and pretrained models are not
distributed with this repository.

## Run the agent

Back up your GD saves before experimenting. Disable other mods that alter input
timing, physics or collision behavior so that training and evaluation are comparable.

```bat
start-brain.bat
```

Open a level in GD. With **Geode > GD AI > AI enabled** turned on, the brain
controls the inputs as soon as the bridge connects. Turn that setting off to
play manually. Stop the Python process with `Ctrl+C`.

The bridge connects to the Python server on `127.0.0.1:5555` (loopback only).
If the server does not connect, check that the port is available, both pieces
use protocol 4, and the mod loaded successfully. Do not expose this port publicly.

| Command | Behavior |
| --- | --- |
| `start-brain.bat --brain experiment` | Use a separate model directory |
| `start-brain.bat --play` | Play with saved weights without PPO training |
| `start-brain.bat --eval 20` | Evaluate 20 full runs and save a report |
| `start-brain.bat --fresh` | Back up the selected brain and start a new one |
| `start-brain.bat --review` | Periodically revisit previously completed levels |
| `start-brain.bat --memory` | Enable a small recurrent memory |
| `start-brain.bat --experiments 0.3` | Enable replay variations on difficult sections |
| `start-brain.bat --pause-to-train` | Pause the game during PPO updates |
| `start-deadlocked.bat` | Select Deadlocked (main level 20) for training |

Reviews, recurrent memory and replay experiments are off by default. Training
normally runs in the background; `--pause-to-train` selects the synchronous path.
Use `start-brain.bat --help` for the complete CLI.

Checkpoints and logs are stored under `brain/brains2/<brain-name>/`. The Geode
mod keeps its own settings, statistics and replays under its normal save directory
(`%LOCALAPPDATA%\GeometryDash\geode\mods\takto.gd-ai\` on Windows). Saved
data, videos and local logs are excluded from Git. Only load model files you
trust: checkpoint loading currently uses PyTorch's full serialization mode.

## Project layout

```text
brain/brain2.py         PPO policy, training, coach, persistence and TCP server
mod/src/               Game observations, input bridge, statistics and replays
mod/mod.json           Geode metadata and settings
tests/sim.py           Standalone deterministic test environment
tests/test_brain.py    Model upgrades, memory, replay logic and training checks
tests/test_ppo.py      PPO probabilities, returns, updates and regression checks
tests/test_sim_session.py  Tick-by-tick simulator integration check
tests/bench.py         Optional longer learning benchmarks
docs/ARCHITECTURE.md   Data flow and protocol overview
```

Optional analysis tools in `tests/` inspect replay and evaluation data. They are
not part of the automated test suite; `level_map.py` currently assumes a default
Steam game location and reads resources from your own game installation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports are welcome: include the
versions, reproduction steps and relevant sanitized logs. Describe actual
measured results separately from ideas or expected improvements.

## License and credits

Original project source is available under the [MIT License](LICENSE).
Third-party dependencies keep their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
Geometry Dash belongs to RobTop Games. This is an independent community project.
