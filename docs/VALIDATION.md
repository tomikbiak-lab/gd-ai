# Validation record

Checked on 2026-10-31 on Windows x64 with Python 3.12.9,
PyTorch 2.14.1+cpu and NumPy 2.5.4.

| Check | Result |
| --- | --- |
| `python tests/test_brain.py` | Passed all checks |
| `python tests/test_ppo.py` | Passed all PPO checks |
| `python tests/test_sim_session.py 60` | Passed: 390 attempts, 242,761 physics ticks, 8 training rounds |
| CPU dependency resolution from the documented index | Resolved both pinned packages |
| `mod/build.bat` from an x64 VS developer environment | Built and packaged `takto.gd-ai.geode` |

The simulator run verified that the played inputs matched the recorded decisions
tick by tick. It recorded zero normal/practice wins during this short check; this
is an integration result, not a claim that the AI learned the simulated level.

The native build used MSVC 19.52.37025, Geode CLI 3.9.0 and these checkouts:

- Geode SDK 5.11.2: `3909856e8714b503945d5cea12ba1bddf5d5b1b0`
- Geode bindings: `1605f500d053f8de122bf5c7dc6d17c24866e02e`

Codegen was compiled from the selected bindings checkout. The build script limits
parallel compilation to two jobs; unrestricted parallel compilation exhausted
memory on the validation machine. A prebuilt newer Codegen generated headers
incompatible with this SDK, so the script explicitly builds Codegen locally.

No live-game run of this publication snapshot was performed during preparation.
The original working project's models and saved progress were not changed.
