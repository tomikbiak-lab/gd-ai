# Contributing

Start with an issue describing the problem or proposed improvement. For a pull
request, keep the scope focused and explain what behavior changes and why.

Install the dependencies and run the three checks documented in README.md.
Changes to the game bridge also need a Windows build against the supported
Geode SDK. If you change the protocol or observation/action layout, update both
the bridge and brain, test their compatibility and document migration behavior.

For learning changes, compare repeat runs with the same evaluation conditions;
report seeds, budget, environment and input-altering mods. Do not describe one
successful attempt as a reliable completion rate.

Do not commit game assets, API keys, personal logs, checkpoints, recordings,
downloaded SDKs or generated build directories. Use synthetic or sanitized data
for examples. Contributions to this repository are provided under its MIT license.
