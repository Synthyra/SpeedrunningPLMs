# SpeedrunningPLMs: how to work here

Read [`program.md`](program.md) first: it holds the benchmark's rules, the files a change may touch, the hook reference
and the search, and your place in the loop. [`CONTRIBUTING.md`](CONTRIBUTING.md) holds every command: install, the
tiny data, the CPU smoke, the tests, adding a module, running a genome and submitting a record attempt. What an agent
needs beyond those two:

- A new idea is a strategy module listed in the registry, never a fork of `train.py`. Smoke every new module on the
  CPU before it goes to the GPU, and run the narrow test selection for its family; the whole suite takes about an hour
  on a laptop CPU.
- You reason on top of the search: read the state and the screening effects, add modules, retire inert ones, and let
  the proposers run. Every run is recorded with its verdict and reason; the verdicts are defined in
  [program.md](program.md#the-loop-by-hand).
- [`ideas/README.md`](ideas/README.md) is the module catalog: record a module's evidence there when it runs.
  [`results.md`](results.md) holds the records, the hand-run candidates and what each track learned. A new record
  changes `results.md` and the record table of the public README (`README.md` on GitHub, `PUBLIC_README.md` in the
  workspace this repository is published from); `tests/test_documents.py` checks that the two agree.
- Python follows [`CODING_STANDARDS.md`](CODING_STANDARDS.md): import order, precise types, intent comments, and a
  shape trace on every tensor line. `candidates/` snapshots are frozen bytes and exempt.

In the workspace this repository is published from (not part of the repository): run the commands of
`CONTRIBUTING.md` as `./ws.cmd run speedrunning_plms -- python ...` from the workspace root; the search states and the
campaign ledgers are in `experiments/representation/speedrunning_plms/` (`2026-10-28_afdb_v6_speedrun` for the
10-minute track, `2026-10-31_afdb_v6_speedrun_1h` for the 1-hour track); and the project's `README.md` is the
workspace's manifest, never published.
