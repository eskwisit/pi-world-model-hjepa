# Changelog

## 0.10.1

### Fixes

- **main:** pass the store audit by loading the model library statically (`155abb9`)

## 0.10.0

### Features

- **lib:** add a hierarchical jepa world model in plain javascript (`885d0fe`)
- **subagents:** add a read-only scout that probes before planning (`1262d99`)
- **scripts:** install the official h-jepa release at pinned versions (`506e0d9`)
- **scripts:** run h-jepa evaluations on an apple gpu or the cpu (`877ba9e`)
- **scripts:** run h-jepa evaluations locally or on a remote nvidia gpu (`652acbe`)
- **main:** register the world model and h-jepa agent tools (`9985691`)
- **skills:** plan, observe and train with the world model (`960609b`)
- **skills:** install and run h-jepa world models from the settings (`864c051`)
- **manifest:** declare tools, skills, settings and workspace file scope (`0506aa1`)

### Documentation

- **readme:** document the world model, h-jepa runs, settings and tests (`d214633`)

### Tests

- **tests:** add shared test helpers with a fake host (`4ef3e95`)
- **tests:** add the node test entry point (`1d28fd0`)
- **tests:** cover the h-jepa model math and learning (`82b631b`)
- **tests:** cover the world model tools (`b44ac87`)
- **tests:** cover plugin load, unload and the scout sub-agent (`c096ca6`)
- **tests:** cover the h-jepa installer, plan and status (`c5b3641`)
- **tests:** cover h-jepa runs locally and over ssh (`32e6e0f`)

### CI

- print the changelog section of a version (`426e190`)
- publish a github release when the manifest version changes (`4d684df`)

### Chores

- ignore build output, workspace data and python caches (`888c532`)
- add the apache 2.0 license (`3a4209a`)
- add the changelog for 0.1.0 (`59ae0e7`)
