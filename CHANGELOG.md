# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1](https://github.com/alphal00p/spenso/compare/v0.2.0...v0.2.1) - 2024-07-12

### Added
- :sparkles: bump version
- *(symbolica)* :alembic: evaluate now compiles, but fails. Need to have evaluate return a result.
- *(symbolica)* :sparkles: EvalTensors, using the same API as symbolica
- *(symbolica)* :sparkles: evaluator tensor from levels of nested tensor networks
- :building_construction: make levels use the new with_map shadowing.
- *(symbolica)* :sparkles: Added tooling for nested evaluate
- fix test
- *(iterators)* :sparkles: added IteratableTensor trait
- *(symbolica)* :sparkles: allow other arguments in function view when converting back and forth
- *(network)* :sparkles: added a scalar field to tensor networks to enable good parsing of symbolica mul
- *(symbolica)* :arrow_up: update to symbolica main
- *(network)* :sparkles: added generic evaluate to fully parametric tensor networks
- *(network)* :sparkles: added a type homogenising function to tensor net
- *(CI)* :sparkles: Codecov
- *(iterators)* :sparkles: lending mut iterators

### Fixed
- :bug: fix compilation error in 1.79
- :bug: fix complex arithmetic
- *(symbolica)* :bug: fix wrapped versions aswell
- *(symbolica)* :bug: fix ufo preprocessing to add the Abstractid function
- :arrow_up: up to date with current symbolica
- :bug: should now work with gammaloop
- *(CI)* :bug: fixed formatting to make nix flake check work
- *(CI)* :bug: fix environment var setting and improve complex
- *(clippy)* :rotating_light: fix clipper warnings and fixed feature flag cfg
- remove prints
- *(CI)* :bug: removed result dir
- *(CI)* :bug: added codecov token
- *(CI)* :bug: moved coverage to its own derivation
- *(CI)* :bug: let tarpaulin test with symbolica, and removed the pres test
- *(CI)* test new codecov
- *(test)* :rewind: revert failingr
- *(CI)* :pencil2: typo again
- *(CI)* :bug: typo
- *(test)* :bug: put intoid under feature flag
- *(test)* :bug: added feature flag on parametric tensor tests
- logo points to raw githubusercontent

### Other
- added scalar arithmetic on complex
- *(test)* :bug: All benchmarks now run
- :truck: rename refzero to avoid conflicts
- *(symbolica)* back to crates symbolica
- *(contraction)* :sparkles: added refzero trait and standardized contraction bounds
- *(symbolica)* :lock: check license
- *(contraction)* :recycle: new contractable with trait, simplifies traits
- upgrade actions
- added codecov badge
- Update rust.yml
- Update rust.yml
- modify linguist classification
- *(CI)* try to upload coverage
- :memo: added presentation for FORM and Symbolica dev conf
- *(CI)* :zap: remove benchmarks from test action
- test(test):
- *(CI)* :sparkles: use nix
- release
- *(iterators)* :white_check_mark: added mutating iterator test
- Modified README such that the logo shows up on crates.io
- Update release-plz.yml
- Update CHANGELOG.md
- release
- Update release-plz.yml
- Update release-plz.yml
- Update release-plz.yml
- Create dependabot.yml
- Update release-plz.yml
- Update release-plz.yml
- Update release-plz.yml
- Create release-plz.yml
- update logo
- added docs bagde and repo to cargo
- Update README.md
- add logo
- Delete spenso.rs.svg
- Delete spenso.png
- added authors, license and description
- can use symbolica 0.5.0
- clippy fix
- made symbolica optional
- trying the benchmarks
- only tensor net fails
- non permuted works
- sparse dense works
- single iterators work

## [0.2.0](https://github.com/alphal00p/spenso/compare/v0.1.1...v0.2.0) - 2024-05-24

### Added
- *(iterators)* :sparkles: lending mut iterators

### Fixed
- *(test)* :bug: put intoid under feature flag
- *(test)* :bug: added feature flag on parametric tensor tests
- logo points to raw githubusercontent

### Other
- *(iterators)* :white_check_mark: added mutating iterator test
- Modified README such that the logo shows up on crates.io

## [0.1.1](https://github.com/alphal00p/spenso/compare/v0.1.0...v0.1.1) - 2024-05-20

### Other
- Added release-plz action
- Update logo
- Update README.md


## [1.0.0] - 2024-05-20

Initial release
