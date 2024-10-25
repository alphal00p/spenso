# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/alphal00p/spenso/compare/v0.2.0...v0.3.0) - 2024-10-25

### Added

- *(symbolica)* :sparkles: user defined concretizations and representations
- *(symbolica)* :arrow_up: update symbolica to 0.12
- *(rust)* :sparkles: more arithmetic on serializable atoms
- *(symbolica)* :arrow_up: update to 0.11 symbolica
- *(network)* :sparkles: rich graph output for network
- *(arithmetic)* :sparkles: neg assign for dense
- *(symbolica)* :sparkles: iterative hornering
- *(symbolica)* :sparkles: added compatibility traits for spenso complex
- *(symbolica)* :sparkles: better serde atoms and more forwarded api
- *(symbolica)* :sparkles: add replace repeat multiple to serializable atom
- *(symbolica)* :sparkles: SerializableCompiledEvaluators
- *(structures)* :sparkles: derive more stuff and make IntoSymbol more generic
- *(symbolica)* :sparkles: implement display on serializable atom
- *(network)* :sparkles: implement serialize for more types, using SerializableAtom struct
- *(symbolica)* :sparkles: TensorSet enum for scalars or tensors along with multiple new structs to work with gammaloop
- *(iterators)* :sparkles: add lending iterator based sum
- *(structures)* :sparkles: impl hash for structures
- *(structures)* :sparkles: impl hash for paramtensor
- *(symbolica)* :sparkles: remove default for map_coeff
- *(structures)* :sparkles: new scalar trait and TensorEvalSets for combined eval
- *(symbolica)* :sparkles: impl floatcomparison for complex (not very well though)
- *(network)* :sparkles: more derives
- *(network)* :sparkles: horner_scheme
- *(network)* :sparkles: compile asm for set
- *(network)* :sparkles: add result to set, and derived asmuch as possible
- *(network)* :sparkles: networksets that share the evaluator
- *(symbolica)* :sparkles: implement real on my complex and forwarded more of the evaluate API
- :sparkles: added AtomStructure to simplify type
- *(symbolica)* :sparkles: pattern replacement trait for paramtensor and forwarding to tensornet
- *(symbolica)* :sparkles: asm compilation, benches work
- *(symbolica)* added benchmarks
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

- fix smart result tensor and fix clippy warnings
- fix non Rep atom parsing in the dual case
- fix concretization
- *(symbolica)* :bug: fix feature flag separation
- *(network)* :bug: remove print in atom parser
- *(ufo)* :bug: fix concrete form of gamma
- *(symbolica)* :bug: fix addview
- *(rust)* :bug: fix network parsing of sums with respect to tensors
- *(contraction)* :bug: multiply in the scalar if it exists, when getting the resulting tensor
- *(symbolica)* :arrow_up: upgrade symbolica
- *(symbolica)* :arrow_up: upgrade to crates symbolica
- *(symbolica)* :arrow_up: update symbolica and forward api
- *(iterators)* :ambulance: fixed deref
- *(symbolica)* :bug: fix CompileEvalFloat trait impl
- *(symbolica)* :bug: update traits and fix node merger for halfedge graph
- *(symbolica)* :bug: fix asm compilation
- *(benchmarks)* fixed benchmark to use the f64 fn_map
- :rotating_light: fix clippy warnings
- *(network)* :bug: fix addview handling when no tensors are involved
- *(gammaloop)* :bug: fix ufo prepocessing var names to be consistent upper and lower
- *(gammaloop)* :ambulance: fix recursion for name
- *(gammaloop)* :bug: fix dual printing
- *(gammaloop)* :bug: fix baserepnames
- *(gammaloop)* :bug: fix ufo preprocessing to be able to set upper or lower indices
- *(gammaloop)* :bug: fixed numerator parsing
- *(test)* :bug: fixed approxeq
- *(test)* :white_check_mark: add full benchmarking with approx asserts
- *(symbolica)* :bug: fix ufo preprocessing and added linearized bench
- *(symbolica)* :zap: added optimisations, precontraction yields an improvement from 4ms to 10mus (x1000!)
- [**breaking**] :bug: fix ufo preprocessor#
- *(CI)* :ambulance: fix lockfile
- *(build)* :building_construction: fix feature cfg
- *(symbolica)* :bug: fix tensornetwork parsing
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

- all features pass
- pin release-plz to fix ci bug
- cargo update and clippy fixes
- make nix flake check pass
- added fallible maps and made tensor network parsing structure generic
- Allow rep type to be totally generic when parsing
- actually implement is_neg for minkowski in PhysReps enum
- update to symbolica 0.13
- added a to_dense to all datatensors
- actually make auto trace work
- auto trace when parsing network
- All tests pass from both feature flag settings
- correctly handle symbolic dimension
- much more robust slot parsing
- concretize using mink
- add minkowski rep, and remove minus sign for loru and lord
- remove prints
- refactor structure.rs
- allow arbitrary integer (pos) powers when parsing tensor network
- mixedtensor setable
- implement GetData for RealOrComplexTensor
- Modified the GetData trait to allow for enums with lifetimes
- Turn mixed tensor scalar into atom if the type allows
- scalar of mixed tensor can be built from atoms
- Added a map_scalar to TensorNetworks
- Fixed tests to see if CI passes.
- Map for paramtensor
- Fix mul_add complex implementation to the correct convention.
- update to 0.12.1
- kroneker structures
- fully switch to weyl rep for ufo. (sigma still missing)
- Merge branch 'master' into symbolica_nightly
- make spenso compile with new symbolica
- *(symbolica)* [**breaking**] :arrow_up: update symbolica api
- :recycle: remove reexports
- Merge branch 'update_to_symb_08'
- *(symbolica)* :recycle: refactor paramtensor to a struct with empty enums
- :rotating_light: fix all clippy warnings
- Merge branch 'master' into update_to_symb_08
- Fix function name passed to export_cpp.
- Updated some API breaking changes introduced in symbolica 0.8.0
- Update Cargo.toml
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
