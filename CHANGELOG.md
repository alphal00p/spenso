# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/alphal00p/spenso/compare/v0.2.0...v0.3.0) - 2024-10-25

### Added

- *(symbolica)* :sparkles: user defined concretizations and representations
- *(symbolica)* :arrow_up: update symbolica to 0.13
- *(rust)* :sparkles: more arithmetic on serializable atoms
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
- *(network)* :sparkles: compile asm for set
- *(network)* :sparkles: add result to set, and derived asmuch as possible
- *(network)* :sparkles: networksets that share the evaluator
- *(symbolica)* :sparkles: implement real on my complex and forwarded more of the evaluate API
- :sparkles: added AtomStructure to simplify type
- *(symbolica)* :sparkles: pattern replacement trait for paramtensor and forwarding to tensornet
- *(symbolica)* added benchmarks
- *(symbolica)* :sparkles: EvalTensors, using the same API as symbolica
- *(symbolica)* :sparkles: evaluator tensor from levels of nested tensor networks
- :building_construction: make levels use the new with_map shadowing.
- *(symbolica)* :sparkles: Added tooling for nested evaluate
- *(iterators)* :sparkles: added IteratableTensor trait
- *(symbolica)* :sparkles: allow other arguments in function view when converting back and forth
- *(network)* :sparkles: added a scalar field to tensor networks to enable good parsing of symbolica mul
- *(network)* :sparkles: added generic evaluate to fully parametric tensor networks
- *(network)* :sparkles: added a type homogenising function to tensor net
- *(iterators)* :sparkles: lending mut iterators






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
