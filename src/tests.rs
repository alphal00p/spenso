use crate::{
    ufo::mink_four_vector, Contract, DenseTensor, FallibleAddAssign, FallibleMul, FallibleSub,
    GetTensorData, HasTensorData, Representation, SparseTensor, StructureContract,
};
use crate::{
    AbstractFiber, BaseRepName, ColorAdjoint, CoreExpandedFiberIterator, CoreFlatFiberIterator,
    DualSlotTo, Euclidean, ExpandedIndex, Fiber, FiberClass, FlatIndex, IteratesAlongFibers,
    Lorentz, PhysReps, PhysicalSlots, RepName, TensorStructure,
};
#[cfg(feature = "shadowing")]
use ahash::{HashMap, HashMapExt};

use indexmap::{IndexMap, IndexSet};

use crate::Complex;
use insta::{assert_ron_snapshot, assert_yaml_snapshot};
use rand::{distributions::Uniform, Rng, SeedableRng};
use rand_xoshiro::Xoroshiro64Star;

use smartstring::alias::String;

#[cfg(feature = "shadowing")]
use super::{symbolic::SymbolicTensor, MixedTensor, Shadowable, TryIntoUpgrade};
#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, AtomView},
    state::State,
};

use super::FallibleAdd;
use super::{
    ufo, AbstractIndex, DataTensor, Dimension, HistoryStructure, NamedStructure, NumTensor,
    SetTensorData, Slot, TensorNetwork, VecStructure,
};

fn test_tensor<D, S>(structure: S, seed: u64, range: Option<(D, D)>) -> SparseTensor<D, S>
where
    S: TensorStructure,
    D: rand::distributions::uniform::SampleUniform,
    Uniform<D>: Copy,

    rand::distributions::Standard: rand::distributions::Distribution<D>,
{
    let mut rng: Xoroshiro64Star = Xoroshiro64Star::seed_from_u64(seed);

    let mut tensor = SparseTensor::empty(structure);

    let density = rng.gen_range(0..tensor.size().unwrap());

    if let Some((low, high)) = range {
        let multipliable = Uniform::new(low, high);
        for _ in 0..density {
            tensor
                .set_flat(
                    rng.gen_range(0..tensor.size().unwrap()).into(),
                    rng.sample(multipliable),
                )
                .unwrap();
        }
    } else {
        for _ in 0..density {
            tensor
                .set_flat(rng.gen_range(0..tensor.size().unwrap()).into(), rng.gen())
                .unwrap();
        }
    }

    tensor
}

fn test_structure(length: usize, seed: u64) -> VecStructure {
    let mut rng = Xoroshiro64Star::seed_from_u64(seed);
    let mut s = IndexSet::new();

    let rank = length;
    while s.len() < rank {
        let rep = rng.gen_range(0..=1);
        let dim = Dimension::Concrete(rng.gen_range(1..=9));
        let id = AbstractIndex(rng.gen_range(0..256));
        let rep: Representation<PhysReps> = match rep {
            0 => Euclidean::new_dimed_rep_selfless(dim).cast(),
            _ => Lorentz::new_dimed_rep_selfless(dim).cast(),
        };

        s.insert(rep.new_slot(id));
    }

    s.into_iter().collect()
}

fn test_structure_with_dims(dims: &[usize], seed: u64) -> VecStructure {
    let mut s = IndexSet::new();
    let mut rng = Xoroshiro64Star::seed_from_u64(seed);

    for d in dims {
        loop {
            let dim: Dimension = (*d).into();
            let rep = rng.gen_range(0..=1);
            let id = AbstractIndex(rng.gen_range(0..256));

            let rep: Representation<PhysReps> = match rep {
                0 => Euclidean::new_dimed_rep_selfless(dim).cast(),
                _ => Lorentz::new_dimed_rep_selfless(dim).cast(),
            };
            if s.insert(rep.new_slot(id)) {
                break;
            }
        }
    }

    s.into_iter().collect()
}

#[test]
fn rng_is_deterministic() {
    let valid = IndexMap::from([
        (vec![3, 0, 3].into(), 53),
        (vec![1, 1, 0].into(), 45),
        (vec![1, 1, 2].into(), -99),
        (vec![2, 1, 0].into(), -59),
        (vec![0, 1, 1].into(), -93),
        (vec![2, 0, 1].into(), 105),
        (vec![4, 1, 0].into(), 125),
        (vec![1, 0, 0].into(), -118),
        (vec![0, 0, 3].into(), -26),
        (vec![4, 0, 0].into(), 59),
        (vec![3, 1, 2].into(), 84),
        (vec![3, 1, 0].into(), -13),
        (vec![1, 0, 3].into(), 119),
        (vec![0, 1, 2].into(), 48),
        (vec![1, 0, 2].into(), 17),
        (vec![0, 0, 0].into(), 34),
        (vec![3, 0, 2].into(), 20),
        (vec![4, 0, 2].into(), -3),
        (vec![3, 1, 3].into(), 69),
        (vec![4, 0, 1].into(), 125),
    ]);
    for _ in 0..10 {
        let a = test_structure(3, 11);

        let a: SparseTensor<i8> = test_tensor(a, 1, None);

        assert_eq!(a.hashmap(), valid);
    }
}

#[test]
fn indexflatten() {
    let a = test_structure(4, 31);
    let idx = vec![1, 2, 3, 1];
    let flatidx = a.flat_index(&idx).unwrap();
    assert_eq!(ExpandedIndex::from(idx), a.expanded_index(flatidx).unwrap());
}

#[test]
fn single_fiber() {
    let a = test_structure(5, 5);

    let fiber = Fiber::from_filter(&[true, false, false, false, false], &a);
    assert!(fiber.single().is_some());

    let iter = fiber.clone().iter();

    let fciter: Vec<FlatIndex> = iter.collect();

    let fiberclass: Vec<FlatIndex> = FiberClass::from(fiber)
        .iter()
        .map(|fc| fc.iter.zero_index)
        .collect::<Vec<_>>();

    assert_ron_snapshot!((a, fciter, fiberclass));
}

#[test]
fn expanded_vs_flat_iter() {
    let a = test_structure(5, 5);
    let fiber = Fiber::from_filter(&[false, true, true, false, true], &a);

    let flat_iter = CoreFlatFiberIterator::new(&fiber, false);
    let expanded_iter = CoreExpandedFiberIterator::new(&fiber, false);

    let flat: Vec<FlatIndex> = flat_iter.collect();
    let expanded: Vec<FlatIndex> = expanded_iter.collect();

    assert_eq!(flat, expanded);
}

#[test]
fn fiber_class_vs_fiber_iterator() {
    let a = test_structure(5, 5);
    let fiber_class: FiberClass<'_, VecStructure> =
        Fiber::from_filter(&[false, true, true, false, true], &a).into();

    let fiber = Fiber::from_filter(&[true, false, false, true, false], &a);

    let flat_iter = CoreFlatFiberIterator::new(&fiber, false);
    let flat_class_iter = CoreFlatFiberIterator::new(&fiber_class, false);
    let flat: Vec<FlatIndex> = flat_iter.collect();
    let flat_class: Vec<FlatIndex> = flat_class_iter.collect();

    assert_eq!(flat, flat_class);
    let expanded_iter = CoreExpandedFiberIterator::new(&fiber, false);
    let expanded_class_iter = CoreExpandedFiberIterator::new(&fiber_class, false);

    let expanded: Vec<FlatIndex> = expanded_iter.collect();
    let expanded_class: Vec<FlatIndex> = expanded_class_iter.collect();
    assert_eq!(expanded, expanded_class);
}

#[test]
fn fibers() {
    let a = test_structure(5, 5);
    let fiber = Fiber::from_filter(&[true, true, false, false, true], &a);
    let fiberclass: FiberClass<'_, VecStructure> = fiber.into();
    let iter = fiberclass.clone().iter();
    let fciter: Vec<FlatIndex> = iter.map(|fc| fc.iter.zero_index).collect();

    let fiter: Vec<FlatIndex> = fiberclass
        .iter()
        .flat_map(|fc| fc.collect::<Vec<FlatIndex>>())
        .collect();
    assert_ron_snapshot!((a, fiter, fciter));
}

#[test]
fn fiber_from_structure() {
    let a = test_structure(5, 5);

    let fiber = Fiber::from(([1u8, 1, 0, 1, 0].as_slice()).into(), &a);

    let fiber_iter = fiber
        .clone()
        .iter()
        .map(|f| a.expanded_index(f).unwrap())
        .collect::<Vec<_>>();

    let fiberclass: FiberClass<'_, VecStructure> = fiber.into();

    let fiberclass_iter: Vec<ExpandedIndex> = fiberclass
        .iter()
        .map(|i| i.map(|f| a.expanded_index(f).unwrap()).collect::<Vec<_>>())
        .flatten()
        .collect();

    assert_eq!(fiberclass_iter.len(), a.size().unwrap());
    assert_yaml_snapshot!((fiberclass_iter, fiber_iter));
}

#[test]
fn permutation() {
    let a: Vec<Slot<Euclidean>> = vec![
        Euclidean::new_slot_selfless(1, 2),
        Euclidean::new_slot_selfless(3, 4),
        Euclidean::new_slot_selfless(5, 6),
    ];

    let b: Vec<Slot<Euclidean>> = vec![
        Euclidean::new_slot_selfless(3, 4),
        Euclidean::new_slot_selfless(5, 6),
        Euclidean::new_slot_selfless(1, 2),
    ];

    let permutation = a.find_permutation(&b).unwrap();

    let c = permutation.iter().map(|x| b[*x]).collect::<Vec<_>>();

    assert_eq!(c, a);
}

#[test]
fn trace() {
    let structura: HistoryStructure<&str, ()> = HistoryStructure::from(NamedStructure::from_iter(
        [
            Euclidean::new_slot_selfless(1, 5),
            Euclidean::new_slot_selfless(1, 5),
        ],
        "a",
        None,
    ));
    let a = test_tensor::<i8, _>(structura, 3, None);
    let f = a.internal_contract();

    assert!(f.is_scalar());
    assert_eq!(f.data(), vec![79]);
}

#[test]
fn construct_dense_tensor() {
    let a = test_structure(4, 32);
    let data = vec![1.0; a.size().unwrap()];
    let tensor = super::DenseTensor::from_data(&data, a).unwrap();
    let num_tensor: NumTensor = tensor.clone().into();
    let data_tensor: DataTensor<f64, _> = tensor.clone().into();

    #[cfg(feature = "shadowing")]
    {
        let mixed_tensor: MixedTensor<f64, _> = tensor.clone().into();
        assert_eq!(
            mixed_tensor
                .try_into_concrete()
                .unwrap()
                .try_as_real()
                .unwrap()
                .data(),
            data
        );
    }

    assert_eq!(data_tensor.data(), data);
    assert_eq!(num_tensor.try_as_float().unwrap().data(), data);
}

use anyhow::Result;
#[test]
fn construct_sparse_tensor() -> Result<()> {
    let structure = test_structure(3, 11);
    println!("{:?}", structure);

    let mut a = SparseTensor::empty(structure);
    a.set(&[1, 0, 1], 1.)?;
    a.set(&[0, 0, 2], 2.)?;
    a.set(&[1, 1, 2], 3.)?;
    a.set(&[1, 0, 2], 4.)?;

    let num_tensor: NumTensor = a.clone().into();
    let data_tensor: DataTensor<f64, _> = a.clone().into();

    #[cfg(feature = "shadowing")]
    {
        let mixed_tensor: MixedTensor<f64, _> = a.clone().into();
        assert_eq!(
            mixed_tensor
                .try_into_concrete()
                .unwrap()
                .try_as_real()
                .unwrap()
                .hashmap(),
            a.hashmap()
        );
    }
    assert_eq!(
        num_tensor.try_as_float().unwrap().hashmap(),
        data_tensor.hashmap()
    );
    assert_eq!(data_tensor.hashmap(), a.hashmap());

    Ok(())
}

#[test]
fn tensor_structure_forwarding() {
    let a = test_structure(6, 1);
    let range = Some((-1000, 1000));

    let sparse: SparseTensor<i16> = test_tensor(a.clone(), 1, range);
    let dense: DenseTensor<i16> = test_tensor(a.clone(), 2, range).to_dense();

    assert_eq!(a.strides().unwrap(), sparse.strides().unwrap());
    assert_eq!(dense.reps(), a.reps());
}

#[test]
fn scalar_and_dim1_conract() {
    let common = test_structure_with_dims(&[1, 3, 1, 2], 6);
    let mut structa = test_structure(1, 32);
    structa.merge(&common);
    let mut structb = test_structure(1, 22);
    structb.merge(&common);
    let range = Some((-100, 100));

    let mut tensor_1: SparseTensor<i16> = test_tensor(structa, 3, range);
    tensor_1.set_flat(0.into(), 45).unwrap();

    let mut tensor_2: SparseTensor<i16> = test_tensor(structb, 2, range);
    tensor_2.set_flat(0.into(), 2).unwrap();
    let f = tensor_1.contract(&tensor_2).unwrap();

    let g = tensor_1.contract(&tensor_2.to_dense()).unwrap();
    let h = tensor_1.to_dense().contract(&tensor_2).unwrap();

    let i = tensor_1.to_dense().contract(&tensor_2.to_dense()).unwrap();
    assert_eq!(f.to_dense().data(), h.data());
    assert_eq!(f.to_dense().data(), g.data());
    assert_eq!(f.to_dense().data(), i.data());

    let valid = IndexMap::from([
        (vec![0, 3].into(), 5908),
        (vec![2, 1].into(), 2491),
        (vec![3, 1].into(), -1081),
        (vec![0, 0].into(), 90),
        (vec![3, 0].into(), -1200),
        (vec![3, 3].into(), -788),
        (vec![1, 1].into(), 2961),
        (vec![2, 3].into(), -4004),
        (vec![2, 0].into(), 160),
        (vec![0, 1].into(), -987),
    ]);

    assert_eq!(f.hashmap(), valid);
}

#[test]
fn dense_dense_single_contract() {
    let structura = VecStructure::new(vec![
        Euclidean::new_slot_selfless(1, 4).into(),
        Euclidean::new_slot_selfless(2, 4).into(),
    ]);
    let structurb = VecStructure::new(vec![
        Euclidean::new_slot_selfless(2, 4).into(),
        Euclidean::new_slot_selfless(3, 3).into(),
    ]);

    let a = DenseTensor::from_data(
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        structura,
    )
    .unwrap();

    let b = DenseTensor::from_data(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], structurb.clone())
        .unwrap();

    let f = a.contract(&b).unwrap();

    assert_yaml_snapshot!(f);
}

#[test]
fn sparse_diag_dense_contract() {
    let structura = VecStructure::new(vec![
        Euclidean::new_slot_selfless(1, 4).into(),
        Euclidean::new_slot_selfless(2, 4).into(),
    ]);
    let structurb = VecStructure::new(vec![
        Euclidean::new_slot_selfless(2, 4).into(),
        Euclidean::new_slot_selfless(3, 3).into(),
    ]);

    let a = SparseTensor::from_data(
        &[
            (vec![0, 0].into(), 1),
            (vec![1, 1].into(), 2),
            (vec![2, 2].into(), 3),
            (vec![3, 3].into(), 4),
        ],
        structura.clone(),
    )
    .unwrap();

    let b = DenseTensor::from_data(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], structurb.clone())
        .unwrap();

    let f = a.contract(&b).unwrap();
    let g = b.contract(&a).unwrap();
    let h = a.contract(&b.to_sparse()).unwrap();
    let i = b.contract(&a.to_dense()).unwrap();

    assert_eq!(i.data(), g.data());
    assert_eq!(i.data(), h.to_dense().data());
    assert_eq!(f.data(), i.data());

    assert_yaml_snapshot!(f);
}

#[test]
fn contract_with_rank_one_in_middle() {
    let s = 12;
    let common = test_structure(3, s);
    let mut structa: VecStructure = test_structure(2, s + 1);
    structa.merge(&common);
    let mut structb: VecStructure = test_structure(1, s + 2);
    structb.merge(&common);

    // println!("seed: {s}");

    // println!("--");
    // println!("{structa}");
    // println!("--");
    // println!("{structb}");
    let range = Some((-1000, 1000));
    let tensor_a: SparseTensor<i32, VecStructure> = test_tensor(structa, s + 3, range);
    let dense_a: DenseTensor<i32, VecStructure> = tensor_a.to_dense();
    let tensor_b: SparseTensor<i32, VecStructure> = test_tensor(structb, s + 4, range);
    let dense_b: DenseTensor<i32, VecStructure> = tensor_b.to_dense();

    let f = tensor_b.contract(&tensor_a).unwrap().to_dense();
    let g = dense_b.contract(&dense_a).unwrap();

    assert_eq!(f.data, g.data);
}

fn test_structure_with_id<T>(ids: T, seed: u64) -> Vec<Slot<PhysReps>>
where
    T: Iterator<Item = usize>,
{
    let mut rng = Xoroshiro64Star::seed_from_u64(seed);
    let mut s = Vec::new();

    for id in ids {
        let rep = rng.gen_range(0..=1);
        let dim = Dimension::Concrete(rng.gen_range(1..=9));
        let id = AbstractIndex(id);
        let rep: Representation<PhysReps> = match rep {
            0 => Euclidean::new_dimed_rep_selfless(dim).cast(),
            _ => Lorentz::new_dimed_rep_selfless(dim).cast(),
        };

        s.push(rep.new_slot(id));
    }
    s
}

#[test]
fn single_contract() {
    let s = 18;
    let range = Some((-1000, 1000));
    let common = test_structure_with_id(0..1, s);
    let mut structa = test_structure_with_id(1..2, s);
    let mut structb = test_structure_with_id(2..3, s);
    let mut rng = Xoroshiro64Star::seed_from_u64(s);

    structa.insert(rng.gen_range(0..structa.len()), common[0]);
    structb.insert(rng.gen_range(0..structb.len()), common[0]);
    structa.sort();
    let structa: VecStructure = structa.into();

    let structb: VecStructure = structb.into();
    let spensor_a: SparseTensor<i32, VecStructure> = test_tensor(structa.clone(), s + 3, range);

    let densor_a: DenseTensor<i32, VecStructure> = spensor_a.to_dense();
    // println!("A={:?}", densor_a);

    let spensor_b: SparseTensor<i32, VecStructure> = test_tensor(structb.clone(), s + 4, range);
    assert_ron_snapshot!((structa, structb));
    let densor_b: DenseTensor<i32, VecStructure> = spensor_b.to_dense();
    // println!("B={:?}", densor_b);

    let dense_dense = densor_b.contract(&densor_a).unwrap();
    // println!("A*B {:?}", dense_dense);
    let sparse_sparse = spensor_b.contract(&spensor_a).unwrap().to_dense();
    let dense_sparse = densor_b.contract(&spensor_a).unwrap();
    let sparse_dense = spensor_b.contract(&densor_a).unwrap();

    assert_eq!(
        dense_dense.data(),
        sparse_sparse.data(),
        "S-S not match at seed: {s}"
    );
    assert_eq!(
        dense_dense.data(),
        dense_sparse.data(),
        "D-S not match at seed: {s}"
    );
    assert_eq!(
        dense_dense.data(),
        sparse_dense.data(),
        "S-D not match at seed: {s}"
    );
}

#[test]
fn all_single_contractions() {
    let range = Some((-1000, 1000));

    let mut dseq = vec![];
    let mut sseq = vec![];
    let mut sdeq = vec![];

    for s in 0..1000 {
        let common = test_structure_with_id(0..1, s);
        let mut structa = test_structure_with_id(1..2, s);
        let mut structb = test_structure_with_id(2..3, s);
        let mut rng = Xoroshiro64Star::seed_from_u64(s);

        structa.insert(rng.gen_range(0..structa.len()), common[0]);
        structb.insert(rng.gen_range(0..structb.len()), common[0]);
        structa.sort();
        let structa: VecStructure = structa.into();
        let structb: VecStructure = structb.into();

        let spensor_a: SparseTensor<i32, VecStructure> = test_tensor(structa.clone(), s + 3, range);
        let densor_a: DenseTensor<i32, VecStructure> = spensor_a.to_dense();
        let spensor_b: SparseTensor<i32, VecStructure> = test_tensor(structb.clone(), s + 4, range);
        let densor_b: DenseTensor<i32, VecStructure> = spensor_b.to_dense();

        let dense_dense = densor_b.contract(&densor_a).unwrap();
        // println!("{}", dense_dense.structure());
        let zeros: DenseTensor<i32> = DenseTensor::zero(dense_dense.structure.clone());
        let sparse_sparse = spensor_b.contract(&spensor_a).unwrap().to_dense();
        let dense_sparse = densor_b.contract(&spensor_a).unwrap_or(zeros.clone());
        let sparse_dense = spensor_b.contract(&densor_a).unwrap_or(zeros.clone());

        if dense_dense.data() != sparse_sparse.data() {
            sseq.push(s);
        }
        if dense_dense.data() != dense_sparse.data() {
            dseq.push(s);
        }
        if dense_dense.data() != sparse_dense.data() {
            sdeq.push(s);
        }
    }

    assert_eq!(sseq.len(), 0, "Sparse-Sparse failed at seeds {sseq:?}");
    assert_eq!(dseq.len(), 0, "Dense-Sparse failed at seeds {dseq:?}");
    assert_eq!(sdeq.len(), 0, "Sparse-Dense failed at seeds {sdeq:?}");
}

#[test]
fn simple_multi_contract() {
    let structa = VecStructure::new(vec![
        Euclidean::new_slot_selfless(1, 3).into(),
        Euclidean::new_slot_selfless(2, 4).into(),
        Euclidean::new_slot_selfless(3, 4).into(),
    ]);
    let structb = VecStructure::new(vec![
        Euclidean::new_slot_selfless(2, 4).into(),
        Euclidean::new_slot_selfless(4, 3).into(),
        Euclidean::new_slot_selfless(3, 4).into(),
    ]);

    let a = DenseTensor::from_data(
        &[
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 15, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
            12, 13, 14, 15, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 15, 16,
        ],
        structa,
    )
    .unwrap();

    let b = DenseTensor::from_data(
        &[
            3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 3, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0,
            3, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 3, 2, 1,
        ],
        structb,
    )
    .unwrap();

    let f = a.contract(&b).unwrap();

    let g = b.contract(&a).unwrap();
    assert_eq!(f.data(), g.data(), "A-B not equal to B-A");

    let h = a.contract(&b.to_sparse()).unwrap();
    assert_eq!(f.data(), h.data(), "dense dense not equal to dense sparse");

    let i = b.contract(&a.to_sparse()).unwrap();

    assert_eq!(f.data(), i.data(), "dense dense not equal to sparse dense");
    let j = a.to_sparse().contract(&b.to_sparse()).unwrap();

    assert_eq!(
        f.data(),
        j.to_dense().data(),
        "dense dense not equal to sparse sparse"
    );

    assert_yaml_snapshot!(f.data());
}

#[test]
fn multi_contract_permuted() {
    let range = Some((-1000, 1000));
    let s = 18;
    let mut rng = Xoroshiro64Star::seed_from_u64(s);
    let ncommon = rng.gen_range(2..5);

    let common = test_structure_with_id(0..ncommon, s);
    let mut structa = test_structure_with_id(ncommon..ncommon + 1, s);
    let mut structb = test_structure_with_id(ncommon + 1..ncommon + 2, s);

    for c in common {
        structa.insert(rng.gen_range(0..structa.len()), c);
        structb.insert(rng.gen_range(0..structb.len()), c);
    }
    structa.sort();
    let structa: VecStructure = structa.into();
    let structb: VecStructure = structb.into();

    println!("structa = {}", structa);
    println!("structb = {}", structb);

    let spensor_a: SparseTensor<i32, VecStructure> = test_tensor(structa.clone(), s + 3, range);
    let densor_a: DenseTensor<i32, VecStructure> = spensor_a.to_dense();
    let spensor_b: SparseTensor<i32, VecStructure> = test_tensor(structb.clone(), s + 4, range);
    let densor_b: DenseTensor<i32, VecStructure> = spensor_b.to_dense();

    let dense_dense = densor_b.contract(&densor_a).unwrap();
    // println!("{}", dense_dense.structure());
    let sparse_sparse = spensor_b.contract(&spensor_a).unwrap().to_dense();
    let dense_sparse = densor_b.contract(&spensor_a).unwrap();
    let sparse_dense = spensor_b.contract(&densor_a).unwrap();

    assert_eq!(
        dense_dense.data(),
        sparse_dense.data(),
        "S-D not match at seed: {s}"
    );
    assert_eq!(
        dense_dense.data(),
        dense_sparse.data(),
        "D-S not match at seed: {s}"
    );
    assert_eq!(
        dense_dense.data(),
        sparse_sparse.data(),
        "S-S not match at seed: {s}"
    );
}

#[test]
fn multi_contract() {
    let range = Some((-1000, 1000));
    let s = 18;
    let mut rng = Xoroshiro64Star::seed_from_u64(s);
    let ncommon = rng.gen_range(2..5);

    let common = test_structure_with_id(0..ncommon, s);
    let mut structa = test_structure_with_id(ncommon..ncommon + 1, s);
    let mut structb = test_structure_with_id(ncommon + 1..ncommon + 2, s);

    for c in common {
        structa.insert(rng.gen_range(0..structa.len()), c);
        structb.insert(rng.gen_range(0..structb.len()), c);
    }
    let structa: VecStructure = structa.into();
    let structb: VecStructure = structb.into();

    println!("structa = {}", structa);
    println!("structb = {}", structb);

    let spensor_a: SparseTensor<i32, VecStructure> = test_tensor(structa.clone(), s + 3, range);
    let densor_a: DenseTensor<i32, VecStructure> = spensor_a.to_dense();
    let spensor_b: SparseTensor<i32, VecStructure> = test_tensor(structb.clone(), s + 4, range);
    let densor_b: DenseTensor<i32, VecStructure> = spensor_b.to_dense();

    let dense_dense = densor_b.contract(&densor_a).unwrap();
    // println!("{}", dense_dense.structure());
    let sparse_sparse = spensor_b.contract(&spensor_a).unwrap().to_dense();
    let dense_sparse = densor_b.contract(&spensor_a).unwrap();
    let sparse_dense = spensor_b.contract(&densor_a).unwrap();

    assert_eq!(
        dense_dense.data(),
        sparse_dense.data(),
        "S-D not match at seed: {s}"
    );
    assert_eq!(
        dense_dense.data(),
        sparse_sparse.data(),
        "S-S not match at seed: {s}"
    );
    assert_eq!(
        dense_dense.data(),
        dense_sparse.data(),
        "D-S not match at seed: {s}"
    );

    insta::assert_ron_snapshot!(dense_dense);
}

#[test]
fn all_multi_contractions() {
    let _seeds = [48, 50, 118, 225, 234, 310];
    let range = Some((-1000, 1000));

    let mut dseq = vec![];
    let mut sseq = vec![];
    let mut sdeq = vec![];
    for s in 0..1000 {
        let mut rng = Xoroshiro64Star::seed_from_u64(s);
        let ncommon = rng.gen_range(2..5);

        let common = test_structure_with_id(0..ncommon, s);
        let mut structa = test_structure_with_id(ncommon..ncommon + 1, s);
        let mut structb = test_structure_with_id(ncommon + 1..ncommon + 2, s);

        for c in common {
            structa.insert(rng.gen_range(0..structa.len()), c);
            structb.insert(rng.gen_range(0..structb.len()), c);
        }
        structa.sort();
        let structa: VecStructure = structa.into();
        let structb: VecStructure = structb.into();

        let spensor_a: SparseTensor<i32, VecStructure> = test_tensor(structa.clone(), s + 3, range);
        let densor_a: DenseTensor<i32, VecStructure> = spensor_a.to_dense();
        let spensor_b: SparseTensor<i32, VecStructure> = test_tensor(structb.clone(), s + 4, range);
        let densor_b: DenseTensor<i32, VecStructure> = spensor_b.to_dense();

        let dense_dense = densor_b.contract(&densor_a).unwrap();
        // println!("{}", dense_dense.structure());
        let zeros: DenseTensor<i32> = DenseTensor::zero(dense_dense.structure.clone());
        let sparse_sparse = spensor_b.contract(&spensor_a).unwrap().to_dense();
        let dense_sparse = densor_b.contract(&spensor_a).unwrap_or(zeros.clone());
        let sparse_dense = spensor_b.contract(&densor_a).unwrap_or(zeros.clone());

        if dense_dense.data() != sparse_sparse.data() {
            sseq.push(s);
        }
        if dense_dense.data() != dense_sparse.data() {
            dseq.push(s);
        }
        if dense_dense.data() != sparse_dense.data() {
            sdeq.push(s);
        }
    }
    assert_eq!(sdeq.len(), 0, "Sparse-Dense failed at seeds {sdeq:?}");
    // assert_eq!(dseq.len(), 0, "Dense-Sparse failed at seeds {dseq:?}");
    // assert_eq!(sseq.len(), 0, "Sparse-Sparse failed at seeds {sseq:?}");
}

#[test]
fn gamma() {
    let g1: SparseTensor<Complex<f64>> = ufo::gamma(0.into(), [0.into(), 1.into()]);
    let g2: SparseTensor<Complex<f64>> = ufo::gamma(1.into(), [1.into(), 2.into()]);
    let g3: SparseTensor<Complex<f64>> = ufo::gamma(2.into(), [2.into(), 0.into()]);

    let c = g1.contract(&g2).unwrap().contract(&g3).unwrap();
    assert_eq!(
        Vec::<Complex<f64>>::new(),
        c.data(),
        "Odd traces must vanish"
    );

    let d: SparseTensor<Complex<f64>> =
        ufo::gamma(0.into(), [0.into(), 0.into()]).internal_contract();

    assert_eq!(Vec::<Complex<f64>>::new(), d.data(), "Gammas are traceless");
}

#[test]
fn matches() {
    let structur_a: HistoryStructure<&str, ()> = HistoryStructure::from(NamedStructure::from_iter(
        [
            Slot::<PhysReps>::from(Lorentz::new_slot_selfless(3, 2)),
            Lorentz::new_slot_selfless(2, 3).into(),
            Euclidean::new_slot_selfless(2, 2).into(),
            Euclidean::new_slot_selfless(1, 2).into(),
        ],
        "a",
        None,
    ));
    let structur_b = HistoryStructure::from(NamedStructure::from_iter(
        [
            Slot::<PhysReps>::from(Lorentz::new_slot_selfless(1, 2)),
            Lorentz::new_slot_selfless(3, 2).into(),
            Lorentz::new_slot_selfless(2, 2).into(),
            Euclidean::new_slot_selfless(1, 2).into(),
        ],
        "b",
        None,
    ));

    let a = structur_a.match_index(&structur_b);

    assert_eq!(Some((false, 3, 0)), a);
}

#[test]
fn mixed_tensor_contraction() {
    let im = Complex::new(1.5, 1.25);
    let data_a = [(vec![0, 0], 1.0), (vec![1, 1], 2.0)];

    let structur_a: HistoryStructure<&str, ()> = HistoryStructure::from(NamedStructure::from_iter(
        [
            Slot::<PhysReps>::from(Euclidean::new_slot_selfless(2, 2)),
            Slot::<PhysReps>::from(Euclidean::new_slot_selfless(1, 2)),
        ],
        "a",
        None,
    ));

    let a = SparseTensor::from_data(&data_a, structur_a.clone()).unwrap();

    let structur_b = HistoryStructure::from(NamedStructure::from_iter(
        [
            Slot::<PhysReps>::from(Euclidean::new_slot_selfless(2, 2)),
            Slot::<PhysReps>::from(Euclidean::new_slot_selfless(4, 2)),
        ],
        "b",
        None,
    ));

    let b = DenseTensor::from_data(
        &[
            im.mul_fallible(&1.0).unwrap(),
            2.0.mul_fallible(&im).unwrap(),
            3.0.mul_fallible(&im).unwrap(),
            4.0.mul_fallible(&im).unwrap(),
        ],
        structur_b.clone(),
    )
    .unwrap();

    let f = b.contract(&a).unwrap();

    assert_eq!(
        f.data,
        [
            1.0.mul_fallible(&im).unwrap(),
            6.0.mul_fallible(&im).unwrap(),
            2.0.mul_fallible(&im).unwrap(),
            8.0.mul_fallible(&im).unwrap()
        ]
    );

    let data_a = [
        (vec![0, 0], 1.0.mul_fallible(&im).unwrap()),
        (vec![1, 1], 2.0.mul_fallible(&im).unwrap()),
    ];

    let a = SparseTensor::from_data(&data_a, structur_a).unwrap();

    let b = DenseTensor::from_data(&[1.0, 2.0, 3.0, 4.0], structur_b).unwrap();

    let f = a.contract(&b).unwrap();
    assert_eq!(
        f.data,
        [
            1.0.mul_fallible(&im).unwrap(),
            2.0.mul_fallible(&im).unwrap(),
            6.0.mul_fallible(&im).unwrap(),
            8.0.mul_fallible(&im).unwrap()
        ]
    );
}

#[test]
fn tensor_net() {
    let a: NumTensor = ufo::gamma(1.into(), [2.into(), 3.into()]).into();
    let b: NumTensor = ufo::gamma(2.into(), [3.into(), 4.into()]).into();
    let c: NumTensor = ufo::gamma(3.into(), [4.into(), 5.into()]).into();
    let d: NumTensor = ufo::gamma(4.into(), [5.into(), 2.into()]).into();
    let p: NumTensor = mink_four_vector(1.into(), &[2., 3., 2., 1.]).into();
    let q: NumTensor = mink_four_vector(2.into(), &[2., 3., 2., 1.]).into();
    let r: NumTensor = mink_four_vector(3.into(), &[2., 3., 2., 1.]).into();
    let s: NumTensor = mink_four_vector(4.into(), &[2., 3., 2., 1.]).into();

    a.contract(&b)
        .unwrap()
        .contract(&c)
        .unwrap()
        .contract(&d)
        .unwrap();

    let mut n: TensorNetwork<NumTensor, Complex<f64>> =
        TensorNetwork::from(vec![a, b, c, p, q, d, r, s]);

    assert!(n.graph.validate_neighbors());

    // println!("{}", n.dot());

    assert_eq!(16, n.graph.neighbors.len());

    n.contract();

    assert_eq!(0, n.graph.neighbors.len());
    assert_eq!(
        Complex::new(400., 0.),
        n.result_tensor().unwrap().try_as_complex().unwrap().data()[0]
    )
}

#[test]
fn contract_spensor() {
    let data_a = [(vec![0, 0], 1.0), (vec![1, 1], 2.0)];
    let structur_a: HistoryStructure<&str, ()> = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 2, 2),
            Euclidean::new_slot_selfless(1, 2).into(),
        ],
        "a",
        None,
    ));

    let a = SparseTensor::from_data(&data_a, structur_a).unwrap();

    let data_b = [(vec![1, 0], 1.0), (vec![0, 1], 2.0)];
    let structur_b = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
            Euclidean::new_slot_selfless(3, 2).into(),
        ],
        "b",
        None,
    ));

    let b = SparseTensor::from_data(&data_b, structur_b).unwrap();

    let f = a.contract(&b).unwrap();

    let result = IndexMap::from([(vec![0, 1].into(), 2.0), (vec![1, 0].into(), 2.0)]);

    assert_eq!(f.hashmap(), result)
}

#[test]
fn sparse_addition() {
    let data_a = [(vec![1, 0], 1.0), (vec![0, 1], 2.0)];
    let structur_a: HistoryStructure<&str, ()> = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 2, 2),
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
        ],
        "a",
        None,
    ));

    let a = SparseTensor::from_data(&data_a, structur_a).unwrap();

    let data_b = [(vec![1, 0], 1.0), (vec![0, 1], 2.0)];
    let structur_b = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
            PhysReps::new_slot(Euclidean {}.into(), 2, 2),
        ],
        "b",
        None,
    ));

    let b = SparseTensor::from_data(&data_b, structur_b).unwrap();

    let f = a.add_fallible(&b).unwrap();

    let result = IndexMap::from([(vec![0, 1].into(), 3.0), (vec![1, 0].into(), 3.0)]);

    assert_eq!(f.hashmap(), result)
}

#[test]
fn sparse_sub() {
    let data_a = [(vec![1, 0], 1.0), (vec![0, 1], 2.0)];
    let structur_a: HistoryStructure<&str, ()> = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 2, 2),
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
        ],
        "a",
        None,
    ));

    let a = SparseTensor::from_data(&data_a, structur_a).unwrap();

    let data_b = [(vec![1, 0], 1.0), (vec![0, 1], 3.0)];

    let structur_b = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 2, 2),
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
        ],
        "a",
        None,
    ));

    let b = SparseTensor::from_data(&data_b, structur_b).unwrap();

    let f = a.sub_fallible(&b).unwrap();

    let result = IndexMap::from([(vec![0, 1].into(), -1.0)]);
    assert_eq!(f.hashmap(), result);
    // println!("{:?}", f);
}

#[test]
fn arithmetic_data() {
    let sa = test_structure(3, 1);
    let range = Some((-1000, 1000));

    let a: DataTensor<i32> = test_tensor(sa.clone(), 1, range).into();

    let b: DataTensor<i32> = test_tensor(sa.clone(), 2, range).into();

    println!("{:?}", a);
    println!("{:?}", b);

    let c = a.add_fallible(&b).unwrap();

    assert_eq!(
        vec![-267, 634, 0, 0, 650, 0, 520, 326, 0, 0, -120, -294, -907, 0, 0],
        c.to_dense().data()
    );

    // let syma = sa.clone().shadow_with("a".into_id());
    // let symb = sa.clone().shadow_with("b".into_id());

    //  let symc = &syma + &symb;
}

#[test]
fn contract_densor_with_spensor() {
    let data_a = [(vec![0, 0], 1.0), (vec![1, 1], 2.0)];

    let structur_a: HistoryStructure<&str, ()> = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 2, 2),
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
        ],
        "a",
        None,
    ));

    let a = SparseTensor::from_data(&data_a, structur_a).unwrap();

    let data_b = [1.0, 2.0, 3.0, 4.0];
    let structur_b = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
            PhysReps::new_slot(Euclidean {}.into(), 4, 2),
        ],
        "b",
        None,
    ));

    let b = DenseTensor::from_data(&data_b, structur_b).unwrap();

    let f = a.contract(&b).unwrap();

    assert_eq!(f.data, [1.0, 2.0, 6.0, 8.0]);
}

// #[test]
// fn symbolic_zeros() {
//     let mut state = State::get_global_state().write().unwrap();
//     let ws = Workspace::new();
//     let structure = TensorSkeleton::from(NamedStructure::from_iter([(1, 2), (3, 2)]seTen,None)sor::symbolic_zeros(structure.clone());

//     let zeros: DenseTensor<f64> = DenseTensor::default(structure);

//     assert_eq!(sym_zeros, zeros.to_symbolic(&ws, &mut state));
// }

#[test]
#[cfg(feature = "shadowing")]
fn evaluate() {
    use crate::TensorShell;

    let structure: NamedStructure<&str, ()> = test_structure(3, 1).to_named("a", None);

    let a: TensorShell<_> = structure.clone().into();

    let a = a.expanded_shadow().unwrap();

    let adata = test_tensor(structure, 1, Some((-100., 100.))).to_dense();

    let mut const_map: HashMap<AtomView<'_>, f64> = HashMap::new();

    a.append_const_map(&adata, &mut const_map);

    let aev: DenseTensor<f64, _> = a.evaluate(|r| r.into(), &const_map);

    assert_eq!(aev.data(), adata.data());
}

#[test]
#[cfg(feature = "shadowing")]
fn convert_sym() {
    use crate::IteratableTensor;

    let i: Complex<f64> = Complex::new(0.0, 1.0);
    let mut data_b: Vec<Complex<f64>> = vec![i * Complex::from(5.0), Complex::<f64>::from(2.6) + i];
    data_b.append(
        &mut [3.34, -17.125, 5.0, 6.0]
            .iter()
            .map(|x| Complex::from(*x))
            .collect::<Vec<_>>(),
    );
    let structur_b: HistoryStructure<&str, ()> = HistoryStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
            PhysReps::new_slot(Euclidean {}.into(), 4, 3),
        ],
        "b",
        None,
    ));
    let b = DenseTensor::from_data(&data_b, structur_b).unwrap();

    let symb: DenseTensor<Atom, _> = b.try_into_upgrade().unwrap();

    let expected_data: Vec<Atom> = [
        "5*𝑖",
        "𝑖+5854679515581645/2251799813685248",
        "940126422213591/281474976710656",
        "-137/8",
        "5",
        "6",
    ]
    .iter()
    .map(|x| Atom::parse(x).unwrap())
    .collect();

    assert_eq!(
        symb.iter_expanded()
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>(),
        expected_data
    );
}

#[test]
#[cfg(feature = "shadowing")]
fn simple_multi_contract_sym() {
    use crate::ToSymbolic;

    let structa = VecStructure::new(vec![
        Euclidean::new_slot_selfless(1, 3).into(),
        Euclidean::new_slot_selfless(2, 4).into(),
        Euclidean::new_slot_selfless(3, 4).into(),
    ]);
    // let structa = structa.to_named("a");
    let structb = VecStructure::new(vec![
        Euclidean::new_slot_selfless(2, 4).into(),
        Euclidean::new_slot_selfless(4, 3).into(),
        Euclidean::new_slot_selfless(3, 4).into(),
    ]);
    // let structb = structb.to_named("b");

    let _a = DenseTensor::from_data(
        &[
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 15, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
            12, 13, 14, 15, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 13, 14, 15, 16,
        ],
        structa.clone(),
    )
    .unwrap();

    let _b = DenseTensor::from_data(
        &[
            3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 3, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0,
            3, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 3, 2, 1,
        ],
        structb.clone(),
    )
    .unwrap();

    let a: DataTensor<Atom, NamedStructure<&str, ()>> = structa
        .to_named("a", None)
        .to_dense_expanded_labels()
        .unwrap()
        .into();
    let b: DataTensor<Atom, NamedStructure<&str, ()>> = structb
        .to_named("b", None)
        .to_dense_expanded_labels()
        .unwrap()
        .into();

    let _f = a.contract(&b).unwrap();
}

#[test]
fn empty_densor() {
    let empty_structure = Vec::<Slot<_>>::new();

    let empty: DenseTensor<f64> = DenseTensor::default(empty_structure.into());

    assert_eq!(*empty.get(&[]).unwrap(), 0.0);
}

#[test]
fn complex() {
    let _structur = test_structure(2, 1);

    let _r = Complex::new(1.0, 2.0);
    let _p = Complex::new(3.0, 4.0);
}

#[test]
#[cfg(feature = "shadowing")]
fn symbolic_contract() {
    use crate::Bispinor;

    let structura: NamedStructure<&str, ()> = NamedStructure::from(NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 1, 2),
            PhysReps::new_slot(Euclidean {}.into(), 4, 3),
        ],
        "T",
        None,
    ));

    let structurb: NamedStructure<&str, ()> = NamedStructure::from_iter(
        [
            PhysReps::new_slot(Euclidean {}.into(), 3, 2),
            PhysReps::new_slot(Euclidean {}.into(), 2, 3),
        ],
        "P",
        None,
    );

    let mink = Lorentz {}.new_dimed_rep(4);
    let mu: Slot<PhysReps> = mink.new_slot(0).into();
    let bis = Bispinor {}.new_dimed_rep(4);

    let i = bis.new_slot(1).into();
    let j = bis.new_slot(2).into();
    let k = bis.new_slot(9).into();

    let _structure: NamedStructure<&str, ()> = NamedStructure::from_iter([mu, i, j], "γ", None);
    let _p_struct: NamedStructure<&str, ()> = NamedStructure::from_iter([mu], "p", None);
    let _t_struct: NamedStructure<&str, ()> = NamedStructure::from_iter([i, j, k], "T", None);

    let a = SymbolicTensor::from_named(&structura.to_shell()).unwrap();
    let b = SymbolicTensor::from_named(&structurb.to_shell()).unwrap();
    let f = a.contract(&b).unwrap();

    println!("{}", f);

    assert_eq!(
        *f.get_atom(),
        Atom::parse("T(aind(euc(2,1),euc(3,4)))*P(aind(euc(2,3),euc(3,2)))").unwrap()
    );

    let a = f.to_network().unwrap();

    // let syms = a.to_symbolic_tensor_vec();

    // for s in syms {
    //     println!("{:?}", s.structure());
    // }

    println!("{}", a.dot());
}

#[test]

fn test_fallible_mul() {
    let a: i32 = 4;
    let b: f64 = 4.;
    let mut c: f64 = a.mul_fallible(&b).unwrap();
    c.add_assign_fallible(&a);
    let d: Option<f64> = b.mul_fallible(&a);
    let a: &i32 = &a;
    let e: Option<f64> = a.mul_fallible(&b);
    assert_eq!(c, 20.);
    assert_eq!(d, Some(16.));
    assert_eq!(e, Some(16.));

    #[cfg(feature = "shadowing")]
    {
        let a = &Atom::parse("a(2)").unwrap();

        let b = &Atom::parse("b(1)").unwrap();
        let i = Atom::new_var(State::I);
        let mut f = a.mul_fallible(&4.).unwrap();
        f.add_assign_fallible(b);
        f.add_assign_fallible(&i);

        let function_map = HashMap::new();
        let mut cache = HashMap::new();

        let mut const_map = HashMap::new();
        const_map.insert(i.as_view(), Complex::<f64>::new(0., 1.).into());

        const_map.insert(a.as_view(), Complex::<f64>::new(3., 1.).into());

        const_map.insert(b.as_view(), Complex::<f64>::new(3., 1.).into());

        let ev: symbolica::domains::float::Complex<f64> =
            f.as_view()
                .evaluate(|r| r.into(), &const_map, &function_map, &mut cache);

        println!("{}", ev);
        // print!("{}", f.unwrap());

        let g = Complex::new(0.1, 3.);

        let mut h = a.sub_fallible(&g).unwrap();

        h.add_assign_fallible(a);
        let _f = a.mul_fallible(a);

        Atom::default();

        println!("{}", h);
    }
}

#[test]
#[cfg(feature = "shadowing")]
fn get_license_key() {
    use gat_lending_iterator::Enumerate;
    use serde::de::IntoDeserializer;
    use symbolica::LicenseManager;

    use crate::{
        ColorAdjoint, ColorFundamental, DualSlotTo, Euclidean, IsAbstractSlot, Lorentz,
        PhysicalSlots, RepName, Slot,
    };

    LicenseManager::new();

    let a = Lorentz {}.dual().dual().dual().dual().new_dimed_rep(4);

    let b: PhysicalSlots = Lorentz {}.dual().new_slot(4, 4).into();
    let aslot: PhysicalSlots = a.new_slot(4).into();
    let c = a.new_slot(4);
    let domatchdisp = aslot.matches(&PhysicalSlots::from(c));
    let domatch = b.matches(&aslot);

    let mu: PhysicalSlots = a.new_slot(34).into();
    let nu: PhysicalSlots = Lorentz {}.dual().dual().new_slot(5, 45645).into();
    let matches = mu.matches(&nu);
    let e = Euclidean {}.dual();

    // let adual = a; //.dual();
    // let b = Euclidean {}.new(4.into());

    // let matches = a.matches(&adual);
}

#[test]
fn duals() {
    let a = Lorentz {}.dual().new_dimed_rep(4);
    let mu = a.new_slot(1);
    let nu = a.dual().new_slot(1);

    assert!(mu.matches(&nu));

    let mu: PhysicalSlots = mu.into();

    assert!(mu.matches(&PhysicalSlots::from(nu)));

    let rho: PhysicalSlots = ColorAdjoint {}.dual().new_slot(4, 1).into();

    assert!(!mu.matches(&rho))
}
