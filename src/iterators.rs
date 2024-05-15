//! Iterators for tensors
//!
//! Iterators for tensors are used to iterate over the elements of a tensor.
//! More specialized iterators are provided that fix a certain subset of indices, and iterate over the remaining indices.
//! At each iteration, the iterator returns a vector of references to the elements of the tensor along the fixed indices (so called fibers).
//!
//! The iterators are built using the basic index iterators provided by the `TensorStructureIterator`s.
//!

use std::{
    fmt::{write, Debug, Display},
    ops::{AddAssign, Index, Neg, SubAssign},
};

use crate::VecStructure;

use super::{
    ConcreteIndex, DenseTensor, Dimension, GetTensorData, Representation, Slot, SparseTensor,
    TensorStructure,
};
use ahash::{AHashMap, AHashSet};

use bitvec::{
    order,
    vec::{self, BitVec},
};
use permutation::Permutation;
use serde::{ser::SerializeMap, Deserialize, Serialize};

use bitvec::prelude::*;

pub trait AbstractFiberIndex {
    fn is_free(&self) -> bool;

    fn is_fixed(&self) -> bool {
        !self.is_free()
    }
}

pub trait AbstractFiber {
    fn strides(&self) -> Vec<usize>;
    fn shape(&self) -> Vec<Dimension>;
    fn order(&self) -> usize;
    fn single(&self) -> Option<usize>;
}

/// An iterator over all indices of a tensor structure
///
/// `Item` is a flat index

pub struct TensorStructureIndexIterator<'a> {
    structure: &'a [Slot],
    current_flat_index: usize,
}

impl<'a> Iterator for TensorStructureIndexIterator<'a> {
    type Item = Vec<ConcreteIndex>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(indices) = self.structure.expanded_index(self.current_flat_index) {
            self.current_flat_index += 1;

            Some(indices)
        } else {
            None
        }
    }
}

impl<'a> TensorStructureIndexIterator<'a> {
    #[must_use]
    pub fn new(structure: &'a [Slot]) -> Self {
        TensorStructureIndexIterator {
            structure,
            current_flat_index: 0,
        }
    }
}
#[derive(Debug, Clone)]
struct BareFiber {
    indices: Vec<FiberIndex>,
    is_single: FiberIndex,
}

impl BareFiber {
    pub fn conj(self) -> Self {
        self
    }

    pub fn bitvec(&self) -> BitVec {
        self.indices.iter().map(|x| x.is_free()).collect()
    }

    pub fn bitvecinv(&self) -> BitVec {
        self.indices.iter().map(|x| x.is_fixed()).collect()
    }
    pub fn from_flat<I>(flat: usize, structure: &I) -> BareFiber
    where
        I: TensorStructure,
    {
        let expanded = structure.expanded_index(flat).unwrap();

        BareFiber {
            indices: expanded.into_iter().map(|i| FiberIndex::from(i)).collect(),
            is_single: FiberIndex::Free,
        }
    }
    /// true is free, false is fixed
    pub fn from_filter(filter: &[bool]) -> BareFiber {
        let mut f = BareFiber {
            indices: filter
                .into_iter()
                .map(|i| {
                    if *i {
                        FiberIndex::Free
                    } else {
                        FiberIndex::Fixed(0)
                    }
                })
                .collect(),
            is_single: FiberIndex::Free,
        };
        f.is_single();
        f
    }
    pub fn zeros<I: TensorStructure>(structure: &I) -> BareFiber {
        BareFiber {
            indices: vec![FiberIndex::Fixed(0); structure.order()],
            is_single: FiberIndex::Free,
        }
    }

    pub fn fix(&mut self, pos: usize, val: usize) {
        if let FiberIndex::Fixed(single_pos) = self.is_single {
            if single_pos == pos {
                self.is_single = FiberIndex::Free;
            }
        }

        self.indices[pos] = val.into();
    }

    pub fn is_single(&mut self) -> FiberIndex {
        if let FiberIndex::Fixed(pos) = self.is_single {
            FiberIndex::Fixed(pos)
        } else {
            let mut has_one = false;
            let mut has_two = false;
            let mut pos = 0;
            for (posi, index) in self.indices.iter().enumerate() {
                if let FiberIndex::Free = index {
                    if !has_one {
                        has_one = true;
                        pos = posi;
                    } else {
                        has_two = true;
                    }
                }
            }
            if has_one && !has_two {
                self.is_single = FiberIndex::Fixed(pos);
                return FiberIndex::Fixed(pos);
            }
            self.is_single = FiberIndex::Free;
            FiberIndex::Free
        }
    }

    pub fn free(&mut self, pos: usize) {
        self.indices[pos] = FiberIndex::Free;
    }
}

impl Index<usize> for BareFiber {
    type Output = FiberIndex;

    fn index(&self, index: usize) -> &Self::Output {
        &(self.indices[index])
    }
}

#[derive(Debug)]
pub struct Fiber<'a, I: TensorStructure> {
    structure: &'a I,
    bare_fiber: BareFiber,
}

impl<'a, I: TensorStructure> Clone for Fiber<'a, I> {
    fn clone(&self) -> Self {
        Fiber {
            structure: self.structure,
            bare_fiber: self.bare_fiber.clone(),
        }
    }
}

impl<'a, I> Index<usize> for Fiber<'a, I>
where
    I: TensorStructure,
{
    type Output = FiberIndex;

    fn index(&self, index: usize) -> &Self::Output {
        &(self.bare_fiber[index])
    }
}

impl<'a, I> AbstractFiber for Fiber<'a, I>
where
    I: TensorStructure,
{
    fn strides(&self) -> Vec<usize> {
        self.structure.strides()
    }

    fn shape(&self) -> Vec<Dimension> {
        self.structure.shape()
    }

    fn order(&self) -> usize {
        self.structure.order()
    }

    fn single(&self) -> Option<usize> {
        if let FiberIndex::Fixed(pos) = self.bare_fiber.is_single {
            Some(pos)
        } else {
            None
        }
    }
}

impl<'a, I> Fiber<'a, I>
where
    I: TensorStructure,
{
    pub fn conj(self) -> Self {
        self
    }

    pub fn iter(self) -> FiberIterator<'a, I> {
        FiberIterator::new(self)
    }

    pub fn bitvec(&self) -> BitVec {
        self.bare_fiber.bitvec()
    }

    pub fn bitvecinv(&self) -> BitVec {
        self.bare_fiber.bitvecinv()
    }
    pub fn from_flat(flat: usize, structure: &'a I) -> Fiber<'a, I> {
        Fiber {
            bare_fiber: BareFiber::from_flat(flat, structure),
            structure,
        }
    }
    /// true is free, false is fixed
    pub fn from_filter(filter: &[bool], structure: &'a I) -> Fiber<'a, I> {
        //check compatibility
        Fiber {
            bare_fiber: BareFiber::from_filter(filter),
            structure,
        }
    }
    pub fn zeros(structure: &'a I) -> Fiber<'a, I> {
        Fiber {
            bare_fiber: BareFiber::zeros(structure),
            structure,
        }
    }
    pub fn fix(&mut self, pos: usize, val: usize) {
        self.bare_fiber.fix(pos, val);
    }
    pub fn is_single(&mut self) -> FiberIndex {
        self.bare_fiber.is_single()
    }
    pub fn free(&mut self, pos: usize) {
        self.bare_fiber.free(pos);
    }
}

impl<'a, I: TensorStructure> Display for Fiber<'a, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for index in self.bare_fiber.indices.iter() {
            write!(f, "{} ", index)?
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FiberMut<'a, I: TensorStructure> {
    structure: &'a mut I,
    bare_fiber: BareFiber,
}

impl<'a, I> Index<usize> for FiberMut<'a, I>
where
    I: TensorStructure,
{
    type Output = FiberIndex;

    fn index(&self, index: usize) -> &Self::Output {
        &(self.bare_fiber[index])
    }
}

impl<'a, I> AbstractFiber for FiberMut<'a, I>
where
    I: TensorStructure,
{
    fn strides(&self) -> Vec<usize> {
        self.structure.strides()
    }

    fn shape(&self) -> Vec<Dimension> {
        self.structure.shape()
    }

    fn order(&self) -> usize {
        self.structure.order()
    }

    fn single(&self) -> Option<usize> {
        if let FiberIndex::Fixed(pos) = self.bare_fiber.is_single {
            Some(pos)
        } else {
            None
        }
    }
}

impl<'a, I> FiberMut<'a, I>
where
    I: TensorStructure,
{
    pub fn conj(self) -> Self {
        self
    }

    pub fn bitvec(&self) -> BitVec {
        self.bare_fiber.bitvec()
    }

    pub fn bitvecinv(&self) -> BitVec {
        self.bare_fiber.bitvecinv()
    }
    pub fn from_flat(flat: usize, structure: &'a I) -> Fiber<'a, I> {
        Fiber {
            bare_fiber: BareFiber::from_flat(flat, structure),
            structure,
        }
    }
    /// true is free, false is fixed
    pub fn from_filter(filter: &[bool], structure: &'a I) -> Fiber<'a, I> {
        //check compatibility
        Fiber {
            bare_fiber: BareFiber::from_filter(filter),
            structure,
        }
    }
    pub fn zeros(structure: &'a I) -> Fiber<'a, I> {
        Fiber {
            bare_fiber: BareFiber::zeros(structure),
            structure,
        }
    }
    pub fn fix(&mut self, pos: usize, val: usize) {
        self.bare_fiber.fix(pos, val);
    }
    pub fn is_single(&mut self) -> FiberIndex {
        self.bare_fiber.is_single()
    }
    pub fn free(&mut self, pos: usize) {
        self.bare_fiber.free(pos);
    }
}

impl<'a, I: TensorStructure> Display for FiberMut<'a, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for index in self.bare_fiber.indices.iter() {
            write!(f, "{} ", index)?
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct SingleStrideShift {
    stride: usize,
    shift: usize,
}

#[derive(Debug, Clone)]
struct MultiStrideShift {
    strides: Vec<usize>,
    shifts: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum StrideShift {
    Single(Option<SingleStrideShift>),
    Multi(MultiStrideShift),
}

impl StrideShift {
    pub fn new_single(stride: Option<usize>, shift: Option<usize>) -> Self {
        let stride_shift = stride.zip(shift);
        if let Some(stride_shift) = stride_shift {
            StrideShift::Single(Some(SingleStrideShift {
                stride: stride_shift.0,
                shift: stride_shift.1,
            }))
        } else {
            StrideShift::Single(None)
        }
    }

    pub fn new_single_none() -> Self {
        StrideShift::Single(None)
    }

    pub fn new_multi(strides: Vec<usize>, shifts: Vec<usize>) -> Self {
        StrideShift::Multi(MultiStrideShift { strides, shifts })
    }
}
#[derive(Debug, Clone)]
pub struct MinimumFiberIterator {
    pub varying_fiber_index: usize,
    pub increment: usize,
    pub stride_shift: StrideShift,
    pub max: usize,
    pub zero_index: usize,
}

impl MinimumFiberIterator {
    fn init_multi_fiber_iter<I, J>(
        strides: Vec<usize>,
        dims: Vec<Dimension>,
        order: usize,
        fiber: &I,
        conj: bool,
    ) -> (usize, Vec<usize>, Vec<usize>, usize)
    where
        I: Index<usize, Output = J>,
        J: AbstractFiberIndex,
    {
        let mut max = 0;
        // max -= 1;

        let mut increment = 1;

        let mut fixed_strides = vec![];
        let mut shifts = vec![];

        let mut before = true;
        let mut has_seen_stride = false;
        let mut first = true;

        for pos in (0..order).rev() {
            let fi = &fiber[pos];

            if fi.is_fixed() ^ conj && !before && !first {
                has_seen_stride = true;
                fixed_strides.push(strides[pos]);
            }
            if fi.is_free() ^ conj && before && has_seen_stride {
                shifts.push(strides[pos]);
            }

            if fi.is_free() ^ conj {
                max += (usize::from(dims[pos]) - 1) * strides[pos];
                if first {
                    increment = strides[pos];
                    first = false;
                }
            }

            before = fi.is_fixed() ^ conj;
        }

        if fixed_strides.len() > shifts.len() {
            fixed_strides.pop();
        }
        (increment, fixed_strides, shifts, max)
    }

    fn init_single_fiber_iter(
        strides: Vec<usize>,
        fiber_position: usize,
        dims: Vec<Dimension>,
        conj: bool,
    ) -> (usize, Option<usize>, Option<usize>, usize) {
        // max -= 1;

        let fiber_stride = strides[fiber_position];
        let dim: usize = dims[fiber_position].into();
        let size = dims.iter().map(|x| usize::from(*x)).product::<usize>();
        let mut stride = None;
        let mut shift = None;

        if conj {
            let max = size - fiber_stride * (dim - 1) - 1;

            let mut increment = 1;

            if fiber_position == dims.len() - 1 {
                increment = *strides.get(dims.len().wrapping_sub(2)).unwrap_or(&1);
            } else if fiber_position != 0 {
                shift = Some(strides[fiber_position - 1]);
                stride = Some(strides[fiber_position]);
            }

            (increment, stride, shift, max)
        } else {
            let increment = fiber_stride;
            let max = fiber_stride * (dim - 1);

            (increment, stride, shift, max)
        }
    }

    pub fn new<I, J>(fiber: &I) -> Self
    where
        I: Index<usize, Output = J> + AbstractFiber,
        J: AbstractFiberIndex,
    {
        if let Some(single) = fiber.single() {
            let (increment, fixed_strides, shifts, max) =
                Self::init_single_fiber_iter(fiber.strides(), single, fiber.shape(), false);

            MinimumFiberIterator {
                increment,
                stride_shift: StrideShift::new_single(fixed_strides, shifts),
                max,
                zero_index: 0,
                varying_fiber_index: 0,
            }
        } else {
            let (increment, fixed_strides, shifts, max) = Self::init_multi_fiber_iter(
                fiber.strides(),
                fiber.shape(),
                fiber.order(),
                fiber,
                false,
            );

            MinimumFiberIterator {
                increment,
                stride_shift: StrideShift::new_multi(fixed_strides, shifts),
                max,
                zero_index: 0,
                varying_fiber_index: 0,
            }
        }
    }

    pub fn new_paired_conjugates<I, J>(fiber: &I) -> (Self, Self)
    where
        I: Index<usize, Output = J> + AbstractFiber,
        J: AbstractFiberIndex,
    {
        let strides = fiber.strides();
        let dims = fiber.shape();
        let order = fiber.order();
        let mut max = 0;

        // max -= 1;

        let mut increment = 1;

        // println!("{:?}{}", fiber_positions, increment);
        let mut fixed_strides = vec![];
        let mut fixed_strides_conj = vec![];
        let mut shifts = vec![];
        let mut shifts_conj = vec![];

        let mut before = true;
        let mut has_seen_stride = false;
        let mut has_seen_stride_conj = false;
        let mut first = true;
        let mut first_conj = true;
        let mut increment_conj = 1;

        let mut max_conj = 0;

        for pos in (0..order).rev() {
            let fi = &fiber[pos];

            if fi.is_fixed() && !before {
                if !first {
                    has_seen_stride = true;
                    fixed_strides.push(strides[pos]);
                }

                if has_seen_stride_conj {
                    shifts_conj.push(strides[pos]);
                }
            }
            if fi.is_free() && before {
                if has_seen_stride {
                    shifts.push(strides[pos]);
                }

                if !first_conj {
                    fixed_strides_conj.push(strides[pos]);
                    has_seen_stride_conj = true;
                }
            }

            if fi.is_fixed() {
                max_conj += (usize::from(dims[pos]) - 1) * strides[pos];
                if first_conj {
                    increment_conj = strides[pos];
                    first_conj = false;
                }
            } else {
                max += (usize::from(dims[pos]) - 1) * strides[pos];
                if first {
                    increment = strides[pos];
                    first = false;
                }
            }

            before = fi.is_fixed();
        }

        if fixed_strides.len() > shifts.len() {
            fixed_strides.pop();
        }

        if fixed_strides_conj.len() > shifts_conj.len() {
            fixed_strides_conj.pop();
        }

        (
            MinimumFiberIterator {
                varying_fiber_index: 0,
                stride_shift: StrideShift::new_multi(fixed_strides_conj, shifts_conj),
                increment: increment_conj,
                max: max_conj,
                zero_index: 0,
            },
            MinimumFiberIterator {
                varying_fiber_index: 0,
                increment,
                stride_shift: StrideShift::new_multi(fixed_strides, shifts),
                max,
                zero_index: 0,
            },
        )
    }

    pub fn new_conjugate<I, J>(fiber: &I) -> Self
    where
        I: Index<usize, Output = J> + AbstractFiber,
        J: AbstractFiberIndex,
    {
        let (increment, fixed_strides, shifts, max) = Self::init_multi_fiber_iter(
            fiber.strides(),
            fiber.shape(),
            fiber.order(),
            fiber,
            false,
        );

        MinimumFiberIterator {
            increment,
            stride_shift: StrideShift::new_multi(fixed_strides, shifts),
            max,
            zero_index: 0,
            varying_fiber_index: 0,
        }
    }

    pub fn reset(&mut self) {
        self.varying_fiber_index = 0;
    }

    pub fn shift(&mut self, shift: usize) {
        self.zero_index = shift;
    }
}

impl Iterator for MinimumFiberIterator {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.varying_fiber_index > self.max {
            return None;
        }
        let index = self.varying_fiber_index + self.zero_index;

        self.varying_fiber_index += self.increment;

        match self.stride_shift {
            StrideShift::Multi(ref ss) => {
                for (i, s) in ss.strides.iter().enumerate() {
                    if self.varying_fiber_index % s == 0 {
                        self.varying_fiber_index += ss.shifts[i] - s;
                    } else {
                        break;
                    }
                }
            }
            StrideShift::Single(Some(ss)) => {
                if self.varying_fiber_index % ss.stride == 0 {
                    self.varying_fiber_index += ss.shift - ss.stride;
                }
            }
            _ => {}
        }
        Some(index)
    }
}

#[derive(Debug)]
pub struct FiberIterator<'a, I: TensorStructure> {
    pub fiber: Fiber<'a, I>,
    pub iter: MinimumFiberIterator,
}

impl<'a, I: TensorStructure> Clone for FiberIterator<'a, I> {
    fn clone(&self) -> Self {
        FiberIterator {
            fiber: self.fiber.clone(),
            iter: self.iter.clone(),
        }
    }
}

impl<'a, I: TensorStructure> FiberIterator<'a, I> {
    pub fn new(fiber: Fiber<'a, I>) -> Self {
        FiberIterator {
            iter: MinimumFiberIterator::new(&fiber),
            fiber,
        }
    }

    pub fn reset(&mut self) {
        self.iter.reset();
    }

    pub fn shift(&mut self, shift: usize) {
        self.iter.shift(shift);
    }
}

impl<'a> Iterator for FiberIterator<'a, VecStructure> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FiberClassIndex {
    Free,
    Fixed,
}

impl AbstractFiberIndex for FiberClassIndex {
    fn is_free(&self) -> bool {
        if let FiberClassIndex::Free = self {
            return true;
        } else {
            return false;
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FiberIndex {
    Free,
    Fixed(usize),
}

impl AbstractFiberIndex for FiberIndex {
    fn is_free(&self) -> bool {
        if let FiberIndex::Free = self {
            return true;
        } else {
            return false;
        }
    }
}

impl From<usize> for FiberIndex {
    fn from(value: usize) -> Self {
        FiberIndex::Fixed(value)
    }
}

impl Display for FiberIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FiberIndex::Fixed(val) => {
                write!(f, "{val}")
            }
            FiberIndex::Free => {
                write!(f, ":")
            }
        }
    }
}

pub struct FiberClass<'a, I: TensorStructure> {
    fiber: Fiber<'a, I>,
    /// true is fixed (but varying when iterating) and false is free (but fixed to 0 when iterating)
    free: BitVec, //check performance when it is AHashSet<usize>
}

impl<'a, I: TensorStructure> Clone for FiberClass<'a, I> {
    fn clone(&self) -> Self {
        FiberClass {
            fiber: self.fiber.clone(),
            free: self.free.clone(),
        }
    }
}

impl<'a, I> Index<usize> for FiberClass<'a, I>
where
    I: TensorStructure,
{
    type Output = FiberClassIndex;

    fn index(&self, index: usize) -> &Self::Output {
        if self.free[index] {
            return &FiberClassIndex::Free;
        } else {
            return &FiberClassIndex::Fixed;
        }
    }
}

impl<'a, I: TensorStructure> From<Fiber<'a, I>> for FiberClass<'a, I> {
    fn from(fiber: Fiber<'a, I>) -> Self {
        let free = fiber.bitvecinv();
        FiberClass { fiber, free }
    }
}

impl<'a, I: TensorStructure> AbstractFiber for FiberClass<'a, I> {
    fn strides(&self) -> Vec<usize> {
        self.fiber.strides()
    }

    fn shape(&self) -> Vec<Dimension> {
        self.fiber.shape()
    }

    fn order(&self) -> usize {
        self.fiber.order()
    }

    fn single(&self) -> Option<usize> {
        self.fiber.single()
    }
}

impl<'a, I: TensorStructure> FiberClass<'a, I> {
    pub fn iter(self) -> FiberClassIterator<'a, I> {
        FiberClassIterator::new(self)
    }

    pub fn fiber(self) -> Fiber<'a, I> {
        self.fiber
    }
}

pub struct FiberClassIterator<'b, N: TensorStructure> {
    pub fiber: FiberIterator<'b, N>,
    pub iter: MinimumFiberIterator,
}

impl<'b, N: TensorStructure> FiberClassIterator<'b, N> {
    pub fn new(class: FiberClass<'b, N>) -> Self {
        let (iter, iter_conj) = MinimumFiberIterator::new_paired_conjugates(&class);

        let fiber = FiberIterator {
            fiber: class.fiber(),
            iter,
        };

        FiberClassIterator {
            fiber,
            iter: iter_conj,
        }
    }
}

impl<'a, I: TensorStructure> Iterator for FiberClassIterator<'a, I> {
    type Item = FiberIterator<'a, I>;

    fn next(&mut self) -> Option<Self::Item> {
        let shift = self.iter.next()?;
        self.fiber.shift(shift);
        Some(self.fiber.clone())
    }
}

/// An iterator over all indices of a tensor structure, keeping an index fixed

pub struct TensorStructureFiberIterator {
    pub varying_fiber_index: usize,
    pub increment: usize,
    pub stride_shift: Option<(usize, usize)>,
    pub max: usize,
}

/// Iterates over all indices not in the fiber, and at each index, iterates over the fiber
impl TensorStructureFiberIterator {
    /// Create a new fiber iterator
    ///
    /// # Arguments
    ///
    /// * `structure` - The tensor structure
    /// * `fiber_position` - The position of the fiber in the structure
    ///
    /// # Panics
    ///
    /// If the fiber index is out of bounds
    pub fn new<S>(structure: &S, fiber_position: usize) -> Self
    where
        S: TensorStructure,
    {
        assert!(fiber_position < structure.order(), "Invalid fiber index");

        let strides = structure.strides();
        let fiber_stride = strides[fiber_position];
        let dim: usize = structure.shape()[fiber_position].into();

        let max = structure.size() - fiber_stride * (dim - 1) - 1;

        let mut stride = None;
        let mut shift = None;
        let mut increment = 1;

        if fiber_position == structure.order() - 1 {
            increment = *strides.get(structure.order().wrapping_sub(2)).unwrap_or(&1);
        } else if fiber_position != 0 {
            shift = Some(strides[fiber_position - 1]);
            stride = Some(strides[fiber_position]);
        }

        TensorStructureFiberIterator {
            varying_fiber_index: 0,
            increment,
            stride_shift: stride.zip(shift),
            max,
        }
    }

    /// Reset the iterator, so that it can be used again.
    ///
    /// This is cheaper than creating a new iterator.
    pub fn reset(&mut self) {
        self.varying_fiber_index = 0;
    }
}

impl Iterator for TensorStructureFiberIterator {
    type Item = usize;

    /// Icrement the iterator
    ///
    /// Inrements the varying fiber index by the increment, and checks if the index is divisible by the stride, in which case it adds the shift
    fn next(&mut self) -> Option<Self::Item> {
        if self.varying_fiber_index > self.max {
            return None;
        }
        let ret = self.varying_fiber_index;

        self.varying_fiber_index += self.increment;

        if let Some((stride, shift)) = self.stride_shift {
            if self.varying_fiber_index % stride == 0 {
                self.varying_fiber_index += shift - stride;
            }
        }

        Some(ret)
    }
}

pub struct TensorStructureFiberClassIterator<'a, S: TensorStructure> {
    pub fiber: Fiber<'a, S>,
    pub increment: usize,
    pub fixed_strides: Vec<usize>,
    pub shifts: Vec<usize>,
    pub max: usize,
}

/// Iterator over all indices of a tensor structure, fixing a subset of indices
///
/// This version directly iterates over the expanded indices.
///
pub struct TensorStructureMultiFiberIteratorExpanded {
    indices: Vec<usize>,
    dims: Vec<Dimension>,
    positions: Vec<usize>,
    length: usize,
    carry: bool,
}

impl TensorStructureMultiFiberIteratorExpanded {
    pub fn new<N>(structure: &N, fiber_positions: &[bool]) -> Self
    where
        N: TensorStructure,
    {
        let positions = fiber_positions
            .iter()
            .enumerate()
            .filter_map(|(i, p)| if *p { None } else { Some(i) })
            .collect();
        let mut dims = structure.shape();
        let mut filter = fiber_positions.iter();
        dims.retain(|_| !*filter.next().unwrap_or_else(|| unreachable!()));

        Self {
            indices: vec![0; dims.len()],
            dims,
            positions,
            length: fiber_positions.len(),
            carry: false,
        }
    }
}

impl Iterator for TensorStructureMultiFiberIteratorExpanded {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.carry {
            return None;
        }
        self.carry = true;

        let mut out = vec![0; self.length];

        for (i, p) in self.positions.iter().zip(self.indices.iter()) {
            out[*i] = *p;
        }

        for (i, r) in self.indices.iter_mut().zip(self.dims.iter()).rev() {
            if self.carry {
                *i += 1;
                self.carry = *i == usize::from(*r);
                *i %= usize::from(*r);
            }
        }

        Some(out)
    }
}

/// Iterator over all indices of a tensor structure, fixing a subset of indices
///
/// `Item` is a flat index, corresponding to the index where the fixed indices are set to 0.
#[derive(Debug)]
pub struct TensorStructureMultiFiberIterator {
    pub varying_fiber_index: usize,
    pub increment: usize,
    pub fixed_strides: Vec<usize>,
    pub shifts: Vec<usize>,
    pub max: usize,
}

use thiserror::Error;

#[derive(Error, Debug)]

enum FiberError {
    #[error("Fiber is not the right size")]
    IncompatibleSize,
}
impl TensorStructureMultiFiberIterator {
    /// Create a new multi fiber iterator
    ///
    /// # Arguments
    ///
    /// * `structure` - The tensor structure
    /// * `fiber_positions` - A boolean array indicating which indices are fixed
    ///
    /// # Algorithm
    ///
    /// Smartly constructs the shifts and strides for each fixed index.
    /// It skippes over adjacent fixed indices for the strides.
    ///
    pub fn new<N>(fiber: &Fiber<N>) -> TensorStructureMultiFiberIterator
    where
        N: TensorStructure,
    {
        let strides = fiber.structure.strides();
        let dims = fiber.structure.shape();
        let order = fiber.structure.order();
        let mut max = 0;

        // max -= 1;

        let mut increment = 1;

        let mut fixed_strides = vec![];
        let mut shifts = vec![];

        let mut before = true;
        let mut has_seen_stride = false;
        let mut first = true;

        for pos in (0..order).rev() {
            let fi = fiber[pos];

            if fi.is_fixed() && !before && !first {
                has_seen_stride = true;
                fixed_strides.push(strides[pos]);
            }
            if fi.is_free() && before && has_seen_stride {
                shifts.push(strides[pos]);
            }

            if fi.is_free() {
                max += (usize::from(dims[pos]) - 1) * strides[pos];
                if first {
                    increment = strides[pos];
                    first = false;
                }
            }

            before = fi.is_fixed();
        }

        if fixed_strides.len() > shifts.len() {
            fixed_strides.pop();
        }

        TensorStructureMultiFiberIterator {
            varying_fiber_index: 0,
            increment,
            fixed_strides,
            shifts,
            max,
        }
    }

    /// Construct a pair of multi fiber iterators, each one taking as fixed what the other takes as free.
    ///
    /// # Arguments
    ///
    /// * `structure` - The tensor structure
    /// * `fiber_positions` - A boolean array indicating which indices are fixed for the second iterator
    ///
    /// # Returns
    ///
    /// A pair of multi fiber iterators.
    /// The first iterator considers the true values of `fiber_positions` as free, and the second iterator considers them as fixed.
    pub fn new_conjugate<N>(structure: &N, fiber_positions: &[bool]) -> (Self, Self)
    where
        N: TensorStructure,
    {
        let strides = structure.strides();
        let dims = structure.shape();
        let order = structure.order();
        let mut max = 0;

        // max -= 1;

        let mut increment = 1;

        // println!("{:?}{}", fiber_positions, increment);
        let mut fixed_strides = vec![];
        let mut fixed_strides_conj = vec![];
        let mut shifts = vec![];
        let mut shifts_conj = vec![];

        let mut before = true;
        let mut has_seen_stride = false;
        let mut has_seen_stride_conj = false;
        let mut first = true;
        let mut first_conj = true;
        let mut increment_conj = 1;

        let mut max_conj = 0;

        for pos in (0..order).rev() {
            let is_fixed = fiber_positions[pos];

            if is_fixed && !before {
                if !first {
                    has_seen_stride = true;
                    fixed_strides.push(strides[pos]);
                }

                if has_seen_stride_conj {
                    shifts_conj.push(strides[pos]);
                }
            }
            if !is_fixed && before {
                if has_seen_stride {
                    shifts.push(strides[pos]);
                }

                if !first_conj {
                    fixed_strides_conj.push(strides[pos]);
                    has_seen_stride_conj = true;
                }
            }

            if is_fixed {
                max_conj += (usize::from(dims[pos]) - 1) * strides[pos];
                if first_conj {
                    increment_conj = strides[pos];
                    first_conj = false;
                }
            } else {
                max += (usize::from(dims[pos]) - 1) * strides[pos];
                if first {
                    increment = strides[pos];
                    first = false;
                }
            }

            before = is_fixed;
        }

        if fixed_strides.len() > shifts.len() {
            fixed_strides.pop();
        }

        if fixed_strides_conj.len() > shifts_conj.len() {
            fixed_strides_conj.pop();
        }

        (
            TensorStructureMultiFiberIterator {
                varying_fiber_index: 0,
                increment: increment_conj,
                fixed_strides: fixed_strides_conj,
                shifts: shifts_conj,
                max: max_conj,
            },
            TensorStructureMultiFiberIterator {
                varying_fiber_index: 0,
                increment,
                fixed_strides,
                shifts,
                max,
            },
        )
    }

    /// Reset the iterator, so that it can be used again.
    ///
    /// This is cheaper than creating a new iterator.
    pub fn reset(&mut self) {
        self.varying_fiber_index = 0;
    }
}

impl Iterator for TensorStructureMultiFiberIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.varying_fiber_index > self.max {
            return None;
        }
        let ret = self.varying_fiber_index;

        self.varying_fiber_index += self.increment;

        for (i, s) in self.fixed_strides.iter().enumerate() {
            if self.varying_fiber_index % s == 0 {
                self.varying_fiber_index += self.shifts[i] - s;
            } else {
                break;
            }
        }

        Some(ret)
    }
}

/// Iterator over all indices of a tensor structure, fixing a subset of indices.
///
/// This version is mostly used as the iterator over the fixed indices (considering them free), when paired with `TensorStructureMultiFiberIterator` that considers the fixed indices as truely fixed.
///
/// `Item` is a flat index, a count, and a boolean indicating if the metric is negative.
/// The flat index corresponds to the index where the fixed indices are set to 0.
///
///
/// It generates the indices to then compute the tensor product of the metric along the fiber.
///
#[derive(Debug)]
pub struct TensorStructureMultiFiberMetricIterator {
    pub iterator: TensorStructureMultiFiberIterator,
    pos: usize,
    pub reps: Vec<Representation>,
    pub indices: Vec<usize>,
    pub is_neg: AHashMap<usize, bool>,
    map: Vec<usize>,
    permutation: Permutation,
}

impl TensorStructureMultiFiberMetricIterator {
    /// Create a new multi fiber metric iterator
    ///
    ///
    ///
    /// # Arguments
    ///
    pub fn new<N>(
        fiber: &Fiber<N>,
        permutation: Permutation,
    ) -> TensorStructureMultiFiberMetricIterator
    where
        N: TensorStructure,
    {
        // for f in fiber_positions {
        //     filter[*f] = true;
        // }
        let iterator = TensorStructureMultiFiberIterator::new(fiber);

        let mut f = fiber.bare_fiber.indices.iter();
        let mut reps = fiber.structure.reps();
        reps.retain(|_| f.next().unwrap_or_else(|| unreachable!()).is_free());
        let capacity = reps.iter().map(usize::from).product();
        let indices = vec![0; reps.len()];
        // println!("Reps : {:?}", reps);

        TensorStructureMultiFiberMetricIterator {
            iterator,
            reps,
            indices,
            is_neg: AHashMap::with_capacity(capacity),
            pos: 0,
            permutation,
            map: Vec::new(),
        }
    }

    /// Construct a pair of multi fiber iterators.
    ///
    /// The first is a [`TensorStructureMultiFiberMetricIterator`], that iterates along the fiber, i.e. considers the fixed indices as free.
    ///
    /// The second is a [`TensorStructureMultiFiberIterator`] that considers the fixed indices as fixed, iterating over the remaining indices, not in the fiber.
    ///
    pub fn new_conjugates<N>(
        structure: &N,
        fiber_positions: &[bool],
        permutation: Permutation,
    ) -> (Self, TensorStructureMultiFiberIterator)
    where
        N: TensorStructure,
    {
        let iters = TensorStructureMultiFiberIterator::new_conjugate(structure, fiber_positions);
        let mut f = fiber_positions.iter();
        let mut reps = structure.reps();
        reps.retain(|_| *f.next().unwrap_or_else(|| unreachable!()));
        let capacity = reps.iter().map(usize::from).product();
        let indices = vec![0; reps.len()];
        // println!("Reps : {:?}", reps);
        (
            TensorStructureMultiFiberMetricIterator {
                iterator: iters.0,
                reps,
                indices,
                is_neg: AHashMap::with_capacity(capacity),
                pos: 0,
                permutation,
                map: Vec::new(),
            },
            iters.1,
        )
    }

    fn increment_indices(&mut self) {
        let mut carry = true;
        for (i, r) in self.indices.iter_mut().rev().zip(self.reps.iter().rev()) {
            if carry {
                *i += 1;
                carry = *i == usize::from(r);
                *i %= usize::from(r);
            }
        }
    }

    fn has_neg(&mut self, i: usize) -> bool {
        if let Some(is_neg) = self.is_neg.get(&i) {
            return *is_neg;
        }
        let mut neg = false;
        for (i, r) in self
            .indices
            .iter()
            .zip(self.reps.iter())
            .filter(|(_, r)| !matches!(r, Representation::Euclidean(_)))
        {
            neg ^= r.is_neg(*i);
        }

        self.is_neg.insert(i, neg);

        neg
    }

    fn generate_map(&mut self) {
        // println!("{:?}{:?}", self.map.len(), self.pos);
        if self.map.len() <= self.pos {
            let mut stride = 1;
            // println!("{:?}", self.indices);
            // println!("{:?}", self.reps);
            // println!("{:?}", self.permutation);
            let mut ind = 0;
            for (i, d) in self
                .permutation
                .apply_slice(self.indices.as_slice())
                .iter()
                .zip(self.permutation.apply_slice(self.reps.as_slice()).iter())
                .rev()
            {
                ind += stride * i;
                stride *= usize::from(d);
            }
            self.map.push(ind);
            // println!("{:?}", self.map)
        }
    }

    pub fn reset(&mut self) -> &[usize] {
        self.iterator.reset();
        self.indices = vec![0; self.reps.len()];
        self.pos = 0;
        // println!("reset!");
        &self.map
    }
}

impl Iterator for TensorStructureMultiFiberMetricIterator {
    type Item = (usize, usize, bool);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.iterator.next() {
            let pos = self.pos;
            self.generate_map();
            self.pos += 1;
            let neg = self.has_neg(i);
            self.increment_indices();
            Some((pos, i, neg))
        } else {
            None
        }
    }
}

/// Iterator over all indices of a tensor structure, fixing an index
///
/// The iterator returns a vector of references to the elements of the tensor along the fixed index (so called fiber).
///
/// In the case of a sparse tensor, the iterator returns a vector of references to the non-zero elements of the tensor along the fixed index, and the amount of skipped indices.
/// For the dense tensor, the iterator only returns the the vectro of references.
pub struct TensorFiberIterator<'a, T>
where
    T: TensorStructure,
{
    tensor: &'a T,
    fiber_iter: TensorStructureFiberIterator,
    skipped: usize,
    pub fiber_dimension: Dimension,
    increment: usize,
}

impl<'a, T> TensorFiberIterator<'a, T>
where
    T: TensorStructure,
{
    /// Create a new fiber iterator, from a tensor and a fixed index
    pub fn new(tensor: &'a T, fiber_position: usize) -> Self {
        let fiber_iter =
            TensorStructureFiberIterator::new(&tensor.external_structure(), fiber_position);
        let increment = tensor.strides()[fiber_position];

        TensorFiberIterator {
            tensor,
            fiber_iter,
            skipped: 0,
            fiber_dimension: tensor.shape()[fiber_position],
            increment,
        }
    }

    /// Reset the iterator, so that it can be used again.
    #[must_use]
    pub fn reset(&mut self) -> usize {
        self.fiber_iter.reset();
        let skipped = self.skipped;
        self.skipped = 0;
        skipped
    }
}

impl<'a, T, N> Iterator for TensorFiberIterator<'a, SparseTensor<T, N>>
where
    N: TensorStructure,
{
    type Item = (usize, Vec<ConcreteIndex>, Vec<&'a T>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(s) = self.fiber_iter.next() {
            let mut out = Vec::new();
            let mut nonzeros = Vec::new();
            for i in 0..self.fiber_dimension.into() {
                if let Some(v) = self.tensor.elements.get(&(s + i * self.increment)) {
                    nonzeros.push(i);
                    out.push(v);
                }
            }
            if out.is_empty() {
                self.skipped += 1;
                self.next()
            } else {
                let skipped = self.skipped;
                self.skipped = 0;
                Some((skipped, nonzeros, out))
            }
        } else {
            None
        }
    }
}

impl<'a, T, N> Iterator for TensorFiberIterator<'a, DenseTensor<T, N>>
where
    N: TensorStructure,
{
    type Item = Vec<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(s) = self.fiber_iter.next() {
            let mut out = Vec::with_capacity(self.fiber_dimension.into());
            for i in 0..self.fiber_dimension.into() {
                if let Some(v) = self.tensor.get_linear(s + i * self.increment) {
                    out.push(v);
                }
            }

            Some(out)
        } else {
            None
        }
    }
}

/// Iterator over all indices of a tensor, fixing a subset of indices
///
/// The iterator returns a vector of tuples of references to the elements of the tensor along the fixed indices (so called fibers) and bools indicating the sign of the metric.
/// It flattens the indices of the fibers. To accomodate for a permutation of the ordering of the fixed indices, the iterator builds a map between the indices of output vector and the permuted version.
///
/// This map can be accessed using the `map` field of the iterator.
///
/// For the sparse tensor, the iterator also returns a vector of references to the non-zero elements of the tensor along the fixed indices, and the amount of skipped indices.
/// For the dense tensor, the iterator only returns the the vector of references.
///
///
pub struct TensorMultiFiberMetricIterator<'a, T>
where
    T: TensorStructure,
{
    tensor: &'a T,
    fiber_iter: TensorStructureMultiFiberMetricIterator,
    free_iter: TensorStructureMultiFiberIterator,
    skipped: usize,
    pub map: Vec<usize>,
    capacity: usize,
}

impl<'a, T> TensorMultiFiberMetricIterator<'a, T>
where
    T: TensorStructure,
{
    /// Create a new multi fiber metric iterator
    ///
    /// # Arguments
    ///
    /// * `tensor` - A reference to the tensor
    /// * `fiber_positions` - A boolean array indicating which indices are fixed
    /// * `permutation` - A permutation of the fixed indices, that matches another tensor.
    pub fn new(tensor: &'a T, fiber_positions: &[bool], permutation: Permutation) -> Self {
        let iters = TensorStructureMultiFiberMetricIterator::new_conjugates(
            &tensor.external_structure(),
            fiber_positions,
            permutation,
        );

        let mut f = fiber_positions.iter();
        let mut dims = tensor.shape();
        dims.retain(|_| !*f.next().unwrap_or_else(|| unreachable!()));
        let capacity = dims.iter().map(|d| usize::from(*d)).product();
        TensorMultiFiberMetricIterator {
            tensor,
            map: vec![],
            fiber_iter: iters.0,
            free_iter: iters.1,
            skipped: 0,
            capacity,
        }
    }

    /// Reset the iterator, so that it can be used again.
    #[must_use]
    pub fn reset(&mut self) -> usize {
        self.fiber_iter.reset();
        self.free_iter.reset();
        let skipped = self.skipped;
        self.skipped = 0;
        skipped
    }
}

impl<'a, T, N> Iterator for TensorMultiFiberMetricIterator<'a, SparseTensor<T, N>>
where
    N: TensorStructure,
{
    type Item = (usize, Vec<ConcreteIndex>, Vec<(&'a T, bool)>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.free_iter.next() {
            let mut out = Vec::new();
            let mut nonzeros = Vec::new();
            for (pos, ind, met) in self.fiber_iter.by_ref() {
                if let Some(v) = self.tensor.get_linear(ind + i) {
                    out.push((v, met));
                    nonzeros.push(pos);
                    // println!("hi");
                }
            }
            if self.map.is_empty() {
                self.map = self.fiber_iter.reset().to_owned();
            } else {
                self.fiber_iter.reset();
            }
            if out.is_empty() {
                self.skipped += 1;
                self.next()
            } else {
                let skipped = self.skipped;
                self.skipped = 0;
                Some((skipped, nonzeros, out))
            }
        } else {
            None
        }
    }
}

impl<'a, T, N> Iterator for TensorMultiFiberMetricIterator<'a, DenseTensor<T, N>>
where
    N: TensorStructure,
{
    type Item = Vec<(&'a T, bool)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.free_iter.next() {
            let mut out = Vec::with_capacity(self.capacity);
            for (_, ind, met) in self.fiber_iter.by_ref() {
                if let Some(v) = self.tensor.get_linear(ind + i) {
                    out.push((v, met));
                }
            }
            if self.map.is_empty() {
                self.map = self.fiber_iter.reset().to_owned();
            } else {
                self.fiber_iter.reset();
            }

            self.fiber_iter.reset();
            if out.is_empty() {
                return self.next();
            }
            Some(out)
        } else {
            None
        }
    }
}

/// Iterator over all indices of a tensor, fixing a subset of indices
///
/// The iterator returns a vector of references to the elements of the tensor along the fixed indices (so called fibers).
/// It flattens the indices of the fibers. To accomodate for a permutation of the ordering of the fixed indices, the iterator builds a map between the indices of output vector and the permuted version.
///
/// This map can be accessed using the `map` field of the iterator.
///
/// For the sparse tensor, the iterator also returns a vector of references to the non-zero elements of the tensor along the fixed indices, and the amount of skipped indices.
/// For the dense tensor, the iterator only returns the the vector of references.
///
pub struct TensorMultiFiberIterator<'a, T>
where
    T: TensorStructure,
{
    tensor: &'a T,
    fiber_iter: TensorStructureMultiFiberIterator,
    free_iter: TensorStructureMultiFiberIterator,
    skipped: usize,
}

impl<'a, T> TensorMultiFiberIterator<'a, T>
where
    T: TensorStructure,
{
    pub fn new(tensor: &'a T, fiber_positions: &[bool]) -> Self {
        let iters = TensorStructureMultiFiberIterator::new_conjugate(
            &tensor.external_structure(),
            fiber_positions,
        );
        TensorMultiFiberIterator {
            tensor,
            fiber_iter: iters.0,
            free_iter: iters.1,
            skipped: 0,
        }
    }

    #[must_use]
    pub fn reset(&mut self) -> usize {
        self.fiber_iter.reset();
        self.free_iter.reset();
        let skipped = self.skipped;
        self.skipped = 0;
        skipped
    }
}

impl<'a, T, N> Iterator for TensorMultiFiberIterator<'a, SparseTensor<T, N>>
where
    N: TensorStructure,
{
    type Item = (usize, Vec<ConcreteIndex>, Vec<&'a T>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.free_iter.next() {
            let mut out = Vec::new();
            let mut nonzeros = Vec::new();
            for (pos, ind) in self.fiber_iter.by_ref().enumerate() {
                if let Some(v) = self.tensor.get_linear(ind + i) {
                    out.push(v);
                    nonzeros.push(pos);
                }
            }
            self.fiber_iter.reset();
            if out.is_empty() {
                self.skipped += 1;
                self.next()
            } else {
                let skipped = self.skipped;
                self.skipped = 0;
                Some((skipped, nonzeros, out))
            }
        } else {
            None
        }
    }
}

impl<'a, T, N> Iterator for TensorMultiFiberIterator<'a, DenseTensor<T, N>>
where
    N: TensorStructure,
{
    type Item = Vec<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.free_iter.next() {
            let mut out = Vec::new();
            for ind in self.fiber_iter.by_ref() {
                if let Some(v) = self.tensor.get_linear(ind + i) {
                    out.push(v);
                }
            }
            self.fiber_iter.reset();
            Some(out)
        } else {
            None
        }
    }
}

/// Iterator over all the elements of a sparse tensor
///
/// Returns the expanded indices and the element at that index
pub struct SparseTensorIterator<'a, T, N> {
    iter: std::collections::hash_map::Iter<'a, usize, T>,
    structure: &'a N,
}

impl<'a, T, N> SparseTensorIterator<'a, T, N> {
    fn new(tensor: &'a SparseTensor<T, N>) -> Self {
        SparseTensorIterator {
            iter: tensor.elements.iter(),
            structure: &tensor.structure,
        }
    }
}

impl<'a, T, N> Iterator for SparseTensorIterator<'a, T, N>
where
    N: TensorStructure,
{
    type Item = (Vec<ConcreteIndex>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((k, v)) = self.iter.next() {
            let indices = self.structure.expanded_index(*k).unwrap();
            Some((indices, v))
        } else {
            None
        }
    }
}

/// Iterator over all the elements of a sparse tensor
///
/// Returns the flat index and the element at that index

pub struct SparseTensorLinearIterator<'a, T> {
    iter: std::collections::hash_map::Iter<'a, usize, T>,
}

impl<'a, T> SparseTensorLinearIterator<'a, T> {
    pub fn new<N>(tensor: &'a SparseTensor<T, N>) -> Self {
        SparseTensorLinearIterator {
            iter: tensor.elements.iter(),
        }
    }
}

impl<'a, T> Iterator for SparseTensorLinearIterator<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, v)| (*k, v))
    }
}

// impl<'a, T, I> IntoIterator for &'a SparseTensor<T, I> {
//     type Item = (&'a Vec<ConcreteIndex>, &'a T);
//     type IntoIter = SparseTensorIterator<'a, T>;

//     fn into_iter(self) -> Self::IntoIter {
//         SparseTensorIterator::new(self)
//     }
// }

/// Iterator over all but two indices of a sparse tensor, where the two indices are traced
///
/// The iterator next returns the value of the trace at the current indices, and the current indices

pub struct SparseTensorTraceIterator<'a, T, I> {
    tensor: &'a SparseTensor<T, I>,
    trace_indices: [usize; 2],
    current_indices: Vec<ConcreteIndex>,
    done: bool,
}

impl<'a, T, I> SparseTensorTraceIterator<'a, T, I>
where
    I: TensorStructure,
{
    /// Create a new trace iterator
    ///
    /// # Arguments
    ///
    /// * `tensor` - A reference to the tensor
    /// * `trace_indices` - The indices to be traced
    ///
    /// # Panics
    ///
    /// Panics if the trace indices do not point to the same dimension
    fn new(tensor: &'a SparseTensor<T, I>, trace_indices: [usize; 2]) -> Self {
        //trace positions must point to the same dimension
        assert!(
            trace_indices
                .iter()
                .map(|&pos| tensor.external_structure()[pos].representation)
                .collect::<Vec<_>>()
                .iter()
                .all(|&sig| sig == tensor.external_structure()[trace_indices[0]].representation),
            "Trace indices must point to the same dimension"
        );
        SparseTensorTraceIterator {
            tensor,
            trace_indices,
            current_indices: vec![0; tensor.order()],
            done: false,
        }
    }

    fn increment_indices(&mut self) -> bool {
        for (i, index) in self
            .current_indices
            .iter_mut()
            .enumerate()
            .rev()
            .filter(|(pos, _)| !self.trace_indices.contains(pos))
        // Filter out the trace indices
        {
            *index += 1;
            // If the index goes beyond the shape boundary, wrap around to 0
            if index >= &mut usize::from(self.tensor.shape()[i]) {
                *index = 0;
                continue; // carry over to the next dimension
            }
            return true; // We've successfully found the next combination
        }
        false // No more combinations left
    }
}

impl<'a, T, I> Iterator for SparseTensorTraceIterator<'a, T, I>
where
    T: for<'c> std::ops::AddAssign<&'c T>
        + for<'b> std::ops::SubAssign<&'b T>
        + std::ops::Neg<Output = T>
        + Clone,
    I: TensorStructure + Clone,
{
    type Item = (Vec<ConcreteIndex>, T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let trace_dimension =
            self.tensor.external_structure()[self.trace_indices[0]].representation;
        let trace_sign = trace_dimension.negative();
        let mut iter = trace_sign.iter().enumerate();
        let mut indices = self.current_indices.clone();
        let (i, mut sign) = iter.next().unwrap(); //First element (to eliminate the need for default)

        indices[self.trace_indices[0]] = i;
        indices[self.trace_indices[1]] = i;

        // Data might not exist at that concrete index usize, we advance it till it does, and if not we skip

        while self.tensor.is_empty_at(&indices) {
            let Some((i, signint)) = iter.next() else {
                self.done = !self.increment_indices();
                return self.next(); // skip
            };
            indices[self.trace_indices[0]] = i;
            indices[self.trace_indices[1]] = i;
            sign = signint;
        }

        let value = (*self.tensor.get(&indices).unwrap()).clone(); //Should now be safe to unwrap
        let mut trace = if *sign { value.neg() } else { value };

        for (i, sign) in iter {
            indices[self.trace_indices[0]] = i;
            indices[self.trace_indices[1]] = i;
            if let Ok(value) = self.tensor.get(&indices) {
                if *sign {
                    trace -= value;
                } else {
                    trace += value;
                }
            }
        }

        //make a vector withouth the trace indices
        let trace_indices: Vec<ConcreteIndex> = self
            .current_indices
            .clone()
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| !self.trace_indices.contains(&i))
            .map(|(_, x)| x)
            .collect();

        self.done = !self.increment_indices();

        Some((trace_indices, trace))
    }
}

impl<T, I> SparseTensor<T, I>
where
    I: TensorStructure,
{
    pub fn iter_fiber(&self, fiber_index: usize) -> TensorFiberIterator<Self> {
        TensorFiberIterator::new(self, fiber_index)
    }

    pub fn iter_trace(&self, trace_indices: [usize; 2]) -> SparseTensorTraceIterator<T, I> {
        SparseTensorTraceIterator::new(self, trace_indices)
    }

    pub fn iter(&self) -> SparseTensorIterator<T, I> {
        SparseTensorIterator::new(self)
    }

    pub fn iter_flat(&self) -> SparseTensorLinearIterator<T> {
        SparseTensorLinearIterator::new(self)
    }

    pub fn iter_multi_fibers_metric(
        &self,
        fiber_positions: &[bool],
        permutation: Permutation,
    ) -> TensorMultiFiberMetricIterator<Self> {
        TensorMultiFiberMetricIterator::new(self, fiber_positions, permutation)
    }

    pub fn iter_multi_fibers(&self, fiber_positions: &[bool]) -> TensorMultiFiberIterator<Self> {
        TensorMultiFiberIterator::new(self, fiber_positions)
    }
}

/// Iterator over all the elements of a dense tensor
///
/// Returns the expanded indices and the element at that index
pub struct DenseTensorIterator<'a, T, I> {
    tensor: &'a DenseTensor<T, I>,
    current_flat_index: usize,
}

impl<'a, T, I> DenseTensorIterator<'a, T, I> {
    /// Create a new dense tensor iterator
    ///
    /// # Arguments
    ///
    /// * `tensor` - A reference to the tensor
    fn new(tensor: &'a DenseTensor<T, I>) -> Self {
        DenseTensorIterator {
            tensor,
            current_flat_index: 0,
        }
    }
}

impl<'a, T, I> Iterator for DenseTensorIterator<'a, T, I>
where
    I: TensorStructure,
{
    type Item = (Vec<ConcreteIndex>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(indices) = self.tensor.expanded_index(self.current_flat_index) {
            let value = self.tensor.get_linear(self.current_flat_index).unwrap();

            self.current_flat_index += 1;

            Some((indices, value))
        } else {
            None
        }
    }
}

/// Iterator over all the elements of a dense tensor
///
/// Returns the flat index and the element at that index

pub struct DenseTensorLinearIterator<'a, T, I> {
    tensor: &'a DenseTensor<T, I>,
    current_flat_index: usize,
}

impl<'a, T, I> DenseTensorLinearIterator<'a, T, I> {
    pub fn new(tensor: &'a DenseTensor<T, I>) -> Self {
        DenseTensorLinearIterator {
            tensor,
            current_flat_index: 0,
        }
    }
}

impl<'a, T, I> Iterator for DenseTensorLinearIterator<'a, T, I>
where
    I: TensorStructure,
{
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.tensor.get_linear(self.current_flat_index)?;
        let index = self.current_flat_index;
        self.current_flat_index += 1;
        Some((index, value))
    }
}

impl<'a, T, I> IntoIterator for &'a DenseTensor<T, I>
where
    I: TensorStructure,
{
    type Item = (Vec<ConcreteIndex>, &'a T);
    type IntoIter = DenseTensorIterator<'a, T, I>;

    fn into_iter(self) -> Self::IntoIter {
        DenseTensorIterator::new(self)
    }
}

impl<T, I> IntoIterator for DenseTensor<T, I>
where
    I: TensorStructure,
{
    type Item = (Vec<ConcreteIndex>, T);
    type IntoIter = DenseTensorIntoIterator<T, I>;

    fn into_iter(self) -> Self::IntoIter {
        DenseTensorIntoIterator::new(self)
    }
}

/// An consuming iterator over the elements of a dense tensor
///
/// Returns the expanded indices and the element at that index
///
///
pub struct DenseTensorIntoIterator<T, I> {
    tensor: DenseTensor<T, I>,
    current_flat_index: usize,
}

impl<T, I> DenseTensorIntoIterator<T, I> {
    fn new(tensor: DenseTensor<T, I>) -> Self {
        DenseTensorIntoIterator {
            tensor,
            current_flat_index: 0,
        }
    }
}

impl<T, I> Iterator for DenseTensorIntoIterator<T, I>
where
    I: TensorStructure,
{
    type Item = (Vec<ConcreteIndex>, T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(indices) = self.tensor.expanded_index(self.current_flat_index) {
            let indices = indices.clone();
            let value = self.tensor.data.remove(self.current_flat_index);

            self.current_flat_index += 1;

            Some((indices, value))
        } else {
            None
        }
    }
}

/// Iterator over all indices of a dense tensor, keeping two indices fixed and tracing over them
///
/// The next method returns the value of the trace at the current indices, and the current indices
pub struct DenseTensorTraceIterator<'a, T, I> {
    tensor: &'a DenseTensor<T, I>,
    trace_indices: [usize; 2],
    current_indices: Vec<ConcreteIndex>,
    done: bool,
}

impl<'a, T, I> DenseTensorTraceIterator<'a, T, I>
where
    I: TensorStructure,
{
    /// Create a new trace iterator
    ///
    /// # Arguments
    ///
    /// * `tensor` - A reference to the tensor
    /// * `trace_indices` - The indices to be traced
    ///
    fn new(tensor: &'a DenseTensor<T, I>, trace_indices: [usize; 2]) -> Self {
        assert!(trace_indices.len() >= 2, "Invalid trace indices");
        //trace positions must point to the same dimension
        assert!(
            trace_indices
                .iter()
                .map(|&pos| tensor.external_structure()[pos].representation)
                .collect::<Vec<_>>()
                .iter()
                .all(|&sig| sig == tensor.external_structure()[trace_indices[0]].representation),
            "Trace indices must point to the same dimension"
        );
        DenseTensorTraceIterator {
            tensor,
            trace_indices,
            current_indices: vec![0; tensor.order()],
            done: false,
        }
    }

    fn increment_indices(&mut self) -> bool {
        for (i, index) in self
            .current_indices
            .iter_mut()
            .enumerate()
            .rev()
            .filter(|(pos, _)| !self.trace_indices.contains(pos))
        {
            *index += 1;
            // If the index goes beyond the shape boundary, wrap around to 0
            if index >= &mut self.tensor.shape()[i] {
                *index = 0;
                continue; // carry over to the next dimension
            }
            return true; // We've successfully found the next combination
        }
        false // No more combinations left
    }
}

impl<'a, T, I> Iterator for DenseTensorTraceIterator<'a, T, I>
where
    T: for<'c> AddAssign<&'c T>
        + for<'b> SubAssign<&'b T>
        + Neg<Output = T>
        + Clone
        + std::fmt::Debug,
    I: TensorStructure,
{
    type Item = (Vec<ConcreteIndex>, T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let trace_dimension =
            self.tensor.external_structure()[self.trace_indices[0]].representation;
        let trace_sign = trace_dimension.negative();

        let mut iter = trace_sign.iter().enumerate();
        let mut indices = self.current_indices.clone();
        let (_, sign) = iter.next().unwrap(); //First sign

        for pos in self.trace_indices {
            indices[pos] = 0;
        }

        let value = self.tensor.get(&indices).unwrap().clone();

        let mut trace = if *sign { value.neg() } else { value };

        for (i, sign) in iter {
            for pos in self.trace_indices {
                indices[pos] = i;
            }

            if let Ok(value) = self.tensor.get(&indices) {
                if *sign {
                    trace -= value;
                } else {
                    trace += value;
                }
            }
        }

        //make a vector without the trace indices
        let trace_indices: Vec<ConcreteIndex> = self
            .current_indices
            .clone()
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| !self.trace_indices.contains(&i))
            .map(|(_, x)| x)
            .collect();

        self.done = !self.increment_indices();

        Some((trace_indices, trace))
    }
}

impl<T, I> DenseTensor<T, I>
where
    I: TensorStructure,
{
    pub fn iter(&self) -> DenseTensorIterator<T, I> {
        DenseTensorIterator::new(self)
    }

    pub fn iter_flat(&self) -> DenseTensorLinearIterator<T, I> {
        DenseTensorLinearIterator::new(self)
    }

    pub fn iter_fiber(&self, fixedindex: usize) -> TensorFiberIterator<Self> {
        TensorFiberIterator::new(self, fixedindex)
    }

    pub fn iter_trace(&self, trace_indices: [usize; 2]) -> DenseTensorTraceIterator<T, I> {
        DenseTensorTraceIterator::new(self, trace_indices)
    }

    pub fn iter_multi_fibers_metric(
        &self,
        fiber_positions: &[bool],
        permutation: Permutation,
    ) -> TensorMultiFiberMetricIterator<Self> {
        TensorMultiFiberMetricIterator::new(self, fiber_positions, permutation)
    }

    pub fn iter_multi_fibers(&self, fiber_positions: &[bool]) -> TensorMultiFiberIterator<Self> {
        TensorMultiFiberIterator::new(self, fiber_positions)
    }
}
