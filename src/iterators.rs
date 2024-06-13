//! Iterators for tensors
//!
//! Iterators for tensors are used to iterate over the elements of a tensor.
//! More specialized iterators are provided that fix a certain subset of indices, and iterate over the remaining indices.
//! At each iteration, the iterator returns a vector of references to the elements of the tensor along the fixed indices (so called fibers).
//!
//! The iterators are built using the basic index iterators provided by the `TensorStructureIterator`s.
//!

use std::{
    fmt::{Debug, Display},
    ops::Index,
};

use crate::{
    ContractableWith, ExpandedIndex, FallibleAddAssign, FallibleSubAssign, FlatIndex, RefZero,
    VecStructure,
};

use super::{
    ConcreteIndex, DenseTensor, Dimension, GetTensorData, HasStructure, Representation, Slot,
    SparseTensor,
};

use gat_lending_iterator::LendingIterator;

use crate::Permutation;
use bitvec::vec::BitVec;
use serde::{Deserialize, Serialize};

pub trait AbstractFiberIndex {
    fn is_free(&self) -> bool;

    fn is_fixed(&self) -> bool {
        !self.is_free()
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
            true
        } else {
            false
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
            true
        } else {
            false
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

pub enum FiberData<'a> {
    Single(usize),
    Flat(FlatIndex),
    BoolFilter(&'a [bool]),
    Pos(&'a [isize]),
    IntFilter(&'a [u8]),
}

impl From<usize> for FiberData<'_> {
    fn from(value: usize) -> Self {
        Self::Single(value)
    }
}

impl<'a> From<&'a [bool]> for FiberData<'a> {
    fn from(value: &'a [bool]) -> Self {
        Self::BoolFilter(value)
    }
}

impl<'a> From<&'a [u8]> for FiberData<'a> {
    fn from(value: &'a [u8]) -> Self {
        Self::IntFilter(value)
    }
}

impl<'a> From<&'a [isize]> for FiberData<'a> {
    fn from(value: &'a [isize]) -> Self {
        Self::Pos(value)
    }
}

impl From<FlatIndex> for FiberData<'_> {
    fn from(value: FlatIndex) -> Self {
        Self::Flat(value)
    }
}

pub trait AbstractFiber<Out: AbstractFiberIndex>: Index<usize, Output = Out> {
    fn strides(&self) -> Vec<usize>;
    fn shape(&self) -> Vec<Dimension>;
    fn reps(&self) -> Vec<Representation>;
    fn order(&self) -> usize;
    fn single(&self) -> Option<usize>;
    fn bitvec(&self) -> BitVec;
}

#[derive(Debug, Clone)]
struct BareFiber {
    indices: Vec<FiberIndex>,
    is_single: FiberIndex,
}

impl BareFiber {
    #[allow(dead_code)]
    pub fn conj(self) -> Self {
        self
    }
    pub fn from<I: HasStructure>(data: FiberData, structure: &I) -> Self {
        match data {
            FiberData::Flat(i) => Self::from_flat(i, structure),
            FiberData::BoolFilter(b) => Self::from_filter(b),
            FiberData::Single(i) => {
                let mut out = Self::zeros(structure);
                out.free(i);
                out
            }
            FiberData::IntFilter(i) => {
                let mut out = Self::zeros(structure);
                for (pos, val) in i.iter().enumerate() {
                    if *val > 0 {
                        out.free(pos);
                    }
                }
                out
            }
            FiberData::Pos(i) => {
                let mut out = Self::zeros(structure);
                for (pos, val) in i.iter().enumerate() {
                    if *val < 0 {
                        out.free(pos);
                    } else {
                        out.fix(pos, *val as usize);
                    }
                }
                out
            }
        }
    }

    pub fn bitvec(&self) -> BitVec {
        self.indices.iter().map(|x| x.is_free()).collect()
    }

    pub fn bitvecinv(&self) -> BitVec {
        self.indices.iter().map(|x| x.is_fixed()).collect()
    }
    pub fn from_flat<I>(flat: FlatIndex, structure: &I) -> BareFiber
    where
        I: HasStructure,
    {
        let expanded = structure.expanded_index(flat).unwrap();

        BareFiber {
            indices: expanded.into_iter().map(FiberIndex::from).collect(),
            is_single: FiberIndex::Free,
        }
    }
    /// true is free, false is fixed
    pub fn from_filter(filter: &[bool]) -> BareFiber {
        let mut f = BareFiber {
            indices: filter
                .iter()
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
    pub fn zeros<I: HasStructure>(structure: &I) -> BareFiber {
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
pub struct Fiber<'a, I: HasStructure> {
    structure: &'a I,
    bare_fiber: BareFiber,
}

impl<'a, I: HasStructure> Clone for Fiber<'a, I> {
    fn clone(&self) -> Self {
        Fiber {
            structure: self.structure,
            bare_fiber: self.bare_fiber.clone(),
        }
    }
}

impl<'a, I> Index<usize> for Fiber<'a, I>
where
    I: HasStructure,
{
    type Output = FiberIndex;

    fn index(&self, index: usize) -> &Self::Output {
        &(self.bare_fiber[index])
    }
}

impl<'a, I> AbstractFiber<FiberIndex> for Fiber<'a, I>
where
    I: HasStructure,
{
    fn strides(&self) -> Vec<usize> {
        self.structure.strides()
    }

    fn reps(&self) -> Vec<Representation> {
        self.structure.reps()
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

    fn bitvec(&self) -> BitVec {
        self.bare_fiber.bitvec()
    }
}

impl<'a, S> Fiber<'a, S>
where
    S: HasStructure,
{
    pub fn conj(self) -> Self {
        self
    }

    pub fn iter(self) -> FiberIterator<'a, S, CoreFlatFiberIterator> {
        FiberIterator::new(self, false)
    }

    pub fn iter_conj(self) -> FiberIterator<'a, S, CoreFlatFiberIterator> {
        FiberIterator::new(self, true)
    }

    pub fn iter_perm(
        self,
        permutation: Permutation,
    ) -> FiberIterator<'a, S, CoreExpandedFiberIterator> {
        FiberIterator::new_permuted(self, permutation, false)
    }

    pub fn iter_perm_metric(
        self,
        permutation: Permutation,
    ) -> FiberIterator<'a, S, MetricFiberIterator> {
        FiberIterator::new_permuted(self, permutation, false)
    }

    pub fn from<'b>(data: FiberData<'b>, structure: &'a S) -> Self {
        Fiber {
            bare_fiber: BareFiber::from(data, structure),
            structure,
        }
    }

    pub fn bitvec(&self) -> BitVec {
        self.bare_fiber.bitvec()
    }

    pub fn bitvecinv(&self) -> BitVec {
        self.bare_fiber.bitvecinv()
    }
    pub fn from_flat(flat: FlatIndex, structure: &'a S) -> Fiber<'a, S> {
        Fiber {
            bare_fiber: BareFiber::from_flat(flat, structure),
            structure,
        }
    }
    /// true is free, false is fixed
    pub fn from_filter(filter: &[bool], structure: &'a S) -> Fiber<'a, S> {
        //check compatibility
        Fiber {
            bare_fiber: BareFiber::from_filter(filter),
            structure,
        }
    }
    pub fn zeros(structure: &'a S) -> Fiber<'a, S> {
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

impl<'a, I: HasStructure> Display for Fiber<'a, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for index in self.bare_fiber.indices.iter() {
            write!(f, "{} ", index)?
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FiberMut<'a, I: HasStructure> {
    structure: &'a mut I,
    bare_fiber: BareFiber,
}

impl<'a, I> Index<usize> for FiberMut<'a, I>
where
    I: HasStructure,
{
    type Output = FiberIndex;

    fn index(&self, index: usize) -> &Self::Output {
        &(self.bare_fiber[index])
    }
}

impl<'a, I> AbstractFiber<FiberIndex> for FiberMut<'a, I>
where
    I: HasStructure,
{
    fn strides(&self) -> Vec<usize> {
        self.structure.strides()
    }

    fn reps(&self) -> Vec<Representation> {
        self.structure.reps()
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

    fn bitvec(&self) -> BitVec {
        self.bare_fiber.bitvec()
    }
}

impl<'a, I> FiberMut<'a, I>
where
    I: HasStructure,
{
    pub fn from<'b>(data: FiberData<'b>, structure: &'a mut I) -> Self {
        //TODO check fiberdata compatibility
        FiberMut {
            bare_fiber: BareFiber::from(data, &*structure),
            structure,
        }
    }

    pub fn conj(self) -> Self {
        self
    }

    pub fn bitvec(&self) -> BitVec {
        self.bare_fiber.bitvec()
    }

    pub fn bitvecinv(&self) -> BitVec {
        self.bare_fiber.bitvecinv()
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

impl<'a, I: HasStructure> Display for FiberMut<'a, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for index in self.bare_fiber.indices.iter() {
            write!(f, "{} ", index)?
        }
        Ok(())
    }
}

impl<'a, I: HasStructure> FiberMut<'a, I> {
    pub fn iter(self) -> MutFiberIterator<'a, I, CoreFlatFiberIterator> {
        MutFiberIterator::new(self, false)
    }
}

pub struct FiberClass<'a, I: HasStructure> {
    structure: &'a I,
    bare_fiber: BareFiber, // A representant of the class

                           // /// true is fixed (but varying when iterating) and false is free (but fixed to 0 when iterating)
                           // free: BitVec, //check performance when it is AHashSet<usize>
}

impl<'a, I: HasStructure> Clone for FiberClass<'a, I> {
    fn clone(&self) -> Self {
        FiberClass {
            bare_fiber: self.bare_fiber.clone(),
            structure: self.structure,
        }
    }
}

impl<'a, I> Index<usize> for FiberClass<'a, I>
where
    I: HasStructure,
{
    type Output = FiberClassIndex;

    fn index(&self, index: usize) -> &Self::Output {
        if self.bare_fiber[index].is_fixed() {
            &FiberClassIndex::Free
        } else {
            &FiberClassIndex::Fixed
        }
    }
}

impl<'a, I: HasStructure> From<Fiber<'a, I>> for FiberClass<'a, I> {
    fn from(fiber: Fiber<'a, I>) -> Self {
        FiberClass {
            bare_fiber: fiber.bare_fiber,
            structure: fiber.structure,
        }
    }
}

impl<'a, I: HasStructure> From<FiberClass<'a, I>> for Fiber<'a, I> {
    fn from(fiber: FiberClass<'a, I>) -> Self {
        Fiber {
            bare_fiber: fiber.bare_fiber,
            structure: fiber.structure,
        }
    }
}

impl<'a, I: HasStructure> AbstractFiber<FiberClassIndex> for FiberClass<'a, I> {
    fn strides(&self) -> Vec<usize> {
        self.structure.strides()
    }

    fn shape(&self) -> Vec<Dimension> {
        self.structure.shape()
    }

    fn reps(&self) -> Vec<Representation> {
        self.structure.reps()
    }

    fn order(&self) -> usize {
        self.structure.order()
    }

    fn single(&self) -> Option<usize> {
        match self.bare_fiber.is_single {
            FiberIndex::Fixed(i) => Some(i),
            _ => None,
        }
    }

    fn bitvec(&self) -> BitVec {
        !self.bare_fiber.bitvec()
    }
}

impl<'a, S: HasStructure> FiberClass<'a, S> {
    pub fn iter(self) -> FiberClassIterator<'a, S> {
        FiberClassIterator::new(self)
    }

    pub fn iter_perm(
        self,
        permutation: Permutation,
    ) -> FiberClassIterator<'a, S, CoreExpandedFiberIterator> {
        FiberClassIterator::new_permuted(self, permutation)
    }

    pub fn iter_perm_metric(
        self,
        permutation: Permutation,
    ) -> FiberClassIterator<'a, S, MetricFiberIterator> {
        FiberClassIterator::new_permuted(self, permutation)
    }
}

pub struct FiberClassMut<'a, I: HasStructure> {
    structure: &'a mut I,
    bare_fiber: BareFiber, // A representant of the class

                           // /// true is fixed (but varying when iterating) and false is free (but fixed to 0 when iterating)
                           // free: BitVec, //check performance when it is AHashSet<usize>
}

impl<'a, I> Index<usize> for FiberClassMut<'a, I>
where
    I: HasStructure,
{
    type Output = FiberClassIndex;

    fn index(&self, index: usize) -> &Self::Output {
        if self.bare_fiber[index].is_fixed() {
            &FiberClassIndex::Free
        } else {
            &FiberClassIndex::Fixed
        }
    }
}

impl<'a, I: HasStructure> From<FiberMut<'a, I>> for FiberClassMut<'a, I> {
    fn from(fiber: FiberMut<'a, I>) -> Self {
        FiberClassMut {
            bare_fiber: fiber.bare_fiber,
            structure: fiber.structure,
        }
    }
}

impl<'a, I: HasStructure> From<FiberClassMut<'a, I>> for FiberMut<'a, I> {
    fn from(fiber: FiberClassMut<'a, I>) -> Self {
        FiberMut {
            bare_fiber: fiber.bare_fiber,
            structure: fiber.structure,
        }
    }
}

impl<'a, I: HasStructure> AbstractFiber<FiberClassIndex> for FiberClassMut<'a, I> {
    fn strides(&self) -> Vec<usize> {
        self.structure.strides()
    }

    fn shape(&self) -> Vec<Dimension> {
        self.structure.shape()
    }

    fn reps(&self) -> Vec<Representation> {
        self.structure.reps()
    }

    fn order(&self) -> usize {
        self.structure.order()
    }

    fn single(&self) -> Option<usize> {
        match self.bare_fiber.is_single {
            FiberIndex::Fixed(i) => Some(i),
            _ => None,
        }
    }

    fn bitvec(&self) -> BitVec {
        !self.bare_fiber.bitvec()
    }
}

// Iterators for fibers

#[derive(Debug, Clone, Copy)]
pub struct SingleStrideShift {
    stride: usize,
    shift: usize,
}

#[derive(Debug, Clone)]
pub struct MultiStrideShift {
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

pub trait FiberIteratorItem {
    type OtherData;
    fn flat_idx(&self) -> FlatIndex;

    fn other_data(self) -> Self::OtherData;
}

impl FiberIteratorItem for FlatIndex {
    type OtherData = ();
    fn flat_idx(&self) -> FlatIndex {
        *self
    }

    fn other_data(self) -> Self::OtherData {}
}

struct SkippingItem<I: FiberIteratorItem> {
    skips: usize,
    item: I,
}

impl<I: FiberIteratorItem> FiberIteratorItem for SkippingItem<I> {
    type OtherData = (usize, I::OtherData);
    fn flat_idx(&self) -> FlatIndex {
        self.item.flat_idx()
    }
    fn other_data(self) -> Self::OtherData {
        (self.skips, self.item.other_data())
    }
}

pub struct MetricItem<I: FiberIteratorItem> {
    neg: bool,
    item: I,
}

impl<I: FiberIteratorItem> FiberIteratorItem for MetricItem<I> {
    type OtherData = (bool, I::OtherData);
    fn flat_idx(&self) -> FlatIndex {
        self.item.flat_idx()
    }
    fn other_data(self) -> Self::OtherData {
        (self.neg, self.item.other_data())
    }
}

pub trait IteratesAlongFibers: Iterator {
    fn reset(&mut self);

    fn shift(&mut self, shift: usize);

    fn new<I, J>(fiber: &I, conj: bool) -> Self
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex;

    /// should be equivalent to (new(fiber, true), new(fiber, false)), but could be faster in certain cases
    fn new_paired_conjugates<I, J>(fiber: &I) -> (Self, Self)
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
        Self: Sized;
}

pub trait IteratesAlongPermutedFibers: IteratesAlongFibers {
    fn new_permuted<I, J>(fiber: &I, conj: bool, permutation: Permutation) -> Self
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex;
}

#[derive(Debug, Clone)]
pub struct CoreFlatFiberIterator {
    pub varying_fiber_index: FlatIndex,
    pub increment: FlatIndex,
    pub stride_shift: StrideShift,
    pub max: FlatIndex,
    pub zero_index: FlatIndex,
}

impl CoreFlatFiberIterator {
    fn init_multi_fiber_iter<I, J>(
        strides: Vec<usize>,
        dims: Vec<Dimension>,
        order: usize,
        fiber: &I,
        conj: bool,
    ) -> (FlatIndex, Vec<usize>, Vec<usize>, FlatIndex)
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
        (increment.into(), fixed_strides, shifts, max.into())
    }

    fn init_single_fiber_iter(
        strides: Vec<usize>,
        fiber_position: usize,
        dims: Vec<Dimension>,
        conj: bool,
    ) -> (FlatIndex, Option<usize>, Option<usize>, FlatIndex) {
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

            (increment.into(), stride, shift, max.into())
        } else {
            let increment = fiber_stride;
            let max = fiber_stride * (dim - 1);

            (increment.into(), stride, shift, max.into())
        }
    }
}

impl Iterator for CoreFlatFiberIterator {
    type Item = FlatIndex;
    fn next(&mut self) -> Option<Self::Item> {
        if self.varying_fiber_index > self.max {
            return None;
        }
        let index = self.varying_fiber_index + self.zero_index;

        self.varying_fiber_index += self.increment;

        match self.stride_shift {
            StrideShift::Multi(ref ss) => {
                for (i, s) in ss.strides.iter().enumerate() {
                    if self.varying_fiber_index % s == 0.into() {
                        self.varying_fiber_index += (ss.shifts[i] - s).into();
                    } else {
                        break;
                    }
                }
            }
            StrideShift::Single(Some(ss)) => {
                if self.varying_fiber_index % ss.stride == 0.into() {
                    self.varying_fiber_index += (ss.shift - ss.stride).into();
                }
            }
            _ => {}
        }
        Some(index)
    }
}

impl IteratesAlongFibers for CoreFlatFiberIterator {
    fn reset(&mut self) {
        self.varying_fiber_index = 0.into();
    }

    fn shift(&mut self, shift: usize) {
        self.zero_index = shift.into();
    }

    fn new<I, J>(fiber: &I, conj: bool) -> Self
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
        Self: Sized,
    {
        if let Some(single) = fiber.single() {
            let (increment, fixed_strides, shifts, max) =
                Self::init_single_fiber_iter(fiber.strides(), single, fiber.shape(), conj);

            CoreFlatFiberIterator {
                increment,
                stride_shift: StrideShift::new_single(fixed_strides, shifts),
                max,
                zero_index: 0.into(),
                varying_fiber_index: 0.into(),
            }
        } else {
            let (increment, fixed_strides, shifts, max) = Self::init_multi_fiber_iter(
                fiber.strides(),
                fiber.shape(),
                fiber.order(),
                fiber,
                conj,
            );

            CoreFlatFiberIterator {
                increment,
                stride_shift: StrideShift::new_multi(fixed_strides, shifts),
                max,
                zero_index: 0.into(),
                varying_fiber_index: 0.into(),
            }
        }
    }

    // faster than seperate functions
    fn new_paired_conjugates<I, J>(fiber: &I) -> (Self, Self)
    where
        I: AbstractFiber<J>,
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
            CoreFlatFiberIterator {
                varying_fiber_index: 0.into(),
                stride_shift: StrideShift::new_multi(fixed_strides_conj, shifts_conj),
                increment: increment_conj.into(),
                max: max_conj.into(),
                zero_index: 0.into(),
            },
            CoreFlatFiberIterator {
                varying_fiber_index: 0.into(),
                increment: increment.into(),
                stride_shift: StrideShift::new_multi(fixed_strides, shifts),
                max: max.into(),
                zero_index: 0.into(),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct CoreExpandedFiberIterator {
    pub varying_fiber_index: Vec<ConcreteIndex>,
    pub dims: Vec<Representation>,
    pub strides: Vec<usize>,
    pub zero_index: FlatIndex,
    pub flat: FlatIndex,
    exhausted: bool,
}

impl CoreExpandedFiberIterator {
    fn init_iter<I, J>(fiber: &I, conj: bool, permutation: Option<Permutation>) -> Self
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
    {
        let varying_indices = fiber.bitvec();
        let mut dims = Self::filter(&varying_indices, &fiber.reps(), conj);

        let mut strides = Self::filter(&varying_indices, &fiber.strides(), conj);
        let varying_fiber_index = vec![0; dims.len()];

        if let Some(perm) = permutation {
            perm.apply_slice_in_place(&mut dims);
            perm.apply_slice_in_place(&mut strides);
        }

        CoreExpandedFiberIterator {
            varying_fiber_index,
            zero_index: 0.into(),
            dims,
            strides,
            flat: 0.into(),
            exhausted: false,
        }
    }

    fn filter<T: Clone>(filter: &BitVec, vec: &[T], conj: bool) -> Vec<T> {
        let mut res = vec![];
        for (i, x) in filter.iter().enumerate() {
            if conj ^ *x {
                res.push(vec[i].clone());
            }
        }
        res
    }
}

impl Iterator for CoreExpandedFiberIterator {
    type Item = FlatIndex;
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let current_flat = self.flat + self.zero_index; // Store the current flat value before modifications

        let mut carry = true;
        for ((pos, dim), stride) in self
            .varying_fiber_index
            .iter_mut()
            .zip(self.dims.iter())
            .zip(self.strides.iter())
            .rev()
        {
            if carry {
                *pos += 1;
                if *pos >= usize::from(*dim) {
                    *pos = 0;
                    self.flat -= (stride * (usize::from(*dim) - 1)).into();
                } else {
                    self.flat += (*stride).into();
                    carry = false;
                }
            }
        }

        if carry {
            self.exhausted = true; // Set the flag to prevent further iterations after this one
        }

        Some(current_flat)
    }
}

#[test]
fn test() {
    use std::collections::HashSet;

    let structura = VecStructure::new(vec![
        (0, 4).into(),
        (4, 4).into(),
        (1, 5).into(),
        (3, 7).into(),
        (2, 8).into(),
    ]);

    let structurb = VecStructure::new(vec![
        (2, 8).into(),
        (3, 7).into(),
        (0, 4).into(),
        (1, 5).into(),
        (5, 4).into(),
    ]);

    let fibera = Fiber::from(
        [true, false, true, true, true].as_slice().into(),
        &structura,
    );
    let fiberb = Fiber::from(
        [true, true, true, true, false].as_slice().into(),
        &structurb,
    );

    let (permuta, _filter_a, _filter_b) = structura.match_indices(&structurb).unwrap();
    let itera = CoreExpandedFiberIterator::new_permuted(&fibera, false, permuta.clone());
    let iterb = CoreExpandedFiberIterator::new(&fiberb, false);

    let collecteda: Vec<HashSet<usize>> = itera
        .map(|f| HashSet::from_iter(structura.expanded_index(f).unwrap().into_iter()))
        .collect::<Vec<_>>();
    let collectedb: Vec<HashSet<usize>> = iterb
        .map(|f| HashSet::from_iter(structurb.expanded_index(f).unwrap().into_iter()))
        .collect::<Vec<_>>();

    for (k, i) in collecteda.iter().zip(collectedb.iter()).enumerate() {
        assert_eq!(i.0, i.1, "Error at index {}", k)
    }

    // assert_eq!(collecteda, collectedb);

    // assert_ron_snapshot!(collecteda);
}

impl IteratesAlongFibers for CoreExpandedFiberIterator {
    fn new<I, J>(fiber: &I, conj: bool) -> Self
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
    {
        Self::init_iter(fiber, conj, None)
    }

    fn new_paired_conjugates<I, J>(fiber: &I) -> (Self, Self)
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
        Self: Sized,
    {
        (
            Self::init_iter(fiber, true, None),
            Self::init_iter(fiber, false, None),
        )
    }

    fn reset(&mut self) {
        self.flat = 0.into();
        self.exhausted = false;
        self.varying_fiber_index = vec![0; self.dims.len()];
    }

    fn shift(&mut self, shift: usize) {
        self.zero_index = shift.into();
    }
}

impl IteratesAlongPermutedFibers for CoreExpandedFiberIterator {
    fn new_permuted<I, J>(fiber: &I, conj: bool, permutation: Permutation) -> Self
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
    {
        Self::init_iter(fiber, conj, Some(permutation))
    }
}

#[derive(Debug, Clone)]
pub struct MetricFiberIterator {
    pub iter: CoreExpandedFiberIterator,
    neg: bool,
}

impl IteratesAlongFibers for MetricFiberIterator {
    fn new<I, J>(fiber: &I, conj: bool) -> Self
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
    {
        MetricFiberIterator {
            iter: CoreExpandedFiberIterator::new(fiber, conj),
            neg: false,
        }
    }

    fn new_paired_conjugates<I, J>(fiber: &I) -> (Self, Self)
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
        Self: Sized,
    {
        (
            MetricFiberIterator {
                iter: CoreExpandedFiberIterator::new(fiber, true),
                neg: false,
            },
            MetricFiberIterator {
                iter: CoreExpandedFiberIterator::new(fiber, false),
                neg: false,
            },
        )
    }

    fn reset(&mut self) {
        self.iter.reset();

        // self.neg = false;
    }

    fn shift(&mut self, shift: usize) {
        self.iter.shift(shift);
    }
}

impl IteratesAlongPermutedFibers for MetricFiberIterator {
    fn new_permuted<I, J>(fiber: &I, conj: bool, permutation: Permutation) -> Self
    where
        I: AbstractFiber<J>,
        J: AbstractFiberIndex,
    {
        MetricFiberIterator {
            iter: CoreExpandedFiberIterator::new_permuted(fiber, conj, permutation),
            neg: false,
        }
    }
}

impl Iterator for MetricFiberIterator {
    type Item = MetricItem<FlatIndex>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.iter.exhausted {
            return None;
        }

        let current_flat = self.iter.flat + self.iter.zero_index; // Store the current flat value before modifications

        let mut carry = true;
        self.neg = false;
        for ((pos, dim), stride) in self
            .iter
            .varying_fiber_index
            .iter_mut()
            .zip(self.iter.dims.iter())
            .zip(self.iter.strides.iter())
            .rev()
        {
            self.neg ^= dim.is_neg(*pos);
            if carry {
                *pos += 1;
                if *pos >= usize::from(*dim) {
                    *pos = 0;
                    self.iter.flat -= (stride * (usize::from(*dim) - 1)).into();
                } else {
                    self.iter.flat += (*stride).into();
                    carry = false;
                }
            }
        }

        if carry {
            self.iter.exhausted = true; // Set the flag to prevent further iterations after this one
        }

        Some(MetricItem {
            neg: self.neg,
            item: current_flat,
        })
    }
}

#[derive(Debug)]
pub struct FiberIterator<'a, S: HasStructure, I: IteratesAlongFibers> {
    pub fiber: Fiber<'a, S>,
    pub iter: I,
    pub skipped: usize,
}

impl<'a, S: HasStructure, I: IteratesAlongFibers + Clone> Clone for FiberIterator<'a, S, I> {
    fn clone(&self) -> Self {
        FiberIterator {
            fiber: self.fiber.clone(),
            iter: self.iter.clone(),
            skipped: self.skipped,
        }
    }
}

impl<'a, S: HasStructure, I: IteratesAlongFibers> FiberIterator<'a, S, I> {
    pub fn new(fiber: Fiber<'a, S>, conj: bool) -> Self {
        FiberIterator {
            iter: I::new(&fiber, conj),
            fiber,
            skipped: 0,
        }
    }

    pub fn reset(&mut self) {
        self.iter.reset();
        self.skipped = 0;
    }

    pub fn shift(&mut self, shift: usize) {
        self.reset();
        self.iter.shift(shift);
    }
}

impl<'a, S: HasStructure, I: IteratesAlongPermutedFibers> FiberIterator<'a, S, I> {
    pub fn new_permuted(fiber: Fiber<'a, S>, permutation: Permutation, conj: bool) -> Self {
        FiberIterator {
            iter: I::new_permuted(&fiber, conj, permutation),
            fiber,
            skipped: 0,
        }
    }
}

impl<'a, I: IteratesAlongFibers> Iterator for FiberIterator<'a, VecStructure, I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, I: IteratesAlongFibers<Item = It>, S: HasStructure, T, It> Iterator
    for FiberIterator<'a, DenseTensor<T, S>, I>
where
    It: FiberIteratorItem,
{
    type Item = (&'a T, It::OtherData);
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| {
            // println!("dense {}", x.flat_idx());
            // println!("{}", self.fiber.structure.size());
            (
                self.fiber.structure.get_linear(x.flat_idx()).unwrap(),
                x.other_data(),
            )
        })
    }
}

impl<'a, I: IteratesAlongFibers<Item = It>, S: HasStructure, T, It> Iterator
    for FiberIterator<'a, SparseTensor<T, S>, I>
where
    It: FiberIteratorItem,
{
    type Item = (&'a T, usize, It::OtherData);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.iter.next() {
            // println!("sparse {}", i.flat_idx());
            if let Some(t) = self.fiber.structure.get_linear(i.flat_idx()) {
                let skipped = self.skipped;
                self.skipped = 0;
                return Some((t, skipped, i.other_data()));
            } else {
                self.skipped += 1;
                return self.next();
            }
        }
        None
    }
}

pub struct MutFiberIterator<'a, S: HasStructure, I: IteratesAlongFibers> {
    iter: I,
    fiber: FiberMut<'a, S>,
    skipped: usize,
}

impl<'a, I: IteratesAlongFibers<Item = It>, S: HasStructure, T, It> LendingIterator
    for MutFiberIterator<'a, SparseTensor<T, S>, I>
where
    It: FiberIteratorItem,
{
    type Item<'r> = (&'r mut T, usize, It::OtherData) where Self: 'r;
    fn next(&mut self) -> Option<Self::Item<'_>> {
        let flat = self.iter.next()?;
        if self.fiber.structure.is_empty_at_flat(flat.flat_idx()) {
            let skipped = self.skipped;
            self.skipped = 0;
            return Some((
                self.fiber
                    .structure
                    .get_linear_mut(flat.flat_idx())
                    .unwrap(),
                skipped,
                flat.other_data(),
            ));
        } else {
            self.skipped += 1;
            return self.next();
        }
    }
}

impl<'a, I: IteratesAlongFibers<Item = It>, S: HasStructure, T, It> LendingIterator
    for MutFiberIterator<'a, DenseTensor<T, S>, I>
where
    It: FiberIteratorItem,
{
    type Item<'r> = (&'r mut T,  It::OtherData) where Self: 'r;
    fn next(&mut self) -> Option<Self::Item<'_>> {
        self.iter.next().map(|x| {
            // println!("dense {}", x.flat_idx());
            // println!("{}", self.fiber.structure.size());
            (
                self.fiber.structure.get_linear_mut(x.flat_idx()).unwrap(),
                x.other_data(),
            )
        })
    }
}

impl<'a, S: HasStructure, I: IteratesAlongFibers> MutFiberIterator<'a, S, I> {
    pub fn new(fiber: FiberMut<'a, S>, conj: bool) -> Self {
        MutFiberIterator {
            iter: I::new(&fiber, conj),
            fiber,
            skipped: 0,
        }
    }

    pub fn reset(&mut self) {
        self.iter.reset();
        self.skipped = 0;
    }

    pub fn shift(&mut self, shift: usize) {
        self.iter.shift(shift);
    }
}

impl<'a, S: HasStructure, I: IteratesAlongPermutedFibers> MutFiberIterator<'a, S, I> {
    pub fn new_permuted(fiber: FiberMut<'a, S>, permutation: Permutation, conj: bool) -> Self {
        MutFiberIterator {
            iter: I::new_permuted(&fiber, conj, permutation),
            fiber,
            skipped: 0,
        }
    }
}

#[test]
fn mutiter() {
    let structa = VecStructure::new(vec![
        (0, 4).into(),
        (4, 4).into(),
        (1, 5).into(),
        (3, 7).into(),
        (2, 8).into(),
    ]);

    let mut a: DenseTensor<f64> = DenseTensor::zero(structa);

    let fiber = a.fiber_mut([1u8, 0, 1, 1, 1].as_slice().into());

    let mut iter = fiber.iter();

    while let Some(i) = iter.next() {
        *i.0 = 1.0;
    }

    println!("{:?}", a);
}

pub struct FiberClassIterator<'b, S: HasStructure, I: IteratesAlongFibers = CoreFlatFiberIterator> {
    pub fiber_iter: FiberIterator<'b, S, I>,
    pub class_iter: CoreFlatFiberIterator,
}

impl<'b, N: HasStructure> FiberClassIterator<'b, N, CoreFlatFiberIterator> {
    pub fn new(class: FiberClass<'b, N>) -> Self {
        let (iter, iter_conj) = CoreFlatFiberIterator::new_paired_conjugates(&class);

        let fiber = FiberIterator {
            fiber: class.into(),
            iter,
            skipped: 0,
        };

        FiberClassIterator {
            fiber_iter: fiber,
            class_iter: iter_conj,
        }
    }
}

impl<'b, N: HasStructure, I: IteratesAlongFibers> FiberClassIterator<'b, N, I> {
    pub fn reset(&mut self) {
        self.class_iter.reset();
        self.fiber_iter.reset();
        self.fiber_iter.shift(0);
    }
}

impl<'b, N: HasStructure, I: IteratesAlongPermutedFibers> FiberClassIterator<'b, N, I> {
    pub fn new_permuted(class: FiberClass<'b, N>, permutation: Permutation) -> Self {
        let iter = CoreFlatFiberIterator::new(&class, false);

        let fiber = FiberIterator::new_permuted(class.into(), permutation, false);

        FiberClassIterator {
            fiber_iter: fiber,
            class_iter: iter,
        }
    }
}

impl<'a, S: HasStructure + 'a, I: IteratesAlongFibers + Clone + Debug> Iterator
    for FiberClassIterator<'a, S, I>
{
    type Item = FiberIterator<'a, S, I>;

    fn next(&mut self) -> Option<Self::Item> {
        let shift = self.class_iter.next()?;
        self.fiber_iter.reset();
        self.fiber_iter.shift(shift.into());
        // println!("new {:?}", self.fiber_iter.iter);
        Some(self.fiber_iter.clone())
    }
}

impl<'a, S: HasStructure + 'a, I: IteratesAlongFibers> LendingIterator
    for FiberClassIterator<'a, S, I>
{
    type Item<'r> = &'r mut FiberIterator<'a, S, I> where Self: 'r;

    fn next(&mut self) -> Option<Self::Item<'_>> {
        let shift = self.class_iter.next()?;
        self.fiber_iter.reset();
        self.fiber_iter.shift(shift.into());
        Some(&mut self.fiber_iter)
    }
}

/// Iterator over all the elements of a sparse tensor
///
/// Returns the expanded indices and the element at that index
pub struct SparseTensorIterator<'a, T, N> {
    iter: std::collections::hash_map::Iter<'a, FlatIndex, T>,
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
    N: HasStructure,
{
    type Item = (ExpandedIndex, &'a T);

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
    iter: std::collections::hash_map::Iter<'a, FlatIndex, T>,
}

impl<'a, T> SparseTensorLinearIterator<'a, T> {
    pub fn new<N>(tensor: &'a SparseTensor<T, N>) -> Self {
        SparseTensorLinearIterator {
            iter: tensor.elements.iter(),
        }
    }
}

impl<'a, T> Iterator for SparseTensorLinearIterator<'a, T> {
    type Item = (FlatIndex, &'a T);

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
    I: HasStructure,
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
    T: ContractableWith<T> + FallibleAddAssign<T> + FallibleSubAssign<T> + Clone + RefZero,
    I: HasStructure + Clone,
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

        while self.tensor.is_empty_at_expanded(&indices) {
            let Some((i, signint)) = iter.next() else {
                self.done = !self.increment_indices();
                return self.next(); // skip
            };
            indices[self.trace_indices[0]] = i;
            indices[self.trace_indices[1]] = i;
            sign = signint;
        }

        let value = (*self.tensor.get(&indices).unwrap()).clone(); //Should now be safe to unwrap
        let zero = value.ref_zero();

        let mut trace = if *sign {
            let mut zero = zero.clone();
            zero.sub_assign_fallible(value);
            zero
        } else {
            value
        };

        for (i, sign) in iter {
            indices[self.trace_indices[0]] = i;
            indices[self.trace_indices[1]] = i;
            if let Ok(value) = self.tensor.get(&indices) {
                if *sign {
                    trace.sub_assign_fallible(value.clone());
                } else {
                    trace.add_assign_fallible(value.clone());
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
    I: HasStructure,
{
    pub fn fiber<'r>(&'r self, fiber_data: FiberData<'_>) -> Fiber<'r, Self> {
        Fiber::from(fiber_data, self)
    }

    pub fn fiber_class<'r>(&'r self, fiber_data: FiberData<'_>) -> FiberClass<'r, Self> {
        Fiber::from(fiber_data, self).into()
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
}

/// Iterator over all the elements of a dense tensor
///
/// Returns the expanded indices and the element at that index
pub struct DenseTensorIterator<'a, T, I> {
    tensor: &'a DenseTensor<T, I>,
    current_flat_index: FlatIndex,
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
            current_flat_index: 0.into(),
        }
    }
}

impl<'a, T, I> Iterator for DenseTensorIterator<'a, T, I>
where
    I: HasStructure,
{
    type Item = (ExpandedIndex, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(indices) = self.tensor.expanded_index(self.current_flat_index) {
            let value = self.tensor.get_linear(self.current_flat_index).unwrap();

            self.current_flat_index += 1.into();

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
    current_flat_index: FlatIndex,
}

impl<'a, T, I> DenseTensorLinearIterator<'a, T, I> {
    pub fn new(tensor: &'a DenseTensor<T, I>) -> Self {
        DenseTensorLinearIterator {
            tensor,
            current_flat_index: 0.into(),
        }
    }
}

impl<'a, T, I> Iterator for DenseTensorLinearIterator<'a, T, I>
where
    I: HasStructure,
{
    type Item = (FlatIndex, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.tensor.get_linear(self.current_flat_index)?;
        let index = self.current_flat_index;
        self.current_flat_index += 1.into();
        Some((index, value))
    }
}

impl<'a, T, I> IntoIterator for &'a DenseTensor<T, I>
where
    I: HasStructure,
{
    type Item = (ExpandedIndex, &'a T);
    type IntoIter = DenseTensorIterator<'a, T, I>;

    fn into_iter(self) -> Self::IntoIter {
        DenseTensorIterator::new(self)
    }
}

impl<T, I> IntoIterator for DenseTensor<T, I>
where
    I: HasStructure,
{
    type Item = (ExpandedIndex, T);
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
    current_flat_index: FlatIndex,
}

impl<T, I> DenseTensorIntoIterator<T, I> {
    fn new(tensor: DenseTensor<T, I>) -> Self {
        DenseTensorIntoIterator {
            tensor,
            current_flat_index: 0.into(),
        }
    }
}

impl<T, I> Iterator for DenseTensorIntoIterator<T, I>
where
    I: HasStructure,
{
    type Item = (ExpandedIndex, T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(indices) = self.tensor.expanded_index(self.current_flat_index) {
            let indices = indices.clone();
            let value = self.tensor.data.remove(self.current_flat_index.into());

            self.current_flat_index += 1.into();

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
    I: HasStructure,
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
    T: ContractableWith<T, Out = T> + FallibleAddAssign<T> + FallibleSubAssign<T> + Clone + RefZero,
    I: HasStructure,
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
        let zero = value.ref_zero();

        let mut trace = if *sign {
            let mut zero = zero.clone();
            zero.sub_assign_fallible(value);
            zero
        } else {
            value
        };

        for (i, sign) in iter {
            for pos in self.trace_indices {
                indices[pos] = i;
            }

            if let Ok(value) = self.tensor.get(&indices) {
                if *sign {
                    trace.sub_assign_fallible(value.clone());
                } else {
                    trace.add_assign_fallible(value.clone());
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
    I: HasStructure,
{
    pub fn iter(&self) -> DenseTensorIterator<T, I> {
        DenseTensorIterator::new(self)
    }

    pub fn fiber<'r>(&'r self, fiber_data: FiberData<'_>) -> Fiber<'r, Self> {
        Fiber::from(fiber_data, self)
    }

    pub fn fiber_mut<'r>(&'r mut self, fiber_data: FiberData<'_>) -> FiberMut<'r, Self> {
        FiberMut::from(fiber_data, self)
    }

    pub fn fiber_class<'r>(&'r self, fiber_data: FiberData<'_>) -> FiberClass<'r, Self> {
        Fiber::from(fiber_data, self).into()
    }

    pub fn fiber_class_mut<'r>(&'r mut self, fiber_data: FiberData<'_>) -> FiberClassMut<'r, Self> {
        FiberMut::from(fiber_data, self).into()
    }

    pub fn iter_flat(&self) -> DenseTensorLinearIterator<T, I> {
        DenseTensorLinearIterator::new(self)
    }

    pub fn iter_trace(&self, trace_indices: [usize; 2]) -> DenseTensorTraceIterator<T, I> {
        DenseTensorTraceIterator::new(self, trace_indices)
    }
}

/// An iterator over all indices of a tensor structure
///
/// `Item` is a flat index

pub struct TensorStructureIndexIterator<'a> {
    structure: &'a [Slot],
    current_flat_index: FlatIndex,
}

impl<'a> Iterator for TensorStructureIndexIterator<'a> {
    type Item = ExpandedIndex;
    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(indices) = self.structure.expanded_index(self.current_flat_index) {
            self.current_flat_index += 1.into();

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
            current_flat_index: 0.into(),
        }
    }
}
