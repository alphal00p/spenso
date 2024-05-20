use std::ops::Neg;

use super::{
    AbstractIndex, DenseTensor, HasStructure, HistoryStructure,
    Representation::{self, Euclidean, Lorentz},
    SetTensorData, Slot, SparseTensor,
};

use num::{NumCast, One, Zero};

use crate::Complex;

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, Symbol},
    state::State,
};

#[cfg(feature = "shadowing")]
use super::{IntoId, Shadowable};

// pub fn init_state() {
//     assert!(EUC == State::get_symbol("euc", None).unwrap());
//     assert!(LOR == State::get_symbol("lor", None).unwrap());
//     assert!(SPIN == State::get_symbol("spin", None).unwrap());
//     assert!(CADJ == State::get_symbol("CAdj", None).unwrap());
//     assert!(CF == State::get_symbol("CF", None).unwrap());
//     assert!(CAF == State::get_symbol("CAF", None).unwrap());
//     assert!(CS == State::get_symbol("CS", None).unwrap());
//     assert!(CAS == State::get_symbol("CAS", None).unwrap());

//     assert!(ID == State::get_symbol("id", None).unwrap());
//     assert!(GAMMA == State::get_symbol("γ", None).unwrap());
//     assert!(GAMMA5 == State::get_symbol("γ5", None).unwrap());
//     assert!(PROJM == State::get_symbol("ProjM", None).unwrap());
//     assert!(PROJP == State::get_symbol("ProjP", None).unwrap());
//     assert!(SIGMA == State::get_symbol("σ", None).unwrap());
// }

#[allow(dead_code)]
#[must_use]
pub fn identity<T, I>(
    indices: (AbstractIndex, AbstractIndex),
    signature: Representation,
) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero,
    I: HasStructure + FromIterator<Slot>,
{
    //TODO: make it just swap indices
    let structure = [(indices.0, signature), (indices.1, signature)]
        .into_iter()
        .map(Slot::from)
        .collect();
    let mut identity = SparseTensor::empty(structure);
    for i in 0..signature.into() {
        identity
            .set(&[i, i], Complex::<T>::new(T::one(), T::zero()))
            .unwrap_or_else(|_| unreachable!());
    }
    identity
}

/// Create a rank 2 identity tensor
///
/// # Arguments
///
/// * `structure` - The structure of the tensor
///
/// # Panics
///
/// * If the structure is not rank 2
/// * If the structure has different indices

pub fn identity_data<T, N>(structure: N) -> SparseTensor<T, N>
where
    T: One,
    N: HasStructure,
{
    assert!(structure.order() == 2, "Identity tensor must be rank 2");

    assert!(
        structure.reps()[0] == structure.reps()[1],
        "Identity tensor must have equal indices"
    );

    let mut identity = SparseTensor::empty(structure);

    for i in 0..identity.shape()[0].into() {
        identity
            .set(&[i, i], T::one())
            .unwrap_or_else(|_| unreachable!());
    }
    identity
}

#[allow(dead_code)]
#[must_use]
pub fn lorentz_identity<T, I>(
    indices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero,
    I: HasStructure + FromIterator<Slot>,
{
    // IdentityL(1,2) (Lorentz) Kronecker delta δ^μ1_μ1
    let signature = Lorentz(4.into());
    identity(indices, signature)
}

pub fn mink_four_vector<T, I>(index: AbstractIndex, p: &[T; 4]) -> DenseTensor<T, I>
where
    T: Clone,
    I: HasStructure + FromIterator<Slot>,
{
    DenseTensor::from_data(
        p,
        [Slot::from((index, Lorentz(4.into())))]
            .into_iter()
            .collect(),
    )
    .unwrap_or_else(|_| unreachable!())
}

#[cfg(feature = "shadowing")]
pub fn mink_four_vector_sym<T>(
    index: AbstractIndex,
    p: &[T; 4],
) -> DenseTensor<T, HistoryStructure<Symbol>>
where
    T: Clone,
{
    DenseTensor::from_data(
        p,
        HistoryStructure::new(&[(index, Lorentz(4.into()))], State::get_symbol("p")),
    )
    .unwrap_or_else(|_| unreachable!())
}

pub fn euclidean_four_vector<T, I>(index: AbstractIndex, p: &[T; 4]) -> DenseTensor<T, I>
where
    T: Clone,
    I: HasStructure + FromIterator<Slot>,
{
    DenseTensor::from_data(
        p,
        [Slot::from((index, Euclidean(4.into())))]
            .into_iter()
            .collect(),
    )
    .unwrap_or_else(|_| unreachable!())
}

#[cfg(feature = "shadowing")]
pub fn euclidean_four_vector_sym<T>(
    index: AbstractIndex,
    p: &[T; 4],
) -> DenseTensor<T, HistoryStructure<Symbol>>
where
    T: Clone,
{
    DenseTensor::from_data(
        p,
        HistoryStructure::new(&[(index, Euclidean(4.into()))], State::get_symbol("p")),
    )
    .unwrap_or_else(|_| unreachable!())
}

#[cfg(feature = "shadowing")]
pub fn param_mink_four_vector<N>(
    index: AbstractIndex,
    name: N,
) -> DenseTensor<Atom, HistoryStructure<N>>
where
    N: Clone + IntoId,
{
    HistoryStructure::new(&[(index, Lorentz(4.into()))], name)
        .shadow()
        .unwrap_or_else(|| unreachable!())
}

#[cfg(feature = "shadowing")]
pub fn param_euclidean_four_vector<N>(
    index: AbstractIndex,
    name: N,
) -> DenseTensor<Atom, HistoryStructure<N>>
where
    N: Clone + IntoId,
{
    HistoryStructure::new(&[(index, Euclidean(4.into()))], name)
        .shadow()
        .unwrap_or_else(|| unreachable!())
}

#[allow(dead_code)]
#[must_use]
pub fn euclidean_identity<T, I>(
    indices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero,
    I: HasStructure + FromIterator<Slot>,
{
    // Identity(1,2) (Spinorial) Kronecker delta δ_s1_s2
    let signature = Euclidean(4.into());
    identity(indices, signature)
}

#[allow(dead_code)]
pub fn gamma<T, I>(
    minkindex: AbstractIndex,
    indices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero + Copy + std::ops::Neg<Output = T>,
    I: HasStructure + FromIterator<Slot>,
{
    // Gamma(1,2,3) Dirac matrix (γ^μ1)_s2_s3
    let structure = [
        (indices.0, Euclidean(4.into())),
        (indices.1, Euclidean(4.into())),
        (minkindex, Lorentz(4.into())),
    ]
    .into_iter()
    .map(Slot::from)
    .collect();

    gamma_data(structure)
}
#[cfg(feature = "shadowing")]
pub fn gammasym<T>(
    minkindex: AbstractIndex,
    indices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol>>
where
    T: One + Zero + Copy + std::ops::Neg<Output = T>,
{
    let structure = HistoryStructure::new(
        &[
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
            (minkindex, Lorentz(4.into())),
        ],
        State::get_symbol("γ"),
    );

    gamma_data(structure)
}

#[allow(clippy::similar_names)]
pub fn gamma_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + Neg<Output = T> + Clone,
    N: HasStructure,
{
    let c1 = Complex::<T>::new(T::one(), T::zero());
    let cn1 = Complex::<T>::new(-T::one(), T::zero());
    let ci = Complex::<T>::new(T::zero(), T::one());
    let cni = Complex::<T>::new(T::zero(), -T::one());
    let mut gamma = SparseTensor::empty(structure);

    // dirac gamma matrices

    gamma.set(&[0, 0, 0], c1.clone()).unwrap();
    gamma.set(&[1, 1, 0], c1.clone()).unwrap();
    gamma.set(&[2, 2, 0], cn1.clone()).unwrap();
    gamma.set(&[3, 3, 0], cn1.clone()).unwrap();

    gamma.set(&[0, 3, 1], c1.clone()).unwrap();
    gamma.set(&[1, 2, 1], c1.clone()).unwrap();
    gamma.set(&[2, 1, 1], cn1.clone()).unwrap();
    gamma.set(&[3, 0, 1], cn1.clone()).unwrap();

    gamma.set(&[0, 3, 2], cni.clone()).unwrap();
    gamma.set(&[1, 2, 2], ci.clone()).unwrap();
    gamma.set(&[2, 1, 2], ci.clone()).unwrap();
    gamma.set(&[3, 0, 2], cni.clone()).unwrap();

    gamma.set(&[0, 2, 3], c1.clone()).unwrap();
    gamma.set(&[1, 3, 3], cn1.clone()).unwrap();
    gamma.set(&[2, 0, 3], cn1.clone()).unwrap();
    gamma.set(&[3, 1, 3], c1.clone()).unwrap();

    gamma //.to_dense()
}

pub fn gamma5<T, I>(indices: (AbstractIndex, AbstractIndex)) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero + Copy,
    I: HasStructure + FromIterator<Slot>,
{
    let structure = [
        (indices.0, Euclidean(4.into())),
        (indices.1, Euclidean(4.into())),
    ]
    .into_iter()
    .map(Slot::from)
    .collect();

    gamma5_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn gamma5sym<T>(
    indices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol>>
where
    T: One + Zero + Copy,
{
    let structure = HistoryStructure::new(
        &[
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
        ],
        State::get_symbol("γ5"),
    );

    gamma5_data(structure)
}

pub fn gamma5_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + Clone,
    N: HasStructure,
{
    let c1 = Complex::<T>::new(T::one(), T::zero());

    let mut gamma5 = SparseTensor::empty(structure);

    gamma5.set(&[0, 2], c1.clone()).unwrap();
    gamma5.set(&[1, 3], c1.clone()).unwrap();
    gamma5.set(&[2, 0], c1.clone()).unwrap();
    gamma5.set(&[3, 1], c1.clone()).unwrap();

    gamma5
}

pub fn proj_m<T, I>(indices: (AbstractIndex, AbstractIndex)) -> SparseTensor<Complex<T>, I>
where
    T: Zero + One + NumCast + Clone,
    I: HasStructure + FromIterator<Slot>,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let structure = [
        (indices.0, Euclidean(4.into())),
        (indices.1, Euclidean(4.into())),
    ]
    .into_iter()
    .map(Slot::from)
    .collect();

    proj_m_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn proj_msym<T>(
    indices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol>>
where
    T: Zero + One + NumCast + Clone,
{
    let structure = HistoryStructure::new(
        &[
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
        ],
        State::get_symbol("ProjM"),
    );

    proj_m_data(structure)
}

#[allow(clippy::similar_names)]
pub fn proj_m_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + NumCast + Clone,
    N: HasStructure,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let chalf = Complex::<T>::new(T::from(0.5).unwrap(), T::zero());
    let cnhalf = Complex::<T>::new(T::from(-0.5).unwrap(), T::zero());

    let mut proj_m = SparseTensor::empty(structure);

    proj_m.set(&[0, 0], chalf.clone()).unwrap();
    proj_m.set(&[1, 1], chalf.clone()).unwrap();
    proj_m.set(&[2, 2], chalf.clone()).unwrap();
    proj_m.set(&[3, 3], chalf.clone()).unwrap();

    proj_m.set(&[0, 2], cnhalf.clone()).unwrap();
    proj_m.set(&[1, 3], cnhalf.clone()).unwrap();
    proj_m.set(&[2, 0], cnhalf.clone()).unwrap();
    proj_m.set(&[3, 1], cnhalf.clone()).unwrap();

    proj_m
}

pub fn proj_p<T, I>(indices: (AbstractIndex, AbstractIndex)) -> SparseTensor<Complex<T>, I>
where
    T: NumCast + Zero + Clone,
    I: HasStructure + FromIterator<Slot>,
{
    // ProjP(1,2) Right chirality projector (( 1+γ5)/ 2 )_s1_s2
    let structure = [
        (indices.0, Euclidean(4.into())),
        (indices.1, Euclidean(4.into())),
    ]
    .into_iter()
    .map(Slot::from)
    .collect();

    proj_p_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn proj_psym<T>(
    indices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol>>
where
    T: Zero + Clone + NumCast,
{
    let structure = HistoryStructure::new(
        &[
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
        ],
        State::get_symbol("ProjP"),
    );

    proj_p_data(structure)
}

pub fn proj_p_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: NumCast + Zero + Clone,
    N: HasStructure,
{
    // ProjP(1,2) Right chirality projector (( 1+γ5)/ 2 )_s1_s2
    let chalf = Complex::<T>::new(T::from(0.5).unwrap_or_else(|| unreachable!()), T::zero());

    let mut proj_p = SparseTensor::empty(structure);

    proj_p
        .set(&[0, 0], chalf.clone())
        .unwrap_or_else(|_| unreachable!());
    proj_p
        .set(&[1, 1], chalf.clone())
        .unwrap_or_else(|_| unreachable!());
    proj_p
        .set(&[2, 2], chalf.clone())
        .unwrap_or_else(|_| unreachable!());
    proj_p
        .set(&[3, 3], chalf.clone())
        .unwrap_or_else(|_| unreachable!());

    proj_p
        .set(&[0, 2], chalf.clone())
        .unwrap_or_else(|_| unreachable!());
    proj_p
        .set(&[1, 3], chalf.clone())
        .unwrap_or_else(|_| unreachable!());
    proj_p
        .set(&[2, 0], chalf.clone())
        .unwrap_or_else(|_| unreachable!());
    proj_p
        .set(&[3, 1], chalf.clone())
        .unwrap_or_else(|_| unreachable!());

    proj_p
}

pub fn sigma<T, I>(
    indices: (AbstractIndex, AbstractIndex),
    minkdices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, I>
where
    T: Copy + Zero + One + Neg<Output = T>,
    I: HasStructure + FromIterator<Slot>,
{
    let structure = [
        (indices.0, Euclidean(4.into())),
        (indices.1, Euclidean(4.into())),
        (minkdices.0, Lorentz(4.into())),
        (minkdices.1, Lorentz(4.into())),
    ]
    .into_iter()
    .map(Slot::from)
    .collect();

    sigma_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn sigmasym<T>(
    indices: (AbstractIndex, AbstractIndex),
    minkdices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol>>
where
    T: Copy + Zero + Clone + One + Neg<Output = T>,
{
    let structure = HistoryStructure::new(
        &[
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
            (minkdices.0, Lorentz(4.into())),
            (minkdices.1, Lorentz(4.into())),
        ],
        State::get_symbol("σ"),
    );

    sigma_data(structure)
}

#[allow(clippy::similar_names)]
pub fn sigma_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Copy + Zero + One + Neg<Output = T>,
    N: HasStructure,
{
    let c1 = Complex::<T>::new(T::one(), T::zero());
    let cn1 = Complex::<T>::new(-T::one(), T::zero());
    let ci = Complex::<T>::new(T::zero(), T::one());
    let cni = Complex::<T>::new(T::zero(), -T::one());

    let mut sigma = SparseTensor::empty(structure);
    sigma.set(&[0, 2, 0, 1], c1).unwrap();
    sigma.set(&[0, 2, 3, 0], c1).unwrap();
    sigma.set(&[0, 3, 1, 2], c1).unwrap();
    sigma.set(&[1, 0, 2, 2], c1).unwrap();
    sigma.set(&[1, 1, 1, 2], c1).unwrap();
    sigma.set(&[1, 3, 0, 2], c1).unwrap();
    sigma.set(&[2, 2, 1, 0], c1).unwrap();
    sigma.set(&[2, 2, 2, 1], c1).unwrap();
    sigma.set(&[2, 3, 3, 2], c1).unwrap();
    sigma.set(&[3, 0, 0, 2], c1).unwrap();
    sigma.set(&[3, 3, 2, 2], c1).unwrap();
    sigma.set(&[3, 1, 3, 2], c1).unwrap();
    sigma.set(&[0, 1, 3, 0], ci).unwrap();
    sigma.set(&[0, 3, 1, 1], ci).unwrap();
    sigma.set(&[0, 3, 2, 0], ci).unwrap();
    sigma.set(&[1, 0, 3, 3], ci).unwrap();
    sigma.set(&[1, 1, 0, 3], ci).unwrap();
    sigma.set(&[1, 1, 2, 0], ci).unwrap();
    sigma.set(&[2, 1, 1, 0], ci).unwrap();
    sigma.set(&[2, 3, 0, 0], ci).unwrap();
    sigma.set(&[2, 3, 3, 1], ci).unwrap();
    sigma.set(&[3, 0, 1, 3], ci).unwrap();
    sigma.set(&[3, 1, 0, 0], ci).unwrap();
    sigma.set(&[3, 1, 2, 3], ci).unwrap();
    sigma.set(&[0, 0, 3, 2], cn1).unwrap();
    sigma.set(&[0, 1, 0, 2], cn1).unwrap();
    sigma.set(&[0, 2, 1, 3], cn1).unwrap();
    sigma.set(&[1, 2, 0, 3], cn1).unwrap();
    sigma.set(&[1, 2, 1, 1], cn1).unwrap();
    sigma.set(&[1, 2, 2, 0], cn1).unwrap();
    sigma.set(&[2, 0, 1, 2], cn1).unwrap();
    sigma.set(&[2, 1, 2, 2], cn1).unwrap();
    sigma.set(&[2, 2, 3, 3], cn1).unwrap();
    sigma.set(&[3, 2, 0, 0], cn1).unwrap();
    sigma.set(&[3, 2, 2, 3], cn1).unwrap();
    sigma.set(&[3, 2, 3, 1], cn1).unwrap();
    sigma.set(&[0, 0, 2, 3], cni).unwrap();
    sigma.set(&[0, 0, 3, 1], cni).unwrap();
    sigma.set(&[0, 1, 1, 3], cni).unwrap();
    sigma.set(&[1, 0, 2, 1], cni).unwrap();
    sigma.set(&[1, 3, 0, 1], cni).unwrap();
    sigma.set(&[1, 3, 3, 0], cni).unwrap();
    sigma.set(&[2, 0, 0, 3], cni).unwrap();
    sigma.set(&[2, 0, 1, 1], cni).unwrap();
    sigma.set(&[2, 1, 3, 3], cni).unwrap();
    sigma.set(&[3, 0, 0, 1], cni).unwrap();
    sigma.set(&[3, 3, 1, 0], cni).unwrap();
    sigma.set(&[3, 3, 2, 1], cni).unwrap();

    sigma
}
