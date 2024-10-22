use std::ops::Neg;

use num::{NumCast, One, Zero};

use crate::{
    complex::Complex,
    data::{DenseTensor, SetTensorData, SparseTensor},
    structure::{
        abstract_index::AbstractIndex,
        dimension::Dimension,
        representation::{
            BaseRepName, Bispinor, Dual, Euclidean, Lorentz, PhysReps, RepName, Representation,
        },
        slot::Slot,
        TensorStructure,
    },
};

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, Symbol},
    state::State,
};

#[cfg(feature = "shadowing")]
use crate::{
    shadowing::Shadowable,
    structure::{HistoryStructure, NamedStructure},
    symbolica_utils::{IntoArgs, IntoSymbol},
};

#[allow(dead_code)]
#[must_use]
pub fn identity<T, I, Rep: RepName>(
    indices: [AbstractIndex; 2],
    signature: Representation<Rep>,
) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero,
    I: TensorStructure + FromIterator<Slot<Rep>>,
{
    //TODO: make it just swap indices
    let structure = indices
        .into_iter()
        .map(|i| Representation::new_slot(&signature, i))
        .collect();
    let mut identity = SparseTensor::empty(structure);
    for i in 0..signature.try_into().unwrap() {
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
    N: TensorStructure,
{
    assert!(structure.order() == 2, "Identity tensor must be rank 2");

    // println!("{:?}", structure.reps());

    assert!(
        Dimension::from(structure.reps()[0]) == Dimension::from(structure.reps()[1]),
        "Identity tensor must have equal indices"
    );

    let mut identity = SparseTensor::empty(structure);

    for i in 0..identity.shape()[0].try_into().unwrap() {
        identity
            .set(&[i, i], T::one())
            .unwrap_or_else(|_| unreachable!());
    }
    identity
}

#[allow(dead_code)]
#[must_use]
pub fn lorentz_identity<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero,
    I: TensorStructure + FromIterator<Slot<Lorentz>>,
{
    // IdentityL(1,2) (Lorentz) Kronecker delta δ^μ1_μ1
    let signature = Lorentz::rep(4);
    identity(indices, signature)
}

pub fn mink_four_vector<T, I>(index: AbstractIndex, p: [T; 4]) -> DenseTensor<T, I>
where
    T: Clone,
    I: TensorStructure + FromIterator<Slot<Lorentz>> + FromIterator<Slot<Dual<Lorentz>>>,
{
    let structure: I = match index {
        AbstractIndex::Dualize(d) => {
            [Lorentz::selfless_dual().new_slot(4, AbstractIndex::Normal(d))]
                .into_iter()
                .collect()
        }
        AbstractIndex::Normal(_) => [Lorentz::slot(4, index)].into_iter().collect(),
        _ => panic!("Invalid index"),
    };
    DenseTensor::from_data(p.to_vec(), structure).unwrap_or_else(|_| unreachable!())
}

#[cfg(feature = "shadowing")]
pub fn mink_four_vector_sym<T>(
    index: AbstractIndex,
    p: [T; 4],
) -> DenseTensor<T, HistoryStructure<Symbol, ()>>
where
    T: Clone,
{
    DenseTensor::from_data(
        p.to_vec(),
        HistoryStructure::from(NamedStructure::from_iter(
            [Lorentz::slot(4, index)],
            State::get_symbol("p"),
            None,
        )),
    )
    .unwrap_or_else(|_| unreachable!())
}

pub fn euclidean_four_vector<T, I>(index: AbstractIndex, p: [T; 4]) -> DenseTensor<T, I>
where
    T: Clone,
    I: TensorStructure + FromIterator<Slot<Euclidean>>,
{
    DenseTensor::from_data(
        p.to_vec(),
        [Euclidean::slot(4, index)].into_iter().collect(),
    )
    .unwrap_or_else(|_| unreachable!())
}

#[cfg(feature = "shadowing")]
pub fn euclidean_four_vector_sym<T>(
    index: AbstractIndex,
    p: [T; 4],
) -> DenseTensor<T, HistoryStructure<Symbol, ()>>
where
    T: Clone,
{
    DenseTensor::from_data(
        p.to_vec(),
        HistoryStructure::from(NamedStructure::from_iter(
            [Euclidean::slot(4, index)],
            State::get_symbol("p"),
            None,
        )),
    )
    .unwrap_or_else(|_| unreachable!())
}

#[cfg(feature = "shadowing")]
pub fn param_mink_four_vector<N, A>(
    index: AbstractIndex,
    name: N,
    args: Option<A>,
) -> DenseTensor<Atom, HistoryStructure<N, A>>
where
    N: Clone + IntoSymbol,
    A: Clone + IntoArgs,
{
    HistoryStructure::from(NamedStructure::from_iter(
        [Lorentz::slot(4, index)],
        name,
        args,
    ))
    .to_shell()
    .expanded_shadow()
    .unwrap_or_else(|_| unreachable!())
}

#[cfg(feature = "shadowing")]
pub fn param_euclidean_four_vector<N, A>(
    index: AbstractIndex,
    name: N,
) -> DenseTensor<Atom, HistoryStructure<N, A>>
where
    N: Clone + IntoSymbol,
    A: Clone + IntoArgs,
{
    HistoryStructure::from(NamedStructure::from_iter(
        [Euclidean::slot(4, index)],
        name,
        None,
    ))
    .to_shell()
    .expanded_shadow()
    .unwrap_or_else(|_| unreachable!())
}

#[allow(dead_code)]
#[must_use]
pub fn euclidean_identity<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero,
    I: TensorStructure + FromIterator<Slot<Euclidean>>,
{
    // Identity(1,2) (Spinorial) Kronecker delta δ_s1_s2
    let signature = Euclidean::rep(4);
    identity(indices, signature)
}

#[allow(dead_code)]
pub fn gamma<T, I>(
    minkindex: AbstractIndex,
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero + Copy + std::ops::Neg<Output = T>,
    I: TensorStructure + FromIterator<Slot<PhysReps>>,
{
    // Gamma(1,2,3) Dirac matrix (γ^μ1)_s2_s3
    let mu = match minkindex {
        AbstractIndex::Dualize(d) => Lorentz::selfless_dual().new_slot(4, d).into(),
        AbstractIndex::Normal(n) => Lorentz::slot(4, n).into(),
        _ => panic!("Invalid index"),
    };
    let structure = [
        mu, // Lorentz::new_slot_selfless(4, minkindex).into(),
        Slot::<PhysReps>::from(Euclidean::slot(4, indices[0])),
        Euclidean::slot(4, indices[1]).into(),
    ]
    .into_iter()
    .map(Slot::from)
    .collect();

    gamma_data_dirac(structure)
}
#[cfg(feature = "shadowing")]
pub fn gammasym<T>(
    minkindex: AbstractIndex,
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: One + Zero + Copy + std::ops::Neg<Output = T>,
{
    let mu = match minkindex {
        AbstractIndex::Dualize(d) => Lorentz::selfless_dual().new_slot(4, d).into(),
        AbstractIndex::Normal(n) => Lorentz::slot(4, n).into(),
        _ => panic!("Invalid index"),
    };
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            mu,
            Slot::<PhysReps>::from(Euclidean::slot(4, indices[0])),
            Euclidean::slot(4, indices[1]).into(),
        ],
        State::get_symbol("γ"),
        None,
    ));

    gamma_data_dirac(structure)
}

#[allow(clippy::similar_names)]
pub fn gamma_data_dirac<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + Neg<Output = T> + Clone,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(T::one(), T::zero());
    let cn1 = Complex::<T>::new(-T::one(), T::zero());
    let ci = Complex::<T>::new(T::zero(), T::one());
    let cni = Complex::<T>::new(T::zero(), -T::one());
    let mut gamma = SparseTensor::empty(structure);
    // ! No check on actual structure, should expext bis,bis,lor

    // dirac gamma matrices

    gamma.set(&[0, 0, 0], c1.clone()).unwrap();
    gamma.set(&[0, 1, 1], c1.clone()).unwrap();
    gamma.set(&[0, 2, 2], cn1.clone()).unwrap();
    gamma.set(&[0, 3, 3], cn1.clone()).unwrap();

    gamma.set(&[1, 0, 3], c1.clone()).unwrap();
    gamma.set(&[1, 1, 2], c1.clone()).unwrap();
    gamma.set(&[1, 2, 1], cn1.clone()).unwrap();
    gamma.set(&[1, 3, 0], cn1.clone()).unwrap();

    gamma.set(&[2, 0, 3], cni.clone()).unwrap();
    gamma.set(&[2, 1, 2], ci.clone()).unwrap();
    gamma.set(&[2, 2, 1], ci.clone()).unwrap();
    gamma.set(&[2, 3, 0], cni.clone()).unwrap();

    gamma.set(&[3, 0, 2], c1.clone()).unwrap();
    gamma.set(&[3, 1, 3], cn1.clone()).unwrap();
    gamma.set(&[3, 2, 0], cn1.clone()).unwrap();
    gamma.set(&[3, 3, 1], c1.clone()).unwrap();

    gamma //.to_dense()
}

#[allow(clippy::similar_names)]
pub fn gamma_data_weyl<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + Neg<Output = T> + Clone,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(T::one(), T::zero());
    let cn1 = Complex::<T>::new(-T::one(), T::zero());
    let ci = Complex::<T>::new(T::zero(), T::one());
    let cni = Complex::<T>::new(T::zero(), -T::one());
    let mut gamma = SparseTensor::empty(structure);
    // ! No check on actual structure, should expext bis,bis,lor

    // dirac gamma matrices

    gamma.set(&[0, 0, 2], c1.clone()).unwrap();
    gamma.set(&[0, 1, 3], c1.clone()).unwrap();
    gamma.set(&[0, 2, 0], c1.clone()).unwrap();
    gamma.set(&[0, 3, 1], c1.clone()).unwrap();

    gamma.set(&[1, 0, 3], c1.clone()).unwrap();
    gamma.set(&[1, 1, 2], c1.clone()).unwrap();
    gamma.set(&[1, 2, 1], cn1.clone()).unwrap();
    gamma.set(&[1, 3, 0], cn1.clone()).unwrap();

    gamma.set(&[2, 0, 3], cni.clone()).unwrap();
    gamma.set(&[2, 1, 2], ci.clone()).unwrap();
    gamma.set(&[2, 2, 1], ci.clone()).unwrap();
    gamma.set(&[2, 3, 0], cni.clone()).unwrap();

    gamma.set(&[3, 0, 2], c1.clone()).unwrap();
    gamma.set(&[3, 1, 3], cn1.clone()).unwrap();
    gamma.set(&[3, 2, 0], cn1.clone()).unwrap();
    gamma.set(&[3, 3, 1], c1.clone()).unwrap();

    gamma //.to_dense()
}

pub fn gamma0_weyl<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + Clone,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(T::one(), T::zero());
    let mut gamma0 = SparseTensor::empty(structure);
    // ! No check on actual structure, should expext bis,bis,lor

    // dirac gamma0 matrices

    gamma0.set(&[0, 2], c1.clone()).unwrap();
    gamma0.set(&[1, 3], c1.clone()).unwrap();
    gamma0.set(&[2, 0], c1.clone()).unwrap();
    gamma0.set(&[3, 1], c1.clone()).unwrap();

    gamma0
}

pub fn gamma5<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero + Copy,
    I: TensorStructure + FromIterator<Slot<Euclidean>>,
{
    let structure = indices.into_iter().map(|i| Euclidean::slot(4, i)).collect();

    gamma5_dirac_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn gamma5sym<T>(
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: One + Zero + Copy,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        indices.into_iter().map(|i| Euclidean::slot(4, i)),
        State::get_symbol("γ5"),
        None,
    ));

    gamma5_dirac_data(structure)
}

pub fn gamma5_dirac_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + Clone,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(T::one(), T::zero());

    let mut gamma5 = SparseTensor::empty(structure);

    gamma5.set(&[0, 2], c1.clone()).unwrap();
    gamma5.set(&[1, 3], c1.clone()).unwrap();
    gamma5.set(&[2, 0], c1.clone()).unwrap();
    gamma5.set(&[3, 1], c1.clone()).unwrap();

    gamma5
}

pub fn gamma5_weyl_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + Clone + Neg<Output = T>,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(T::one(), T::zero());

    let mut gamma5 = SparseTensor::empty(structure);

    gamma5.set(&[0, 0], -c1.clone()).unwrap();
    gamma5.set(&[1, 1], -c1.clone()).unwrap();
    gamma5.set(&[2, 2], c1.clone()).unwrap();
    gamma5.set(&[3, 3], c1.clone()).unwrap();

    gamma5
}

pub fn proj_m<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: Zero + One + NumCast + Clone,
    I: TensorStructure + FromIterator<Slot<Euclidean>>,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let structure = indices.into_iter().map(|i| Euclidean::slot(4, i)).collect();

    proj_m_data_dirac(structure)
}

#[cfg(feature = "shadowing")]
pub fn proj_msym<T>(
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: Zero + One + NumCast + Clone,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        indices.into_iter().map(|i| Euclidean::slot(4, i)),
        State::get_symbol("ProjM"),
        None,
    ));

    proj_m_data_dirac(structure)
}

#[allow(clippy::similar_names)]
pub fn proj_m_data_dirac<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + NumCast + Clone,
    N: TensorStructure,
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

#[allow(clippy::similar_names)]
pub fn proj_m_data_weyl<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + NumCast + Clone,
    N: TensorStructure,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let c1 = Complex::<T>::new(T::one(), T::zero());
    let mut proj_m = SparseTensor::empty(structure);

    proj_m.set(&[0, 0], c1.clone()).unwrap();
    proj_m.set(&[1, 1], c1.clone()).unwrap();

    proj_m
}

pub fn proj_p<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: NumCast + Zero + Clone,
    I: TensorStructure + FromIterator<Slot<Bispinor>>,
{
    // ProjP(1,2) Right chirality projector (( 1+γ5)/ 2 )_s1_s2
    let structure = indices.into_iter().map(|i| Bispinor::slot(4, i)).collect();

    proj_p_data_dirac(structure)
}

#[cfg(feature = "shadowing")]
pub fn proj_psym<T>(
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: Zero + Clone + NumCast,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        indices.into_iter().map(|i| Bispinor::slot(4, i)),
        State::get_symbol("ProjP"),
        None,
    ));

    proj_p_data_dirac(structure)
}

pub fn proj_p_data_dirac<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: NumCast + Zero + Clone,
    N: TensorStructure,
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

#[allow(clippy::similar_names)]
pub fn proj_p_data_weyl<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + NumCast + Clone,
    N: TensorStructure,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let c1 = Complex::<T>::new(T::one(), T::zero());
    let mut proj_m = SparseTensor::empty(structure);

    proj_m.set(&[2, 2], c1.clone()).unwrap();
    proj_m.set(&[3, 3], c1.clone()).unwrap();

    proj_m
}

pub fn sigma<T, I>(
    indices: [AbstractIndex; 2],
    minkdices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, I>
where
    T: Copy + Zero + One + Neg<Output = T>,
    I: TensorStructure + FromIterator<Slot<PhysReps>>,
{
    let structure = [
        Slot::<PhysReps>::from(Bispinor::slot(4, indices[0])),
        Bispinor::slot(4, indices[1]).into(),
        Lorentz::slot(4, minkdices[0]).into(),
        Lorentz::slot(4, minkdices[1]).into(),
    ]
    .into_iter()
    .map(Slot::from)
    .collect();

    sigma_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn sigmasym<T>(
    indices: [AbstractIndex; 2],
    minkdices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: Copy + Zero + Clone + One + Neg<Output = T>,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            Slot::<PhysReps>::from(Bispinor::slot(4, indices[0])),
            Bispinor::slot(4, indices[1]).into(),
            Lorentz::slot(4, minkdices[0]).into(),
            Lorentz::slot(4, minkdices[1]).into(),
        ],
        State::get_symbol("σ"),
        None,
    ));

    sigma_data(structure)
}

#[allow(clippy::similar_names)]
pub fn sigma_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Copy + Zero + One + Neg<Output = T>,
    N: TensorStructure,
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

#[cfg(test)]
mod test {
    #[cfg(feature = "shadowing")]
    use symbolica::{atom::Atom, state::State};

    #[cfg(feature = "shadowing")]
    use super::*;

    #[cfg(feature = "shadowing")]
    use crate::{
        complex::RealOrComplexTensor,
        contraction::Contract,
        network::TensorNetwork,
        parametric::ParamOrConcrete,
        shadowing::Shadowable,
        structure::{HasStructure, VecStructure},
        symbolica_utils::{SerializableAtom, SerializableSymbol},
    };

    #[test]
    #[cfg(feature = "shadowing")]
    fn clifford() {
        use crate::network::TensorNetwork;

        let expr = Atom::parse(
            "γ(mink(4,1),bis(4,4),bis(4,3))*γ(mink(4,2),bis(4,3),bis(4,4))+γ(mink(4,2),bis(4,4),bis(4,3))*γ(mink(4,1),bis(4,3),bis(4,4))",
        )
        .unwrap();
        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();

        net.contract();

        insta::assert_ron_snapshot!(net
            .result()
            .unwrap()
            .0
            .try_into_concrete()
            .unwrap()
            .try_into_complex()
            .unwrap()
            .to_bare_dense());
    }

    #[test]
    #[cfg(feature = "shadowing")]
    fn clifford2() {
        let expr = Atom::parse(
            "γ(mink(4,1),bis(4,4),bis(4,3))*γ(mink(4,2),bis(4,3),bis(4,1))+γ(mink(4,2),bis(4,4),bis(4,3))*γ(mink(4,1),bis(4,3),bis(4,1))",
        )
        .unwrap();
        // +γ(aind(bis(4,4),bis(4,3),mink(4,2)))*γ(aind(bis(4,3),bis(4,4),mink(4,1))))
        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();

        net.contract();
        let other = Atom::parse("2*Metric(mink(4,1),mink(4,2))*id(bis(4,4),bis(4,3))").unwrap();

        let mut net = TensorNetwork::try_from(other.as_view()).unwrap();

        net.contract();

        // println!(
        //     "{}",
        //     net.to_fully_parametric().result_tensor_smart().unwrap()
        // );
        //
        //TODO need to be able to compare these!
    }

    #[test]
    #[cfg(feature = "shadowing")]
    fn gamma_algebra() {
        let _ = env_logger::builder().is_test(true).try_init();
        let expr = Atom::parse("γ(aind(mink(4,1),bis(4,4),bis(4,3)))*Q(1,aind(mink(4,1)))*γ(aind(mink(4,2),bis(4,3),bis(4,4)))*Q(2,aind(mink(4,2)))").unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();

        net.contract();

        println!("{}", net.result().unwrap().0);

        let expr = Atom::parse("γ(aind(bis(4,1),bis(4,4),bis(4,3)))*Q(1,aind(bis(4,1)))*γ(aind(bis(4,2),bis(4,3),bis(4,4)))*Q(2,aind(bis(4,2)))").unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();

        net.contract();

        println!("{}", net.result().unwrap().0);
    }

    #[test]
    #[cfg(feature = "shadowing")]
    fn gamma_algebra2() {
        let expr = Atom::parse(
            "γ(aind(mink(4,1),bis(4,4),bis(4,3)))*γ(aind(mink(4,1),bis(4,1),bis(4,2)))",
        )
        .unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();

        net.contract();

        println!("{}", net.result().unwrap().0);
    }

    #[test]
    #[cfg(feature = "shadowing")]
    fn data() {
        let _ = env_logger::builder().is_test(true).try_init();
        let expr = Atom::parse(
            "γ(aind(mink(4,1),bis(4,4),bis(4,3)))*γ(aind(mink(4,2),bis(4,3),bis(4,4)))",
        )
        .unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();
        println!("{}", net.dot());
        net.contract();

        let mut other =
            TensorNetwork::try_from(Atom::parse("p(aind(mink (4,1)))").unwrap().as_view()).unwrap();
        other.contract();

        net.push(other.result().unwrap().0);

        net.contract();

        println!("{}", net.result().unwrap().0);
    }

    #[test]
    #[cfg(feature = "shadowing")]
    fn data2() {
        let _ = env_logger::builder().is_test(true).try_init();
        let expr = Atom::parse(
            "γ(aind(mink(4,1),bis(4,4),bis(4,3)))*γ(aind(mink(4,2),bis(4,3),bis(4,4)))",
        )
        .unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();
        println!("{}", net.dot());
        net.contract();

        println!("{}", net.result().unwrap().0);
    }
    #[allow(clippy::type_complexity)]
    #[test]
    #[cfg(feature = "shadowing")]
    fn data3() {
        let _ = env_logger::builder().is_test(true).try_init();
        let g1 = Atom::parse("γ(aind(mink(4,1),bis(4,3),bis(4,4)))").unwrap();
        let g2 = Atom::parse("γ(aind(mink(4,2),bis(4,3),bis(4,4)))").unwrap();
        let p = Atom::parse("p(aind(mink (4,1)))").unwrap();
        let u = Atom::parse("u(aind(bis (4,3)))").unwrap();
        let v = Atom::parse("v(aind(bis (4,4)))").unwrap();

        let q = Atom::parse("q(aind(mink (4,2)))").unwrap();

        let g1_tensor: ParamOrConcrete<
            RealOrComplexTensor<
                f64,
                crate::structure::NamedStructure<
                    crate::symbolica_utils::SerializableSymbol,
                    Vec<crate::symbolica_utils::SerializableAtom>,
                >,
            >,
            crate::structure::NamedStructure<
                crate::symbolica_utils::SerializableSymbol,
                Vec<crate::symbolica_utils::SerializableAtom>,
            >,
        > = TensorNetwork::try_from(g1.as_view())
            .unwrap()
            .result()
            .unwrap()
            .0;
        let g2_tensor = TensorNetwork::try_from(g2.as_view())
            .unwrap()
            .result()
            .unwrap()
            .0;
        let p_tensor = TensorNetwork::try_from(p.as_view())
            .unwrap()
            .result()
            .unwrap()
            .0;

        let q_tensor = TensorNetwork::try_from(q.as_view())
            .unwrap()
            .result()
            .unwrap()
            .0;

        let u_tensor = TensorNetwork::try_from(u.as_view())
            .unwrap()
            .result()
            .unwrap()
            .0;
        let v_tensor = TensorNetwork::try_from(v.as_view())
            .unwrap()
            .result()
            .unwrap()
            .0;

        println!(
            "{}",
            g1_tensor
                .contract(&p_tensor)
                .unwrap()
                .contract(&u_tensor)
                .unwrap()
                .contract(&v_tensor)
                .unwrap()
                .scalar()
                .unwrap()
                .try_as_param()
                .unwrap()
                .expand()
                .factor()
        );

        println!(
            "{}",
            g2_tensor
                .contract(&q_tensor)
                .unwrap()
                .contract(&u_tensor)
                .unwrap()
                .contract(&v_tensor)
                .unwrap()
                .scalar()
                .unwrap()
                .try_as_param()
                .unwrap()
                .expand()
                .factor()
        );
        // println!(
        //     "{}",
        //     g1_tensor
        //         .contract(&g2_tensor)
        //         .unwrap()
        //         .contract(&p_tensor)
        //         .unwrap()
        // );

        // println!(
        //     "pslash gamma \n {}",
        //     g1_tensor
        //         .contract(&p_tensor)
        //         .unwrap()
        //         .contract(&g2_tensor)
        //         .unwrap()
        // );

        // for (i, a) in g1_tensor
        //     .contract(&p_tensor)
        //     .unwrap()
        //     .contract(&g2_tensor)
        //     .unwrap()
        //     .try_into_parametric()
        //     .unwrap()
        //     .tensor
        //     .map_data(|a| a.expand())
        //     .iter_flat()
        // {
        //     print!("{}, ", a);
        // }

        // let prob = g1_tensor.contract(&p_tensor).unwrap();

        // println!("{}", prob);
        // println!("{:?}\n", prob.structure());

        // let q0 = Atom::parse("q(0)").unwrap();
        // let q1 = Atom::parse("q(1)").unwrap();
        // let q2 = Atom::parse("q(2)").unwrap();
        // let q3 = Atom::parse("q(3)").unwrap();
        // let i = Atom::parse("i").unwrap();
        // let z = Atom::new_num(0);
        // let q12 = &q1 + &i * &q2;
        // let bbar = &q1 - &i * &q2;

        // let spinstructureb = VecStructure::new(vec![
        //     Bispinor::new_slot_selfless(4, 4).into(),
        //     Bispinor::new_slot_selfless(4, 3).into(),
        // ]);

        // let b: DenseTensor<
        //     Atom,
        //     crate::structure::NamedStructure<SerializableSymbol, Vec<SerializableAtom>>,
        // > = DenseTensor::from_data(
        //     vec![
        //         q0.clone(),
        //         z.clone(),
        //         -q3.clone(),
        //         -bbar.clone(), //
        //         z.clone(),
        //         q0.clone(),
        //         -q12.clone(),
        //         q3.clone(),
        //         q3.clone(),
        //         bbar,
        //         -q0.clone(),
        //         z.clone(),
        //         q12,
        //         -q3.clone(),
        //         z,
        //         -q0,
        //     ],
        //     spinstructureb.to_named(State::get_symbol("b").into(), None),
        // )
        // .unwrap();

        // println!("{}", b);

        // println!(
        //     "pslash gamma \n {}",
        //     g2_tensor
        //         .contract(&p_tensor)
        //         .unwrap()
        //         .contract(&g1_tensor)
        //         .unwrap()
        // );

        // let pslash = g1_tensor.contract(&p_tensor).unwrap();
        // let qslash = g2_tensor.contract(&q_tensor).unwrap();

        // println!(
        //     "pslash * qslash\n {}",
        //     pslash
        //         .contract(&qslash)
        //         .unwrap()
        //         .scalar()
        //         .unwrap()
        //         .try_as_param()
        //         .unwrap()
        //         .expand()
        // );
    }

    #[test]
    #[cfg(feature = "shadowing")]
    fn data4() {
        let _ = env_logger::builder().is_test(true).try_init();
        let expr = Atom::parse("A(aind(mink(4,1),bis(4,4),bis(4,3)))*B(aind(mink(4,1)))").unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();
        println!("{}", net.dot());
        net.contract();
        let res = net.result().unwrap().0;

        println!("{}", res);

        println!("{:?}", res.structure().external_structure());
    }

    #[test]
    #[cfg(feature = "shadowing")]
    fn data5() {
        let spinstructure = VecStructure::new(vec![
            Bispinor::slot(4, 3).into(),
            Bispinor::slot(4, 4).into(),
        ]);

        let p0 = Atom::parse("p(0)").unwrap();
        let p1 = Atom::parse("p(1)").unwrap();
        let p2 = Atom::parse("p(2)").unwrap();
        let p3 = Atom::parse("p(3)").unwrap();
        let i = Atom::parse("i").unwrap();
        let p12 = &p1 + &i * &p2;
        let abar = &p1 - &i * &p2;
        let z = Atom::new_num(0);

        let _q0 = Atom::parse("q(0)").unwrap();
        let _q1 = Atom::parse("q(1)").unwrap();
        let _q2 = Atom::parse("q(2)").unwrap();
        let _q3 = Atom::parse("q(3)").unwrap();

        let a: DenseTensor<
            Atom,
            crate::structure::NamedStructure<SerializableSymbol, Vec<SerializableAtom>>,
        > = DenseTensor::from_data(
            vec![
                p0.clone(),
                z.clone(),
                -p3.clone(),
                -abar.clone(), //
                z.clone(),
                p0.clone(),
                -p12.clone(),
                p3.clone(), //
                p3.clone(),
                abar,
                -p0.clone(),
                z.clone(), //
                p12,
                -p3.clone(),
                z.clone(),
                -p0,
            ],
            spinstructure.to_named(State::get_symbol("a").into(), None),
        )
        .unwrap();

        let ashadow = a.expanded_shadow().unwrap();

        let spinstructureb = VecStructure::new(vec![
            Bispinor::slot(4, 4).into(),
            Bispinor::slot(4, 3).into(),
        ]);

        let q0 = Atom::parse("q(0)").unwrap();
        let q1 = Atom::parse("q(1)").unwrap();
        let q2 = Atom::parse("q(2)").unwrap();
        let q3 = Atom::parse("q(3)").unwrap();
        let i = Atom::parse("i").unwrap();
        let q12 = &q1 + &i * &q2;
        let bbar = &q1 - &i * &q2;

        let b: DenseTensor<
            Atom,
            crate::structure::NamedStructure<SerializableSymbol, Vec<SerializableAtom>>,
        > = DenseTensor::from_data(
            vec![
                q0.clone(),
                z.clone(),
                -q3.clone(),
                -bbar.clone(), //
                z.clone(),
                q0.clone(),
                -q12.clone(),
                q3.clone(),
                q3.clone(),
                bbar,
                -q0.clone(),
                z.clone(),
                q12,
                -q3.clone(),
                z,
                -q0,
            ],
            spinstructureb.to_named(State::get_symbol("b").into(), None),
        )
        .unwrap();

        let bshadow = b.expanded_shadow().unwrap();

        println!("{}", a.contract(&b).unwrap().scalar().unwrap().expand());

        println!(
            "{}",
            ashadow
                .contract(&bshadow)
                .unwrap()
                .scalar()
                .unwrap()
                .expand()
        );
    }

    #[test]
    #[cfg(feature = "shadowing")]
    fn data6() {
        let spinstructure = VecStructure::new(vec![
            Bispinor::slot(4, 3).into(),
            Bispinor::slot(4, 4).into(),
        ]);

        let a00 = Atom::parse("a(0,0)").unwrap();
        let a01 = Atom::parse("a(0,1)").unwrap();
        let a02 = Atom::parse("a(0,2)").unwrap();
        let a03 = Atom::parse("a(0,3)").unwrap();
        let a10 = Atom::parse("a(1,0)").unwrap();
        let a11 = Atom::parse("a(1,1)").unwrap();
        let a12 = Atom::parse("a(1,2)").unwrap();
        let a13 = Atom::parse("a(1,3)").unwrap();
        let a20 = Atom::parse("a(2,0)").unwrap();
        let a21 = Atom::parse("a(2,1)").unwrap();
        let a22 = Atom::parse("a(2,2)").unwrap();
        let a23 = Atom::parse("a(2,3)").unwrap();
        let a30 = Atom::parse("a(3,0)").unwrap();
        let a31 = Atom::parse("a(3,1)").unwrap();
        let a32 = Atom::parse("a(3,2)").unwrap();
        let a33 = Atom::parse("a(3,3)").unwrap();

        let a: DenseTensor<
            Atom,
            crate::structure::NamedStructure<SerializableSymbol, Vec<SerializableAtom>>,
        > = DenseTensor::from_data(
            vec![
                a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33,
            ],
            spinstructure.to_named(State::get_symbol("a").into(), None),
        )
        .unwrap();

        let ashadow = a.expanded_shadow().unwrap();

        let spinstructureb = VecStructure::new(vec![
            Bispinor::slot(4, 4).into(),
            Bispinor::slot(4, 3).into(),
        ]);

        let b00 = Atom::parse("b(0,0)").unwrap();
        let b01 = Atom::parse("b(0,1)").unwrap();
        let b02 = Atom::parse("b(0,2)").unwrap();
        let b03 = Atom::parse("b(0,3)").unwrap();
        let b10 = Atom::parse("b(1,0)").unwrap();
        let b11 = Atom::parse("b(1,1)").unwrap();
        let b12 = Atom::parse("b(1,2)").unwrap();
        let b13 = Atom::parse("b(1,3)").unwrap();
        let b20 = Atom::parse("b(2,0)").unwrap();
        let b21 = Atom::parse("b(2,1)").unwrap();
        let b22 = Atom::parse("b(2,2)").unwrap();
        let b23 = Atom::parse("b(2,3)").unwrap();
        let b30 = Atom::parse("b(3,0)").unwrap();
        let b31 = Atom::parse("b(3,1)").unwrap();
        let b32 = Atom::parse("b(3,2)").unwrap();
        let b33 = Atom::parse("b(3,3)").unwrap();

        let b: DenseTensor<
            Atom,
            crate::structure::NamedStructure<SerializableSymbol, Vec<SerializableAtom>>,
        > = DenseTensor::from_data(
            vec![
                b00, b01, b02, b03, b10, b11, b12, b13, b20, b21, b22, b23, b30, b31, b32, b33,
            ],
            spinstructureb.to_named(State::get_symbol("b").into(), None),
        )
        .unwrap();

        let bshadow = b.expanded_shadow().unwrap();

        println!("{}", a.contract(&b).unwrap().scalar().unwrap().expand());

        println!(
            "{}",
            ashadow
                .contract(&bshadow)
                .unwrap()
                .scalar()
                .unwrap()
                .expand()
        );
    }
}
