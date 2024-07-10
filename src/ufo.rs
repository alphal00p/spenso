use std::ops::Neg;

use super::{
    AbstractIndex, DenseTensor,
    Representation::{self, Euclidean, Lorentz},
    SetTensorData, Slot, SparseTensor,
};
use crate::ABSTRACTIND;
use crate::BISPINOR;
use crate::COLORADJ;
use crate::COLORANTIFUND;
use crate::COLORANTISEXT;
use crate::COLORFUND;
use crate::COLORSEXT;
use crate::LORENTZ;

#[cfg(feature = "shadowing")]
use crate::{HistoryStructure, NamedStructure};
use num::{NumCast, One, Zero};

use crate::{Complex, Dimension, TensorStructure};

#[cfg(feature = "shadowing")]
use super::{IntoArgs, IntoSymbol, Shadowable};
use constcat::concat;
#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, Symbol},
    id::Pattern,
    state::State,
};

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
    I: TensorStructure + FromIterator<Slot>,
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
    N: TensorStructure,
{
    assert!(structure.order() == 2, "Identity tensor must be rank 2");

    // println!("{:?}", structure.reps());

    assert!(
        Dimension::from(structure.reps()[0]) == Dimension::from(structure.reps()[1]),
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

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_spin(atom: Atom) -> Atom {
    let replacements = [
        (
            "Identity(i_,j_)",
            concat!(
                "id(",
                ABSTRACTIND,
                "(",
                BISPINOR,
                "(4,i_),",
                BISPINOR,
                "(4,j_)))"
            ),
        ),
        (
            "IdentityL(mu_,nu_)",
            concat!(
                "id(",
                ABSTRACTIND,
                "(",
                LORENTZ,
                "(4,mu_),",
                LORENTZ,
                "(4,nu_)))"
            ),
        ),
        (
            "Gamma(mu_,i_,j_)",
            concat!(
                "γ(",
                ABSTRACTIND,
                "(",
                LORENTZ,
                "(4,mu_),",
                BISPINOR,
                "(4,i_),",
                BISPINOR,
                "(4,j_)))"
            ),
        ),
        (
            "Gamma5(i_,j_)",
            concat!(
                "γ5(",
                ABSTRACTIND,
                "(",
                BISPINOR,
                "(4,i_),",
                BISPINOR,
                "(4,j_)))"
            ),
        ),
        (
            "ProjM(i_,j_)",
            concat!(
                "ProjM(",
                ABSTRACTIND,
                "(",
                BISPINOR,
                "(4,i_),",
                BISPINOR,
                "(4,j_)))"
            ),
        ),
        (
            "ProjP(i_,j_)",
            concat!(
                "ProjP(",
                ABSTRACTIND,
                "(",
                BISPINOR,
                "(4,i_),",
                BISPINOR,
                "(4,j_)))"
            ),
        ),
        (
            "Sigma(mu_,nu_,i_,j_)",
            concat!(
                "σ(",
                ABSTRACTIND,
                "(",
                LORENTZ,
                "(4,mu_),",
                LORENTZ,
                "(4,nu_),",
                BISPINOR,
                "(4,i_),",
                BISPINOR,
                "(4,j_)))"
            ),
        ),
        (
            "C(i_,j_)",
            concat!(
                "C(",
                ABSTRACTIND,
                "(",
                BISPINOR,
                "(4,i_),",
                BISPINOR,
                "(4,j_)))"
            ),
        ),
        (
            "Metric(mu_,nu_)",
            concat!(
                "Metric(",
                ABSTRACTIND,
                "(",
                LORENTZ,
                "(4,mu_),",
                LORENTZ,
                "(4,nu_)))"
            ),
        ),
    ];

    batch_replace(&replacements, atom)
}

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_color(atom: Atom) -> Atom {
    let replacements = [
        (
            "T(a_,i_,ia_)",
            concat!(
                "T(",
                ABSTRACTIND,
                "(",
                COLORADJ,
                "(a_),",
                COLORFUND,
                "(i_),",
                COLORANTIFUND,
                "(ia_)))"
            ),
        ),
        (
            "f(a1_,a2_,a3_)",
            concat!(
                "f(",
                ABSTRACTIND,
                "(",
                COLORADJ,
                "(a1_),",
                COLORADJ,
                "(a2_),",
                COLORADJ,
                "(a3_)))"
            ),
        ),
        (
            "d(a1_,a2_,a3_)",
            concat!(
                "d(",
                ABSTRACTIND,
                "(",
                COLORADJ,
                "(a1_),",
                COLORADJ,
                "(a2_),",
                COLORADJ,
                "(a3_)))"
            ),
        ),
        (
            "Epsilon(i1_,i2_,i3_)",
            concat!(
                "EpsilonBar(",
                ABSTRACTIND,
                "(",
                COLORFUND,
                "(i1_),",
                COLORFUND,
                "(i2_),",
                COLORFUND,
                "(i3_)))"
            ),
        ),
        (
            "EpsilonBar(ia1_,ia2_,ia3_)",
            concat!(
                "Epsilon(",
                ABSTRACTIND,
                "(",
                COLORANTIFUND,
                "(ia1_),",
                COLORANTIFUND,
                "(ia2_),",
                COLORANTIFUND,
                "(ia3_)))"
            ),
        ),
        (
            "T6(a_,s_,as_)",
            concat!(
                "T6(",
                ABSTRACTIND,
                "(",
                COLORADJ,
                "(a_),",
                COLORSEXT,
                "(s_),",
                COLORANTISEXT,
                "(as_)))"
            ),
        ),
        (
            "K6(ia1_,ia2_,s_)",
            concat!(
                "K6(",
                ABSTRACTIND,
                "(",
                COLORANTIFUND,
                "(ia1_),",
                COLORANTIFUND,
                "(ia2_),",
                COLORSEXT,
                "(s_)))"
            ),
        ),
        (
            "K6Bar(as_,i1_,i2_)",
            concat!(
                "K6Bar(",
                ABSTRACTIND,
                "(",
                COLORANTISEXT,
                "(as_),",
                COLORFUND,
                "(i1_),",
                COLORFUND,
                "(i2_)))"
            ),
        ),
    ];

    batch_replace(&replacements, atom)
}

#[cfg(feature = "shadowing")]
pub fn batch_replace(replacements: &[(&str, &str)], mut atom: Atom) -> Atom {
    for (pattern, replacement) in replacements {
        let pattern = Pattern::parse(pattern).unwrap();
        let replacement = Pattern::parse(replacement).unwrap();
        atom = pattern.replace_all(atom.as_view(), &replacement, None, None);
    }

    atom
}

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_color_wrapped(atom: Atom) -> Atom {
    let replacements = [
        (
            "T(a_,i_,ia_)",
            "T(coad(8,indexid(a_)),cof(3,indexid(i_)),coaf(3,indexid(ia_)))",
        ),
        (
            "f(a1_,a2_,a3_)",
            "f(coad(8,indexid(a1_)),coad(8,indexid(a2_)),coad(8,indexid(a3_)))",
        ),
        (
            "d(a1_,a2_,a3_)",
            "d(coad(8,indexid(a1_)),coad(8,indexid(a2_)),coad(8,indexid(a3_)))",
        ),
        (
            "Epsilon(i1_,i2_,i3_)",
            "EpsilonBar(cof(3,indexid(i1_)),cof(3,indexid(i2_)),cof(3,indexid(i3_)))",
        ),
        (
            "EpsilonBar(ia1_,ia2_,ia3_)",
            "Epsilon(coaf(3,indexid(ia1_)),coaf(3,indexid(ia2_)),coaf(3,indexid(ia3_)))",
        ),
        (
            "T6(a_,s_,as_)",
            "T6(coad(8,indexid(a_)),cos(6,indexid(s_)),coas(6,indexid(as_)))",
        ),
        (
            "K6(ia1_,ia2_,s_)",
            "K6(coaf(3,indexid(ia1_)),coaf(3,indexid(ia2_)),cos(6,indexid(s_)))",
        ),
        (
            "K6Bar(as_,i1_,i2_)",
            "K6Bar(coas(6,indexid(as_)),cof(3,indexid(i1_)),cof(3,indexid(i2_)))",
        ),
    ];

    batch_replace(&replacements, atom)
}

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_spin_wrapped(atom: Atom) -> Atom {
    let replacements = [
        (
            "Identity(i_,j_)",
            "id(bis(4,indexid(i_)),bis(4,indexid(j_)))",
        ),
        (
            "IdentityL(mu_,nu_)",
            "id(lor(4,indexid(mu_)),lor(4,indexid(nu_)))",
        ),
        (
            "Gamma(mu_,i_,j_)",
            "γ(lor(4,indexid(mu_)),bis(4,indexid(i_)),bis(4,indexid(j_)))",
        ),
        ("Gamma5(i_,j_)", "γ5(bis(4,indexid(i_)),bis(4,indexid(j_)))"),
        (
            "ProjM(i_,j_)",
            "ProjM(bis(4,indexid(i_)),bis(4,indexid(j_)))",
        ),
        (
            "ProjP(i_,j_)",
            "ProjP(bis(4,indexid(i_)),bis(4,indexid(j_)))",
        ),
        (
            "Sigma(mu_,nu_,i_,j_)",
            "σ(lor(4,indexid(mu_)),lor(4,indexid(nu_)),bis(4,indexid(i_)),bis(4,indexid(j_)))",
        ),
        ("C(i_,j_)", "C(bis(4,indexid(i_)),bis(4,indexid(j_)))"),
        (
            "Metric(mu_,nu_)",
            "Metric(lor(4,indexid(mu_)),lor(4,indexid(nu_)))",
        ),
    ];

    batch_replace(&replacements, atom)
}

#[allow(dead_code)]
#[must_use]
pub fn lorentz_identity<T, I>(
    indices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero,
    I: TensorStructure + FromIterator<Slot>,
{
    // IdentityL(1,2) (Lorentz) Kronecker delta δ^μ1_μ1
    let signature = Lorentz(4.into());
    identity(indices, signature)
}

pub fn mink_four_vector<T, I>(index: AbstractIndex, p: &[T; 4]) -> DenseTensor<T, I>
where
    T: Clone,
    I: TensorStructure + FromIterator<Slot>,
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
) -> DenseTensor<T, HistoryStructure<Symbol, ()>>
where
    T: Clone,
{
    use crate::NamedStructure;

    DenseTensor::from_data(
        p,
        HistoryStructure::from(NamedStructure::from_iter(
            [(index, Lorentz(4.into()))],
            State::get_symbol("p"),
            None,
        )),
    )
    .unwrap_or_else(|_| unreachable!())
}

pub fn euclidean_four_vector<T, I>(index: AbstractIndex, p: &[T; 4]) -> DenseTensor<T, I>
where
    T: Clone,
    I: TensorStructure + FromIterator<Slot>,
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
) -> DenseTensor<T, HistoryStructure<Symbol, ()>>
where
    T: Clone,
{
    use crate::NamedStructure;

    DenseTensor::from_data(
        p,
        HistoryStructure::from(NamedStructure::from_iter(
            [(index, Euclidean(4.into()))],
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
        [(index, Lorentz(4.into()))],
        name,
        args,
    ))
    .to_shell()
    .shadow()
    .unwrap_or_else(|| unreachable!())
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
        [(index, Euclidean(4.into()))],
        name,
        None,
    ))
    .to_shell()
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
    I: TensorStructure + FromIterator<Slot>,
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
    I: TensorStructure + FromIterator<Slot>,
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
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: One + Zero + Copy + std::ops::Neg<Output = T>,
{
    use crate::NamedStructure;

    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
            (minkindex, Lorentz(4.into())),
        ],
        State::get_symbol("γ"),
        None,
    ));

    gamma_data(structure)
}

#[allow(clippy::similar_names)]
pub fn gamma_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
where
    T: Zero + One + Neg<Output = T> + Clone,
    N: TensorStructure,
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
    I: TensorStructure + FromIterator<Slot>,
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
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: One + Zero + Copy,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
        ],
        State::get_symbol("γ5"),
        None,
    ));

    gamma5_data(structure)
}

pub fn gamma5_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
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

pub fn proj_m<T, I>(indices: (AbstractIndex, AbstractIndex)) -> SparseTensor<Complex<T>, I>
where
    T: Zero + One + NumCast + Clone,
    I: TensorStructure + FromIterator<Slot>,
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
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: Zero + One + NumCast + Clone,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
        ],
        State::get_symbol("ProjM"),
        None,
    ));

    proj_m_data(structure)
}

#[allow(clippy::similar_names)]
pub fn proj_m_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
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

pub fn proj_p<T, I>(indices: (AbstractIndex, AbstractIndex)) -> SparseTensor<Complex<T>, I>
where
    T: NumCast + Zero + Clone,
    I: TensorStructure + FromIterator<Slot>,
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
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: Zero + Clone + NumCast,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
        ],
        State::get_symbol("ProjP"),
        None,
    ));

    proj_p_data(structure)
}

pub fn proj_p_data<T, N>(structure: N) -> SparseTensor<Complex<T>, N>
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

pub fn sigma<T, I>(
    indices: (AbstractIndex, AbstractIndex),
    minkdices: (AbstractIndex, AbstractIndex),
) -> SparseTensor<Complex<T>, I>
where
    T: Copy + Zero + One + Neg<Output = T>,
    I: TensorStructure + FromIterator<Slot>,
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
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: Copy + Zero + Clone + One + Neg<Output = T>,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            (indices.0, Euclidean(4.into())),
            (indices.1, Euclidean(4.into())),
            (minkdices.0, Lorentz(4.into())),
            (minkdices.1, Lorentz(4.into())),
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

pub fn metric_data<T, N>(structure: N) -> SparseTensor<T, N>
where
    T: One + Clone + Neg<Output = T>,
    N: TensorStructure,
{
    let reps = structure.reps();

    if reps[0] == reps[1] && reps.len() == 2 {
        match reps[0] {
            Lorentz(d) => {
                let mut metric = SparseTensor::empty(structure);

                for i in 1..d.into() {
                    metric
                        .set(&[i, i], -T::one())
                        .unwrap_or_else(|_| unreachable!());
                }

                metric
                    .set(&[0, 0], T::one())
                    .unwrap_or_else(|_| unreachable!());

                metric
            }
            _ => panic!("Metric tensor must have Lorentz indices"),
        }
    } else {
        panic!("Metric tensor must have equal indices")
    }
}
