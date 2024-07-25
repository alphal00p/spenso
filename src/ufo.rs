use std::ops::Neg;

use super::{
    AbstractIndex, DenseTensor, RepName, Representation, SetTensorData, Slot, SparseTensor,
};

#[cfg(feature = "shadowing")]
use crate::{
    ABSTRACTIND, BISPINOR, COLORADJ, COLORANTIFUND, COLORANTISEXT, COLORFUND, COLORSEXT, EUCLIDEAN,
    LORENTZ, SPINFUND,
};

#[cfg(feature = "shadowing")]
use crate::{HistoryStructure, NamedStructure};
use num::{NumCast, One, Zero};

use crate::{
    BaseRepName, Bispinor, Complex, Dimension, Dual, Euclidean, Lorentz, PhysReps, TensorStructure,
};

#[cfg(feature = "shadowing")]
use super::{IntoArgs, IntoSymbol, Shadowable};
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

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_spin(atom: Atom, wrapped: bool) -> Atom {
    let replacements = [
        (
            "Identity(i_,j_)",
            named_tensor(
                "id".into(),
                &[
                    &bis(wrapped_to_four("i_", wrapped)),
                    &bis(wrapped_to_four("j_", wrapped)),
                ],
            ),
        ),
        (
            "IdentityL(mu_,nu_)",
            named_tensor(
                "id".into(),
                &[
                    &lor(wrapped_to_four("mu_", wrapped)),
                    &lor(wrapped_to_four("nu_", wrapped)),
                ],
            ),
        ),
        (
            "Gamma(mu_,i_,j_)",
            named_tensor(
                "γ".into(),
                &[
                    &lor(wrapped_to_four("mu_", wrapped)),
                    &bis(wrapped_to_four("i_", wrapped)),
                    &bis(wrapped_to_four("j_", wrapped)),
                ],
            ),
        ),
        (
            "Gamma5(i_,j_)",
            named_tensor(
                "γ5".into(),
                &[
                    &bis(wrapped_to_four("i_", wrapped)),
                    &bis(wrapped_to_four("j_", wrapped)),
                ],
            ),
        ),
        (
            "ProjM(i_,j_)",
            named_tensor(
                "ProjM".into(),
                &[
                    &bis(wrapped_to_four("i_", wrapped)),
                    &bis(wrapped_to_four("j_", wrapped)),
                ],
            ),
        ),
        (
            "ProjP(i_,j_)",
            named_tensor(
                "ProjP".into(),
                &[
                    &bis(wrapped_to_four("i_", wrapped)),
                    &bis(wrapped_to_four("j_", wrapped)),
                ],
            ),
        ),
        (
            "Sigma(mu_,nu_,i_,j_)",
            named_tensor(
                "σ".into(),
                &[
                    &lor(wrapped_to_four("mu_", wrapped)),
                    &lor(wrapped_to_four("nu_", wrapped)),
                    &bis(wrapped_to_four("i_", wrapped)),
                    &bis(wrapped_to_four("j_", wrapped)),
                ],
            ),
        ),
        (
            "C(i_,j_)",
            named_tensor(
                "C".into(),
                &[
                    &bis(wrapped_to_four("i_", wrapped)),
                    &bis(wrapped_to_four("j_", wrapped)),
                ],
            ),
        ),
        (
            "Metric(mu_,nu_)",
            named_tensor(
                "Metric".into(),
                &[
                    &lor(wrapped_to_four("mu_", wrapped)),
                    &lor(wrapped_to_four("nu_", wrapped)),
                ],
            ),
        ),
    ];

    batch_replace(&replacements, atom)
}
#[cfg(feature = "shadowing")]
fn named_tensor(name: String, args: &[&str]) -> String {
    name + "(" + ABSTRACTIND + "(" + args.join(",").as_str() + "))"
}
#[cfg(feature = "shadowing")]
enum ReplacementArgs {
    Wrapped(usize, &'static str),
    Unwrapped(usize, &'static str),
    Bare(&'static str),
}
#[cfg(feature = "shadowing")]
fn rep_string(rep: &str, rep_args: ReplacementArgs) -> String {
    rep.to_string()
        + "("
        + match rep_args {
            ReplacementArgs::Wrapped(dim, ind) => dim.to_string() + "," + "indexid(" + ind + "))",
            ReplacementArgs::Unwrapped(dim, ind) => dim.to_string() + "," + ind + ")",
            ReplacementArgs::Bare(ind) => ind.to_string() + ")",
        }
        .as_str()
}

#[allow(dead_code)]
#[cfg(feature = "shadowing")]
fn euc(rep_args: ReplacementArgs) -> String {
    rep_string(EUCLIDEAN, rep_args)
}
#[cfg(feature = "shadowing")]
fn lor(rep_args: ReplacementArgs) -> String {
    rep_string(LORENTZ, rep_args)
}
#[cfg(feature = "shadowing")]
fn bis(rep_args: ReplacementArgs) -> String {
    rep_string(BISPINOR, rep_args)
}

#[allow(dead_code)]
#[cfg(feature = "shadowing")]
fn spin(rep_args: ReplacementArgs) -> String {
    rep_string(SPINFUND, rep_args)
}
#[cfg(feature = "shadowing")]
fn coad(ind: &'static str, wrapped: bool) -> String {
    if wrapped {
        rep_string(COLORADJ, ReplacementArgs::Wrapped(8, ind))
    } else {
        rep_string(COLORADJ, ReplacementArgs::Bare(ind))
    }
}
#[cfg(feature = "shadowing")]
fn cof(ind: &'static str, wrapped: bool) -> String {
    if wrapped {
        rep_string(COLORFUND, ReplacementArgs::Wrapped(3, ind))
    } else {
        rep_string(COLORFUND, ReplacementArgs::Bare(ind))
    }
}
#[cfg(feature = "shadowing")]
fn coaf(ind: &'static str, wrapped: bool) -> String {
    if wrapped {
        rep_string(COLORANTIFUND, ReplacementArgs::Wrapped(3, ind))
    } else {
        rep_string(COLORANTIFUND, ReplacementArgs::Bare(ind))
    }
}
#[cfg(feature = "shadowing")]
fn cos(ind: &'static str, wrapped: bool) -> String {
    if wrapped {
        rep_string(COLORSEXT, ReplacementArgs::Wrapped(6, ind))
    } else {
        rep_string(COLORSEXT, ReplacementArgs::Bare(ind))
    }
}
#[cfg(feature = "shadowing")]
fn coas(ind: &'static str, wrapped: bool) -> String {
    if wrapped {
        rep_string(COLORANTISEXT, ReplacementArgs::Wrapped(6, ind))
    } else {
        rep_string(COLORANTISEXT, ReplacementArgs::Bare(ind))
    }
}

#[cfg(feature = "shadowing")]
fn wrapped_to_four(ind: &'static str, wrapped: bool) -> ReplacementArgs {
    if wrapped {
        ReplacementArgs::Wrapped(4, ind)
    } else {
        ReplacementArgs::Unwrapped(4, ind)
    }
}

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_color(atom: Atom, wrapped: bool) -> Atom {
    let replacements = [
        (
            "T(a_,i_,ia_)",
            named_tensor(
                "T".into(),
                &[
                    &coad("a_", wrapped),
                    &cof("i_", wrapped),
                    &coaf("ia_", wrapped),
                ],
            ),
        ),
        (
            "f(a1_,a2_,a3_)",
            named_tensor(
                "f".into(),
                &[
                    &coad("a1_", wrapped),
                    &coad("a2_", wrapped),
                    &coad("a3_", wrapped),
                ],
            ),
        ),
        (
            "d(a1_,a2_,a3_)",
            named_tensor(
                "d".into(),
                &[
                    &coad("a1_", wrapped),
                    &coad("a2_", wrapped),
                    &coad("a3_", wrapped),
                ],
            ),
        ),
        (
            "Epsilon(i1_,i2_,i3_)",
            named_tensor(
                "EpsilonBar".into(),
                &[
                    &cof("i1_", wrapped),
                    &cof("i2_", wrapped),
                    &cof("i3_", wrapped),
                ],
            ),
        ),
        (
            "EpsilonBar(ia1_,ia2_,ia3_)",
            named_tensor(
                "Epsilon".into(),
                &[
                    &coaf("ia1_", wrapped),
                    &coaf("ia2_", wrapped),
                    &coaf("ia3_", wrapped),
                ],
            ),
        ),
        (
            "T6(a_,s_,as_)",
            named_tensor(
                "T6".into(),
                &[
                    &coad("a_", wrapped),
                    &cos("s_", wrapped),
                    &coas("as_", wrapped),
                ],
            ),
        ),
        (
            "K6(ia1_,ia2_,s_)",
            named_tensor(
                "K6".into(),
                &[
                    &coaf("ia1_", wrapped),
                    &coaf("ia2_", wrapped),
                    &cos("s_", wrapped),
                ],
            ),
        ),
        (
            "K6Bar(as_,i1_,i2_)",
            named_tensor(
                "K6Bar".into(),
                &[
                    &coas("as_", wrapped),
                    &cof("i1_", wrapped),
                    &cof("i2_", wrapped),
                ],
            ),
        ),
    ];

    batch_replace(&replacements, atom)
}

#[cfg(feature = "shadowing")]
pub fn batch_replace<T: AsRef<str>, U: AsRef<str>>(
    replacements: &[(T, U)],
    mut atom: Atom,
) -> Atom {
    for (pattern, replacement) in replacements {
        let pattern = Pattern::parse(pattern.as_ref()).unwrap();
        let replacement = Pattern::parse(replacement.as_ref()).unwrap();
        atom = pattern.replace_all(atom.as_view(), &replacement, None, None);
    }

    atom
}

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_color_wrapped(atom: Atom) -> Atom {
    preprocess_ufo_color(atom, true)
}

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_spin_wrapped(atom: Atom) -> Atom {
    preprocess_ufo_spin(atom, true)
}

#[allow(dead_code)]
#[must_use]
pub fn lorentz_identity<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero,
    I: TensorStructure + FromIterator<Slot<Lorentz>>,
{
    // IdentityL(1,2) (Lorentz) Kronecker delta δ^μ1_μ1
    let signature = Lorentz::new_dimed_rep_selfless(4);
    identity(indices, signature)
}

pub fn mink_four_vector<T, I>(index: AbstractIndex, p: &[T; 4]) -> DenseTensor<T, I>
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
        AbstractIndex::Normal(_) => [Lorentz::new_slot_selfless(4, index)].into_iter().collect(),
    };
    DenseTensor::from_data(p, structure).unwrap_or_else(|_| unreachable!())
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
            [Lorentz::new_slot_selfless(4, index)],
            State::get_symbol("p"),
            None,
        )),
    )
    .unwrap_or_else(|_| unreachable!())
}

pub fn euclidean_four_vector<T, I>(index: AbstractIndex, p: &[T; 4]) -> DenseTensor<T, I>
where
    T: Clone,
    I: TensorStructure + FromIterator<Slot<Euclidean>>,
{
    DenseTensor::from_data(
        p,
        [Euclidean::new_slot_selfless(4, index)]
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
            [Euclidean::new_slot_selfless(4, index)],
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
        [Lorentz::new_slot_selfless(4, index)],
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
        [Euclidean::new_slot_selfless(4, index)],
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
    let signature = Euclidean::new_dimed_rep_selfless(4);
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
        AbstractIndex::Normal(n) => Lorentz::new_slot_selfless(4, n).into(),
    };
    let structure = [
        Slot::<PhysReps>::from(Euclidean::new_slot_selfless(4, indices[0])),
        Euclidean::new_slot_selfless(4, indices[1]).into(),
        mu, // Lorentz::new_slot_selfless(4, minkindex).into(),
    ]
    .into_iter()
    .map(Slot::from)
    .collect();

    gamma_data(structure)
}
#[cfg(feature = "shadowing")]
pub fn gammasym<T>(
    minkindex: AbstractIndex,
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: One + Zero + Copy + std::ops::Neg<Output = T>,
{
    use crate::NamedStructure;
    let mu = match minkindex {
        AbstractIndex::Dualize(d) => Lorentz::selfless_dual().new_slot(4, d).into(),
        AbstractIndex::Normal(n) => Lorentz::new_slot_selfless(4, n).into(),
    };
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            Slot::<PhysReps>::from(Euclidean::new_slot_selfless(4, indices[0])),
            Euclidean::new_slot_selfless(4, indices[1]).into(),
            mu,
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

pub fn gamma5<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: One + Zero + Copy,
    I: TensorStructure + FromIterator<Slot<Euclidean>>,
{
    let structure = indices
        .into_iter()
        .map(|i| Euclidean::new_slot_selfless(4, i))
        .collect();

    gamma5_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn gamma5sym<T>(
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: One + Zero + Copy,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        indices
            .into_iter()
            .map(|i| Euclidean::new_slot_selfless(4, i)),
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

pub fn proj_m<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: Zero + One + NumCast + Clone,
    I: TensorStructure + FromIterator<Slot<Euclidean>>,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let structure = indices
        .into_iter()
        .map(|i| Euclidean::new_slot_selfless(4, i))
        .collect();

    proj_m_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn proj_msym<T>(
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: Zero + One + NumCast + Clone,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        indices
            .into_iter()
            .map(|i| Euclidean::new_slot_selfless(4, i)),
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

pub fn proj_p<T, I>(indices: [AbstractIndex; 2]) -> SparseTensor<Complex<T>, I>
where
    T: NumCast + Zero + Clone,
    I: TensorStructure + FromIterator<Slot<Bispinor>>,
{
    // ProjP(1,2) Right chirality projector (( 1+γ5)/ 2 )_s1_s2
    let structure = indices
        .into_iter()
        .map(|i| Bispinor::new_slot_selfless(4, i))
        .collect();

    proj_p_data(structure)
}

#[cfg(feature = "shadowing")]
pub fn proj_psym<T>(
    indices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, HistoryStructure<Symbol, ()>>
where
    T: Zero + Clone + NumCast,
{
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        indices
            .into_iter()
            .map(|i| Bispinor::new_slot_selfless(4, i)),
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
    indices: [AbstractIndex; 2],
    minkdices: [AbstractIndex; 2],
) -> SparseTensor<Complex<T>, I>
where
    T: Copy + Zero + One + Neg<Output = T>,
    I: TensorStructure + FromIterator<Slot<PhysReps>>,
{
    let structure = [
        Slot::<PhysReps>::from(Bispinor::new_slot_selfless(4, indices[0])),
        Bispinor::new_slot_selfless(4, indices[1]).into(),
        Lorentz::new_slot_selfless(4, minkdices[0]).into(),
        Lorentz::new_slot_selfless(4, minkdices[1]).into(),
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
            Slot::<PhysReps>::from(Bispinor::new_slot_selfless(4, indices[0])),
            Bispinor::new_slot_selfless(4, indices[1]).into(),
            Lorentz::new_slot_selfless(4, minkdices[0]).into(),
            Lorentz::new_slot_selfless(4, minkdices[1]).into(),
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
        reps[0].metric_data(structure)
    } else {
        panic!("Metric tensor must have equal indices")
    }
}
