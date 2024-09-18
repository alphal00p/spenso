use std::ops::Neg;

use num::{NumCast, One, Zero};

use crate::{
    complex::Complex,
    data::{DenseTensor, SetTensorData, SparseTensor},
    structure::{
        AbstractIndex, BaseRepName, Bispinor, Dimension, Dual, Euclidean, Lorentz, PhysReps,
        RepName, Representation, Slot, TensorStructure,
    },
};

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, Symbol},
    id::Pattern,
    state::State,
};

#[cfg(feature = "shadowing")]
use crate::structure::{
    HistoryStructure, IntoArgs, IntoSymbol, NamedStructure, Shadowable, ABSTRACTIND, BISPINOR,
    COLORADJ, COLORANTIFUND, COLORANTISEXT, COLORFUND, COLORSEXT, EUCLIDEAN, SPINFUND,
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

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_spin(atom: Atom, wrapped: bool, down: bool) -> Atom {
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
                    &if down {
                        Dual::<Lorentz>::rep_string(wrapped_to_four("mu_", wrapped))
                    } else {
                        Lorentz::rep_string(wrapped_to_four("mu_", wrapped))
                    },
                    &if down {
                        Lorentz::rep_string(wrapped_to_four("nu_", wrapped))
                    } else {
                        Dual::<Lorentz>::rep_string(wrapped_to_four("nu_", wrapped))
                    },
                ],
            ),
        ),
        (
            "Gamma(mu_,i_,j_)",
            named_tensor(
                "γ".into(),
                &[
                    &if down {
                        Dual::<Lorentz>::rep_string(wrapped_to_four("mu_", wrapped))
                    } else {
                        Lorentz::rep_string(wrapped_to_four("mu_", wrapped))
                    },
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
                    &if down {
                        Dual::<Lorentz>::rep_string(wrapped_to_four("mu_", wrapped))
                    } else {
                        Lorentz::rep_string(wrapped_to_four("mu_", wrapped))
                    },
                    &if down {
                        Dual::<Lorentz>::rep_string(wrapped_to_four("nu_", wrapped))
                    } else {
                        Lorentz::rep_string(wrapped_to_four("nu_", wrapped))
                    },
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
                    &if down {
                        Dual::<Lorentz>::rep_string(wrapped_to_four("mu_", wrapped))
                    } else {
                        Lorentz::rep_string(wrapped_to_four("mu_", wrapped))
                    },
                    &if down {
                        Dual::<Lorentz>::rep_string(wrapped_to_four("nu_", wrapped))
                    } else {
                        Lorentz::rep_string(wrapped_to_four("nu_", wrapped))
                    },
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
fn bis(rep_args: ReplacementArgs) -> String {
    rep_string(BISPINOR, rep_args)
}

#[cfg(feature = "shadowing")]
trait ReprRepl {
    fn rep_string(rep_args: ReplacementArgs) -> String;
}

#[cfg(feature = "shadowing")]
impl<R: BaseRepName> ReprRepl for R {
    fn rep_string(rep_args: ReplacementArgs) -> String {
        rep_string(R::NAME, rep_args)
    }
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
        let replacement = Pattern::parse(replacement.as_ref()).unwrap().into();
        atom = pattern.replace_all(atom.as_view(), &replacement, None, None);
    }

    atom
}

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_color_wrapped(atom: Atom) -> Atom {
    preprocess_ufo_color(atom, true)
}

#[cfg(feature = "shadowing")]
pub fn preprocess_ufo_spin_wrapped(atom: Atom, down: bool) -> Atom {
    preprocess_ufo_spin(atom, true, down)
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
        AbstractIndex::Normal(_) => [Lorentz::new_slot_selfless(4, index)].into_iter().collect(),
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
            [Lorentz::new_slot_selfless(4, index)],
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
        [Euclidean::new_slot_selfless(4, index)]
            .into_iter()
            .collect(),
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
        mu, // Lorentz::new_slot_selfless(4, minkindex).into(),
        Slot::<PhysReps>::from(Euclidean::new_slot_selfless(4, indices[0])),
        Euclidean::new_slot_selfless(4, indices[1]).into(),
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
        AbstractIndex::Normal(n) => Lorentz::new_slot_selfless(4, n).into(),
    };
    let structure = HistoryStructure::from(NamedStructure::from_iter(
        [
            mu,
            Slot::<PhysReps>::from(Euclidean::new_slot_selfless(4, indices[0])),
            Euclidean::new_slot_selfless(4, indices[1]).into(),
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

    gamma.set(&[0, 2, 0], c1.clone()).unwrap();
    gamma.set(&[0, 3, 1], c1.clone()).unwrap();
    gamma.set(&[0, 0, 2], c1.clone()).unwrap();
    gamma.set(&[0, 1, 3], c1.clone()).unwrap();

    gamma.set(&[1, 3, 0], c1.clone()).unwrap();
    gamma.set(&[1, 2, 1], c1.clone()).unwrap();
    gamma.set(&[1, 1, 2], cn1.clone()).unwrap();
    gamma.set(&[1, 0, 3], cn1.clone()).unwrap();

    gamma.set(&[2, 3, 0], cni.clone()).unwrap();
    gamma.set(&[2, 2, 1], ci.clone()).unwrap();
    gamma.set(&[2, 1, 2], ci.clone()).unwrap();
    gamma.set(&[2, 0, 3], cni.clone()).unwrap();

    gamma.set(&[3, 2, 0], c1.clone()).unwrap();
    gamma.set(&[3, 3, 1], cn1.clone()).unwrap();
    gamma.set(&[3, 0, 2], cn1.clone()).unwrap();
    gamma.set(&[3, 1, 3], c1.clone()).unwrap();

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
    let structure = indices
        .into_iter()
        .map(|i| Euclidean::new_slot_selfless(4, i))
        .collect();

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
        indices
            .into_iter()
            .map(|i| Euclidean::new_slot_selfless(4, i)),
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
    let structure = indices
        .into_iter()
        .map(|i| Euclidean::new_slot_selfless(4, i))
        .collect();

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
        indices
            .into_iter()
            .map(|i| Euclidean::new_slot_selfless(4, i)),
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
    let structure = indices
        .into_iter()
        .map(|i| Bispinor::new_slot_selfless(4, i))
        .collect();

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
        indices
            .into_iter()
            .map(|i| Bispinor::new_slot_selfless(4, i)),
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

#[cfg(test)]
mod test {
    use bitvec::{access::BitSafeUsize, vec};
    use symbolica::{atom::Atom, state::State};

    use crate::{
        complex::{RealOrComplex, RealOrComplexTensor},
        contraction::Contract,
        data::{DenseTensor, HasTensorData},
        iterators::IteratableTensor,
        network::TensorNetwork,
        parametric::{ParamOrConcrete, SerializableAtom},
        structure::{
            BaseRepName, Bispinor, HasStructure, SerializableSymbol, Shadowable, TensorStructure,
            VecStructure,
        },
    };

    #[test]
    fn clifford() {
        let expr = Atom::parse(
            "γ(aind(bis(4,4),bis(4,3),lord(4,1)))*γ(aind(bis(4,3),bis(4,4),lord(4,2)))+γ(aind(bis(4,4),bis(4,3),lord(4,2)))*γ(aind(bis(4,3),bis(4,4),lord(4,1)))",
        )
        .unwrap();
        // +γ(aind(bis(4,4),bis(4,3),lord(4,2)))*γ(aind(bis(4,3),bis(4,4),lord(4,1))))
        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();
        println!("{}", net.dot());
        net.contract();
        println!("{}", net.dot());

        println!("{}", net.result_tensor().unwrap());
    }

    #[test]
    fn clifford2() {
        let expr = Atom::parse(
            "γ(aind(lord(4,1),bis(4,4),bis(4,3)))*γ(aind(lord(4,2),bis(4,3),bis(4,1)))+γ(aind(lord(4,2),bis(4,4),bis(4,3)))*γ(aind(lord(4,1),bis(4,3),bis(4,1)))",
        )
        .unwrap();
        // +γ(aind(bis(4,4),bis(4,3),lord(4,2)))*γ(aind(bis(4,3),bis(4,4),lord(4,1))))
        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();
        println!("{}", net.dot());
        net.contract();
        println!("{}", net.dot());

        println!("{}", net.result_tensor().unwrap());
    }

    #[test]
    fn gamma_algebra() {
        let _ = env_logger::builder().is_test(true).try_init();
        let expr = Atom::parse("γ(aind(lord(4,1),bis(4,4),bis(4,3)))*Q(1,aind(loru(4,1)))*γ(aind(lord(4,2),bis(4,3),bis(4,4)))*Q(2,aind(loru(4,2)))").unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();

        net.contract();

        println!("{}", net.result_tensor().unwrap());

        let expr = Atom::parse("γ(aind(bis(4,1),bis(4,4),bis(4,3)))*Q(1,aind(bis(4,1)))*γ(aind(bis(4,2),bis(4,3),bis(4,4)))*Q(2,aind(bis(4,2)))").unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();

        net.contract();

        println!("{}", net.result_tensor().unwrap());
    }

    #[test]
    fn gamma_algebra2() {
        let expr = Atom::parse(
            "γ(aind(loru(4,1),bis(4,4),bis(4,3)))*γ(aind(lord(4,1),bis(4,1),bis(4,2)))",
        )
        .unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();

        net.contract();

        println!("{}", net.result_tensor().unwrap());
    }

    #[test]
    fn data() {
        let _ = env_logger::builder().is_test(true).try_init();
        let expr = Atom::parse(
            "γ(aind(loru(4,1),bis(4,4),bis(4,3)))*γ(aind(loru(4,2),bis(4,3),bis(4,4)))",
        )
        .unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();
        println!("{}", net.dot());
        net.contract();

        let mut other =
            TensorNetwork::try_from(Atom::parse("p(aind(lord (4,1)))").unwrap().as_view()).unwrap();
        other.contract();

        net.push(other.result_tensor().unwrap());

        net.contract();

        println!("{}", net.result_tensor().unwrap());
    }

    #[test]
    fn data2() {
        let _ = env_logger::builder().is_test(true).try_init();
        let expr = Atom::parse(
            "γ(aind(loru(4,1),bis(4,4),bis(4,3)))*γ(aind(lord(4,2),bis(4,3),bis(4,4)))",
        )
        .unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();
        println!("{}", net.dot());
        net.contract();

        println!("{}", net.result_tensor().unwrap());
    }

    #[test]
    fn data3() {
        let _ = env_logger::builder().is_test(true).try_init();
        let g1 = Atom::parse("γ(aind(loru(4,1),bis(4,3),bis(4,4)))").unwrap();
        let g2 = Atom::parse("γ(aind(loru(4,2),bis(4,3),bis(4,4)))").unwrap();
        let p = Atom::parse("p(aind(lord (4,1)))").unwrap();
        let u = Atom::parse("u(aind(bis (4,3)))").unwrap();
        let v = Atom::parse("v(aind(bis (4,4)))").unwrap();

        let q = Atom::parse("q(aind(lord (4,2)))").unwrap();

        let g1_tensor: ParamOrConcrete<
            RealOrComplexTensor<
                f64,
                crate::structure::NamedStructure<
                    crate::structure::SerializableSymbol,
                    Vec<crate::parametric::SerializableAtom>,
                >,
            >,
            crate::structure::NamedStructure<
                crate::structure::SerializableSymbol,
                Vec<crate::parametric::SerializableAtom>,
            >,
        > = TensorNetwork::try_from(g1.as_view())
            .unwrap()
            .result_tensor()
            .unwrap();
        let g2_tensor = TensorNetwork::try_from(g2.as_view())
            .unwrap()
            .result_tensor()
            .unwrap();
        let p_tensor = TensorNetwork::try_from(p.as_view())
            .unwrap()
            .result_tensor()
            .unwrap();

        let q_tensor = TensorNetwork::try_from(q.as_view())
            .unwrap()
            .result_tensor()
            .unwrap();

        let u_tensor = TensorNetwork::try_from(u.as_view())
            .unwrap()
            .result_tensor()
            .unwrap();
        let v_tensor = TensorNetwork::try_from(v.as_view())
            .unwrap()
            .result_tensor()
            .unwrap();

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
    fn data4() {
        let _ = env_logger::builder().is_test(true).try_init();
        let expr = Atom::parse("A(aind(loru(4,1),bis(4,4),bis(4,3)))*B(aind(lord(4,1)))").unwrap();

        let mut net = TensorNetwork::try_from(expr.as_view()).unwrap();
        println!("{}", net.dot());
        net.contract();
        let res = net.result_tensor().unwrap();

        println!("{}", res);

        println!("{:?}", res.structure().external_structure());
    }

    #[test]
    fn data5() {
        let spinstructure = VecStructure::new(vec![
            Bispinor::new_slot_selfless(4, 3).into(),
            Bispinor::new_slot_selfless(4, 4).into(),
        ]);

        let p0 = Atom::parse("p(0)").unwrap();
        let p1 = Atom::parse("p(1)").unwrap();
        let p2 = Atom::parse("p(2)").unwrap();
        let p3 = Atom::parse("p(3)").unwrap();
        let i = Atom::parse("i").unwrap();
        let p12 = &p1 + &i * &p2;
        let abar = &p1 - &i * &p2;
        let z = Atom::new_num(0);

        let q0 = Atom::parse("q(0)").unwrap();
        let q1 = Atom::parse("q(1)").unwrap();
        let q2 = Atom::parse("q(2)").unwrap();
        let q3 = Atom::parse("q(3)").unwrap();

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
            Bispinor::new_slot_selfless(4, 4).into(),
            Bispinor::new_slot_selfless(4, 3).into(),
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
    fn data6() {
        let spinstructure = VecStructure::new(vec![
            Bispinor::new_slot_selfless(4, 3).into(),
            Bispinor::new_slot_selfless(4, 4).into(),
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
            Bispinor::new_slot_selfless(4, 4).into(),
            Bispinor::new_slot_selfless(4, 3).into(),
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
