use symbolica::atom::Symbol;
use symbolica::{symbol, tag};

use super::*;

use library::Library;

use crate::network::library::DummyLibrary;
use crate::structure::abstract_index::{AindSymbols, AIND_SYMBOLS};
// use crate::shadowing::Concretize;
use crate::structure::slot::{ParseableAind, Slot, SlotError};
use crate::structure::{
    NamedStructure, OrderedStructure, PermutedStructure, StructureError, TensorShell,
};
use crate::tensors::symbolic::SymbolicTensor;

use std::fmt::Display;

use store::TensorScalarStore;
// use log::trace;

use symbolica::atom::{representation::FunView, AddView, Atom, AtomView, MulView, PowView};

use crate::structure::{HasStructure, TensorStructure};

use crate::{shadowing::Concretize, structure::representation::LibraryRep, structure::HasName};

pub struct SpensoTags {
    pub tag: String,
    pub bracket: Symbol,
    pub pure_scalar: Symbol,
}

pub static SPENSO_TAG: std::sync::LazyLock<SpensoTags> = std::sync::LazyLock::new(|| SpensoTags {
    tag: tag!("broadcast"),
    bracket: symbol!("bracket"),
    pure_scalar: symbol!("pure_scalar"),
});

pub type ShadowedStructure<Aind> = NamedStructure<Symbol, Vec<Atom>, LibraryRep, Aind>;

impl<'a, Aind: ParseableAind + AbsInd> TryFrom<AtomView<'a>>
    for PermutedStructure<ShadowedStructure<Aind>>
{
    type Error = StructureError;
    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Fun(f) = value {
            f.try_into()
        } else {
            Err(StructureError::ParsingError(value.to_plain_string()))
        }
    }
}

impl<Aind: ParseableAind + AbsInd> TryFrom<Atom> for PermutedStructure<ShadowedStructure<Aind>> {
    type Error = StructureError;
    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a, Aind: ParseableAind + AbsInd> TryFrom<&'a Atom>
    for PermutedStructure<ShadowedStructure<Aind>>
{
    type Error = StructureError;
    fn try_from(value: &'a Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a, Aind: ParseableAind + AbsInd> TryFrom<FunView<'a>>
    for PermutedStructure<ShadowedStructure<Aind>>
{
    type Error = StructureError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        match value.get_symbol() {
            s if s == AIND_SYMBOLS.aind => {
                let mut structure: Vec<Slot<LibraryRep, Aind>> = vec![];

                for arg in value.iter() {
                    structure.push(arg.try_into()?);
                }

                let o = OrderedStructure::new(structure);
                let p = o.map_structure(|s| s.into());

                Ok(p)
            }
            name => {
                let mut args = vec![];
                let mut slots = vec![];
                let mut is_structure: Option<StructureError> =
                    Some(SlotError::EmptyStructure.into());

                for arg in value.iter() {
                    let slot: Result<Slot<LibraryRep, Aind>, _> = arg.try_into();

                    match slot {
                        Ok(slot) => {
                            is_structure = None;
                            slots.push(slot);
                        }
                        Err(e) => {
                            if let AtomView::Fun(f) = arg {
                                if f.get_symbol() == AIND_SYMBOLS.aind {
                                    let internal_s = Self::try_from(f);

                                    if let Ok(s) = internal_s {
                                        let p = s.index_permutation;
                                        let mut v = s.structure.structure.structure;
                                        p.apply_slice_in_place_inv(&mut v); //undo sorting
                                        let p = s.rep_permutation;
                                        p.apply_slice_in_place_inv(&mut v); //undo sorting
                                        slots.extend(v);
                                        is_structure = None;
                                        continue;
                                    }
                                }
                            }
                            is_structure = Some(e.into());
                            args.push(arg.to_owned());
                        }
                    }
                }

                if let Some(e) = is_structure {
                    Err(e)
                } else {
                    let mut structure: PermutedStructure<ShadowedStructure<Aind>> =
                        OrderedStructure::new(slots).map_structure(Into::into);
                    structure.structure.set_name(name);
                    if !args.is_empty() {
                        structure.structure.additional_args = Some(args);
                    }
                    Ok(structure)
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParseSettings {
    pub precontract_scalars: bool,
    pub take_first_term_from_sum: bool,
}

impl Default for ParseSettings {
    fn default() -> Self {
        Self {
            precontract_scalars: true,
            take_first_term_from_sum: false,
        }
    }
}

impl<
        'a,
        Sc,
        T: HasStructure + TensorStructure,
        K: Clone + Display + Debug,
        // FK: Clone + Display + Debug,
        Str: TensorScalarStore<Tensor = T, Scalar = Sc> + Clone,
        Aind: AbsInd,
    > Network<Str, K, Symbol, Aind>
where
    Sc: for<'r> TryFrom<AtomView<'r>> + Clone,
    TensorNetworkError<K, Symbol>: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
{
    #[allow(clippy::result_large_err)]
    pub fn try_from_view<S, Lib: Library<S, Key = K>>(
        value: AtomView<'a>,
        library: &Lib,
        settings: &ParseSettings,
    ) -> Result<Self, TensorNetworkError<K, Symbol>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        S::Slot: IsAbstractSlot<Aind = Aind>,
        T::Slot: IsAbstractSlot<Aind = Aind>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        match value {
            AtomView::Mul(m) => Self::try_from_mul(m, library, settings),
            AtomView::Fun(f) => Self::try_from_fun(f, library, settings),
            AtomView::Add(a) => Self::try_from_add(a, library, settings),
            AtomView::Pow(p) => Self::try_from_pow(p, library, settings),
            a => Ok(Network::from_scalar(a.try_into()?)),
        }
    }

    #[allow(clippy::result_large_err)]
    fn try_from_mul<S, Lib: Library<S, Key = K>>(
        value: MulView<'a>,
        library: &Lib,
        settings: &ParseSettings,
    ) -> Result<Self, TensorNetworkError<K, Symbol>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        S::Slot: IsAbstractSlot<Aind = Aind>,
        T::Slot: IsAbstractSlot<Aind = Aind>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        let mut iter = value.iter();

        let first = Self::try_from_view(iter.next().unwrap(), library, settings)?;

        let rest: Result<Vec<_>, _> = iter
            .map(|a| Self::try_from_view(a, library, settings))
            .collect();

        let res = first.n_mul(rest?);
        if settings.precontract_scalars {
            if let NetworkState::PureScalar = res.state {
                return Ok(Self::from_scalar(value.as_view().try_into()?));
            }
        }

        Ok(res)
    }

    #[allow(clippy::result_large_err)]
    fn try_from_fun<S, Lib: Library<S, Key = K>>(
        value: FunView<'a>,
        library: &Lib,
        settings: &ParseSettings,
    ) -> Result<Self, TensorNetworkError<K, Symbol>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        S::Slot: IsAbstractSlot<Aind = Aind>,
        T::Slot: IsAbstractSlot<Aind = Aind>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        let symbol = value.get_symbol();

        if symbol == SPENSO_TAG.bracket {
            let mut n_muls = value
                .iter()
                .map(|a| Self::try_from_view(a, library, settings))
                .collect::<Result<Vec<_>, _>>()?;

            Ok(n_muls.pop().unwrap().n_mul(n_muls))
        } else if symbol == SPENSO_TAG.pure_scalar {
            if value.get_nargs() != 1 {
                return Err(TensorNetworkError::TooManyArgsFunction(
                    value.as_view().to_plain_string(),
                ));
            }
            Ok(Self::from_scalar(value.iter().next().unwrap().try_into()?))
        } else if symbol.has_tag(&SPENSO_TAG.tag) {
            if value.get_nargs() != 1 {
                return Err(TensorNetworkError::TooManyArgsFunction(
                    value.as_view().to_plain_string(),
                ));
            }

            let inner = value.iter().next().unwrap();
            let inner_tensor = Self::try_from_view(inner, library, settings)?;

            Ok(inner_tensor.fun(symbol))
        } else {
            let s: Result<PermutedStructure<S>, _> = value.try_into();

            if let Ok(s) = s {
                // println!("Perm:{}", s.index_permutation);
                // let s = s;
                match library.key_for_structure(&s) {
                    Ok(key) => {
                        // println!("Adding lib");
                        // let t = library.get(&key).unwrap();
                        Ok(Self::library_tensor(
                            &s.structure,
                            PermutedStructure {
                                structure: key,
                                rep_permutation: s.rep_permutation,
                                index_permutation: s.index_permutation,
                            },
                        ))
                    }
                    Err(_) => Ok(Self::from_tensor(
                        s.structure
                            .to_shell()
                            .concretize(Some(s.index_permutation.inverse())),
                    )),
                }
            } else {
                Ok(Self::from_scalar(value.as_view().try_into()?))
            }
        }
    }

    #[allow(clippy::result_large_err)]
    fn try_from_pow<S, Lib: Library<S, Key = K>>(
        value: PowView<'a>,
        library: &Lib,
        settings: &ParseSettings,
    ) -> std::result::Result<Self, TensorNetworkError<K, Symbol>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        S::Slot: IsAbstractSlot<Aind = Aind>,
        T::Slot: IsAbstractSlot<Aind = Aind>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        let (base, exp) = value.get_base_exp();

        if let Ok(n) = i8::try_from(exp) {
            let base = Self::try_from_view(base, library, settings)?;

            if let NetworkState::Tensor = base.state {
                Err(TensorNetworkError::NonSelfDualTensorPower(
                    value.as_view().to_plain_string(),
                ))
            } else if n < 0 {
                // An even power of a self_dual tensor, or scalar is a scalar
                if n % 2 == 0 || base.state.is_scalar() {
                    Ok(base.pow(n))
                } else {
                    Err(TensorNetworkError::NegativeExponentNonScalar(format!(
                        "Atom:{},graph of base: {}, dangling indices: {:?}",
                        value.as_view().to_plain_string(),
                        base.dot(),
                        base.graph.dangling_indices()
                    )))
                }
            } else {
                Ok(base.pow(n))
            }
        } else {
            Ok(Self::from_scalar(value.as_view().try_into()?))
        }
    }

    #[allow(clippy::result_large_err)]
    fn try_from_add<S, Lib: Library<S, Key = K>>(
        value: AddView<'a>,
        library: &Lib,
        settings: &ParseSettings,
    ) -> Result<Self, TensorNetworkError<K, Symbol>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        S::Slot: IsAbstractSlot<Aind = Aind>,
        T::Slot: IsAbstractSlot<Aind = Aind>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        let mut iter = value.iter();

        let first_atom = iter.next().unwrap();

        let first = Self::try_from_view(first_atom, library, settings)?;
        if settings.take_first_term_from_sum {
            Ok(first)
        } else {
            let rest: Result<Vec<_>, _> = iter
                .map(|a| match Self::try_from_view(a, library, settings) {
                    Ok(n) => {
                        if n.state.is_compatible(&first.state) {
                            Ok(n)
                        } else {
                            Err(TensorNetworkError::IncompatibleSummand(format!(
                                "{} is {:?} vs {} is {:?}",
                                a, n.state, first_atom, first.state
                            )))
                        }
                    }
                    Err(e) => Err(e),
                })
                .collect();

            Ok(first.n_add(rest?))
        }
    }
}

pub type SymbolicNet<Aind> =
    Network<NetworkStore<SymbolicTensor<Aind>, Atom>, DummyKey, Symbol, Aind>;
pub trait SymbolicParse {
    fn parse_to_symbolic_net<Aind: AbsInd + ParseableAind>(
        &self,
        settings: &ParseSettings,
    ) -> Result<SymbolicNet<Aind>, TensorNetworkError<DummyKey, Symbol>>;
}

impl SymbolicParse for AtomView<'_> {
    fn parse_to_symbolic_net<Aind: AbsInd + ParseableAind>(
        &self,
        settings: &ParseSettings,
    ) -> Result<SymbolicNet<Aind>, TensorNetworkError<DummyKey, Symbol>> {
        let lib = DummyLibrary::<SymbolicTensor>::new();

        SymbolicNet::<Aind>::try_from_view(*self, &lib, settings)
    }
}

#[cfg(test)]
pub mod test {
    use core::panic;

    use crate::{
        network::library::panicing::ErroringLibrary,
        structure::{
            representation::{initialize, Euclidean, Lorentz, Minkowski, RepName},
            slot::IsAbstractSlot,
            ToSymbolic,
        },
        tensors::symbolic::SymbolicTensor,
    };

    use super::*;
    use insta::assert_snapshot;
    use library::{DummyKey, DummyLibrary};
    use linnet::half_edge::swap::Swap;
    use symbolica::{atom::AtomCore, parse, parse_lit, symbol};

    #[test]
    fn parse_scalar() {
        let expr = parse!("1");

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        );

        let mut net = net.unwrap();

        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();

        if let ExecutionResult::Val(TensorOrScalarOrKey::Scalar(a)) = net.result().unwrap() {
            println!("YaY:{a}")
        } else {
            panic!("Not scalar")
        }
    }

    #[test]
    fn parse_pow() {
        let expr = parse!("ee^3*d(mink(4,1))^-2");

        let lib: DummyLibrary<SymbolicTensor> = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        assert_snapshot!(
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            ),@r#"
        digraph {
          node	 [shape=circle,height=0.1,label=""];
          overlap = "scale";
          layout = "neato";

          0	 [label = "S:ee^3*d(mink(4,1))^-2"];
          ext0	 [style=invis];
          0:0:s	-> ext0	 [id=0 color="red"];
        }
        "#);
    }

    #[test]
    fn parse_ratio() {
        let expr = parse_lit!(
            (P(1, mink(4, 1)) * P(2, mink(4, 1)))
                / f(spenso::mul(P(3, mink(4, 1)) * P(4, mink(4, 1))))
        );
        println!("{expr}");
        let lib: DummyLibrary<SymbolicTensor> = DummyLibrary::<_>::new();
        // let fnlib = ErroringLibrary::<Symbol>::new();
        let net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
    }
    #[test]
    fn parse_div() {
        let expr = parse!("c*a/spenso::mul(d(mink(4,1))* b(mink(4,1)))(spenso::mul(d(mink(4,1))* b(mink(4,1)))^-3)(spenso::mul(d(mink(4,1))* b(mink(4,1)))^-2)");
        println!("{expr}");
        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        assert_snapshot!(
            net.dot_display_impl(
                |a| a.to_canonical_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            ),@r#"
        digraph {
          node	 [shape=circle,height=0.1,label=""];
          overlap = "scale";
          layout = "neato";

          0	 [label = "S:a*c*mul(b(mink(4,1))*d(mink(4,1)))^-6"];
          ext0	 [style=invis];
          0:0:s	-> ext0	 [id=0 color="red"];
        }
        "#
        );
    }

    #[test]
    fn parse_scalar_tensor() {
        let expr = parse!("c*a*b(mink(4,1))");

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<Steps<1>, ContractScalars, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not scalar")
        }
    }

    #[test]
    fn parse_scalar_tensors_step_by() {
        let expr = parse!("c*a*b(mink(4,1))*d(mink(4,2))*d(mink(4,1))*d(mink(4,2))");

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        let mut netc = net.clone();
        println!(
            "//Init:\n{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<Steps<1>, ContractScalars, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "//Contract Scalars:\n{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<Steps<1>, SingleSmallestDegree<false>, _, _, _>(&lib, &fnlib)
            .unwrap();

        // println!("{:#?}", net.graph.graph);
        println!(
            "//Single Smallest Degree 1:\n{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<Steps<1>, SingleSmallestDegree<false>, _, _, _>(&lib, &fnlib)
            .unwrap();

        // println!("{:#?}", net.graph.graph);
        println!(
            "//Single Smallest Degree 2:\n{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<Steps<1>, ContractScalars, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "//Contract Scalars again:\n{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        netc.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            netc.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor {
                tensor: tensor2, ..
            }) = netc.result().unwrap()
            {
                assert_eq!(tensor2.expression, tensor.expression);
            } else {
                panic!("Not scalar")
            }
        } else {
            panic!("Not scalar")
        }
    }

    #[test]
    fn parse_scalar_expr() {
        let expr = parse!("(y+x(mink(4,1))*y(mink(4,1))) *(1+1+2*x*(3*sin(r))/t)");

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        if let ExecutionResult::Val(TensorOrScalarOrKey::Scalar(a)) = net.result().unwrap() {
            // println!("YaY:{a}");
            assert_eq!(&expr, a);
        } else {
            panic!("Not scalar")
        }
    }

    #[test]
    fn parse_tensor_expr() {
        let tensor1 = ShadowedStructure::<AbstractIndex>::from_iter(
            [
                Minkowski {}.new_slot(4, 1).to_lib(),
                Euclidean {}.new_slot(4, 2).to_lib(),
                Minkowski {}.new_slot(symbol!("d"), symbol!("mu")).to_lib(),
            ],
            symbol!("T"),
            None,
        )
        .structure
        .to_symbolic(None)
        .unwrap();

        let tensor2 = ShadowedStructure::<AbstractIndex>::from_iter(
            [
                Minkowski {}.new_slot(4, 1).to_lib(),
                Lorentz {}.new_slot(7, 1).to_lib(),
                Lorentz {}.new_slot(3, 2).to_lib(),
            ],
            symbol!("TT"),
            None,
        )
        .structure
        .to_symbolic(None)
        .unwrap();

        let tensor3 = ShadowedStructure::<AbstractIndex>::from_iter(
            [
                Lorentz {}.dual().new_slot(7, 1).to_lib(),
                Euclidean {}.new_slot(4, 2).to_lib(),
                Minkowski {}.new_slot(symbol!("d"), symbol!("mu")).to_lib(),
            ],
            symbol!("TTT"),
            None,
        )
        .structure
        .to_symbolic(None)
        .unwrap();

        let tensor4 = ShadowedStructure::<AbstractIndex>::from_iter(
            [Lorentz {}.new_slot(3, 2).to_lib()],
            symbol!("L"),
            None,
        )
        .structure
        .to_symbolic(None)
        .unwrap();

        let tensor5 = ShadowedStructure::<AbstractIndex>::from_iter(
            [Lorentz {}.dual().new_slot(3, 2).to_lib()],
            symbol!("P"),
            None,
        )
        .structure
        .to_symbolic(None)
        .unwrap();

        let expr = (parse!("a*sin(x/2)") * tensor1 * tensor2 * tensor3 + tensor4) * tensor5;

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            // println!("YaY:{a}");
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not tensor")
        }
    }

    // -G^2*(-g(mink(4,5),mink(4,6))*Q(2,mink(4,7))+g(mink(4,5),mink(4,6))*Q(3,mink(4,7))+g(mink(4,5),mink(4,7))*Q(2,mink(4,6))+g(mink(4,5),mink(4,7))*Q(4,mink(4,6))-g(mink(4,6),mink(4,7))*Q(3,mink(4,5))-g(mink(4,6),mink(4,7))*Q(4,mink(4,5)))*id(mink(4,2),mink(4,5))*id(mink(4,3),mink(4,6))*id(euc(4,0),euc(4,5))*id(euc(4,1),euc(4,4))*g(mink(4,4),mink(4,7))*vbar(1,euc(4,1))*u(0,euc(4,0))*ebar(2,mink(4,2))*ebar(3,mink(4,3))*gamma(mink(4,4),euc(4,5),euc(4,4))

    #[test]
    fn parse_big_tensors() {
        initialize();
        let expr = parse!("-G^2*(-g(mink(4,5),mink(4,6))*Q(2,mink(4,7))+g(mink(4,5),mink(4,6))*Q(3,mink(4,7))+g(mink(4,5),mink(4,7))*Q(2,mink(4,6))+g(mink(4,5),mink(4,7))*Q(4,mink(4,6))-g(mink(4,6),mink(4,7))*Q(3,mink(4,5))-g(mink(4,6),mink(4,7))*Q(4,mink(4,5)))*id(mink(4,2),mink(4,5))*id(mink(4,3),mink(4,6))*id(euc(4,0),euc(4,5))*id(euc(4,1),euc(4,4))*g(mink(4,4),mink(4,7))*vbar(1,euc(4,1))*u(0,euc(4,0))*ebar(2,mink(4,2))*ebar(3,mink(4,3))*gamma(euc(4,5),euc(4,4),mink(4,4))");
        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        println!("Hi");
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            // println!("YaY:{}", (&expr - &tensor.expression).expand());
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not tensor")
        }
    }

    #[test]
    fn equal_duals() {
        initialize();
        let loop_tn_expr = parse_lit!(
            ((_gammaloop::Q(5, spenso::cind(0)))
                ^ 2 + (_gammaloop::Q(5, spenso::cind(1)))
                ^ 2 * -1 + (_gammaloop::Q(5, spenso::cind(2)))
                ^ 2 * -1 + (_gammaloop::Q(5, spenso::cind(3)))
                ^ 2 * -1)
                ^ (-1)
                    * ((_gammaloop::Q(6, spenso::cind(0)))
                        ^ 2 + (_gammaloop::Q(6, spenso::cind(1)))
                        ^ 2 * -1 + (_gammaloop::Q(6, spenso::cind(2)))
                        ^ 2 * -1 + (_gammaloop::Q(6, spenso::cind(3)))
                        ^ 2 * -1)
                ^ (-1)
                    * ((_gammaloop::Q(7, spenso::cind(0)))
                        ^ 2 + (_gammaloop::Q(7, spenso::cind(1)))
                        ^ 2 * -1 + (_gammaloop::Q(7, spenso::cind(2)))
                        ^ 2 * -1 + (_gammaloop::Q(7, spenso::cind(3)))
                        ^ 2 * -1)
                ^ (-1)
                    * ((_gammaloop::Q(8, spenso::cind(0)))
                        ^ 2 + (_gammaloop::Q(8, spenso::cind(1)))
                        ^ 2 * -1 + (_gammaloop::Q(8, spenso::cind(2)))
                        ^ 2 * -1 + (_gammaloop::Q(8, spenso::cind(3)))
                        ^ 2 * -1)
                ^ (-1)
                    * ((_gammaloop::Q(9, spenso::cind(0)))
                        ^ 2 + (_gammaloop::Q(9, spenso::cind(1)))
                        ^ 2 * -1 + (_gammaloop::Q(9, spenso::cind(2)))
                        ^ 2 * -1 + (_gammaloop::Q(9, spenso::cind(3)))
                        ^ 2 * -1)
                ^ (-1) * -1𝑖 / 3 * UFO::GC_11
                ^ 2 * UFO::GC_1
                ^ 3 * UFO::Gamma(
                    spenso::mink(4, _gammaloop::hedge_13),
                    spenso::bis(4, _gammaloop::hedge_11),
                    2
                ) * _gammaloop::Q(5, spenso::mink(4, _gammaloop::edge_5_1))
                    * _gammaloop::Q(6, spenso::mink(4, _gammaloop::edge_6_1))
                    * _gammaloop::Q(7, spenso::mink(4, _gammaloop::edge_7_1))
                    * _gammaloop::Q(8, spenso::mink(4, _gammaloop::edge_8_1))
                    * _gammaloop::u(1)
                    * _gammaloop::vbar(2, spenso::bis(4, _gammaloop::hedge_2))
                    * _gammaloop::ebar(0, spenso::mink(4, _gammaloop::hedge_0))
                    * _gammaloop::ebar(3, spenso::mink(4, _gammaloop::hedge_3))
                    * _gammaloop::ebar(4, spenso::mink(4, _gammaloop::hedge_4))
                    * spenso::g(
                        spenso::cof(3, _gammaloop::hedge_1),
                        spenso::dind(spenso::cof(3, _gammaloop::hedge_2))
                    )
                    * spenso::g(
                        spenso::mink(4, _gammaloop::hedge_13),
                        spenso::mink(4, _gammaloop::hedge_14)
                    )
                    * spenso::gamma(
                        spenso::bis(4, _gammaloop::hedge_10),
                        spenso::bis(4, _gammaloop::hedge_9),
                        spenso::mink(4, _gammaloop::edge_8_1)
                    )
                    * spenso::gamma(
                        spenso::bis(4, _gammaloop::hedge_12),
                        spenso::bis(4, _gammaloop::hedge_11),
                        spenso::mink(4, _gammaloop::edge_7_1)
                    )
                    * spenso::gamma(
                        spenso::bis(4, _gammaloop::hedge_2),
                        spenso::bis(4, _gammaloop::hedge_10),
                        spenso::mink(4, _gammaloop::hedge_14)
                    )
                    * spenso::gamma(
                        spenso::bis(4, _gammaloop::hedge_5),
                        spenso::bis(4, _gammaloop::hedge_12),
                        spenso::mink(4, _gammaloop::hedge_3)
                    )
                    * spenso::gamma(
                        spenso::bis(4, _gammaloop::hedge_6),
                        spenso::bis(4, _gammaloop::hedge_5),
                        spenso::mink(4, _gammaloop::edge_5_1)
                    )
                    * spenso::gamma(
                        spenso::bis(4, _gammaloop::hedge_7),
                        spenso::bis(4, _gammaloop::hedge_6),
                        spenso::mink(4, _gammaloop::hedge_4)
                    )
                    * spenso::gamma(
                        spenso::bis(4, _gammaloop::hedge_8),
                        spenso::bis(4, _gammaloop::hedge_7),
                        spenso::mink(4, _gammaloop::edge_6_1)
                    )
                    * spenso::gamma(
                        spenso::bis(4, _gammaloop::hedge_9),
                        spenso::bis(4, _gammaloop::hedge_8),
                        spenso::mink(4, _gammaloop::hedge_0)
                    )
                    * spenso::t(
                        spenso::coad(8, _gammaloop::hedge_13),
                        spenso::cof(2),
                        spenso::dind(spenso::cof(3, _gammaloop::hedge_11))
                    )
                    * spenso::t(
                        spenso::coad(8, _gammaloop::hedge_13),
                        spenso::cof(3, _gammaloop::hedge_11),
                        spenso::dind(spenso::cof(3, _gammaloop::hedge_2))
                    )
        );

        let lib: DummyLibrary<_, DummyKey> = DummyLibrary::<_, _>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        // println!("Hi");
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            loop_tn_expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
    }

    #[test]
    fn gammaloop_six_photon() {
        initialize();
        let expr = parse_lit!(
            -64 / 243 * ee
                ^ 6 * (MT * g(euc(4, hedge3), euc(4, hedge4))
                    + gamma(euc(4, hedge3), euc(4, hedge4), mink(4, edge_3_1))
                        * Q(3, mink(4, edge_3_1)))
                    * (MT * g(euc(4, hedge6), euc(4, hedge7))
                        + gamma(euc(4, hedge6), euc(4, hedge7), mink(4, edge_5_1))
                            * Q(5, mink(4, edge_5_1)))
                    * (MT * g(euc(4, hedge9), euc(4, hedge10))
                        + gamma(euc(4, hedge9), euc(4, hedge10), mink(4, edge_7_1))
                            * Q(7, mink(4, edge_7_1)))
                    * (MT * g(euc(4, hedge11), euc(4, hedge12))
                        + gamma(euc(4, hedge11), euc(4, hedge12), mink(4, edge_8_1))
                            * Q(8, mink(4, edge_8_1)))
                    * (MT * g(euc(4, hedge13), euc(4, hedge14))
                        + gamma(euc(4, hedge13), euc(4, hedge14), mink(4, edge_9_1))
                            * Q(9, mink(4, edge_9_1)))
                    * (MT * g(euc(4, hedge16), euc(4, hedge17))
                        + gamma(euc(4, hedge16), euc(4, hedge17), mink(4, edge_11_1))
                            * Q(11, mink(4, edge_11_1)))
                    * gamma(euc(4, hedge4), euc(4, hedge6), mink(4, hedge5))
                    * gamma(euc(4, hedge7), euc(4, hedge9), mink(4, hedge8))
                    * gamma(euc(4, hedge10), euc(4, hedge11), mink(4, hedge0))
                    * gamma(euc(4, hedge12), euc(4, hedge13), mink(4, hedge1))
                    * gamma(euc(4, hedge14), euc(4, hedge16), mink(4, hedge15))
                    * gamma(euc(4, hedge17), euc(4, hedge3), mink(4, hedge2))
                    * eps(0, mink(4, hedge0))
                    * eps(1, mink(4, hedge1))
                    * epsbar(2, mink(4, hedge2))
                    * epsbar(4, mink(4, hedge5))
                    * epsbar(6, mink(4, hedge8))
                    * epsbar(10, mink(4, hedge15))
        );
        let lib: DummyLibrary<_, DummyKey> = DummyLibrary::<_, _>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        // println!("Hi");
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        // println!("{}", expr);
        // println!(
        //     "{}",
        //     net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
        // );

        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        // println!(
        //     "{}",
        //     net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
        // );
        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not tensor")
        }
    }

    #[test]
    fn parse_neg_tensors() {
        initialize();
        let expr =
            parse!("-d(mink(4,6),mink(4,5))*Q(2,mink(4,7))+d(mink(4,6),mink(4,5))*Q(3,mink(4,7))");
        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        println!("Hi");
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            // println!("YaY:{}", (&expr - &tensor.expression).expand());
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not tensor")
        }
    }

    #[test]
    fn many_sums() {
        initialize();
        let expr = parse_lit!(
            (P(4, mink(4, r_2)) + N(4, mink(4, r_2)))
                * (P(5, mink(4, r_3)) + N(5, mink(4, r_3)))
                * (A(mink(4, r_2), mink(4, r_3)) + B(mink(4, r_3), mink(4, r_2)))
        );
        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<StepsDebug<6>, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not tensor")
        }
    }

    #[test]
    fn contract_problem() {
        initialize();

        let expr = parse_lit!(
            (-1 * spenso::Q(spenso::EMRID(0, 4), spenso::mink(4, python::l_20))
                * spenso::gamma(
                    spenso::euc(4, python::l_3),
                    spenso::euc(4, python::l_6),
                    spenso::mink(4, python::l_0)
                )
                * spenso::gamma(
                    spenso::euc(4, python::l_5),
                    spenso::euc(4, python::l_2),
                    spenso::mink(4, python::l_20)
                )
                * spenso::gamma(
                    spenso::euc(4, python::l_6),
                    spenso::euc(4, python::l_5),
                    spenso::mink(4, python::l_1)
                )
                + 2 * spenso::Q(spenso::EMRID(0, 4), spenso::mink(4, python::l_1))
                    * spenso::gamma(
                        spenso::euc(4, python::l_3),
                        spenso::euc(4, python::l_2),
                        spenso::mink(4, python::l_0)
                    ))
                * 1𝑖
                * spenso::G
                ^ 2
        );

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();
        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        net.execute::<Steps<14>, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
    }

    #[test]
    // #[should_panic]
    fn parse_problem() {
        initialize();
        let expr = parse_lit!(
            1 / 18 * ee
                ^ 2 * G
                ^ 4 * ((mUV
                    ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                        + OSE(3, Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                    ^ (1 / 2)
                        + (mUV
                            ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                                + OSE(4, -Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                    ^ (1 / 2))
                ^ -1 * ((mUV
                    ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                        + OSE(3, Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                    ^ (1 / 2)
                        + (mUV
                            ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                                + OSE(5, -Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                    ^ (1 / 2))
                ^ -1 * (-Q3(3, mink(4, edge_4_1))
                    + (mUV
                        ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                            + OSE(4, -Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                    ^ (1 / 2) * sigma(4) * delta(cind(0), mink(4, edge_4_1)))
                    * (-Q3(3, mink(4, edge_5_1))
                        + (mUV
                            ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                                + OSE(5, -Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                        ^ (1 / 2) * sigma(5) * delta(cind(0), mink(4, edge_5_1)))
                    * (mUV
                        ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                            + OSE(3, Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                ^ (-1 / 2)
                    * (mUV
                        ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                            + OSE(4, -Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                ^ (-1 / 2)
                    * (mUV
                        ^ 2 - (mUV ^ 2 - dot(Q3(3), Q3(3))) - dot(Q3(3), Q3(3))
                            + OSE(5, -Q3(3), mUV ^ 2, mUV ^ 2 - dot(Q3(3), Q3(3))))
                ^ (-1 / 2)
                    * g(mink(4, hedge_3), mink(4, hedge_4))
                    * gamma(bis(4, hedge(1)), bis(4, hedge(8)), mink(4, hedge_4))
                ^ 2 * gamma(bis(4, hedge(5)), bis(4, hedge(0)), mink(4, hedge_3))
                ^ 2 * gamma(bis(4, hedge(6)), bis(4, hedge(5)), mink(4, edge_4_1))
                    * gamma(bis(4, hedge(7)), bis(4, hedge(6)), mink(4, hedge_2))
                ^ 2 * gamma(bis(4, hedge(8)), bis(4, hedge(7)), mink(4, edge_5_1)),
            "spenso"
        );

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        println!("{}", expr);

        net.graph.graph.check().unwrap();

        // println!("{:?}", net.graph);

        // let ron_string = ron::ser::to_string(&net.graph).unwrap();

        // Write the RON string to a file
        // let mut file = File::create("graph.ron").unwrap();
        // file.write_all(ron_string.as_bytes()).unwrap();
        net.graph.merge_ops();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        let node: NodeIndex = net.graph.graph.len();
        println!("Nnodes: {node}");

        // net.graph.graph.iter_crown(NodeIndex(200)).for_each(|h| {
        //     println!("Hedge: {:?}", h);
        // });

        // return;

        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            // println!("YaY:{}", (&expr - &tensor.expression).expand());
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not tensor")
        }

        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();
        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            // println!("YaY:{}", (&expr - &tensor.expression).expand());
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not tensor")
        }
    }
    #[test]
    // #[should_panic]
    fn infinite_execution() {
        initialize();
        let _s = symbol!("python::dim");

        let _ = parse_lit!(
            (spenso::N(0, spenso::mink(python::dim, python::l(0)))
                * spenso::P(0, spenso::mink(python::dim, python::r(0)))
                + spenso::N(0, spenso::mink(python::dim, python::r(0)))
                    * spenso::P(0, spenso::mink(python::dim, python::l(0))))
                * spenso::N(0, spenso::mink(python::dim, python::dummy_ss(0, 1)))
                * spenso::P(0, spenso::mink(python::dim, python::dummy_ss(0, 1)))
                + -1 * spenso::N(0, spenso::mink(python::dim, python::dummy_ss(0, 3)))
                    * spenso::N(0, spenso::mink(python::dim, python::dummy_ss(0, 4)))
                    * spenso::P(0, spenso::mink(python::dim, python::dummy_ss(0, 3)))
                    * spenso::P(0, spenso::mink(python::dim, python::dummy_ss(0, 4)))
                    * spenso::g(
                        spenso::mink(python::dim, python::l(0)),
                        spenso::mink(python::dim, python::r(0))
                    )
        )
        .replace(parse_lit!(python::dim))
        .with(Atom::num(4));

        let expr = parse_lit!(
            -1𝑖 * spenso::G
                ^ 3 * (spenso::g(spenso::mink(4, python::l_6), spenso::mink(4, python::l_8))
                    * spenso::g(spenso::mink(4, python::l_7), spenso::mink(4, python::l_9))
                    - spenso::g(spenso::mink(4, python::l_6), spenso::mink(4, python::l_9))
                        * spenso::g(spenso::mink(4, python::l_8), spenso::mink(4, python::l_7)))
                    * spenso::g(spenso::mink(4, python::l_0), spenso::mink(4, python::l_6))
                    * spenso::g(spenso::mink(4, python::l_1), spenso::mink(4, python::l_7))
                    * spenso::g(spenso::mink(4, python::l_4), spenso::mink(4, python::l_8))
                    * spenso::g(spenso::mink(4, python::l_5), spenso::mink(4, python::l_9))
                    * spenso::g(spenso::bis(4, python::l_2), spenso::bis(4, python::l_5))
                    * spenso::g(spenso::bis(4, python::l_3), spenso::bis(4, python::l_6))
                    * spenso::gamma(
                        spenso::bis(4, python::l_6),
                        spenso::bis(4, python::l_5),
                        spenso::mink(4, python::l_5)
                    )
        );

        // let expr = parse_lit!(
        //     (-1 * spenso::g(spenso::mink(4, python::l_6), spenso::mink(4, python::l_9))
        //         * spenso::g(spenso::mink(4, python::l_7), spenso::mink(4, python::l_8))
        //         + spenso::g(spenso::mink(4, python::l_6), spenso::mink(4, python::l_8))
        //             * spenso::g(spenso::mink(4, python::l_7), spenso::mink(4, python::l_9)))
        //         * -1𝑖
        //         * (spenso::G
        //             ^ 3 * spenso::g(spenso::bis(4, python::l_2), spenso::bis(4, python::l_5))
        //                 * spenso::g(spenso::bis(4, python::l_3), spenso::bis(4, python::l_6))
        //                 * spenso::g(spenso::mink(4, python::l_0), spenso::mink(4, python::l_6))
        //                 * spenso::g(spenso::mink(4, python::l_1), spenso::mink(4, python::l_7))
        //                 * spenso::g(spenso::mink(4, python::l_4), spenso::mink(4, python::l_8))
        //                 * spenso::g(spenso::mink(4, python::l_5), spenso::mink(4, python::l_9))
        //                 * spenso::gamma(
        //                     spenso::bis(4, python::l_6),
        //                     spenso::bis(4, python::l_5),
        //                     spenso::mink(4, python::l_5)
        //                 )),
        //     "spenso"
        // );

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();

        let mut net_iter = net.clone();

        loop {
            let old = net_iter.clone();
            net_iter
                .execute::<Steps<1>, SingleSmallestDegree<false>, _, _, _>(&lib, &fnlib)
                .unwrap();
            net_iter
                .execute::<Steps<1>, ContractScalars, _, _, _>(&lib, &fnlib)
                .unwrap();
            if net_iter == old {
                break;
            }
        }
        println!(
            "{}",
            net_iter.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        net.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
    }

    #[test]
    fn gammaloop_input() {
        initialize();
        let expr = parse_lit!(
            16 / 81 * ee
                ^ 4 * (MT * g(euc(4, hedge_3), euc(4, hedge_4))
                    + gamma(euc(4, hedge_3), euc(4, hedge_4), mink(4, edge_3_1))
                        * Q(3, mink(4, edge_3_1)))
                    * (MT * g(euc(4, hedge_6), euc(4, hedge_7))
                    + gamma(euc(4, hedge_6), euc(4, hedge_7), mink(4, edge_5_1))
                            * Q(5, mink(4, edge_5_1)))
                    * (MT * g(euc(4, hedge_10), euc(4, hedge_11))
                    + gamma(euc(4, hedge_10), euc(4, hedge_11), mink(4, edge_8_1))
                            * Q(8, mink(4, edge_8_1)))
                    // * (-1 / 8 * (-Q(1, cind(0)) + Q(7, cind(0)) + OSE(8))
                    //     ^ -1 * (-Q(1, cind(0)) + Q(2, cind(0)) + Q(7, cind(0)) + OSE(3))
                    //     ^ -1 * (-Q(1, cind(0))
                    //         + Q(2, cind(0))
                    //         + Q(4, cind(0))
                    //         + Q(7, cind(0))
                    //         + OSE(5))
                    //     ^ -1 * delta_sigma(1, 1, 1, 1, 1, 1, 1, 1, 1) * OSE(3)
                    //     ^ -1 * OSE(5)
                    //     ^ -1 * OSE(8)
                    //     ^ -1 - 1 / 8 * (-Q(1, cind(0)) + Q(7, cind(0)) + OSE(8))
                    //     ^ -1 * (-Q(1, cind(0)) + Q(2, cind(0)) + Q(7, cind(0)) + OSE(3))
                    //     ^ -1 * (Q(1, cind(0)) - Q(2, cind(0)) - Q(4, cind(0)) - Q(7, cind(0))
                    //         + OSE(5))
                    //     ^ -1 * delta_sigma(1, 1, 1, 1, 1, -1, 1, 1, 1) * OSE(3)
                    //     ^ -1 * OSE(5)
                    //     ^ -1 * OSE(8)
                    //     ^ -1 - 1 / 8 * (-Q(1, cind(0)) + Q(7, cind(0)) + OSE(8))
                    //     ^ -1 * (Q(1, cind(0)) - Q(2, cind(0)) - Q(7, cind(0)) + OSE(3))
                    //     ^ -1 * (-Q(1, cind(0))
                    //         + Q(2, cind(0))
                    //         + Q(4, cind(0))
                    //         + Q(7, cind(0))
                    //         + OSE(5))
                    //     ^ -1 * delta_sigma(1, 1, 1, -1, 1, 1, 1, 1, 1) * OSE(3)
                    //     ^ -1 * OSE(5)
                    //     ^ -1 * OSE(8)
                    //     ^ -1 - 1 / 8 * (-Q(1, cind(0)) + Q(7, cind(0)) + OSE(8))
                    //     ^ -1 * (Q(1, cind(0)) - Q(2, cind(0)) - Q(7, cind(0)) + OSE(3))
                    //     ^ -1 * (Q(1, cind(0)) - Q(2, cind(0)) - Q(4, cind(0)) - Q(7, cind(0))
                    //         + OSE(5))
                    //     ^ -1 * delta_sigma(1, 1, 1, -1, 1, -1, 1, 1, 1) * OSE(3)
                    //     ^ -1 * OSE(5)
                    //     ^ -1 * OSE(8)
                    //     ^ -1 - 1 / 8 * (Q(1, cind(0)) - Q(7, cind(0)) + OSE(8))
                    //     ^ -1 * (-Q(1, cind(0)) + Q(2, cind(0)) + Q(7, cind(0)) + OSE(3))
                    //     ^ -1 * (-Q(1, cind(0))
                    //         + Q(2, cind(0))
                    //         + Q(4, cind(0))
                    //         + Q(7, cind(0))
                    //         + OSE(5))
                    //     ^ -1 * delta_sigma(1, 1, 1, 1, 1, 1, 1, 1, -1) * OSE(3)
                    //     ^ -1 * OSE(5)
                    //     ^ -1 * OSE(8)
                    //     ^ -1 - 1 / 8 * (Q(1, cind(0)) - Q(7, cind(0)) + OSE(8))
                    //     ^ -1 * (-Q(1, cind(0)) + Q(2, cind(0)) + Q(7, cind(0)) + OSE(3))
                    //     ^ -1 * (Q(1, cind(0)) - Q(2, cind(0)) - Q(4, cind(0)) - Q(7, cind(0))
                    //         + OSE(5))
                    //     ^ -1 * delta_sigma(1, 1, 1, 1, 1, -1, 1, 1, -1) * OSE(3)
                    //     ^ -1 * OSE(5)
                    //     ^ -1 * OSE(8)
                    //     ^ -1 - 1 / 8 * (Q(1, cind(0)) - Q(7, cind(0)) + OSE(8))
                    //     ^ -1 * (Q(1, cind(0)) - Q(2, cind(0)) - Q(7, cind(0)) + OSE(3))
                    //     ^ -1 * (-Q(1, cind(0))
                    //         + Q(2, cind(0))
                    //         + Q(4, cind(0))
                    //         + Q(7, cind(0))
                    //         + OSE(5))
                    //     ^ -1 * delta_sigma(1, 1, 1, -1, 1, 1, 1, 1, -1) * OSE(3)
                    //     ^ -1 * OSE(5)
                    //     ^ -1 * OSE(8)
                    //     ^ -1 - 1 / 8 * (Q(1, cind(0)) - Q(7, cind(0)) + OSE(8))
                    //     ^ -1 * (Q(1, cind(0)) - Q(2, cind(0)) - Q(7, cind(0)) + OSE(3))
                    //     ^ -1 * (Q(1, cind(0)) - Q(2, cind(0)) - Q(4, cind(0)) - Q(7, cind(0))
                    //         + OSE(5))
                    //     ^ -1 * delta_sigma(1, 1, 1, -1, 1, -1, 1, 1, -1) * OSE(3)
                    //     ^ -1 * OSE(5)
                    //     ^ -1 * OSE(8)
                    //     ^ -1)
                    * g(dind(cof(3, hedge_8)), cof(3, hedge_1))
                    * gamma(euc(4, hedge_1), euc(4, hedge_10), mink(4, hedge_9))
                    * gamma(euc(4, hedge_4), euc(4, hedge_6), mink(4, hedge_5))
                    * gamma(euc(4, hedge_7), euc(4, hedge_8), mink(4, hedge_0))
                    * gamma(euc(4, hedge_11), euc(4, hedge_3), mink(4, hedge_2))
                    * ubar(6, euc(4, hedge_8))
                    * u(1, euc(4, hedge_1))
                    * eps(0, mink(4, hedge_0))
                    * eps(2, mink(4, hedge_2))
                    * eps(4, mink(4, hedge_5))
                    * eps(7, mink(4, hedge_9))
        );

        let lib = DummyLibrary::<_>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let mut net = Network::<NetworkStore<SymbolicTensor, Atom>, _, Symbol>::try_from_view(
            expr.as_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap();
        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        net.execute::<Steps<14>, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
    }
    #[test]
    fn wrapping() {
        initialize();
        let expr = parse_lit!(A * g(euc(4, hedge_3), euc(4, hedge_5)));

        let expr2 = parse_lit!(B * gg(euc(4, hedge_3), euc(4, hedge_4)));
        let expr3 = parse_lit!(C * ggg(euc(4, hedge_5), euc(4, hedge_4)));

        let lib = DummyLibrary::<SymbolicTensor>::new();
        let fnlib = ErroringLibrary::<Symbol>::new();
        let net = dummy_lib_parse(expr.as_view());
        let net2 = dummy_lib_parse(expr2.as_view());
        let net3 = dummy_lib_parse(expr3.as_view());

        let mut acc = Network::one();
        println!("{}", expr);

        acc *= net;
        acc *= net2;
        acc *= net3;
        println!(
            "{}",
            acc.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        acc.merge_ops();

        println!(
            "{}",
            acc.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );

        // acc.execute::<Sequential, SmallestDegree, _, _,_>(&lib,&fnlib)
        //     .unwrap();

        acc.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
            .unwrap();
        println!(
            "{}",
            acc.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.to_string(),
                |_| "".to_string()
            )
        );
        let obt: Atom = acc.result_scalar().unwrap().into();
        assert_eq!(obt, expr * expr2 * expr3)
    }
    #[test]
    fn scalar_mult() {
        initialize();
        let expr = parse_lit!(A * B * C * g(euc(4, hedge_3)));

        let expr2 = parse_lit!(B * gg(euc(4, hedge_3)));
        let expr3 = parse_lit!(C * ggg(euc(4, hedge_5)) * g(euc(4, hedge_4)));
        let expr4 = parse_lit!(A * B * ggg(euc(4, hedge_5)) * g(euc(4, hedge_4)));
        let fnlib = ErroringLibrary::<Symbol>::new();
        let lib = DummyLibrary::<SymbolicTensor>::new();

        for ex in [expr, expr2, expr3, expr4] {
            println!("Expr:{ex}");
            let mut acc = dummy_lib_parse(&ex);
            // acc *= net3;
            println!(
                "{}",
                acc.dot_display_impl(
                    |a| a.to_string(),
                    |_| None,
                    |a| a.to_string(),
                    |_| "".to_string()
                )
            );

            acc.execute::<Steps<1>, ContractScalars, _, _, _>(&lib, &fnlib)
                .unwrap();

            println!(
                "{}",
                acc.dot_display_impl(
                    |a| a.to_string(),
                    |_| None,
                    |a| a.to_string(),
                    |_| "".to_string()
                )
            );
            acc.execute::<Sequential, SmallestDegree, _, _, _>(&lib, &fnlib)
                .unwrap();

            println!(
                "{}",
                acc.dot_display_impl(
                    |a| a.to_string(),
                    |_| None,
                    |a| a.to_string(),
                    |_| "".to_string()
                )
            );

            if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
                acc.result().unwrap()
            {
                // println!("YaY:{}", (&expr - &tensor.expression).expand());
                assert_eq!(ex, tensor.expression);
            } else {
                panic!("Not tensor")
            }
        }
    }

    fn dummy_lib_parse(
        atom: impl AtomCore,
    ) -> Network<NetworkStore<SymbolicTensor, Atom>, DummyKey, Symbol> {
        let lib = DummyLibrary::<SymbolicTensor>::new();
        Network::<NetworkStore<SymbolicTensor, Atom>, DummyKey, Symbol>::try_from_view(
            atom.as_atom_view(),
            &lib,
            &ParseSettings::default(),
        )
        .unwrap()
    }
}
