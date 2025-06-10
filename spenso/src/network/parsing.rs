use symbolica::atom::{AtomCore, Symbol};

use super::*;

use library::Library;

use crate::network::library::LibraryTensor;
use crate::structure::abstract_index::AIND_SYMBOLS;
// use crate::shadowing::Concretize;
use crate::structure::slot::{Slot, SlotError};
use crate::structure::{
    NamedStructure, OrderedStructure, PermutedStructure, StructureError, TensorShell,
};

use std::fmt::Display;

use store::TensorScalarStore;
// use log::trace;

use symbolica::atom::{representation::FunView, AddView, Atom, AtomView, MulView, PowView};

use crate::structure::{HasStructure, TensorStructure};

use crate::{shadowing::Concretize, structure::representation::LibraryRep, structure::HasName};

pub type ShadowedStructure = NamedStructure<Symbol, Vec<Atom>, LibraryRep>;

impl<'a> TryFrom<AtomView<'a>> for PermutedStructure<ShadowedStructure> {
    type Error = StructureError;
    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Fun(f) = value {
            f.try_into()
        } else {
            Err(StructureError::ParsingError(value.to_plain_string()))
        }
    }
}

impl TryFrom<Atom> for PermutedStructure<ShadowedStructure> {
    type Error = StructureError;
    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for PermutedStructure<ShadowedStructure> {
    type Error = StructureError;
    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<FunView<'a>> for PermutedStructure<ShadowedStructure> {
    type Error = StructureError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        match value.get_symbol() {
            s if s == AIND_SYMBOLS.aind => {
                let mut structure: Vec<Slot<LibraryRep>> = vec![];

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
                let mut is_structure = Some(SlotError::EmptyStructure);

                for arg in value.iter() {
                    let slot: Result<Slot<LibraryRep>, _> = arg.try_into();

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
                                        let p = s.permutation;
                                        let mut v = s.structure.structure.structure;
                                        p.apply_slice_in_place_inv(&mut v); //undo sorting
                                        slots.extend(v);
                                        is_structure = None;
                                        continue;
                                    }
                                }
                            }
                            is_structure = Some(e);
                            args.push(arg.to_owned().into());
                        }
                    }
                }

                if let Some(e) = is_structure {
                    Err(StructureError::EmptyStructure(e))
                } else {
                    let mut structure: PermutedStructure<ShadowedStructure> =
                        OrderedStructure::new(slots).map_structure(Into::into);
                    structure.structure.set_name(name.into());
                    if !args.is_empty() {
                        structure.structure.additional_args = Some(args);
                    }
                    Ok(structure)
                }
            }
        }
    }
}

impl<
        'a,
        Sc,
        T: HasStructure + TensorStructure,
        K: Clone + Display + Debug,
        Str: TensorScalarStore<Tensor = T, Scalar = Sc> + Clone,
    > Network<Str, K>
where
    Sc: for<'r> TryFrom<AtomView<'r>> + Clone,
    TensorNetworkError<K>: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
{
    pub fn try_from_view<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: AtomView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        match value {
            AtomView::Mul(m) => Self::try_from_mul(m, library),
            AtomView::Fun(f) => Self::try_from_fun(f, library),
            AtomView::Add(a) => Self::try_from_add(a, library),
            AtomView::Pow(p) => Self::try_from_pow(p, library),
            a => Ok(Network::from_scalar(a.try_into()?)),
        }
    }

    fn try_from_mul<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: MulView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        let mut iter = value.iter();

        let first = Self::try_from_view(iter.next().unwrap(), library)?;

        let rest: Result<Vec<_>, _> = iter.map(|a| Self::try_from_view(a, library)).collect();

        Ok(first.n_mul(rest?))
    }

    fn try_from_fun<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: FunView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        let s: Result<PermutedStructure<S>, _> = value.try_into();

        if let Ok(s) = s {
            println!("Perm:{}", s.permutation);
            // let s = s;
            match library.key_for_structure(&s.structure) {
                Ok(key) => {
                    // let t = library.get(&key).unwrap();
                    Ok(Self::library_tensor(&s.structure, key))
                }
                Err(_) => Ok(Self::from_tensor(
                    s.structure
                        .to_shell()
                        .concretize(Some(s.permutation.inverse())),
                )),
            }
        } else {
            Ok(Self::from_scalar(
                value.as_view().try_into().map_err(Into::into)?,
            ))
        }
    }

    fn try_from_pow<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: PowView<'a>,
        library: &Lib,
    ) -> std::result::Result<Self, TensorNetworkError<K>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        let (base, exp) = value.get_base_exp();

        if let Ok(n) = i64::try_from(exp) {
            if n < 0 {
                return Ok(Self::from_scalar(value.as_view().try_into()?));
            }
            if n == 0 {
                let one = Atom::num(1);
                return Ok(Self::from_scalar(one.as_view().try_into()?));
            } else if n == 1 {
                return Self::try_from_view(base, library);
            }
            let net = Self::try_from_view(base, library)?;
            let cloned_net = net.clone();

            Ok(net.n_mul((1..n).map(|_| cloned_net.clone())))
        } else {
            Ok(Self::from_scalar(value.as_view().try_into()?))
        }
    }

    fn try_from_add<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: AddView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
        PermutedStructure<S>: TryFrom<FunView<'a>>,
    {
        let mut iter = value.iter();

        let first = Self::try_from_view(iter.next().unwrap(), library)?;

        let rest: Result<Vec<_>, _> = iter.map(|a| Self::try_from_view(a, library)).collect();

        Ok(first.n_add(rest?))
    }
}

#[cfg(test)]
pub mod test {
    use core::panic;

    use crate::{
        structure::{
            representation::{Euclidean, Lorentz, Minkowski, RepName},
            slot::IsAbstractSlot,
            ToSymbolic,
        },
        tensors::symbolic::SymbolicTensor,
    };

    use super::*;
    use library::{symbolic::ETS, DummyLibrary};
    use symbolica::{parse, symbol};

    #[test]
    fn parse_scalar() {
        let expr = parse!("1");

        let lib = DummyLibrary::<_>::new();
        let mut net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib)
                .unwrap();

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();

        if let ExecutionResult::Val(TensorOrScalarOrKey::Scalar(a)) = net.result().unwrap() {
            println!("YaY:{a}")
        } else {
            panic!("Not scalar")
        }
    }

    #[test]
    fn parse_scalar_expr() {
        let expr = parse!("(y+x(mink(4,1))*y(mink(4,1))) *(1+1+2*x*(3*sin(r))/t)");

        let lib = DummyLibrary::<_>::new();
        let mut net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib)
                .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
        );

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();
        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
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
        let tensor1 = ShadowedStructure::from_iter(
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

        let tensor2 = ShadowedStructure::from_iter(
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

        let tensor3 = ShadowedStructure::from_iter(
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

        let tensor4 =
            ShadowedStructure::from_iter([Lorentz {}.new_slot(3, 2).to_lib()], symbol!("L"), None)
                .structure
                .to_symbolic(None)
                .unwrap();

        let tensor5 = ShadowedStructure::from_iter(
            [Lorentz {}.dual().new_slot(3, 2).to_lib()],
            symbol!("P"),
            None,
        )
        .structure
        .to_symbolic(None)
        .unwrap();

        let expr = (parse!("a*sin(x/2)") * tensor1 * tensor2 * tensor3 + tensor4) * tensor5;

        let lib = DummyLibrary::<_>::new();
        let mut net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib)
                .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
        );

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();
        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
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

    // -G^2*(-g(mink(4,5),mink(4,6))*Q(2,mink(4,7))+g(mink(4,5),mink(4,6))*Q(3,mink(4,7))+g(mink(4,5),mink(4,7))*Q(2,mink(4,6))+g(mink(4,5),mink(4,7))*Q(4,mink(4,6))-g(mink(4,6),mink(4,7))*Q(3,mink(4,5))-g(mink(4,6),mink(4,7))*Q(4,mink(4,5)))*𝟙(mink(4,2),mink(4,5))*𝟙(mink(4,3),mink(4,6))*𝟙(euc(4,0),euc(4,5))*𝟙(euc(4,1),euc(4,4))*g(mink(4,4),mink(4,7))*vbar(1,euc(4,1))*u(0,euc(4,0))*ϵbar(2,mink(4,2))*ϵbar(3,mink(4,3))*gamma(mink(4,4),euc(4,5),euc(4,4))

    #[test]
    fn parse_big_tensors() {
        let _ = ETS.id;
        let expr = parse!("-G^2*(-g(mink(4,5),mink(4,6))*Q(2,mink(4,7))+g(mink(4,5),mink(4,6))*Q(3,mink(4,7))+g(mink(4,5),mink(4,7))*Q(2,mink(4,6))+g(mink(4,5),mink(4,7))*Q(4,mink(4,6))-g(mink(4,6),mink(4,7))*Q(3,mink(4,5))-g(mink(4,6),mink(4,7))*Q(4,mink(4,5)))*𝟙(mink(4,2),mink(4,5))*𝟙(mink(4,3),mink(4,6))*𝟙(euc(4,0),euc(4,5))*𝟙(euc(4,1),euc(4,4))*g(mink(4,4),mink(4,7))*vbar(1,euc(4,1))*u(0,euc(4,0))*ϵbar(2,mink(4,2))*ϵbar(3,mink(4,3))*gamma(mink(4,4),euc(4,5),euc(4,4))");
        let lib = DummyLibrary::<_>::new();
        println!("Hi");
        let mut net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib)
                .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
        );

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();
        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
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
    fn parse_neg_tensors() {
        let _ = ETS.id;
        let expr =
            parse!("-d(mink(4,6),mink(4,5))*Q(2,mink(4,7))+d(mink(4,6),mink(4,5))*Q(3,mink(4,7))");
        let lib = DummyLibrary::<_>::new();
        println!("Hi");
        let mut net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib)
                .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
        );

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();
        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
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
}
