use symbolica::atom::Symbol;

use super::*;

use library::Library;

use crate::network::library::LibraryTensor;
use crate::structure::abstract_index::AIND_SYMBOLS;
// use crate::shadowing::Concretize;
use crate::structure::slot::{Slot, SlotError};
use crate::structure::{NamedStructure, StructureError, TensorShell, VecStructure};

use std::fmt::Display;

use store::TensorScalarStore;
// use log::trace;

use symbolica::atom::{representation::FunView, AddView, Atom, AtomView, MulView, PowView};

use crate::structure::{HasStructure, TensorStructure};

use crate::{shadowing::Concretize, structure::representation::LibraryRep, structure::HasName};

pub type ShadowedStructure = NamedStructure<Symbol, Vec<Atom>, LibraryRep>;

impl<'a> TryFrom<FunView<'a>> for ShadowedStructure {
    type Error = StructureError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        match value.get_symbol() {
            s if s == AIND_SYMBOLS.aind => {
                let mut structure: Vec<Slot<LibraryRep>> = vec![];

                for arg in value.iter() {
                    structure.push(arg.try_into()?);
                }

                Ok(VecStructure::from(structure).into())
            }
            name => {
                let mut structure: ShadowedStructure = VecStructure::default().into();
                structure.set_name(name.into());
                let mut args = vec![];
                let mut is_structure = Some(SlotError::EmptyStructure);

                for arg in value.iter() {
                    let slot: Result<Slot<LibraryRep>, _> = arg.try_into();

                    match slot {
                        Ok(slot) => {
                            is_structure = None;
                            structure.structure.push(slot);
                        }
                        Err(e) => {
                            if let AtomView::Fun(f) = arg {
                                if f.get_symbol() == AIND_SYMBOLS.aind {
                                    let internal_s = Self::try_from(f);

                                    if let Ok(s) = internal_s {
                                        structure.extend(s);
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

                if !args.is_empty() {
                    structure.additional_args = Some(args);
                }
                if let Some(e) = is_structure {
                    Err(StructureError::EmptyStructure(e))
                } else {
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
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
    {
        match value {
            AtomView::Mul(m) => Self::try_from_mul(m, library),
            AtomView::Fun(f) => Self::try_from_fun(f, library),
            AtomView::Add(a) => Self::try_from_add(a, library),
            AtomView::Pow(p) => Self::try_from_pow(p, library),
            a => Ok(Network::scalar(a.try_into()?)),
        }
    }

    fn try_from_mul<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: MulView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
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
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
    {
        let s: Result<S, _> = value.try_into();

        if let Ok(s) = s {
            let s = s.clone();
            match library.key_for_structure(s) {
                Ok(key) => {
                    let t = library.get(&key).unwrap();
                    Ok(Self::library_tensor(t.as_ref(), key))
                }
                Err(shell) => Ok(Self::local_tensor(shell.to_shell().concretize())),
            }
        } else {
            Ok(Self::scalar(
                value.as_view().try_into().map_err(Into::into)?,
            ))
        }
    }

    fn try_from_pow<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: PowView<'a>,
        library: &Lib,
    ) -> std::result::Result<Self, TensorNetworkError<K>>
    where
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
    {
        let (base, exp) = value.get_base_exp();

        if let Ok(n) = i64::try_from(exp) {
            if n < 0 {
                return Ok(Self::scalar(value.as_view().try_into()?));
            }
            if n == 0 {
                let one = Atom::new_num(1);
                return Ok(Self::scalar(one.as_view().try_into()?));
            } else if n == 1 {
                return Self::try_from_view(base, library);
            }
            let net = Self::try_from_view(base, library)?;
            let cloned_net = net.clone();

            Ok(net.n_mul((1..n).map(|_| cloned_net.clone())))
        } else {
            Ok(Self::scalar(value.as_view().try_into()?))
        }
    }

    fn try_from_add<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: AddView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
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
        symbolic::SymbolicTensor,
    };

    use super::*;
    use library::DummyLibrary;
    use symbolica::{parse, symbol};

    #[test]
    fn parse_scalar() {
        let expr = parse!("1").unwrap();

        let lib = DummyLibrary::<_>::new();
        let mut net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib)
                .unwrap();

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();

        if let TensorOrScalarOrKey::Scalar(a) = net.result().unwrap() {
            println!("YaY:{a}")
        } else {
            panic!("Not scalar")
        }
    }

    #[test]
    fn parse_scalar_expr() {
        let expr = parse!("(y+x(mink(4,1))*y(mink(4,1))) *(1+1+2*x*(3*sin(r))/t)").unwrap();

        let lib = DummyLibrary::<_>::new();
        let mut net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib)
                .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl::<_, ()>(
                &lib,
                |a| a.to_string(),
                |_| "".to_string(),
                |a| a.to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();
        println!(
            "{}",
            net.dot_display_impl::<_, ()>(
                &lib,
                |a| a.to_string(),
                |_| "".to_string(),
                |a| a.to_string()
            )
        );
        if let TensorOrScalarOrKey::Scalar(a) = net.result().unwrap() {
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
        .to_symbolic()
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
        .to_symbolic()
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
        .to_symbolic()
        .unwrap();

        let tensor4 =
            ShadowedStructure::from_iter([Lorentz {}.new_slot(3, 2).to_lib()], symbol!("L"), None)
                .to_symbolic()
                .unwrap();

        let tensor5 = ShadowedStructure::from_iter(
            [Lorentz {}.dual().new_slot(3, 2).to_lib()],
            symbol!("P"),
            None,
        )
        .to_symbolic()
        .unwrap();

        let expr =
            (parse!("a*sin(x/2)").unwrap() * tensor1 * tensor2 * tensor3 + tensor4) * tensor5;

        let lib = DummyLibrary::<_>::new();
        let mut net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib)
                .unwrap();

        println!("{}", expr);
        println!(
            "{}",
            net.dot_display_impl::<_, ()>(
                &lib,
                |a| a.to_string(),
                |_| "".to_string(),
                |a| a.to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();
        println!(
            "{}",
            net.dot_display_impl::<_, ()>(
                &lib,
                |a| a.to_string(),
                |_| "".to_string(),
                |a| a.to_string()
            )
        );

        if let TensorOrScalarOrKey::Tensor { tensor, .. } = net.result().unwrap() {
            // println!("YaY:{a}");
            assert_eq!(expr, tensor.expression);
        } else {
            panic!("Not tensor")
        }
    }
}
