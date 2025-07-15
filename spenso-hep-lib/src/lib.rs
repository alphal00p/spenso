use std::{ops::Neg, sync::LazyLock};

use idenso::{gamma::AGS, representations::initialize};

use spenso::{
    algebra::complex::Complex,
    network::library::{
        TensorLibraryData,
        symbolic::{ExplicitKey, TensorLibrary},
    },
    structure::{PermutedStructure, TensorStructure, abstract_index::AbstractIndex, slot::AbsInd},
    tensors::{
        data::{SetTensorData, SparseTensor},
        parametric::MixedTensor,
    },
};

#[allow(clippy::similar_names)]
pub fn gamma_data_dirac<T, N>(structure: N, one: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone + Neg<Output = T>,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(one.clone(), zero.clone());
    let cn1 = Complex::<T>::new(-one.clone(), zero.clone());
    let ci = Complex::<T>::new(zero.clone(), one.clone());
    let cni = Complex::<T>::new(zero.clone(), -one.clone());
    let mut gamma = SparseTensor::empty(structure);
    // ! No check on actual structure, should expext mink,bis,bis

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
pub fn gamma_data_weyl<T, N>(structure: N, one: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Neg<Output = T> + Clone,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(one.clone(), zero.clone());
    let cn1 = Complex::<T>::new(-one.clone(), zero.clone());
    let ci = Complex::<T>::new(zero.clone(), one.clone());
    let cni = Complex::<T>::new(zero.clone(), -one.clone());
    let mut gamma = SparseTensor::empty(structure);
    // ! No check on actual structure, should expext mink,bis,bis

    // dirac gamma matrices

    gamma.set(&[0, 2, 0], c1.clone()).unwrap();
    gamma.set(&[1, 3, 0], c1.clone()).unwrap();
    gamma.set(&[2, 0, 0], c1.clone()).unwrap();
    gamma.set(&[3, 1, 0], c1.clone()).unwrap();

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

pub fn gamma0_weyl<T, N>(structure: N, one: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(one, zero);
    let mut gamma0 = SparseTensor::empty(structure);
    // ! No check on actual structure, should expext bis,bis,lor

    // dirac gamma0 matrices

    gamma0.set(&[0, 2], c1.clone()).unwrap();
    gamma0.set(&[1, 3], c1.clone()).unwrap();
    gamma0.set(&[2, 0], c1.clone()).unwrap();
    gamma0.set(&[3, 1], c1.clone()).unwrap();

    gamma0
}

pub fn gamma5_dirac_data<T, N>(structure: N, one: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(one, zero);

    let mut gamma5 = SparseTensor::empty(structure);

    gamma5.set(&[0, 2], c1.clone()).unwrap();
    gamma5.set(&[1, 3], c1.clone()).unwrap();
    gamma5.set(&[2, 0], c1.clone()).unwrap();
    gamma5.set(&[3, 1], c1.clone()).unwrap();

    gamma5
}

pub fn gamma5_weyl_data<T, N>(structure: N, one: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone + Neg<Output = T>,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(one, zero);

    let mut gamma5 = SparseTensor::empty(structure);

    gamma5.set(&[0, 0], -c1.clone()).unwrap();
    gamma5.set(&[1, 1], -c1.clone()).unwrap();
    gamma5.set(&[2, 2], c1.clone()).unwrap();
    gamma5.set(&[3, 3], c1.clone()).unwrap();

    gamma5
}

#[allow(clippy::similar_names)]
pub fn proj_m_data_dirac<T, N>(structure: N, half: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone + Neg<Output = T>,
    N: TensorStructure,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let chalf = Complex::<T>::new(half.clone(), zero.clone());
    let cnhalf = Complex::<T>::new(-half, zero);

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
pub fn proj_m_data_weyl<T, N>(structure: N, one: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone,
    N: TensorStructure,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let c1 = Complex::<T>::new(one, zero);
    let mut proj_m = SparseTensor::empty(structure);

    proj_m.set(&[0, 0], c1.clone()).unwrap();
    proj_m.set(&[1, 1], c1.clone()).unwrap();

    proj_m
}

pub fn proj_p_data_dirac<T, N>(structure: N, half: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone,
    N: TensorStructure,
{
    // ProjP(1,2) Right chirality projector (( 1+γ5)/ 2 )_s1_s2
    let chalf = Complex::<T>::new(half, zero);

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
pub fn proj_p_data_weyl<T, N>(structure: N, one: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone,
    N: TensorStructure,
{
    // ProjM(1,2) Left chirality projector (( 1−γ5)/ 2 )_s1_s2
    let c1 = Complex::<T>::new(one, zero);
    let mut proj_m = SparseTensor::empty(structure);

    proj_m.set(&[2, 2], c1.clone()).unwrap();
    proj_m.set(&[3, 3], c1.clone()).unwrap();

    proj_m
}

#[allow(clippy::similar_names)]
pub fn sigma_data<T, N>(structure: N, one: T, zero: T) -> SparseTensor<Complex<T>, N>
where
    T: Clone + Neg<Output = T>,
    N: TensorStructure,
{
    let c1 = Complex::<T>::new(one.clone(), zero.clone());
    let cn1 = Complex::<T>::new(-one.clone(), zero.clone());
    let ci = Complex::<T>::new(zero.clone(), one.clone());
    let cni = Complex::<T>::new(zero.clone(), -one.clone());

    let mut sigma = SparseTensor::empty(structure);
    sigma.set(&[0, 2, 0, 1], c1.clone()).unwrap();
    sigma.set(&[0, 2, 3, 0], c1.clone()).unwrap();
    sigma.set(&[0, 3, 1, 2], c1.clone()).unwrap();
    sigma.set(&[1, 0, 2, 2], c1.clone()).unwrap();
    sigma.set(&[1, 1, 1, 2], c1.clone()).unwrap();
    sigma.set(&[1, 3, 0, 2], c1.clone()).unwrap();
    sigma.set(&[2, 2, 1, 0], c1.clone()).unwrap();
    sigma.set(&[2, 2, 2, 1], c1.clone()).unwrap();
    sigma.set(&[2, 3, 3, 2], c1.clone()).unwrap();
    sigma.set(&[3, 0, 0, 2], c1.clone()).unwrap();
    sigma.set(&[3, 3, 2, 2], c1.clone()).unwrap();
    sigma.set(&[3, 1, 3, 2], c1.clone()).unwrap();
    sigma.set(&[0, 1, 3, 0], ci.clone()).unwrap();
    sigma.set(&[0, 3, 1, 1], ci.clone()).unwrap();
    sigma.set(&[0, 3, 2, 0], ci.clone()).unwrap();
    sigma.set(&[1, 0, 3, 3], ci.clone()).unwrap();
    sigma.set(&[1, 1, 0, 3], ci.clone()).unwrap();
    sigma.set(&[1, 1, 2, 0], ci.clone()).unwrap();
    sigma.set(&[2, 1, 1, 0], ci.clone()).unwrap();
    sigma.set(&[2, 3, 0, 0], ci.clone()).unwrap();
    sigma.set(&[2, 3, 3, 1], ci.clone()).unwrap();
    sigma.set(&[3, 0, 1, 3], ci.clone()).unwrap();
    sigma.set(&[3, 1, 0, 0], ci.clone()).unwrap();
    sigma.set(&[3, 1, 2, 3], ci.clone()).unwrap();
    sigma.set(&[0, 0, 3, 2], cn1.clone()).unwrap();
    sigma.set(&[0, 1, 0, 2], cn1.clone()).unwrap();
    sigma.set(&[0, 2, 1, 3], cn1.clone()).unwrap();
    sigma.set(&[1, 2, 0, 3], cn1.clone()).unwrap();
    sigma.set(&[1, 2, 1, 1], cn1.clone()).unwrap();
    sigma.set(&[1, 2, 2, 0], cn1.clone()).unwrap();
    sigma.set(&[2, 0, 1, 2], cn1.clone()).unwrap();
    sigma.set(&[2, 1, 2, 2], cn1.clone()).unwrap();
    sigma.set(&[2, 2, 3, 3], cn1.clone()).unwrap();
    sigma.set(&[3, 2, 0, 0], cn1.clone()).unwrap();
    sigma.set(&[3, 2, 2, 3], cn1.clone()).unwrap();
    sigma.set(&[3, 2, 3, 1], cn1.clone()).unwrap();
    sigma.set(&[0, 0, 2, 3], cni.clone()).unwrap();
    sigma.set(&[0, 0, 3, 1], cni.clone()).unwrap();
    sigma.set(&[0, 1, 1, 3], cni.clone()).unwrap();
    sigma.set(&[1, 0, 2, 1], cni.clone()).unwrap();
    sigma.set(&[1, 3, 0, 1], cni.clone()).unwrap();
    sigma.set(&[1, 3, 3, 0], cni.clone()).unwrap();
    sigma.set(&[2, 0, 0, 3], cni.clone()).unwrap();
    sigma.set(&[2, 0, 1, 1], cni.clone()).unwrap();
    sigma.set(&[2, 1, 3, 3], cni.clone()).unwrap();
    sigma.set(&[3, 0, 0, 1], cni.clone()).unwrap();
    sigma.set(&[3, 3, 1, 0], cni.clone()).unwrap();
    sigma.set(&[3, 3, 2, 1], cni.clone()).unwrap();

    sigma
}

pub fn hep_lib<Aind: AbsInd, T: TensorLibraryData + Clone + Default>(
    one: T,
    zero: T,
) -> TensorLibrary<MixedTensor<T, ExplicitKey<Aind>>, Aind>
where
{
    let mut weyl = TensorLibrary::new();
    initialize();
    weyl.update_ids();

    let gamma_key = PermutedStructure::identity(
        gamma_data_weyl(AGS.gamma_strct::<Aind>(4), one.clone(), zero.clone()).into(),
    );
    // println!("permutation{}", gamma_key.rep_permutation);
    weyl.insert_explicit(gamma_key);

    let gamma5_key = PermutedStructure::identity(
        gamma5_weyl_data(AGS.gamma5_strct::<Aind>(4), one.clone(), zero.clone()).into(),
    );
    weyl.insert_explicit(gamma5_key);

    let projm_key = PermutedStructure::identity(
        proj_m_data_weyl(AGS.projm_strct::<Aind>(4), one.clone(), zero.clone()).into(),
    );
    weyl.insert_explicit(projm_key);

    let projp_key = PermutedStructure::identity(
        proj_p_data_weyl(AGS.projp_strct::<Aind>(4), one.clone(), zero.clone()).into(),
    );
    weyl.insert_explicit(projp_key);

    weyl
}

pub static HEP_LIB: LazyLock<
    TensorLibrary<MixedTensor<f64, ExplicitKey<AbstractIndex>>, AbstractIndex>,
> = LazyLock::new(|| hep_lib(1., 0.));

#[cfg(test)]
mod tests {

    use idenso::{
        gamma::GammaSimplifier,
        metric::{MetricSimplifier, PermuteWithMetric},
        representations::Bispinor,
    };
    use spenso::{
        algebra::upgrading_arithmetic::FallibleSub,
        iterators::IteratableTensor,
        network::{
            ExecutionResult, Network, Sequential, SingleSmallestDegree, SmallestDegree,
            SmallestDegreeIter, Steps, parsing::ShadowedStructure, store::NetworkStore,
        },
        shadowing::Concretize,
        structure::{
            HasStructure, IndexlessNamedStructure,
            abstract_index::AbstractIndex,
            permuted::Perm,
            representation::{Minkowski, RepName},
            slot::IsAbstractSlot,
        },
        tensors::{
            data::{DenseTensor, SparseOrDense},
            parametric::atomcore::TensorAtomMaps,
            symbolic::SymbolicTensor,
        },
    };
    use symbolica::{
        atom::{Atom, Symbol},
        function,
        id::ConditionResult,
        parse, parse_lit, symbol,
    };

    use super::*;
    use ahash::{HashMap, HashMapExt};

    #[test]
    fn simple_scalar() {
        initialize();
        let _a = HEP_LIB.get(&AGS.gamma_strct(4)).unwrap();

        let expr = parse!("gamma(bis(4,l_5),bis(4,l_4),mink(4,l_4))*gamma(bis(4,l_6),bis(4,l_5),mink(4,l_4))*gamma(bis(4,l_4),bis(4,l_6),mink(4,l_5))*p(mink(4,l_5))
            ","spenso");
        // let expr = parse!(
        // "gamma(bis(4,l_4),bis(4,l_6),mink(4,l_5))*p(mink(4,l_5))
        // ",
        // "spenso"
        // );
        // println!("{}", expr);

        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure<AbstractIndex>>, Atom>,
            _,
        >::try_from_view(expr.as_view(), &*HEP_LIB)
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |a| Some(format!("{}", a.global_name.unwrap())),
                |a| a.structure().global_name.unwrap().to_string()
            )
        );

        net.execute::<Steps<1>, SmallestDegreeIter<1>, _, _>(&*HEP_LIB)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |a| Some(format!("{}", a.global_name?)),
                |a| a
                    .structure()
                    .global_name
                    .map(|a| a.to_string())
                    .unwrap_or("".to_string())
            )
        );
        net.execute::<Steps<1>, SmallestDegreeIter<2>, _, _>(&*HEP_LIB)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |a| Some(format!("{}", a.global_name?)),
                |a| a
                    .structure()
                    .global_name
                    .map(|a| a.to_string())
                    .unwrap_or("".to_string())
            )
        );

        println!(
            "{}",
            net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
        );
        // if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
        //     net.result().unwrap()
        // {
        //     // println!("YaY:{}", (&expr - &tensor.expression).expand());
        //     // assert_eq!(expr, tensor.expression);
        // } else {
        //     panic!("Not tensor")
        // }
    }

    #[test]
    #[should_panic]
    fn parse_problem() {
        initialize();
        let _a = HEP_LIB.get(&AGS.gamma_strct(4)).unwrap();

        let expr = parse_lit!(
            (-1 * G
                ^ 3 * P(0, mink(4, 0))
                    * P(2, mink(4, 26))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 1))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    + -1 * G
                ^ 3 * P(0, mink(4, 26))
                    * P(1, mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 0))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    + -1 * G
                ^ 3 * P(0, mink(4, 26))
                    * P(1, mink(4, 5))
                    * g(mink(4, 0), mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 5))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    + -1 * G
                ^ 3 * P(0, mink(4, 5))
                    * P(2, mink(4, 26))
                    * g(mink(4, 0), mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 5))
                    + -1 * G
                ^ 3 * P(1, mink(4, 1))
                    * P(1, mink(4, 26))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 0))
                    + -1 * G
                ^ 3 * P(1, mink(4, 26))
                    * P(1, mink(4, 5))
                    * g(mink(4, 0), mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 5))
                    + -2 * G
                ^ 3 * P(0, mink(4, 1))
                    * P(0, mink(4, 26))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 0))
                    + -2 * G
                ^ 3 * P(0, mink(4, 1))
                    * P(1, mink(4, 26))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 0))
                    + -2 * G
                ^ 3 * P(0, mink(4, 5))
                    * Q(0, mink(4, 5))
                    * g(mink(4, 0), mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 2), mink(4, 4))
                    + -2 * G
                ^ 3 * P(1, mink(4, 0))
                    * P(1, mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 2), mink(4, 4))
                    + -2 * G
                ^ 3 * P(1, mink(4, 0))
                    * P(2, mink(4, 26))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 1))
                    + -2 * G
                ^ 3 * P(1, mink(4, 1))
                    * P(2, mink(4, 0))
                    * gamma(bis(4, 3), bis(4, 2), mink(4, 4))
                    + -2 * G
                ^ 3 * P(1, mink(4, 5))
                    * P(2, mink(4, 5))
                    * g(mink(4, 0), mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 2), mink(4, 4))
                    + -4 * G
                ^ 3 * P(0, mink(4, 1))
                    * P(2, mink(4, 0))
                    * gamma(bis(4, 3), bis(4, 2), mink(4, 4))
                    + 2 * G
                ^ 3 * P(0, mink(4, 0))
                    * P(0, mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 2), mink(4, 4))
                    + 2 * G
                ^ 3 * P(0, mink(4, 0))
                    * P(2, mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 2), mink(4, 4))
                    + 2 * G
                ^ 3 * P(0, mink(4, 1))
                    * P(2, mink(4, 26))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 0))
                    + 2 * G
                ^ 3 * P(0, mink(4, 26))
                    * P(1, mink(4, 0))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 1))
                    + 2 * G
                ^ 3 * P(0, mink(4, 5))
                    * P(2, mink(4, 5))
                    * g(mink(4, 0), mink(4, 1))
                    * gamma(bis(4, 3), bis(4, 2), mink(4, 4))
                    + 2 * G
                ^ 3 * P(1, mink(4, 0))
                    * P(1, mink(4, 26))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 1))
                    + 2 * G
                ^ 3 * P(1, mink(4, i))
                ^ 2 * g(mink(4, 0), mink(4, 1)) * gamma(bis(4, 3), bis(4, 2), mink(4, 4)) + G
                ^ 3 * P(1, mink(4, 1))
                    * P(2, mink(4, 26))
                    * gamma(bis(4, 3), bis(4, 7), mink(4, 4))
                    * gamma(bis(4, 6), bis(4, 2), mink(4, 26))
                    * gamma(bis(4, 7), bis(4, 6), mink(4, 0))),
            "spenso"
        );
        // println!("{}", expr);

        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure<AbstractIndex>>, Atom>,
            _,
        >::try_from_view(expr.as_view(), &*HEP_LIB)
        .unwrap();

        net.merge_ops();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |a| Some(format!("{}", a.global_name.unwrap())),
                |a| a.structure().global_name.unwrap().to_string()
            )
        );

        // net.validate();
        net.execute::<Steps<1>, SingleSmallestDegree<true>, _, _>(&*HEP_LIB)
            .unwrap();
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.validate();
        // net.execute::<Steps<1>, ContractScalars, _, _>(&*HEP_LIB);
        // net.execute::<Steps<1>, SmallestDegree, _, _>(&*HEP_LIB);
        // net.execute::<StepsDebug<1>, SingleSmallestDegree<true>, _, _>(&*HEP_LIB);
        // net.execute::<Steps<1>, ContractScalars, _, _>(&*HEP_LIB);

        //     .unwrap();
        // net.execute::<Steps<14>, SingleSmallestDegree<false>, _, _>(&*HEP_LIB)
        //     .unwrap();
        // net.execute::<Steps<1>, SingleSmallestDegree<true>, _, _>(&*HEP_LIB)
        //     .unwrap();
        // // net.execute::<Sequential, SmallestDegree, _, _>(&*HEP_LIB)
        //     .unwrap();
        // println!(
        //     "{}",
        //     net.dot_display_impl(|a| a.to_string(), |_| None, |a| a.to_string())
        // );

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.structure().to_string().replace('\n', "\\n")
            )
        );
        // if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
        //     net.result().unwrap()
        // {
        //     // println!("YaY:{}", (&expr - &tensor.expression).expand());
        //     // assert_eq!(expr, tensor.expression);
        // } else {
        //     panic!("Not tensor")
        // }
    }

    fn validate_gamma(
        expr: Atom,
        const_map: HashMap<Atom, symbolica::domains::float::Complex<f64>>,
    ) {
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure<AbstractIndex>>, Atom>,
            _,
        >::try_from_view(expr.as_view(), &*HEP_LIB)
        .unwrap();

        let simplified = expr.simplify_gamma();

        println!("Simplified to {}", simplified);
        let mut net_simplified = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure<AbstractIndex>>, Atom>,
            _,
        >::try_from_view(simplified.as_view(), &*HEP_LIB)
        .unwrap();

        net.execute::<Sequential, SmallestDegree, _, _>(&*HEP_LIB)
            .unwrap();
        net_simplified
            .execute::<Sequential, SmallestDegree, _, _>(&*HEP_LIB)
            .unwrap();

        let function_map = HashMap::new();

        if let ExecutionResult::Val(v) = net.result_tensor(&*HEP_LIB).unwrap() {
            if let ExecutionResult::Val(v2) = net_simplified.result_tensor(&*HEP_LIB).unwrap() {
                let mut res = v.into_owned();
                let mut res_simplified = v2.into_owned();
                res.evaluate_complex(|c| c.into(), &const_map, &function_map);
                res_simplified.evaluate_complex(|c| c.into(), &const_map, &function_map);
                res = res.to_dense();
                res_simplified = res_simplified.to_dense();

                let mut sub = res.sub_fallible(&res_simplified).unwrap();
                sub.to_param();
                let sub = sub.try_into_parametric().unwrap();
                let zero = sub
                    .zero_test(10, 0.01)
                    .iter_flat()
                    .fold(ConditionResult::True, |a, (_, b)| a & *b);

                match zero {
                    ConditionResult::False => panic!(
                        "Should be zero but \n{}\n minus simplified\n{}\n is \n{}",
                        res, res_simplified, sub
                    ),
                    ConditionResult::Inconclusive => panic!("Inconclusive"),
                    _ => {
                        println!("Works:res\n{}res_simplified\n{}", res, res_simplified)
                    }
                }
            } else {
                panic!("Expected tensor result");
            }
        } else {
            panic!("Expected tensor result");
        }
    }

    #[test]
    fn gamma_algebra_validate() {
        let mut const_map = HashMap::new();
        let pt: DenseTensor<Atom, _> = ShadowedStructure::<AbstractIndex>::from_iter(
            [Minkowski {}.new_slot(4, 1)],
            symbol!("spenso::p"),
            None,
        )
        .structure
        .to_shell()
        .concretize(None);

        for (i, a) in pt.iter_flat() {
            const_map.insert(
                a.clone(),
                symbolica::domains::float::Complex::new(usize::from(i) as f64 * 1., 0.),
            );
        }

        let qt: DenseTensor<Atom, _> = ShadowedStructure::<AbstractIndex>::from_iter(
            [Minkowski {}.new_slot(4, 1)],
            symbol!("spenso::q"),
            None,
        )
        .structure
        .to_shell()
        .concretize(None);

        for (i, a) in qt.iter_flat() {
            const_map.insert(
                a.clone(),
                symbolica::domains::float::Complex::new((usize::from(i) + 1) as f64 * 1., 0.),
            );
        }
        initialize();

        #[allow(non_snake_case)]
        #[allow(unused)]
        fn A(
            i: impl Into<AbstractIndex>,
            j: impl Into<AbstractIndex>,
            k: impl Into<AbstractIndex>,
        ) -> Atom {
            let a_strct = IndexlessNamedStructure::<Symbol, ()>::from_iter(
                [
                    Bispinor {}.new_rep(2).to_lib(),
                    Bispinor {}.new_rep(2).cast(),
                    Bispinor {}.new_rep(2).cast(),
                ],
                symbol!("A"),
                None,
            );
            a_strct
                .reindex([i.into(), j.into(), k.into()])
                .unwrap()
                .map_structure(|a| SymbolicTensor::from_named(&a).unwrap())
                .permute_inds()
                .expression
                .simplify_metrics()
        }
        #[allow(unused)]
        #[allow(non_snake_case)]
        fn B(
            i: impl Into<AbstractIndex>,
            j: impl Into<AbstractIndex>,
            k: impl Into<AbstractIndex>,
        ) -> Atom {
            let a_strct = IndexlessNamedStructure::<Symbol, ()>::from_iter(
                [
                    Bispinor {}.new_rep(2).to_lib(),
                    Bispinor {}.new_rep(2).cast(),
                    Bispinor {}.new_rep(2).cast(),
                ],
                symbol!("B"),
                None,
            );
            a_strct
                .reindex([i.into(), j.into(), k.into()])
                .unwrap()
                .map_structure(|a| SymbolicTensor::from_named(&a).unwrap())
                .permute_inds()
                .expression
                .simplify_metrics()
        }

        fn gamma(
            i: impl Into<AbstractIndex>,
            j: impl Into<AbstractIndex>,
            mu: impl Into<AbstractIndex>,
        ) -> Atom {
            let gamma_strct = IndexlessNamedStructure::<Symbol, ()>::from_iter(
                [
                    Bispinor {}.new_rep(4).to_lib(),
                    Bispinor {}.new_rep(4).cast(),
                    Minkowski {}.new_rep(4).cast(),
                ],
                AGS.gamma,
                None,
            );
            gamma_strct
                .reindex([i.into(), j.into(), mu.into()])
                .unwrap()
                .permute_with_metric()
        }

        fn p(m: impl Into<AbstractIndex>) -> Atom {
            let m_atom: AbstractIndex = m.into();
            let m_atom: Atom = m_atom.into();
            let mink = Minkowski {}.new_rep(4);
            function!(symbol!("spenso::p"), mink.to_symbolic([m_atom]))
        }
        fn q(m: impl Into<AbstractIndex>) -> Atom {
            let m_atom: AbstractIndex = m.into();
            let m_atom: Atom = m_atom.into();
            let mink = Minkowski {}.new_rep(4);
            function!(symbol!("spenso::q"), mink.to_symbolic([m_atom]))
        }

        // gamma.reindex([1,2,3]).unwrap().map_structure(|a|)

        let expr = p(1)
            * (p(3) + q(3))
            * gamma(1, 2, 1)
            * gamma(2, 3, 2)
            * gamma(3, 4, 3)
            * gamma(4, 1, 4);

        validate_gamma(expr, const_map.clone());
        let _expr = p(1) * p(1);

        let bis = Bispinor {}.new_rep(2);
        let mink = Minkowski {}.new_rep(2);
        let _expr = function!(
            symbol!("A"),
            bis.slot::<AbstractIndex, _>(1).to_atom(),
            bis.slot::<AbstractIndex, _>(2).to_atom(),
            mink.slot::<AbstractIndex, _>(1).to_atom()
        ) * function!(
            symbol!("A"),
            bis.slot::<AbstractIndex, _>(2).to_atom(),
            bis.slot::<AbstractIndex, _>(1).to_atom(),
            mink.slot::<AbstractIndex, _>(2).to_atom()
        );

        let expr = gamma(1, 2, 1) * gamma(2, 1, 1);
        validate_gamma(expr, const_map.clone());
        let expr = gamma(1, 2, 2) * gamma(2, 1, 1) + gamma(1, 2, 1) * gamma(2, 1, 2);
        validate_gamma(expr, const_map.clone());
        // let expr = gamma(1, 2, 2) * gamma(2, 1, 1) + gamma(1, 2, 1) * gamma(2, 1, 2);
        // // + gamma(1, 2, 1) * gamma(2, 1, 1);

        // // let expr = A(1, 2, 0) * B(2, 1, 3);
        // validate_gamma(expr, const_map.clone());
        // let expr = gamma(1, 2, 1);

        // validate_gamma(expr, const_map.clone());
        // let expr = gamma(2, 1, 1);

        // validate_gamma(expr, const_map.clone());
        // assert_eq!(pt, qt);

        // let a: DataTensor<_, _> = DenseTensor::fill(
        //     OrderedStructure::<Bispinor>::from_iter([Bispinor {}.new_slot(2, 2)]).structure,
        //     Complex::new(-1., 0.),
        // )
        // .into();
        // let b: DataTensor<_, _> = DenseTensor::fill(
        //     OrderedStructure::<Bispinor>::from_iter([Bispinor {}.new_slot(2, 2)]).structure,
        //     Complex::new(1., 0.),
        // )
        // .into();
        // assert_eq!(
        //     ParamOrConcrete::<_, OrderedStructure<Bispinor>>::Concrete(a),
        //     ParamOrConcrete::<_, OrderedStructure<Bispinor>>::Concrete(b)
        // );
    }
}
