use std::{ops::Neg, sync::LazyLock};

use idenso::representations::{Bispinor, initialize};
use spenso::{
    algebra::complex::Complex,
    network::library::{
        TensorLibraryData,
        symbolic::{ExplicitKey, TensorLibrary},
    },
    structure::{
        PermutedStructure, TensorStructure,
        representation::{LibraryRep, Minkowski, RepName},
    },
    tensors::{
        data::{SetTensorData, SparseTensor},
        parametric::MixedTensor,
    },
};
use symbolica::{atom::Symbol, symbol};

pub struct GammaLibrary {
    pub gamma: Symbol,
    pub projp: Symbol,
    pub projm: Symbol,
    pub gamma5: Symbol,
    pub sigma: Symbol,
}

pub static WEYL: LazyLock<GammaLibrary> = LazyLock::new(|| GammaLibrary {
    gamma: symbol!("weyl::gamma"),
    projp: symbol!("weyl::projp"),
    projm: symbol!("weyl::projm"),
    gamma5: symbol!("weyl::gamma5"),
    sigma: symbol!("weyl::sigma"),
});

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

// pub fn id_impl(rep: SpensoRepresentation) -> Self {
//     ExplicitKey::from_iter(
//         [rep.representation, rep.representation.dual()],
//         ETS.id,
//         None,
//     )
//     .structure
//     .into()
// }

// pub fn metric_impl(rep: SpensoRepresentation) -> Self {
//     ExplicitKey::from_iter([rep.representation, rep.representation], ETS.metric, None)
//         .structure
//         .into()
// }

#[allow(non_snake_case)]
pub fn gamma4D_strct(name: Symbol) -> PermutedStructure<ExplicitKey> {
    ExplicitKey::from_iter(
        [
            LibraryRep::from(Minkowski {}).new_rep(4),
            Bispinor {}.new_rep(4).cast(),
            Bispinor {}.new_rep(4).cast(),
        ],
        name,
        None,
    )
}

pub fn gamma5_strct(name: Symbol) -> PermutedStructure<ExplicitKey> {
    ExplicitKey::from_iter([Bispinor {}.new_rep(4), Bispinor {}.new_rep(4)], name, None)
}

pub fn projm_strct(name: Symbol) -> PermutedStructure<ExplicitKey> {
    ExplicitKey::from_iter([Bispinor {}.new_rep(4), Bispinor {}.new_rep(4)], name, None)
}

pub fn projp_strct(name: Symbol) -> PermutedStructure<ExplicitKey> {
    ExplicitKey::from_iter([Bispinor {}.new_rep(4), Bispinor {}.new_rep(4)], name, None)
}

pub fn weyl<T: TensorLibraryData + Clone + Default>(
    one: T,
    zero: T,
) -> TensorLibrary<MixedTensor<T, ExplicitKey>> {
    let mut weyl = TensorLibrary::new();
    initialize();
    weyl.update_ids();

    let gamma_key = gamma4D_strct(WEYL.gamma).structure;
    weyl.insert_explicit(gamma_data_weyl(gamma_key, one.clone(), zero.clone()).into());

    let gamma5_key = gamma5_strct(WEYL.gamma5).structure;
    weyl.insert_explicit(gamma5_weyl_data(gamma5_key, one.clone(), zero.clone()).into());

    let projm_key = projm_strct(WEYL.projm).structure;
    weyl.insert_explicit(proj_m_data_weyl(projm_key, one.clone(), zero.clone()).into());

    let projp_key = projp_strct(WEYL.projp).structure;
    weyl.insert_explicit(proj_p_data_weyl(projp_key, one.clone(), zero.clone()).into());

    weyl
}

pub static WEYLIB: LazyLock<TensorLibrary<MixedTensor<f64, ExplicitKey>>> =
    LazyLock::new(|| weyl(1., 0.));

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;

    use spenso::{
        network::{
            ContractScalars, ExecutionResult, Network, Sequential, SingleSmallestDegree,
            SmallestDegree, SmallestDegreeIter, Steps, StepsDebug, TensorOrScalarOrKey,
            library::symbolic::ETS, parsing::ShadowedStructure, store::NetworkStore,
        },
        structure::HasStructure,
    };
    use symbolica::{atom::Atom, parse};

    use super::*;

    #[test]
    fn simple_scalar() {
        let _ = ETS.id;
        let a = WEYLIB.get(&gamma4D_strct(WEYL.gamma).structure).unwrap();

        let expr = parse!("weyl::gamma(bis(4,l_5),bis(4,l_4),mink(4,l_4))*weyl::gamma(bis(4,l_6),bis(4,l_5),mink(4,l_4))*weyl::gamma(bis(4,l_4),bis(4,l_6),mink(4,l_5))*p(mink(4,l_5))
            ","spenso");
        // let expr = parse!(
        // "weyl::gamma(bis(4,l_4),bis(4,l_6),mink(4,l_5))*p(mink(4,l_5))
        // ",
        // "spenso"
        // );
        // println!("{}", expr);

        let mut net =
            Network::<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>, _>::try_from_view(
                expr.as_view(),
                &*WEYLIB,
            )
            .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |a| Some(format!("{}", a.global_name.unwrap())),
                |a| a.structure().global_name.unwrap().to_string()
            )
        );

        net.execute::<Steps<1>, SmallestDegreeIter<1>, _>(&*WEYLIB)
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
        net.execute::<Steps<1>, SmallestDegreeIter<2>, _>(&*WEYLIB)
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
    fn parse_problem() {
        let _ = ETS.id;
        let a = WEYLIB.get(&gamma4D_strct(WEYL.gamma).structure).unwrap();

        let expr = parse!("((N(4,mink(4,l_2))*P(4,mink(4,r_2))+N(4,mink(4,r_2))*P(4,mink(4,l_2)))*N(4,mink(4,dummy_ss_4_1))*P(4,mink(4,dummy_ss_4_1))+-1*N(4,mink(4,dummy_ss_4_2))^2*P(4,mink(4,l_2))*P(4,mink(4,r_2))+-1*N(4,mink(4,dummy_ss_4_3))*N(4,mink(4,dummy_ss_4_4))*P(4,mink(4,dummy_ss_4_3))*P(4,mink(4,dummy_ss_4_4))*g(mink(4,l_2),mink(4,r_2)))*((N(5,mink(4,l_3))*P(5,mink(4,r_3))+N(5,mink(4,r_3))*P(5,mink(4,l_3)))*N(5,mink(4,dummy_ss_5_1))*P(5,mink(4,dummy_ss_5_1))+-1*N(5,mink(4,dummy_ss_5_2))^2*P(5,mink(4,l_3))*P(5,mink(4,r_3))+-1*N(5,mink(4,dummy_ss_5_3))*N(5,mink(4,dummy_ss_5_4))*P(5,mink(4,dummy_ss_5_3))*P(5,mink(4,dummy_ss_5_4))*g(mink(4,l_3),mink(4,r_3)))*(-1*G^2*P(0,mink(4,r_20))*𝑖*𝟙(bis(4,r_0),bis(4,r_7))*𝟙(bis(4,r_1),bis(4,r_4))*𝟙(mink(4,r_2),mink(4,r_5))*𝟙(mink(4,r_3),mink(4,r_4))*weyl::gamma(bis(4,r_4),bis(4,r_5),mink(4,r_4))*weyl::gamma(bis(4,r_5),bis(4,r_6),mink(4,r_20))*weyl::gamma(bis(4,r_6),bis(4,r_7),mink(4,r_5))+G^2*P(2,mink(4,r_20))*𝑖*𝟙(bis(4,r_0),bis(4,r_7))*𝟙(bis(4,r_1),bis(4,r_4))*𝟙(mink(4,r_2),mink(4,r_5))*𝟙(mink(4,r_3),mink(4,r_4))*weyl::gamma(bis(4,r_4),bis(4,r_5),mink(4,r_4))*weyl::gamma(bis(4,r_5),bis(4,r_6),mink(4,r_20))*weyl::gamma(bis(4,r_6),bis(4,r_7),mink(4,r_5)))*(-1*P(2,mink(4,l_20))+P(0,mink(4,l_20)))*-1*G^2*P(2,mink(4,dummy_2_0))*P(3,mink(4,dummy_3_1))*𝑖*𝟙(bis(4,l_0),bis(4,l_7))*𝟙(bis(4,l_1),bis(4,l_4))*𝟙(mink(4,l_2),mink(4,l_5))*𝟙(mink(4,l_3),mink(4,l_4))*weyl::gamma(bis(4,l_1),bis(4,r_1),mink(4,dummy_3_1))*weyl::gamma(bis(4,l_5),bis(4,l_4),mink(4,l_4))*weyl::gamma(bis(4,l_6),bis(4,l_5),mink(4,l_20))*weyl::gamma(bis(4,l_7),bis(4,l_6),mink(4,l_5))*weyl::gamma(bis(4,r_0),bis(4,l_0),mink(4,dummy_2_0))
            ","spenso");
        // println!("{}", expr);

        let mut net =
            Network::<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>, _>::try_from_view(
                expr.as_view(),
                &*WEYLIB,
            )
            .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |a| Some(format!("{}", a.global_name.unwrap())),
                |a| a.structure().global_name.unwrap().to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _>(&*WEYLIB)
            .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a
                    .structure()
                    .global_name
                    .map(|a| a.to_string())
                    .unwrap_or("".to_string())
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
}
