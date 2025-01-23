use super::*;
use crate::{
    data::SparseOrDense,
    shadowing::ETS,
    structure::{
        representation::{BaseRepName, Minkowski},
        NamedStructure,
    },
    symbolic::SymbolicTensor,
    upgrading_arithmetic::FallibleSub,
};
use constcat::concat;
use symbolica::fun;

#[test]
fn pslash_parse() {
    use crate::structure::representation::PhysReps;

    let expr = "Q(15,dind(lor(4,75257)))   *γ(lor(4,75257),bis(4,1),bis(4,18))";
    let atom = Atom::parse(expr).unwrap();

    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

    let network = sym_tensor.to_network::<PhysReps>().unwrap();

    println!("{}", network.dot());
}

#[test]
fn three_loop_photon_parse() {
    use crate::structure::representation::PhysReps;

    let expr = concat!(
        "-64/729*ee^6*G^4",
        "*(MT*id(bis(4,1),bis(4,18)))", //+Q(15,aind(loru(4,75257)))    *γ(aind(loru(4,75257),bis(4,1),bis(4,18))))",
        "*(MT*id(bis(4,3),bis(4,0)))", //+Q(6,aind(loru(4,17)))        *γ(aind(loru(4,17),bis(4,3),bis(4,0))))",
        "*(MT*id(bis(4,5),bis(4,2))   )", //+Q(7,aind(loru(4,35)))        *γ(aind(loru(4,35),bis(4,5),bis(4,2))))",
        "*(MT*id(bis(4,7),bis(4,4))   )", //+Q(8,aind(loru(4,89)))        *γ(aind(loru(4,89),bis(4,7),bis(4,4))))",
        "*(MT*id(bis(4,9),bis(4,6))   )", //+Q(9,aind(loru(4,233)))       *γ(aind(loru(4,233),bis(4,9),bis(4,6))))",
        "*(MT*id(bis(4,11),bis(4,8))  )", //+Q(10,aind(loru(4,611)))      *γ(aind(loru(4,611),bis(4,11),bis(4,8))))",
        "*(MT*id(bis(4,13),bis(4,10)) )", //+Q(11,aind(loru(4,1601)))    *γ(aind(loru(4,1601),bis(4,13),bis(4,10))))",
        "*(MT*id(bis(4,15),bis(4,12)) )", //+Q(12,aind(loru(4,4193)))    *γ(aind(loru(4,4193),bis(4,15),bis(4,12))))",
        "*(MT*id(bis(4,17),bis(4,14)) )", //+Q(13,aind(loru(4,10979)))   *γ(aind(loru(4,10979),bis(4,17),bis(4,14))))",
        "*(MT*id(bis(4,19),bis(4,16)) )", //+Q(14,aind(loru(4,28745)))   *γ(aind(loru(4,28745),bis(4,19),bis(4,16))))",
        "*Metric(mink(4,13),mink(4,8))",
        "*Metric(mink(4,15),mink(4,10))",
        // "*T(coad(8,9),cof(3,8),coaf(3,7))",
        // "*T(coad(8,14),cof(3,13),coaf(3,12))",
        // "*T(coad(8,21),cof(3,20),coaf(3,19))",
        // "*T(coad(8,26),cof(3,25),coaf(3,24))",
        // "*id(coaf(3,3),cof(3,4))*id(coaf(3,4),cof(3,24))*id(coaf(3,5),cof(3,6))*id(coaf(3,6),cof(3,3))",
        // "*id(coaf(3,8),cof(3,5))*id(coaf(3,10),cof(3,11))*id(coaf(3,11),cof(3,7))*id(coaf(3,13),cof(3,10))",
        // "*id(coaf(3,15),cof(3,16))*id(coaf(3,16),cof(3,12))*id(coaf(3,17),cof(3,18))*id(coaf(3,18),cof(3,15))",
        // "*id(coaf(3,20),cof(3,17))*id(coaf(3,22),cof(3,23))*id(coaf(3,23),cof(3,19))*id(coaf(3,25),cof(3,22))*id(coad(8,21),coad(8,9))*id(coad(8,26),coad(8,14))",
        "*γ(mink(4,6),bis(4,1),bis(4,0))",
        "*γ(mink(4,7),bis(4,3),bis(4,2))",
        "*γ(mink(4,8),bis(4,5),bis(4,4))",
        "*γ(mink(4,9),bis(4,7),bis(4,6))",
        "*γ(mink(4,10),bis(4,9),bis(4,8))",
        "*γ(mink(4,11),bis(4,11),bis(4,10))",
        "*γ(mink(4,12),bis(4,13),bis(4,12))",
        "*γ(mink(4,13),bis(4,15),bis(4,14))",
        "*γ(mink(4,14),bis(4,17),bis(4,16))",
        "*γ(mink(4,15),bis(4,19),bis(4,18))",
        "*ϵ(0,mink(4,6))",
        "*ϵ(1,mink(4,7))",
        "*ϵbar(2,mink(4,14))",
        "*ϵbar(3,mink(4,12))",
        "*ϵbar(4,mink(4,11))",
        "*ϵbar(5,mink(4,9))"
    );

    let atom = Atom::parse(expr).unwrap();

    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

    let _network = sym_tensor.to_network::<PhysReps>().unwrap();

    // println!("{}", network.rich_graph().dot());
}

// fn g(i: usize,) -> Atom {
//     let mink = Minkowski::rep(4);
//     let bis = Bispinor::rep(4);

//     fun!(
//         ETS.gamma,
//         mink.new_slot(mu).to_atom(),
//         bis.new_slot(i).to_atom(),
//         bis.new_slot(j).to_atom()
//     )
// }
fn g(mu: usize, nu: usize) -> Atom {
    let mink = Minkowski::rep(4);

    fun!(
        ETS.metric,
        mink.new_slot(mu).to_atom(),
        mink.new_slot(nu).to_atom()
    )
}

fn g_concrete(mu: usize, nu: usize) -> RealOrComplexTensor<f64, NamedStructure<Symbol>> {
    let mink = Minkowski::rep(4);

    NamedStructure::from_iter([mink.new_slot(mu), mink.new_slot(nu)], ETS.metric, None)
        .to_shell()
        .to_explicit()
        .unwrap()
        .try_into_concrete()
        .unwrap()
}
#[test]
fn sparse_dense_addition() {
    let a = g_concrete(1, 3)
        .contract(&g_concrete(2, 4))
        .unwrap()
        .add_fallible(&g_concrete(1, 2).contract(&g_concrete(3, 4)).unwrap())
        .unwrap(); // * (g(1, 4) * g(2, 3));

    let b = g_concrete(1, 3)
        .to_dense()
        .contract(&g_concrete(2, 4).to_dense())
        .unwrap()
        .add_fallible(
            &g_concrete(1, 2)
                .to_dense()
                .contract(&g_concrete(3, 4).to_dense())
                .unwrap(),
        )
        .unwrap();

    println!("{}", a.sub_fallible(&b).unwrap().to_sparse());

    // println!("{}", b.to_sparse());
}

// #[test]
// fn expanded_vs_not() {
//     let a = (g(1, 3) * g(2, 4) + g(1, 2) * g(3, 4)); // * (g(1, 4) * g(2, 3));

//     let sym_tensor: SymbolicTensor = a.try_into().unwrap();
//     let mut network = sym_tensor.to_network::<PhysReps>().unwrap();
//     network.contract();
//     let res = network.result_tensor_smart().unwrap();

//     println!("{:?}", res.structure());
//     println!("{}", res);
// }
