use std::{fs::File, io::BufReader};

use ahash::AHashMap;

use spenso::{
    network, Complex, Levels, MixedTensor, SmartShadowStructure, SymbolicTensor, TensorNetwork,
};
use symbolica::{
    atom::{Atom, AtomView, Symbol},
    domains::{
        float::{NumericalFloatComparison, NumericalFloatLike},
        rational::Rational,
    },
    evaluate::FunctionMap,
    state::State,
};

use symbolica::domains::float::Complex as SymComplex;

fn main() {
    let expr = concat!("-64/729*G^4*ee^6",
    "*(MT*id(aind(bis(4,47),bis(4,135)))+Q(15,aind(lor(4,149)))*γ(aind(lor(4,149),bis(4,47),bis(4,135))))",
    "*(MT*id(aind(bis(4,83),bis(4,46)))+Q(6,aind(lor(4,138)))*γ(aind(lor(4,138),bis(4,83),bis(4,46))))",
    "*(MT*id(aind(bis(4,88),bis(4,82)))+γ(aind(lor(4,140),bis(4,88),bis(4,82)))*Q(7,aind(lor(4,140))))",
    "*(MT*id(aind(bis(4,96),bis(4,142)))+γ(aind(lor(4,141),bis(4,96),bis(4,142)))*Q(8,aind(lor(4,141))))",
    "*(MT*id(aind(bis(4,103),bis(4,95)))+γ(aind(lor(4,143),bis(4,103),bis(4,95)))*Q(9,aind(lor(4,143))))",
    "*(MT*id(aind(bis(4,110),bis(4,102)))+γ(aind(lor(4,144),bis(4,110),bis(4,102)))*Q(10,aind(lor(4,144))))",
    "*(MT*id(aind(bis(4,117),bis(4,109)))+γ(aind(lor(4,145),bis(4,117),bis(4,109)))*Q(11,aind(lor(4,145))))",
    "*(MT*id(aind(bis(4,122),bis(4,116)))+γ(aind(lor(4,146),bis(4,122),bis(4,116)))*Q(12,aind(lor(4,146))))",
    "*(MT*id(aind(bis(4,129),bis(4,123)))+γ(aind(lor(4,147),bis(4,129),bis(4,123)))*Q(13,aind(lor(4,147))))",
    "*(MT*id(aind(bis(4,134),bis(4,130)))+γ(aind(lor(4,148),bis(4,134),bis(4,130)))*Q(14,aind(lor(4,148))))",
    // "*id(coaf(3,46),cof(3,47))*id(coaf(3,82),cof(3,83))*id(coaf(3,95),cof(3,96))*id(coaf(3,109),cof(3,110))*id(coaf(3,116),cof(3,117))*id(coaf(3,130),cof(3,129))",
    "*γ(aind(lor(4,45),bis(4,47),bis(4,46)))*γ(aind(lor(4,81),bis(4,83),bis(4,82)))*γ(aind(lor(4,87),bis(4,88),bis(4,142)))*γ(aind(lor(4,94),bis(4,96),bis(4,95)))",
    "*γ(aind(lor(4,101),bis(4,103),bis(4,102)))*γ(aind(lor(4,108),bis(4,110),bis(4,109)))*γ(aind(lor(4,115),bis(4,117),bis(4,116)))*γ(aind(lor(4,121),bis(4,122),bis(4,123)))",
    "*γ(aind(lor(4,128),bis(4,129),bis(4,130)))*γ(aind(lor(4,133),bis(4,134),bis(4,135)))*Metric(aind(lor(4,121),lor(4,87)))*Metric(aind(lor(4,133),lor(4,101)))",
    // "*T(coad(8,87),cof(3,88),coaf(3,46))*T(coad(8,101),cof(3,103),coaf(3,102))*T(coad(8,121),cof(3,122),coaf(3,123))*T(coad(8,133),cof(3,134),coaf(3,135))",
    "*ϵ(0,aind(lor(4,45)))*ϵ(1,aind(lor(4,81)))*ϵbar(2,aind(lor(4,94)))*ϵbar(3,aind(lor(4,108)))*ϵbar(4,aind(lor(4,115)))*ϵbar(5,aind(lor(4,128)))"
);

    let params = ["MT", "G", "ee"].map(|p| Atom::parse(p).unwrap());

    let atom = Atom::parse(expr).unwrap();

    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

    let mut const_map = AHashMap::new();
    const_map.insert(params[0].as_view(), 0.2);
    const_map.insert(params[1].as_view(), 0.32243234);
    const_map.insert(params[2].as_view(), 0.932);

    let network = sym_tensor.to_network().unwrap();

    // for p in &network.params {
    //     println!("Param {}", p);
    // }

    println!("Network dot: {}", network.dot());

    let file = File::open("./examples/data.json").unwrap();
    let reader = BufReader::new(file);

    let data_string_map: AHashMap<String, Complex<f64>> = serde_json::from_reader(reader).unwrap();

    let file = File::open("./examples/const.json").unwrap();
    let reader = BufReader::new(file);

    let const_string_map: AHashMap<String, Complex<f64>> = serde_json::from_reader(reader).unwrap();

    let mut data_atom_map: (Vec<Atom>, Vec<Complex<f64>>) = data_string_map
        .into_iter()
        .map(|(k, v)| (Atom::parse(&k).unwrap(), v))
        .unzip();

    let mut const_atom_map: AHashMap<Atom, Complex<f64>> = const_string_map
        .into_iter()
        .map(|(k, v)| (Atom::parse(&k).unwrap(), v))
        .collect();

    let i = Atom::new_var(State::I);
    const_atom_map.insert(i, Complex::i());

    let mut const_map: AHashMap<AtomView<'_>, symbolica::domains::float::Complex<f64>> =
        data_atom_map
            .0
            .iter()
            .zip(data_atom_map.1.iter())
            .map(|(k, v)| (k.as_view(), (*v).into()))
            .collect();

    for (k, &v) in const_atom_map.iter() {
        const_map.insert(k.as_view(), v.into());
    }

    let mut precontracted = network.clone();
    precontracted.contract();
    precontracted.evaluate_complex(|i| i.into(), &const_map);

    println!(
        "Pre contracted{}",
        precontracted
            .result_tensor()
            .unwrap()
            .try_into_concrete()
            .unwrap()
            .try_into_complex()
            .unwrap()
            .try_into_dense()
            .unwrap()
    );

    let mut postcontracted = network.clone();
    postcontracted.evaluate_complex(|i| i.into(), &const_map);
    postcontracted.contract();

    println!(
        "Post contracted{}",
        postcontracted
            .result_tensor()
            .unwrap()
            .try_into_concrete()
            .unwrap()
            .try_into_complex()
            .unwrap()
            .try_into_dense()
            .unwrap()
    );

    let network: TensorNetwork<MixedTensor<_, SmartShadowStructure<_, _>>, Atom> = network.cast();

    let mut levels: Levels<_, _> = network.into();

    let mut fn_map: FunctionMap<Complex<Rational>> = FunctionMap::new();

    for (k, v) in const_atom_map.iter() {
        fn_map.add_constant(k.as_view().into(), (*v).map(Rational::from_f64))
    }

    let params = data_atom_map.0;

    let values: Vec<Complex<Rational>> = data_atom_map
        .1
        .iter()
        .map(|c| c.map(|f| Rational::from_f64(f)))
        .collect();

    let mut evaluator_tensor = levels.contract(2, &mut fn_map).eval_tree(
        |a| Complex {
            im: a.zero(),
            re: a.clone(),
        },
        &fn_map,
        &params,
    );
    // evaluator_tensor.evaluate(&values);
    let mut neet =
        evaluator_tensor.map_coeff(&|t| SymComplex::<f64>::from(t.map_ref(|r| r.clone().to_f64())));

    let values: Vec<SymComplex<f64>> = data_atom_map.1.iter().map(|c| (*c).into()).collect();
    let out = neet.evaluate(&values);
    // neet.linearize(); //default needs to be derived on partial eq;
}
