use std::{f64::consts::E, fs::File, io::BufReader};

use ahash::AHashMap;

use spenso::{
    network, Complex, HasStructure, Levels, MixedTensor, SmartShadowStructure, SymbolicTensor,
    TensorNetwork,
};
use symbolica::{
    atom::{Atom, AtomView, Symbol},
    domains::{
        float::{NumericalFloatComparison, NumericalFloatLike},
        rational::Rational,
    },
    evaluate::FunctionMap,
    id::Replacement,
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

    let exprnew=concat!("-64/729*ee^6*G^4",
    "*(MT*id(aind(bis(4,105),bis(4,175)))+Q(15,aind(lor(4,192)))*γ(aind(lor(4,192),bis(4,105),bis(4,175))))",
    "*(MT*id(aind(bis(4,137),bis(4,104)))+Q(6,aind(lor(4,182)))*γ(aind(lor(4,182),bis(4,137),bis(4,104))))",
    "*(MT*id(aind(bis(4,141),bis(4,136)))+Q(7,aind(lor(4,183)))*γ(aind(lor(4,183),bis(4,141),bis(4,136))))",
    "*(MT*id(aind(bis(4,146),bis(4,185)))+Q(8,aind(lor(4,184)))*γ(aind(lor(4,184),bis(4,146),bis(4,185))))",
    "*(MT*id(aind(bis(4,151),bis(4,145)))+Q(9,aind(lor(4,186)))*γ(aind(lor(4,186),bis(4,151),bis(4,145))))",
    "*(MT*id(aind(bis(4,156),bis(4,150)))+Q(10,aind(lor(4,187)))*γ(aind(lor(4,187),bis(4,156),bis(4,150))))",
    "*(MT*id(aind(bis(4,161),bis(4,155)))+Q(11,aind(lor(4,188)))*γ(aind(lor(4,188),bis(4,161),bis(4,155))))",
    "*(MT*id(aind(bis(4,166),bis(4,160)))+Q(12,aind(lor(4,189)))*γ(aind(lor(4,189),bis(4,166),bis(4,160))))",
    "*(MT*id(aind(bis(4,171),bis(4,165)))+Q(13,aind(lor(4,190)))*γ(aind(lor(4,190),bis(4,171),bis(4,165))))",
    "*(MT*id(aind(bis(4,176),bis(4,170)))+Q(14,aind(lor(4,191)))*γ(aind(lor(4,191),bis(4,176),bis(4,170))))",
    "*Metric(aind(lor(4,167),lor(4,142)))*Metric(aind(lor(4,177),lor(4,152)))",
    // "*T(aind(coad(8,142),cof(3,141),coaf(3,104)))*T(aind(coad(8,152),cof(3,151),coaf(3,150)))*T(aind(coad(8,167),cof(3,166),coaf(3,165)))*T(aind(coad(8,177),cof(3,176),coaf(3,175)))",
    // "*id(aind(coaf(3,104),cof(3,105)))*id(aind(coaf(3,136),cof(3,137)))*id(aind(coaf(3,145),cof(3,146)))*id(aind(coaf(3,155),cof(3,156)))*id(aind(coaf(3,160),cof(3,161)))*id(aind(coaf(3,170),cof(3,171)))",
    "*γ(aind(lor(4,106),bis(4,105),bis(4,104)))",
    "*γ(aind(lor(4,138),bis(4,137),bis(4,136)))",
    "*γ(aind(lor(4,142),bis(4,141),bis(4,104)))",
    "*γ(aind(lor(4,147),bis(4,146),bis(4,145)))",
    "*γ(aind(lor(4,152),bis(4,151),bis(4,150)))",
    "*γ(aind(lor(4,157),bis(4,156),bis(4,155)))",
    "*γ(aind(lor(4,162),bis(4,161),bis(4,160)))",
    "*γ(aind(lor(4,167),bis(4,166),bis(4,165)))",
    "*γ(aind(lor(4,172),bis(4,171),bis(4,170)))",
    "*γ(aind(lor(4,177),bis(4,176),bis(4,175)))",
    "*ϵ(0,aind(lor(4,106)))*ϵ(1,aind(lor(4,138)))*ϵbar(2,aind(lor(4,147)))*ϵbar(3,aind(lor(4,157)))*ϵbar(4,aind(lor(4,162)))*ϵbar(5,aind(lor(4,172)))");

    let atom = Atom::parse(expr).unwrap();

    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

    let network = sym_tensor.to_network().unwrap();

    // for (n, t) in &network.graph.nodes {
    //     println!("{}", t)
    // }

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

    let mut const_atom_map: AHashMap<Symbol, Complex<f64>> = const_string_map
        .into_iter()
        .map(|(k, v)| (State::get_symbol(&k), v))
        .collect();

    let i = Atom::new_var(State::I);
    const_atom_map.insert(State::I, Complex::i());

    let mut const_map: AHashMap<AtomView<'_>, symbolica::domains::float::Complex<f64>> =
        data_atom_map
            .0
            .iter()
            .zip(data_atom_map.1.iter())
            .map(|(k, v)| (k.as_view(), (*v).into()))
            .collect();

    let mut constvec = AHashMap::new();

    for (k, v) in const_atom_map.iter() {
        constvec.insert(Atom::new_var(*k), *v);
    }
    for (k, &v) in constvec.iter() {
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

    let counting_network: TensorNetwork<MixedTensor<_, SmartShadowStructure<_, _>>, Atom> =
        network.clone().cast();

    let mut replacements = vec![];

    let mut fn_map: FunctionMap<Rational> = FunctionMap::new();

    for (k, v) in const_atom_map.iter() {
        let name_re = Atom::new_var(State::get_symbol(k.to_string() + "_re"));
        let name_im = Atom::new_var(State::get_symbol(k.to_string() + "_im"));
        let i = Atom::new_var(State::I);
        let pat = &name_re + i * &name_im;
        replacements.push((Atom::new_var(*k).into_pattern(), pat.into_pattern()));

        fn_map.add_constant(name_re.into(), Rational::from_f64(v.re));
        fn_map.add_constant(name_im.into(), Rational::from_f64(v.im));
    }
    let reps: Vec<Replacement> = replacements
        .iter()
        .map(|(pat, rhs)| Replacement::new(&pat, &rhs))
        .collect();

    println!(
        "scalar: {}",
        counting_network
            .replace_all_multiple(&reps)
            .scalar
            .as_ref()
            .unwrap()
    );

    let counting_network = counting_network.replace_all_multiple(&reps);
    let mut levels: Levels<_, _> = counting_network.replace_all_multiple(&reps).into();
    let mut params = data_atom_map.0;
    params.push(Atom::new_var(State::I));

    let mut evaluator_tensor = levels
        .contract(1, &mut fn_map)
        .eval_tree(|a| a.clone(), &fn_map, &params)
        .unwrap();

    evaluator_tensor.horner_scheme();
    evaluator_tensor.common_subexpression_elimination();
    // evaluator_tensor.common_pair_elimination();
    // let evaluator_tensor = evaluator_tensor.linearize(params.len());
    let mut neet = evaluator_tensor.map_coeff::<SymComplex<f64>, _>(&|r| r.into());
    let mut ev = neet.linearize(params.len());

    let mut values: Vec<SymComplex<f64>> = data_atom_map.1.iter().map(|c| (*c).into()).collect();
    values.push(SymComplex::from(Complex::i()));
    let out = ev.evaluate(&values);

    println!("{}", out);

    let mut precontracted_new = counting_network.clone();
    precontracted_new.contract();
    let eval_precontracted = precontracted_new
        .to_fully_parametric()
        .eval_tree(|a| a.clone(), &fn_map, &params)
        .unwrap();

    let mut neeet = eval_precontracted
        .map_coeff::<SymComplex<f64>, _>(&|r| r.into())
        .linearize(params.len());

    let out = neeet.evaluate(&values);

    println!("Pre contracted new{}", out.result_tensor().unwrap());

    let postcontracted_new = counting_network.clone();

    let eval_postcontracted = postcontracted_new
        .to_fully_parametric()
        .eval_tree(|a| a.clone(), &fn_map, &params)
        .unwrap();

    let mut neeet = eval_postcontracted.map_coeff::<f64, _>(&|r| r.into());

    let mut neeet = neeet.compile("nested_evaluation", "libneval");

    let mut out = neeet.evaluate_complex(&values);
    out.contract();

    let o = out.result_tensor().unwrap().scalar().unwrap();
    // neet.common_pair_elimination(); //default needs to be derived on complex;
    println!(
        "Post contracted new{}",
        out.result_tensor().unwrap().scalar().unwrap()
    );
}
