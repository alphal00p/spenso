use std::{fs::File, io::BufReader};

use ahash::AHashMap;
use approx::{assert_relative_eq, RelativeEq};
use criterion::{criterion_group, criterion_main, Criterion};
use spenso::{
    complex::Complex,
    network::{Levels, TensorNetwork},
    parametric::MixedTensor,
    structure::{HasStructure, SmartShadowStructure},
    symbolic::SymbolicTensor,
    symbolica_utils::SerializableAtom,
};
use symbolica::{
    atom::{Atom, AtomView, Symbol},
    domains::rational::Rational,
    evaluate::{CompileOptions, FunctionMap, InlineASM},
    id::Replacement,
    state::State,
};

use symbolica::domains::float::Complex as SymComplex;

fn criterion_benchmark(c: &mut Criterion) {
    let expr = concat!("-64/729*G^4*ee^6",
    "*(MT*id(aind(bis(4,47),bis(4,135)))+Q(15,aind(loru(4,149)))*γ(aind(lord(4,149),bis(4,47),bis(4,135))))",
    "*(MT*id(aind(bis(4,83),bis(4,46)))+Q(6,aind(loru(4,138)))*γ(aind(lord(4,138),bis(4,83),bis(4,46))))",
    "*(MT*id(aind(bis(4,88),bis(4,82)))+γ(aind(lord(4,140),bis(4,88),bis(4,82)))*Q(7,aind(loru(4,140))))",
    "*(MT*id(aind(bis(4,96),bis(4,142)))+γ(aind(lord(4,141),bis(4,96),bis(4,142)))*Q(8,aind(loru(4,141))))",
    "*(MT*id(aind(bis(4,103),bis(4,95)))+γ(aind(lord(4,143),bis(4,103),bis(4,95)))*Q(9,aind(loru(4,143))))",
    "*(MT*id(aind(bis(4,110),bis(4,102)))+γ(aind(lord(4,144),bis(4,110),bis(4,102)))*Q(10,aind(loru(4,144))))",
    "*(MT*id(aind(bis(4,117),bis(4,109)))+γ(aind(lord(4,145),bis(4,117),bis(4,109)))*Q(11,aind(loru(4,145))))",
    "*(MT*id(aind(bis(4,122),bis(4,116)))+γ(aind(lord(4,146),bis(4,122),bis(4,116)))*Q(12,aind(loru(4,146))))",
    "*(MT*id(aind(bis(4,129),bis(4,123)))+γ(aind(lord(4,147),bis(4,129),bis(4,123)))*Q(13,aind(loru(4,147))))",
    "*(MT*id(aind(bis(4,134),bis(4,130)))+γ(aind(lord(4,148),bis(4,134),bis(4,130)))*Q(14,aind(loru(4,148))))",
    // "*id(coaf(3,46),cof(3,47))*id(coaf(3,82),cof(3,83))*id(coaf(3,95),cof(3,96))*id(coaf(3,109),cof(3,110))*id(coaf(3,116),cof(3,117))*id(coaf(3,130),cof(3,129))",
    "*γ(aind(loru(4,45),bis(4,47),bis(4,46)))*γ(aind(loru(4,81),bis(4,83),bis(4,82)))*γ(aind(loru(4,87),bis(4,88),bis(4,142)))*γ(aind(loru(4,94),bis(4,96),bis(4,95)))",
    "*γ(aind(loru(4,101),bis(4,103),bis(4,102)))*γ(aind(loru(4,108),bis(4,110),bis(4,109)))*γ(aind(loru(4,115),bis(4,117),bis(4,116)))*γ(aind(loru(4,121),bis(4,122),bis(4,123)))",
    "*γ(aind(loru(4,128),bis(4,129),bis(4,130)))*γ(aind(loru(4,133),bis(4,134),bis(4,135)))*Metric(aind(lord(4,121),lord(4,87)))*Metric(aind(lord(4,133),lord(4,101)))",
    // "*T(coad(8,87),cof(3,88),coaf(3,46))*T(coad(8,101),cof(3,103),coaf(3,102))*T(coad(8,121),cof(3,122),coaf(3,123))*T(coad(8,133),cof(3,134),coaf(3,135))",
    "*ϵ(0,aind(lord(4,45)))*ϵ(1,aind(lord(4,81)))*ϵbar(2,aind(lord(4,94)))*ϵbar(3,aind(lord(4,108)))*ϵbar(4,aind(lord(4,115)))*ϵbar(5,aind(lord(4,128)))"
);

    let atom = Atom::parse(expr).unwrap();

    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

    let mut network = sym_tensor.to_network().unwrap();

    let mut group = c.benchmark_group("nested_evaluate_net");

    let file = File::open("./examples/data.json").unwrap();
    let reader = BufReader::new(file);

    let data_string_map: AHashMap<String, Complex<f64>> = serde_json::from_reader(reader).unwrap();

    let file = File::open("./examples/const.json").unwrap();
    let reader = BufReader::new(file);

    let const_string_map: AHashMap<String, Complex<f64>> = serde_json::from_reader(reader).unwrap();

    let data_atom_map: (Vec<Atom>, Vec<Complex<f64>>) = data_string_map
        .into_iter()
        .map(|(k, v)| (Atom::parse(&k).unwrap(), v))
        .unzip();

    let mut const_atom_map: AHashMap<Symbol, Complex<f64>> = const_string_map
        .into_iter()
        .map(|(k, v)| (State::get_symbol(k), v))
        .collect();

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

    let mut replacements = vec![];
    let mut fn_map: FunctionMap<Rational> = FunctionMap::new();

    for (k, v) in const_atom_map.iter() {
        let name_re = Atom::new_var(State::get_symbol(k.to_string() + "_re"));
        let name_im = Atom::new_var(State::get_symbol(k.to_string() + "_im"));
        let i = Atom::new_var(State::I);
        let pat = &name_re + i * &name_im;
        replacements.push((Atom::new_var(*k).into_pattern(), pat.into_pattern().into()));

        fn_map.add_constant(name_re, Rational::from(v.re));
        fn_map.add_constant(name_im, Rational::from(v.im));
    }

    let reps: Vec<Replacement> = replacements
        .iter()
        .map(|(pat, rhs)| Replacement::new(pat, rhs))
        .collect();

    let mut params = data_atom_map.0.clone();
    params.push(Atom::new_var(State::I));

    let mut truth_net = network.clone();

    truth_net.evaluate_complex(|i| i.into(), &const_map);
    truth_net.contract();
    let truth = truth_net
        .result_tensor()
        .unwrap()
        .scalar()
        .unwrap()
        .try_into_concrete()
        .unwrap()
        .try_into_complex()
        .unwrap();

    group.bench_function("3LPhysical postcontracted", |b| {
        b.iter_batched(
            || network.clone(),
            |mut network| {
                network.evaluate_complex(|i| i.into(), &const_map);
                network.contract();
                assert_relative_eq!(
                    truth,
                    network
                        .result_tensor()
                        .unwrap()
                        .scalar()
                        .unwrap()
                        .try_into_concrete()
                        .unwrap()
                        .try_into_complex()
                        .unwrap(),
                    epsilon = 0.1
                );
            },
            criterion::BatchSize::SmallInput,
        )
    });

    let counting_network: TensorNetwork<
        MixedTensor<_, SmartShadowStructure<_, _>>,
        SerializableAtom,
    > = network.clone().cast().replace_all_multiple(&reps);
    let mut values: Vec<SymComplex<f64>> = data_atom_map.1.iter().map(|c| (*c).into()).collect();
    values.push(SymComplex::from(Complex::i()));

    let mut postcontracted_eval_tree_tensor = counting_network
        .clone()
        .to_fully_parametric()
        .eval_tree(&fn_map, &params)
        .unwrap();

    postcontracted_eval_tree_tensor.horner_scheme();
    // postcontracted_eval_tree_tensor.common_pair_elimination();
    postcontracted_eval_tree_tensor.common_subexpression_elimination();

    let mut mapped_postcontracted_eval_tree_tensor =
        postcontracted_eval_tree_tensor.map_coeff::<SymComplex<f64>, _>(&|r| r.into());

    group.bench_function("3LPhysical postcontracted new", |b| {
        b.iter(|| {
            let mut out = mapped_postcontracted_eval_tree_tensor.evaluate(&values);
            out.contract();
            assert_relative_eq!(
                truth,
                out.result_tensor().unwrap().scalar().unwrap().into(),
                epsilon = 0.1
            );
        })
    });

    let mut levels: Levels<_, _> = counting_network.clone().into();
    let mut levels2 = levels.clone();

    let mut eval_tree_leveled_tensor = levels
        .contract(1, &mut fn_map)
        .eval_tree(&fn_map, &params)
        .unwrap();

    eval_tree_leveled_tensor.horner_scheme();
    // eval_tree_leveled_tensor.common_subexpression_elimination(10);
    // evaluator_tensor.common_pair_elimination();

    let mut eval_tree_leveled_tensor_depth2 = levels2
        .contract(2, &mut fn_map)
        .eval_tree(&fn_map, &params)
        .unwrap();

    eval_tree_leveled_tensor_depth2.horner_scheme();
    // eval_tree_leveled_tensor_depth2.common_subexpression_elimination(10);
    // eval_tree_leveled_tensor_depth2.common_pair_elimination();
    // evaluator_tensor.evaluate(&values);
    let mut neet = eval_tree_leveled_tensor.map_coeff::<SymComplex<f64>, _>(&|r| r.into());

    let mut neet2 = eval_tree_leveled_tensor_depth2.map_coeff::<SymComplex<f64>, _>(&|r| r.into());
    group.bench_function("3LPhysical leveled", |b| {
        b.iter(|| {
            let out = neet.evaluate(&values).scalar().unwrap();
            assert!(truth.relative_eq(&out.into(), 0.1, 1.))
        })
    });

    group.bench_function("3LPhysical leveled 2", |b| {
        b.iter(|| {
            let out = neet2.evaluate(&values).scalar().unwrap();
            assert!(truth.relative_eq(&out.into(), 0.1, 1.))
        })
    });
    network.contract();

    group.bench_function("3LPhysical precontracted", |b| {
        b.iter_batched(
            || network.clone(),
            |mut network| {
                network.evaluate_complex(|i| i.into(), &const_map);
                assert!(truth.relative_eq(
                    &network
                        .result_tensor()
                        .unwrap()
                        .scalar()
                        .unwrap()
                        .try_into_concrete()
                        .unwrap()
                        .try_into_complex()
                        .unwrap(),
                    0.1,
                    1.
                ));
            },
            criterion::BatchSize::SmallInput,
        )
    });

    let mut contracted_counting_network = counting_network.clone();
    contracted_counting_network.contract();

    let mut precontracted_eval_tree_net = contracted_counting_network
        .clone()
        .to_fully_parametric()
        .eval_tree(&fn_map, &params)
        .unwrap();
    precontracted_eval_tree_net.horner_scheme();
    // precontracted_eval_tree_net.common_subexpression_elimination(10);
    // precontracted_eval_tree_net.common_pair_elimination();

    let mut mapped_precontracted_eval_tree_net =
        precontracted_eval_tree_net.map_coeff::<SymComplex<f64>, _>(&|r| r.into());

    group.bench_function("3LPhysical precontracted new optimized", |b| {
        b.iter(|| {
            let out = mapped_precontracted_eval_tree_net.evaluate(&values);
            assert!(truth.relative_eq(
                &out.result_tensor().unwrap().scalar().unwrap().into(),
                0.1,
                1.
            ));
        })
    });

    let mut mapped_precontracted_eval_net = mapped_precontracted_eval_tree_net.linearize(Some(1));

    group.bench_function("3LPhysical precontracted new lin", |b| {
        b.iter(|| {
            let out = mapped_precontracted_eval_net.evaluate(&values);
            assert!(truth.relative_eq(
                &out.result_tensor().unwrap().scalar().unwrap().into(),
                0.1,
                1.
            ));
        })
    });

    // let time = std::time::Instant::now();
    // let mut neeet = precontracted_eval_tree_net
    //     .map_coeff::<f64, _>(&|r| r.into())
    //     .linearize()
    //     .compile("nested_evaluation", "libneval");

    // println!("compile time: {:?}", time.elapsed());
    // group.bench_function("3LPhysical precontracted new compiled", |b| {
    //     b.iter(|| {
    //         let out = neeet.evaluate_complex(&values);
    //         assert!(truth.relative_eq(
    //             &(out.result_tensor().unwrap().scalar().unwrap()).into(),
    //             0.1,
    //             1.
    //         ),);
    //     })
    // });

    let values: Vec<Complex<f64>> = values.into_iter().map(|c| c.into()).collect();
    let time = std::time::Instant::now();
    let mut precontracted_eval_tree_net = contracted_counting_network
        .clone()
        .to_fully_parametric()
        .eval_tree(&fn_map, &params)
        .unwrap();
    precontracted_eval_tree_net.horner_scheme();
    precontracted_eval_tree_net.common_subexpression_elimination();
    let mut neeet = precontracted_eval_tree_net
        .map_coeff::<f64, _>(&|r| r.into())
        .linearize(Some(1))
        .export_cpp(
            "nested_evaluation_asm",
            "nested_evaluation_asm",
            true,
            InlineASM::X64,
        )
        .unwrap()
        .compile_and_load("nested_evaluation_asm", CompileOptions::default())
        .unwrap();

    println!("asm compile time and optimisation: {:?}", time.elapsed());
    group.bench_function("3LPhysical precontracted new compiled asm", |b| {
        b.iter(|| {
            let out = neeet.evaluate(&values);
            assert!(truth.relative_eq(&(out.result_tensor().unwrap().scalar().unwrap()), 0.1, 1.),);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
