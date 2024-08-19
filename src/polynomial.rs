#[cfg(test)]
mod test {

    use constcat::concat;
    #[cfg(feature = "shadowing")]
    use symbolica::{
        atom::Atom,
        id::{Pattern, Replacement},
        poly::{polynomial::MultivariatePolynomial, Variable},
        state::State,
    };

    #[cfg(feature = "shadowing")]
    use crate::symbolic::SymbolicTensor;

    use crate::structure::HasStructure;

    #[cfg(feature = "shadowing")]
    #[test]
    fn three_loop_photon_parse() {
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

        let time = std::time::Instant::now();
        let mut network = sym_tensor.to_network().unwrap();
        println!("Network created {:?}", time.elapsed());

        let time = std::time::Instant::now();
        network.contract();
        println!("Network contracted {:?}", time.elapsed());

        println!("{}", network.graph.nodes.len());

        let mut res = network
            .result_tensor()
            .unwrap()
            .scalar()
            .unwrap()
            .try_as_param()
            .unwrap()
            .clone();

        let mut reps = vec![];
        let mut vars: Vec<Variable> = vec![];
        let mut qs = vec![];

        for i in 0..18 {
            reps.push((
                Pattern::parse(format!("Q({},cind(0))", i).as_str()).unwrap(),
                Pattern::parse(format!("Q{}", i).as_str()).unwrap(),
            ));

            vars.push(Variable::Symbol(State::get_symbol(
                format!("Q{}", i).as_str(),
            )));
            qs.push(Atom::parse(format!("Q{}", i).as_str()).unwrap());
        }

        let reps = reps
            .iter()
            .map(|(pat, rhs)| Replacement::new(pat, rhs))
            .collect::<Vec<_>>();

        res = res.replace_all_multiple(&reps);
        println!(
            "Applied replacements, size: :{}",
            res.as_view().get_byte_size()
        );

        // let poly: MultivariatePolynomial<_, u8> = res.to_polynomial(&Q, None);

        // println!("Converted to polynomial {:?}", time.elapsed());
        // // poly.to_multivariate_polynomial_list(xs, include)

        // let time = std::time::Instant::now();
        // let a = poly.to_multivariate_polynomial_list(
        //     &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        //     true,
        // );

        // println!("Converted to list {:?}", time.elapsed());

        // let mut tot_bytes = 0;

        // let mut poly = Atom::new_num(0);
        // for (i, (a, exp)) in a.iter().enumerate() {
        //     let bytes = exp.to_expression().as_view().get_byte_size();

        //     println!("{:?}:{}", &a[0..18], bytes);
        //     let mut monomial = Atom::parse(format!("C{}", i).as_str()).unwrap();
        //     for (i, pow) in a[0..18].iter().enumerate() {
        //         for _ in 0..*pow {
        //             monomial = monomial * &qs[i];
        //         }
        //     }
        //     poly = poly + monomial;

        //     tot_bytes += bytes;
        //     // println!("{:?}: {}", a, exp);
        // }
        // println!("{}", poly);

        // println!(
        //     "Total bytes: {}",
        //     tot_bytes + poly.as_view().get_byte_size()
        // );
    }

    #[test]
    fn one_loop_photon_parse() {
        #[cfg(feature = "shadowing")]
        use symbolica::domains::rational::Q;

        use crate::structure::HasStructure;

        let expr = concat!("4096/729*Nc*(MT*id(aind(bis(4,3),bis(4,0)))+Q(6,aind(lord(4,13)))*γ(aind(loru(4,13),bis(4,3),bis(4,0))))*(MT*id(aind(bis(4,5),bis(4,2)))+Q(7,aind(lord(4,27)))*γ(aind(loru(4,27),bis(4,5),bis(4,2))))*(MT*id(aind(bis(4,7),bis(4,4)))+Q(8,aind(lord(4,69)))*γ(aind(loru(4,69),bis(4,7),bis(4,4))))*(MT*id(aind(bis(4,9),bis(4,6)))+Q(9,aind(lord(4,181)))*γ(aind(loru(4,181),bis(4,9),bis(4,6))))*(MT*id(aind(bis(4,11),bis(4,8)))+Q(10,aind(lord(4,475)))*γ(aind(loru(4,475),bis(4,11),bis(4,8))))*(-MT*id(aind(bis(4,1),bis(4,10)))-Q(11,aind(lord(4,1245)))*γ(aind(loru(4,1245),bis(4,1),bis(4,10))))*sqrt(pi)^6*sqrt(aEW)^6*γ(aind(lord(4,6),bis(4,1),bis(4,0)))*γ(aind(lord(4,7),bis(4,3),bis(4,2)))*γ(aind(lord(4,8),bis(4,5),bis(4,4)))*γ(aind(lord(4,9),bis(4,7),bis(4,6)))*γ(aind(lord(4,10),bis(4,9),bis(4,8)))*γ(aind(lord(4,11),bis(4,11),bis(4,10)))*ϵ(0,aind(loru(4,6)))*ϵ(1,aind(loru(4,7)))*ϵbar(2,aind(loru(4,11)))*ϵbar(3,aind(loru(4,10)))*ϵbar(4,aind(loru(4,9)))*ϵbar(5,aind(loru(4,8)))"
);

        let atom = Atom::parse(expr).unwrap();

        let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

        let time = std::time::Instant::now();
        let mut network = sym_tensor.to_network().unwrap();
        println!("Network created {:?}", time.elapsed());

        let time = std::time::Instant::now();
        network.contract();
        println!("Network contracted {:?}", time.elapsed());

        println!("{}", network.graph.nodes.len());

        let mut res = network
            .result_tensor()
            .unwrap()
            .scalar()
            .unwrap()
            .try_as_param()
            .unwrap()
            .clone();

        let mut reps = vec![];
        let mut vars: Vec<Variable> = vec![];
        let mut qs = vec![];

        for i in 0..11 {
            reps.push((
                Pattern::parse(format!("Q({},cind(0))", i).as_str()).unwrap(),
                Pattern::parse(format!("Q{}", i).as_str()).unwrap(),
            ));

            vars.push(Variable::Symbol(State::get_symbol(
                format!("Q{}", i).as_str(),
            )));
            qs.push(Atom::parse(format!("Q{}", i).as_str()).unwrap());
        }

        let reps = reps
            .iter()
            .map(|(pat, rhs)| Replacement::new(pat, rhs))
            .collect::<Vec<_>>();

        res = res.replace_all_multiple(&reps);
        println!(
            "Applied replacements, size: :{}",
            res.as_view().get_byte_size()
        );

        let time = std::time::Instant::now();
        let poly: MultivariatePolynomial<_, u8> = res.to_polynomial(&Q, None);

        println!("Converted to polynomial {:?}", time.elapsed());
        // poly.to_multivariate_polynomial_list(xs, include)

        let time = std::time::Instant::now();
        let a = poly.to_multivariate_polynomial_list(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], true);

        println!("Converted to list {:?}", time.elapsed());

        let mut tot_bytes = 0;

        let mut poly = Atom::new_num(0);
        for (i, (a, exp)) in a.iter().enumerate() {
            let bytes = exp.to_expression().as_view().get_byte_size();

            println!("{:?}:{}", &a[0..11], bytes);
            let mut monomial = Atom::parse(format!("C{}", i).as_str()).unwrap();
            for (i, pow) in a[0..11].iter().enumerate() {
                for _ in 0..*pow {
                    monomial = monomial * &qs[i];
                }
            }
            poly = poly + monomial;

            tot_bytes += bytes;
            // println!("{:?}: {}", a, exp);
        }
        println!("{}", poly);

        println!(
            "Total bytes: {}",
            tot_bytes + poly.as_view().get_byte_size()
        );
    }
}
