/// A permutation of `0..n`, with the ability to apply itself (or its inverse) to slices.
///
/// # Examples
///
/// ```
/// use spenso::permutation::Permutation;
///
/// // Create a permutation that maps 0->2, 1->0, 2->1, 3->3
/// let p = Permutation::from_map(vec![2, 0, 1, 3]);
///
/// // Apply the permutation to a slice
/// let data = vec![10, 20, 30, 40];
/// let permuted = p.apply_slice(&data);
/// assert_eq!(permuted, vec![30, 10, 20, 40]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Permutation {
    map: Vec<usize>,
    inv: Vec<usize>,
}

/// Implement ordering comparisons for permutations based on their `map` field.
impl PartialOrd for Permutation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.map.partial_cmp(&other.map)
    }
}

impl Permutation {
    // --------------------------------------------------------------------------------------------
    // Basic Constructors and Accessors
    // --------------------------------------------------------------------------------------------

    /// Creates the identity permutation of length `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::id(4);
    /// assert_eq!(p.apply_slice(&[10,20,30,40]), vec![10,20,30,40]);
    /// ```
    pub fn id(n: usize) -> Self {
        Permutation {
            map: (0..n).collect(),
            inv: (0..n).collect(),
        }
    }

    /// Creates a permutation from a mapping vector.
    /// The `map` vector states where index `i` is sent: `map[i]` is the image of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// assert_eq!(p.apply_slice(&[10,20,30]), vec![30,10,20]);
    /// ```
    pub fn from_map(map: Vec<usize>) -> Self {
        let mut inv = vec![0; map.len()];
        for (i, &j) in map.iter().enumerate() {
            inv[j] = i;
        }
        Permutation { map, inv }
    }

    /// Returns the internal mapping as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// assert_eq!(p.map(), &[2, 0, 1]);
    /// ```
    // -- ADDED
    pub fn map(&self) -> &[usize] {
        &self.map
    }

    /// Returns the inverse mapping as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// assert_eq!(p.inv(), &[1, 2, 0]);
    /// ```
    // -- ADDED
    pub fn inv(&self) -> &[usize] {
        &self.inv
    }

    // --------------------------------------------------------------------------------------------
    // Basic Operations
    // --------------------------------------------------------------------------------------------

    /// Returns the inverse of the permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let inv = p.inverse();
    /// assert_eq!(inv.apply_slice(&[10,20,30]), vec![20,30,10]);
    /// ```
    pub fn inverse(&self) -> Self {
        Permutation {
            map: self.inv.clone(),
            inv: self.map.clone(),
        }
    }

    /// Applies `self` to a slice, returning a new `Vec<T>` in permuted order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let data = vec![10, 20, 30];
    /// assert_eq!(p.apply_slice(&data), vec![30, 10, 20]);
    /// ```
    pub fn apply_slice<T: Clone, S>(&self, slice: S) -> Vec<T>
    where
        S: AsRef<[T]>,
    {
        let s = slice.as_ref();
        self.map.iter().map(|&idx| s[idx].clone()).collect()
    }

    /// Applies the inverse of `self` to a slice, returning a new `Vec<T>` in permuted order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let data = vec![10, 20, 30];
    /// assert_eq!(p.apply_slice_inv(&data), vec![20, 30, 10]);
    /// ```
    pub fn apply_slice_inv<T: Clone, S>(&self, slice: S) -> Vec<T>
    where
        S: AsRef<[T]>,
    {
        let s = slice.as_ref();
        self.inv.iter().map(|&idx| s[idx].clone()).collect()
    }

    /// Applies `self` in-place to the provided slice by using transpositions
    /// derived from the cycle decomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let mut data = vec![10, 20, 30];
    /// p.apply_slice_in_place(&mut data);
    /// assert_eq!(data, vec![20, 30, 10]);
    /// ```
    pub fn apply_slice_in_place<T: Clone, S>(&self, slice: &mut S)
    where
        S: AsMut<[T]>,
    {
        let transpositions = self.transpositions();
        for (i, j) in transpositions.iter().rev() {
            slice.as_mut().swap(*i, *j);
        }
    }

    /// Applies the inverse of `self` in-place to the provided slice by using transpositions
    /// derived from the cycle decomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let mut data = vec![10, 20, 30];
    /// p.apply_slice_in_place_inv(&mut data);
    /// assert_eq!(data, vec![30, 10, 20]);
    /// ```
    pub fn apply_slice_in_place_inv<T: Clone, S>(&self, slice: &mut S)
    where
        S: AsMut<[T]>,
    {
        let transpositions = self.transpositions();
        for (i, j) in transpositions {
            slice.as_mut().swap(i, j);
        }
    }

    /// Composes `self` with another permutation `other`, returning a new permutation:
    /// `(other ◦ self)(i) = other.map[self.map[i]]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// // p1 maps: 0->2,1->1,2->0
    /// // p2 maps: 0->1,1->2,2->0
    /// let p1 = Permutation::from_map(vec![2,1,0]);
    /// let p2 = Permutation::from_map(vec![1,2,0]);
    /// let composition = p1.compose(&p2);
    /// // Check effect on index 0
    /// // p2(0) = 1, then p1(1) = 1 => 0 -> 1
    /// // So composition should map 0->1
    /// assert_eq!(composition.map(), &[0, 2, 1]);
    /// ```
    pub fn compose(&self, other: &Self) -> Self {
        let map = self.map.iter().map(|&i| other.map[i]).collect();
        Self::from_map(map)
    }

    // --------------------------------------------------------------------------------------------
    // Sorting Utilities
    // --------------------------------------------------------------------------------------------

    /// Given a slice of items that implement `Ord`, returns the permutation that sorts them
    /// in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let data = vec![30, 10, 20, 40];
    /// let perm = Permutation::sort(&data);
    /// // perm.map should be [1, 2, 0, 3]
    /// assert_eq!(perm.apply_slice(&data), vec![10, 20, 30, 40]);
    /// ```
    pub fn sort<T, S>(slice: S) -> Permutation
    where
        T: Ord,
        S: AsRef<[T]>,
    {
        let s = slice.as_ref();
        let mut permutation: Vec<usize> = (0..s.len()).collect();
        permutation.sort_by_key(|&i| &s[i]);
        Self::from_map(permutation)
    }

    // --------------------------------------------------------------------------------------------
    // Cycles and Transpositions
    // --------------------------------------------------------------------------------------------

    /// Returns the cycle decomposition of `self` as a `Vec` of cycles,
    /// each cycle represented as a `Vec<usize>`.
    /// Each cycle lists the indices of a single cycle, e.g. `[0, 2, 1]` means `0->2, 2->1, 1->0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1, 3]);
    /// let cycles = p.find_cycles();
    /// // cycles might be [[0, 2, 1], [3]]
    /// assert_eq!(cycles.len(), 2);
    /// ```
    pub fn find_cycles(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.map.len()];
        let mut cycles = Vec::new();
        for i in 0..self.map.len() {
            if visited[i] {
                continue;
            }
            let mut cycle = Vec::new();
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                cycle.push(j);
                j = self.map[j];
            }
            if !cycle.is_empty() {
                cycles.push(cycle);
            }
        }
        cycles
    }

    /// Converts a single cycle to a list of transpositions that produce that cycle.
    /// This is a helper method and typically not used directly.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let cycle = vec![0, 2, 1];
    /// let transpositions = Permutation::cycle_to_transpositions(&cycle);
    /// // cycle 0->2,2->1,1->0 can be built from swaps (0,1) and (0,2)
    /// assert_eq!(transpositions, vec![(0, 1), (0, 2)]);
    /// ```
    pub fn cycle_to_transpositions(cycle: &[usize]) -> Vec<(usize, usize)> {
        let mut transpositions = Vec::new();
        for i in (1..cycle.len()).rev() {
            transpositions.push((cycle[0], cycle[i]));
        }
        transpositions
    }

    /// Returns the list of transpositions for `self`, by decomposing it into cycles
    /// and then converting each cycle to transpositions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let transpositions = p.transpositions();
    /// assert_eq!(transpositions, vec![(0, 1), (0, 2)]);
    /// ```
    pub fn transpositions(&self) -> Vec<(usize, usize)> {
        let cycles = self.find_cycles();
        let mut transpositions = Vec::new();
        for cycle in cycles {
            transpositions.extend(Self::cycle_to_transpositions(&cycle));
        }
        transpositions
    }

    // --------------------------------------------------------------------------------------------
    // Myrvold & Ruskey Ranking/Unranking
    // --------------------------------------------------------------------------------------------

    /// Computes the rank of the permutation in the Myrvold & Ruskey "Rank1" ordering.
    ///
    /// This is a recursive implementation. For permutations of size `n`, it removes
    /// the position of `n-1` from the permutation, multiplies the result by `n`,
    /// and adds the index of `n-1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 1, 3, 0]);
    /// assert_eq!(p.myrvold_ruskey_rank1(), 12);
    /// ```
    pub fn myrvold_ruskey_rank1(mut self) -> usize {
        let n = self.map.len();
        if self.map.len() == 1 {
            return 0;
        }

        let s = self.map[n - 1];
        self.map.swap_remove(self.inv[n - 1]);
        self.inv.swap_remove(s);

        s + n * self.myrvold_ruskey_rank1()
    }

    /// Unranks a permutation of size `n` from its Myrvold & Ruskey "Rank1" index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::myrvold_ruskey_unrank1(4, 12);
    /// assert_eq!(p.map(), &[2, 1, 3, 0]);
    /// ```
    pub fn myrvold_ruskey_unrank1(n: usize, mut rank: usize) -> Self {
        let mut p = (0..n).collect::<Vec<_>>();
        for i in (1..=n).rev() {
            let j = rank % i;
            rank /= i;
            p.swap(i - 1, j);
        }
        Permutation::from_map(p)
    }

    fn factorial(n: usize) -> usize {
        (1..=n).product()
    }

    /// Computes the rank of the permutation in the Myrvold & Ruskey "Rank2" ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 1, 3, 0]);
    /// // Suppose it has rank = R. We can test we get p back by unranking R.
    /// let rank = p.clone().myrvold_ruskey_rank2();
    /// let q = Permutation::myrvold_ruskey_unrank2(4, rank);
    /// assert_eq!(q, p);
    /// ```
    pub fn myrvold_ruskey_rank2(mut self) -> usize {
        let n = self.map.len();
        if n == 1 {
            return 0;
        }
        let s = self.map[n - 1];
        self.map.swap_remove(self.inv[n - 1]);
        self.inv.swap_remove(s);
        s * Self::factorial(n - 1) + self.myrvold_ruskey_rank2()
    }

    /// Unranks a permutation of size `n` from its Myrvold & Ruskey "Rank2" index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::myrvold_ruskey_unrank2(4, 1);
    /// assert_eq!(p.map(), &[2, 1, 3, 0]);
    /// ```
    pub fn myrvold_ruskey_unrank2(n: usize, mut rank: usize) -> Self {
        let mut p = (0..n).collect::<Vec<_>>();
        for i in (1..=n).rev() {
            let s = rank / (Self::factorial(i - 1));
            p.swap(i - 1, s);
            rank %= Self::factorial(i - 1);
        }
        Permutation::from_map(p)
    }

    // --------------------------------------------------------------------------------------------
    // Additional Suggested Methods
    // --------------------------------------------------------------------------------------------

    /// Checks if this permutation is the identity permutation (i.e., does nothing).
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::id(4);
    /// assert!(p.is_identity());
    ///
    /// let q = Permutation::from_map(vec![1,0,2,3]);
    /// assert!(!q.is_identity());
    /// ```
    // -- ADDED
    pub fn is_identity(&self) -> bool {
        self.map.iter().enumerate().all(|(i, &m)| i == m)
    }

    /// Returns the sign (+1 or -1) of the permutation,
    /// indicating whether it is an even (+1) or odd (-1) permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![1,0,3,2]);
    /// assert_eq!(p.sign(), 1); // even
    ///
    /// let q = Permutation::from_map(vec![2,1,0]);
    /// assert_eq!(q.sign(), -1); // odd
    /// ```
    // -- ADDED
    pub fn sign(&self) -> i8 {
        // Count inversions or use transpositions
        let mut sign = 1i8;
        for cycle in self.find_cycles() {
            // Each cycle of length k contributes (k-1) to the total parity
            let k = cycle.len();
            if k > 1 && (k - 1) % 2 == 1 {
                sign = -sign;
            }
        }
        sign
    }

    /// Computes the k-th power of the permutation (composition with itself k times).
    /// For k = 0, it returns the identity of the same size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use spenso::permutation::Permutation;
    /// let p = Permutation::from_map(vec![1, 2, 0]);
    /// // p^2 maps 0->p(1)=2, 1->p(2)=0, 2->p(0)=1 => [2,0,1]
    /// let p2 = p.pow(2);
    /// assert_eq!(p2.map(), &[2, 0, 1]);
    /// ```
    // -- ADDED
    pub fn pow(&self, k: usize) -> Self {
        if k == 0 {
            return Permutation::id(self.map.len());
        }
        let mut result = Permutation::id(self.map.len());
        let mut base = self.clone();
        let mut exp = k;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result.compose(&base);
            }
            base = base.compose(&base);
            exp /= 2;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_myrvold_ruskey_rank1() {
        let p = Permutation::from_map(vec![2, 1, 3, 0]);
        assert_eq!(p.myrvold_ruskey_rank1(), 12);
        for i in 0..=23 {
            assert_eq!(
                i,
                Permutation::myrvold_ruskey_rank1(Permutation::myrvold_ruskey_unrank1(4, i))
            );
        }
    }

    #[test]
    fn test_myrvold_ruskey_rank2() {
        let p = Permutation::myrvold_ruskey_unrank2(4, 1);
        assert_eq!(p.map, vec![2, 1, 3, 0]);
        for i in 0..=23 {
            assert_eq!(
                i,
                Permutation::myrvold_ruskey_rank2(Permutation::myrvold_ruskey_unrank2(4, i))
            );
        }
    }

    #[test]
    fn test_apply_slice() {
        let p = Permutation::from_map(vec![2, 1, 3, 0]);
        let data = vec![10, 20, 30, 40];
        let permuted = p.apply_slice(&data);
        assert_eq!(permuted, vec![30, 20, 40, 10]);
    }

    #[test]
    fn test_apply_slice_inv() {
        let p = Permutation::from_map(vec![2, 1, 3, 0]);
        let data = vec![10, 20, 30, 40];
        let permuted = p.apply_slice_inv(&data);
        assert_eq!(permuted, vec![40, 20, 10, 30]);
    }

    #[test]
    fn test_find_cycles() {
        let p = Permutation::from_map(vec![2, 0, 1, 3]);
        let cycles = p.find_cycles();
        assert_eq!(cycles, vec![vec![0, 2, 1], vec![3]]);
    }

    #[test]
    fn test_cycle_to_transpositions() {
        let cycle = vec![0, 2, 1];
        let transpositions = Permutation::cycle_to_transpositions(&cycle);
        assert_eq!(transpositions, vec![(0, 1), (0, 2)]);
    }

    #[test]
    fn test_transpositions() {
        let p = Permutation::from_map(vec![2, 0, 1, 3]);
        let transpositions = p.transpositions();
        assert_eq!(transpositions, vec![(0, 1), (0, 2)]);
    }

    #[test]
    fn test_apply_slice_in_place() {
        let p = Permutation::from_map(vec![2, 0, 1, 3]);
        let mut data = vec![10, 20, 30, 40];
        p.apply_slice_in_place(&mut data);
        assert_eq!(data, vec![20, 30, 10, 40]);
    }

    #[test]
    fn test_apply_slice_in_place_inv() {
        let p = Permutation::from_map(vec![2, 0, 1, 3]);
        let mut data = vec![10, 20, 30, 40];
        p.apply_slice_in_place_inv(&mut data);
        assert_eq!(data, vec![30, 10, 20, 40]);
    }

    #[test]
    fn test_sort() {
        let data = vec![30, 10, 20, 40];
        let perm = Permutation::sort(&data);
        assert_eq!(perm.map, vec![1, 2, 0, 3]);

        let sorted_data = perm.apply_slice(&data);
        assert_eq!(sorted_data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_sort_inverse() {
        let data = vec![30, 10, 20, 40];
        let perm = Permutation::sort(&data);
        let sorted_data = perm.apply_slice(&data);
        assert_eq!(sorted_data, vec![10, 20, 30, 40]);

        let inv_perm = perm.inverse();
        let original_data = inv_perm.apply_slice(&sorted_data);
        assert_eq!(original_data, data);
    }

    // -- ADDED TESTS

    #[test]
    fn test_is_identity() {
        let p = Permutation::id(5);
        assert!(p.is_identity());

        let q = Permutation::from_map(vec![1, 0, 2]);
        assert!(!q.is_identity());
    }

    #[test]
    fn test_sign() {
        // Even permutation
        let p = Permutation::from_map(vec![1, 0, 3, 2]);
        assert_eq!(p.sign(), 1);

        // Odd permutation
        let q = Permutation::from_map(vec![2, 1, 0]);
        assert_eq!(q.sign(), -1);
    }

    #[test]
    fn test_pow() {
        let p = Permutation::from_map(vec![1, 2, 0]);
        // p^1 = p
        assert_eq!(p.pow(1), p);

        // p^2 = 0->2,1->0,2->1 => [2,0,1]
        let p2 = p.pow(2);
        assert_eq!(p2.map(), &[2, 0, 1]);

        // p^3 = identity
        let p3 = p.pow(3);
        assert_eq!(p3, Permutation::id(3));
    }

    #[test]
    fn test_compose() {
        // p1: 0→1, 1→2, 2→0 (a cycle of length 3)
        let p1 = Permutation::from_map(vec![1, 2, 0]);

        // p2: 0→2, 1→0, 2→1 (the inverse of p1)
        let p2 = Permutation::from_map(vec![2, 0, 1]);

        // By definition in your code, compose(self, other) = x ↦ other(self(x)).
        //
        // So p2 ∘ p1 means apply p1 first, then p2. Since p2 is the inverse of p1,
        // their composition should be the identity permutation.
        let c1 = p1.compose(&p2);
        assert_eq!(c1, Permutation::id(3), "Expected p2 ∘ p1 = identity");

        // Likewise, p1 ∘ p2 = identity
        let c2 = p2.compose(&p1);
        assert_eq!(c2, Permutation::id(3), "Expected p1 ∘ p2 = identity");
    }
}
