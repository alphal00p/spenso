use std::fmt::Debug;

use bitvec::vec::BitVec;

use crate::structure::MergeInfo;

const SUPERSCRIPT_DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
const SUBSCRIPT_DIGITS: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];

fn to_unicode(number: isize, digits: &[char; 10], minus_sign: char) -> String {
    if number == 0 {
        return digits[0].to_string();
    }
    let mut num = number;
    let mut digit_stack = Vec::new();
    while num != 0 {
        let digit = (num % 10).unsigned_abs();
        digit_stack.push(digits[digit]);
        num /= 10;
    }
    let mut result = String::new();
    if number < 0 {
        result.push(minus_sign);
    }
    result.extend(digit_stack.drain(..).rev());
    result
}

pub fn to_superscript(number: isize) -> String {
    to_unicode(number, &SUPERSCRIPT_DIGITS, '⁻')
}

pub fn to_subscript(number: isize) -> String {
    to_unicode(number, &SUBSCRIPT_DIGITS, '₋')
}

#[derive(Debug)]
pub struct DuplicateItemError {
    pub message: String,
}

impl std::fmt::Display for DuplicateItemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for DuplicateItemError {}

// Trait definition
pub trait MergeOrdered<T>: Sized {
    /// Merges self with another vector, separating common items.
    /// Returns: (merged_non_common_items, common_items_values, merge_info_for_non_common)
    fn merge_ordered_ref_with_common_removal(&self, other: &Self) -> (Self, Vec<T>, MergeInfo)
    where
        T: Ord + Clone;

    /// Merges self with another vector, separating common items.
    /// Returns: (merged_non_common_items, BitVec with bits set for common indices in first vec, BitVec with bits set for common indices in second vec, merge_info_for_non_common)
    /// Errors if either self or other contain duplicates.
    fn merge_ordered_ref_with_common_indices(
        &self,
        other: &Self,
    ) -> Result<(Self, BitVec, BitVec, MergeInfo), DuplicateItemError>
    where
        T: Ord + Clone,
    {
        // Reuse the more generic function with standard Ord comparison and equality
        self.merge_ordered_ref_with_comparison_and_matching(other, |a, b| a.cmp(b), |a, b| a == b)
    }

    /// Merges self with another vector, creating a BitVec that flags positions in the
    /// merged result that were deemed "common" at the point of their inclusion from self.
    /// An item from `self` is "common" if `self[i] == other[j]` when `self[i]` is chosen.
    /// Returns: (fully_merged_items, common_item_flags, merge_info_for_full_merge)
    fn merge_ref_flag_common_positions(&self, other: &Self) -> (Self, BitVec, MergeInfo)
    where
        T: Ord + Clone;

    /// Merges two ordered sequences using custom comparison and matching functions.
    /// - `compare`: A function that defines the ordering between elements (similar to Ord::cmp)
    /// - `is_match`: A function that determines if two elements should be considered "matching"
    ///   (and thus removed from both sequences) when they compare equal according to `compare`
    /// Returns: (merged_non_common_items, BitVec with bits set for common indices in first vec, BitVec with bits set for common indices in second vec, merge_info_for_non_common)
    fn merge_ordered_ref_with_comparison_and_matching<C, M>(
        &self,
        other: &Self,
        compare: C,
        is_match: M,
    ) -> Result<(Self, BitVec, BitVec, MergeInfo), DuplicateItemError>
    where
        T: Clone,
        C: Fn(&T, &T) -> std::cmp::Ordering,
        M: Fn(&T, &T) -> bool;
}

impl<T: Debug> MergeOrdered<T> for Vec<T> {
    fn merge_ordered_ref_with_common_removal(&self, other: &Self) -> (Self, Vec<T>, MergeInfo)
    where
        T: Ord + Clone,
    {
        debug_assert!(
            self.windows(2).all(|w| w[0] <= w[1]),
            "Input vector 'self' to merge_ordered_ref_with_common_removal must be sorted!"
        );
        debug_assert!(
            other.windows(2).all(|w| w[0] <= w[1]),
            "Input vector 'other' to merge_ordered_ref_with_common_removal must be sorted!"
        );

        let mut result_non_common = Vec::with_capacity(self.len() + other.len());
        let mut common_items = Vec::new();

        if self.is_empty() {
            result_non_common.extend_from_slice(other);
            let merge_info = if other.is_empty() {
                MergeInfo::FirstBeforeSecond
            } else {
                MergeInfo::SecondBeforeFirst
            };
            return (result_non_common, common_items, merge_info);
        }
        if other.is_empty() {
            result_non_common.extend_from_slice(self);
            return (
                result_non_common,
                common_items,
                MergeInfo::FirstBeforeSecond,
            );
        }

        let mut partition = BitVec::new(); // For non-common items result
        partition.reserve(self.len() + other.len());

        let mut i = 0;
        let mut j = 0;
        let mut transitions = 0;
        let mut last_added_to_result_from_self_val: Option<bool> = None;

        let manage_transition =
            |current_is_from_self: bool, last_opt: &mut Option<bool>, trans_count: &mut usize| {
                if let Some(last_val) = *last_opt {
                    if last_val != current_is_from_self {
                        *trans_count += 1;
                    }
                }
                *last_opt = Some(current_is_from_self);
            };

        let complete_merge_for_removal = |current_i: &mut usize,
                                          current_j: &mut usize,
                                          res: &mut Vec<T>,
                                          common_val_list: &mut Vec<T>,
                                          part: &mut BitVec,
                                          s_vec: &Self,
                                          o_vec: &Self| {
            while *current_i < s_vec.len() && *current_j < o_vec.len() {
                if s_vec[*current_i] < o_vec[*current_j] {
                    res.push(s_vec[*current_i].clone());
                    part.push(true);
                    *current_i += 1;
                } else if s_vec[*current_i] > o_vec[*current_j] {
                    res.push(o_vec[*current_j].clone());
                    part.push(false);
                    *current_j += 1;
                } else {
                    // Common
                    common_val_list.push(s_vec[*current_i].clone());
                    *current_i += 1;
                    *current_j += 1;
                }
            }
            while *current_i < s_vec.len() {
                res.push(s_vec[*current_i].clone());
                part.push(true);
                *current_i += 1;
            }
            while *current_j < o_vec.len() {
                res.push(o_vec[*current_j].clone());
                part.push(false);
                *current_j += 1;
            }
        };

        while i < self.len() && j < other.len() {
            if self[i] < other[j] {
                result_non_common.push(self[i].clone());
                partition.push(true);
                i += 1;
                manage_transition(
                    true,
                    &mut last_added_to_result_from_self_val,
                    &mut transitions,
                );
            } else if self[i] > other[j] {
                result_non_common.push(other[j].clone());
                partition.push(false);
                j += 1;
                manage_transition(
                    false,
                    &mut last_added_to_result_from_self_val,
                    &mut transitions,
                );
            } else {
                // Common item
                common_items.push(self[i].clone());
                i += 1;
                j += 1;
                continue;
            }

            if transitions > 1 {
                complete_merge_for_removal(
                    &mut i,
                    &mut j,
                    &mut result_non_common,
                    &mut common_items,
                    &mut partition,
                    self,
                    other,
                );
                return (
                    result_non_common,
                    common_items,
                    MergeInfo::Interleaved(partition),
                );
            }
        }

        while i < self.len() {
            result_non_common.push(self[i].clone());
            partition.push(true);
            i += 1;
            manage_transition(
                true,
                &mut last_added_to_result_from_self_val,
                &mut transitions,
            );
            if transitions > 1 {
                complete_merge_for_removal(
                    &mut i,
                    &mut j,
                    &mut result_non_common,
                    &mut common_items,
                    &mut partition,
                    self,
                    other,
                );
                return (
                    result_non_common,
                    common_items,
                    MergeInfo::Interleaved(partition),
                );
            }
        }
        while j < other.len() {
            result_non_common.push(other[j].clone());
            partition.push(false);
            j += 1;
            manage_transition(
                false,
                &mut last_added_to_result_from_self_val,
                &mut transitions,
            );
            if transitions > 1 {
                complete_merge_for_removal(
                    &mut i,
                    &mut j,
                    &mut result_non_common,
                    &mut common_items,
                    &mut partition,
                    self,
                    other,
                );
                return (
                    result_non_common,
                    common_items,
                    MergeInfo::Interleaved(partition),
                );
            }
        }

        if result_non_common.is_empty() {
            (
                result_non_common,
                common_items,
                MergeInfo::Interleaved(BitVec::new()),
            )
        } else {
            if partition[0] {
                (
                    result_non_common,
                    common_items,
                    MergeInfo::FirstBeforeSecond,
                )
            } else {
                (
                    result_non_common,
                    common_items,
                    MergeInfo::SecondBeforeFirst,
                )
            }
        }
    }

    fn merge_ref_flag_common_positions(&self, other: &Self) -> (Self, BitVec, MergeInfo)
    where
        T: Ord + Clone,
    {
        debug_assert!(
            self.windows(2).all(|w| w[0] <= w[1]),
            "Input vector 'self' to merge_ref_flag_common_positions must be sorted!"
        );
        debug_assert!(
            other.windows(2).all(|w| w[0] <= w[1]),
            "Input vector 'other' to merge_ref_flag_common_positions must be sorted!"
        );

        let mut result_merged_all = Vec::with_capacity(self.len() + other.len());
        let mut common_item_flags = BitVec::new();
        common_item_flags.reserve(self.len() + other.len());

        if self.is_empty() {
            result_merged_all.extend_from_slice(other);
            common_item_flags.resize(other.len(), false);
            let merge_info = if other.is_empty() {
                MergeInfo::FirstBeforeSecond
            } else {
                MergeInfo::SecondBeforeFirst
            };
            return (result_merged_all, common_item_flags, merge_info);
        }
        if other.is_empty() {
            result_merged_all.extend_from_slice(self);
            common_item_flags.resize(self.len(), false);
            return (
                result_merged_all,
                common_item_flags,
                MergeInfo::FirstBeforeSecond,
            );
        }

        let mut partition = BitVec::new();
        partition.reserve(self.len() + other.len());
        let mut i = 0;
        let mut j = 0;
        let mut transitions = 0;
        let mut last_added_to_result_from_self_val: Option<bool> = None;

        let manage_transition =
            |current_is_from_self: bool, last_opt: &mut Option<bool>, trans_count: &mut usize| {
                if let Some(last_val) = *last_opt {
                    if last_val != current_is_from_self {
                        *trans_count += 1;
                    }
                }
                *last_opt = Some(current_is_from_self);
            };

        let complete_full_merge_flagging =
            |current_i: &mut usize,
             current_j: &mut usize,
             res: &mut Vec<T>,
             common_flags: &mut BitVec,
             part: &mut BitVec,
             s_vec: &Self,
             o_vec: &Self| {
                while *current_i < s_vec.len() && *current_j < o_vec.len() {
                    if s_vec[*current_i] < o_vec[*current_j] {
                        res.push(s_vec[*current_i].clone());
                        part.push(true);
                        common_flags.push(false);
                        *current_i += 1;
                    } else if s_vec[*current_i] > o_vec[*current_j] {
                        res.push(o_vec[*current_j].clone());
                        part.push(false);
                        common_flags.push(false);
                        *current_j += 1;
                    } else {
                        res.push(s_vec[*current_i].clone());
                        part.push(true);
                        common_flags.push(true);
                        *current_i += 1;
                    }
                }
                while *current_i < s_vec.len() {
                    res.push(s_vec[*current_i].clone());
                    part.push(true);
                    common_flags.push(false);
                    *current_i += 1;
                }
                while *current_j < o_vec.len() {
                    res.push(o_vec[*current_j].clone());
                    part.push(false);
                    common_flags.push(false);
                    *current_j += 1;
                }
            };

        while i < self.len() && j < other.len() {
            let current_element_is_from_self: bool;
            if self[i] < other[j] {
                result_merged_all.push(self[i].clone());
                partition.push(true);
                common_item_flags.push(false);
                i += 1;
                current_element_is_from_self = true;
            } else if self[i] > other[j] {
                result_merged_all.push(other[j].clone());
                partition.push(false);
                common_item_flags.push(false);
                j += 1;
                current_element_is_from_self = false;
            } else {
                result_merged_all.push(self[i].clone());
                partition.push(true);
                common_item_flags.push(true);
                i += 1;
                current_element_is_from_self = true;
            }

            manage_transition(
                current_element_is_from_self,
                &mut last_added_to_result_from_self_val,
                &mut transitions,
            );

            if transitions > 1 {
                complete_full_merge_flagging(
                    &mut i,
                    &mut j,
                    &mut result_merged_all,
                    &mut common_item_flags,
                    &mut partition,
                    self,
                    other,
                );
                return (
                    result_merged_all,
                    common_item_flags,
                    MergeInfo::Interleaved(partition),
                );
            }
        }

        while i < self.len() {
            result_merged_all.push(self[i].clone());
            partition.push(true);
            common_item_flags.push(false);
            i += 1;
            manage_transition(
                true,
                &mut last_added_to_result_from_self_val,
                &mut transitions,
            );
            if transitions > 1 {
                complete_full_merge_flagging(
                    &mut i,
                    &mut j,
                    &mut result_merged_all,
                    &mut common_item_flags,
                    &mut partition,
                    self,
                    other,
                );
                return (
                    result_merged_all,
                    common_item_flags,
                    MergeInfo::Interleaved(partition),
                );
            }
        }
        while j < other.len() {
            result_merged_all.push(other[j].clone());
            partition.push(false);
            common_item_flags.push(false);
            j += 1;
            manage_transition(
                false,
                &mut last_added_to_result_from_self_val,
                &mut transitions,
            );
            if transitions > 1 {
                complete_full_merge_flagging(
                    &mut i,
                    &mut j,
                    &mut result_merged_all,
                    &mut common_item_flags,
                    &mut partition,
                    self,
                    other,
                );
                return (
                    result_merged_all,
                    common_item_flags,
                    MergeInfo::Interleaved(partition),
                );
            }
        }

        if result_merged_all.is_empty() {
            (
                result_merged_all,
                common_item_flags,
                MergeInfo::Interleaved(BitVec::new()),
            )
        } else {
            if partition[0] {
                (
                    result_merged_all,
                    common_item_flags,
                    MergeInfo::FirstBeforeSecond,
                )
            } else {
                (
                    result_merged_all,
                    common_item_flags,
                    MergeInfo::SecondBeforeFirst,
                )
            }
        }
    }

    fn merge_ordered_ref_with_comparison_and_matching<C, M>(
        &self,
        other: &Self,
        compare: C,
        is_match: M,
    ) -> Result<(Self, BitVec, BitVec, MergeInfo), DuplicateItemError>
    where
        T: Clone,
        C: Fn(&T, &T) -> std::cmp::Ordering,
        M: Fn(&T, &T) -> bool,
    {
        // Check for duplicates in self
        for i in 1..self.len() {
            if compare(&self[i - 1], &self[i]) == std::cmp::Ordering::Equal
                && is_match(&self[i - 1], &self[i])
            {
                return Err(DuplicateItemError {
                    message: format!(
                        "Found duplicate item in first sequence: {:?} in {self:?}",
                        self[i - 1]
                    ),
                });
            }
        }

        // Check for duplicates in other
        for i in 1..other.len() {
            if compare(&other[i - 1], &other[i]) == std::cmp::Ordering::Equal
                && is_match(&other[i - 1], &other[i])
            {
                return Err(DuplicateItemError {
                    message: format!(
                        "Found duplicate item in second sequence: {:?} in {other:?}",
                        other[i - 1]
                    ),
                });
            }
        }

        debug_assert!(
            self.windows(2).all(|w| compare(&w[0], &w[1]).is_le()),
            "Input vector 'self' to merge_ordered_ref_with_common_indices must be sorted according to compare function"
        );
        debug_assert!(
            other.windows(2).all(|w| compare(&w[0], &w[1]).is_le()),
            "Input vector 'other' to merge_ordered_ref_with_common_indices must be sorted according to compare function"
        );

        let mut result_non_common = Vec::with_capacity(self.len() + other.len());
        let mut common_indices_self = BitVec::with_capacity(self.len()); // BitVec for common elements in self
        common_indices_self.resize(self.len(), false);
        let mut common_indices_other = BitVec::with_capacity(other.len()); // BitVec for common elements in other
        common_indices_other.resize(other.len(), false);

        if self.is_empty() || other.is_empty() {
            // If either is empty, there are no common elements
            // Just merge the non-empty vector into result_non_common
            if self.is_empty() {
                result_non_common.extend_from_slice(other);
                return Ok((
                    result_non_common,
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::SecondBeforeFirst,
                ));
            } else {
                result_non_common.extend_from_slice(self);
                return Ok((
                    result_non_common,
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::FirstBeforeSecond,
                ));
            }
        }

        let mut partition = BitVec::new(); // For non-common items result
        partition.reserve(self.len() + other.len());

        let mut i = 0;
        let mut j = 0;
        let mut transitions = 0;
        let mut last_added_to_result_from_self_val: Option<bool> = None;

        let manage_transition =
            |current_is_from_self: bool, last_opt: &mut Option<bool>, trans_count: &mut usize| {
                if let Some(last_val) = *last_opt {
                    if last_val != current_is_from_self {
                        *trans_count += 1;
                    }
                }
                *last_opt = Some(current_is_from_self);
            };

        let complete_merge = |current_i: &mut usize,
                              current_j: &mut usize,
                              res: &mut Vec<T>,
                              common_idx_self: &mut BitVec,
                              common_idx_other: &mut BitVec,
                              part: &mut BitVec,
                              s_vec: &Self,
                              o_vec: &Self| {
            while *current_i < s_vec.len() && *current_j < o_vec.len() {
                match compare(&s_vec[*current_i], &o_vec[*current_j]) {
                    std::cmp::Ordering::Less => {
                        res.push(s_vec[*current_i].clone());
                        part.push(true);
                        *current_i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        res.push(o_vec[*current_j].clone());
                        part.push(false);
                        *current_j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        if is_match(&s_vec[*current_i], &o_vec[*current_j]) {
                            // Common item found
                            common_idx_self.set(*current_i, true); // Set bit for index in first vector
                            common_idx_other.set(*current_j, true); // Set bit for index in second vector
                            *current_i += 1;
                            *current_j += 1;
                        } else {
                            // They compare equal but aren't matches
                            res.push(s_vec[*current_i].clone());
                            part.push(true);
                            *current_i += 1;

                            res.push(o_vec[*current_j].clone());
                            part.push(false);
                            *current_j += 1;
                        }
                    }
                }
            }
            while *current_i < s_vec.len() {
                res.push(s_vec[*current_i].clone());
                part.push(true);
                *current_i += 1;
            }
            while *current_j < o_vec.len() {
                res.push(o_vec[*current_j].clone());
                part.push(false);
                *current_j += 1;
            }
        };

        while i < self.len() && j < other.len() {
            match compare(&self[i], &other[j]) {
                std::cmp::Ordering::Less => {
                    result_non_common.push(self[i].clone());
                    partition.push(true);
                    i += 1;
                    manage_transition(
                        true,
                        &mut last_added_to_result_from_self_val,
                        &mut transitions,
                    );
                }
                std::cmp::Ordering::Greater => {
                    result_non_common.push(other[j].clone());
                    partition.push(false);
                    j += 1;
                    manage_transition(
                        false,
                        &mut last_added_to_result_from_self_val,
                        &mut transitions,
                    );
                }
                std::cmp::Ordering::Equal => {
                    if is_match(&self[i], &other[j]) {
                        // Common item
                        common_indices_self.set(i, true); // Set bit for index in first vector
                        common_indices_other.set(j, true); // Set bit for index in second vector
                        i += 1;
                        j += 1;
                        continue; // Skip transition check for common items since they're not added to result_non_common
                    } else {
                        // They compare equal by sort order but don't match by our criterion
                        result_non_common.push(self[i].clone());
                        partition.push(true);
                        i += 1;

                        result_non_common.push(other[j].clone());
                        partition.push(false);
                        j += 1;

                        manage_transition(
                            true,
                            &mut last_added_to_result_from_self_val,
                            &mut transitions,
                        );
                        manage_transition(
                            false,
                            &mut last_added_to_result_from_self_val,
                            &mut transitions,
                        );
                    }
                }
            }

            if transitions > 1 {
                complete_merge(
                    &mut i,
                    &mut j,
                    &mut result_non_common,
                    &mut common_indices_self,
                    &mut common_indices_other,
                    &mut partition,
                    self,
                    other,
                );
                return Ok((
                    result_non_common,
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::Interleaved(partition),
                ));
            }
        }

        while i < self.len() {
            result_non_common.push(self[i].clone());
            partition.push(true);
            i += 1;
            manage_transition(
                true,
                &mut last_added_to_result_from_self_val,
                &mut transitions,
            );
            if transitions > 1 {
                complete_merge(
                    &mut i,
                    &mut j,
                    &mut result_non_common,
                    &mut common_indices_self,
                    &mut common_indices_other,
                    &mut partition,
                    self,
                    other,
                );
                return Ok((
                    result_non_common,
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::Interleaved(partition),
                ));
            }
        }
        while j < other.len() {
            result_non_common.push(other[j].clone());
            partition.push(false);
            j += 1;
            manage_transition(
                false,
                &mut last_added_to_result_from_self_val,
                &mut transitions,
            );
            if transitions > 1 {
                complete_merge(
                    &mut i,
                    &mut j,
                    &mut result_non_common,
                    &mut common_indices_self,
                    &mut common_indices_other,
                    &mut partition,
                    self,
                    other,
                );
                return Ok((
                    result_non_common,
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::Interleaved(partition),
                ));
            }
        }

        if result_non_common.is_empty() {
            Ok((
                result_non_common,
                common_indices_self,
                common_indices_other,
                MergeInfo::Interleaved(BitVec::new()),
            ))
        } else {
            if partition[0] {
                Ok((
                    result_non_common,
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::FirstBeforeSecond,
                ))
            } else {
                Ok((
                    result_non_common,
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::SecondBeforeFirst,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn test_merge_with_custom_comparison_and_matching() {
        // Test with custom comparison (absolute value) and matching (sum to zero)
        let first = vec![1, -2, -3, 4, 5];
        let second = vec![1, 2, 3, -4, -7];

        let compare = |a: &i32, b: &i32| a.abs().cmp(&b.abs());
        let is_match = |a: &i32, b: &i32| a + b == 0;

        let result = first
            .merge_ordered_ref_with_comparison_and_matching(&second, compare, is_match)
            .unwrap();
        let (merged, self_common, other_common, _) = result;

        // -3 matches with 3, -2 matches with 2, , -4 matches with 4
        assert_eq!(self_common.count_ones(), 3); // 4 matches in first array
        assert_eq!(other_common.count_ones(), 3); // 4 matches in second array

        // Extract non-common elements by checking self_common and other_common
        let left: Vec<i32> = first
            .iter()
            .enumerate()
            .filter(|(i, _)| !self_common[*i])
            .map(|(_, v)| *v)
            .collect();
        let right: Vec<i32> = second
            .iter()
            .enumerate()
            .filter(|(i, _)| !other_common[*i])
            .map(|(_, v)| *v)
            .collect();

        assert_eq!(left, vec![1, 5]);
        assert_eq!(right, vec![1, -7]);

        // Test case with tuples where we compare first element but match on second
        let first = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let second = vec![(1, 'x'), (2, 'b'), (3, 'y')];

        let compare = |a: &(i32, char), b: &(i32, char)| a.0.cmp(&b.0);
        let is_match = |a: &(i32, char), b: &(i32, char)| a.1 == b.1;

        let result = first
            .merge_ordered_ref_with_comparison_and_matching(&second, compare, is_match)
            .unwrap();
        let (merged, self_common, other_common, _) = result;

        // Only the tuple with 'b' matches
        let left: Vec<(i32, char)> = first
            .iter()
            .enumerate()
            .filter(|(i, _)| !self_common[*i])
            .map(|(_, v)| *v)
            .collect();
        let right: Vec<(i32, char)> = second
            .iter()
            .enumerate()
            .filter(|(i, _)| !other_common[*i])
            .map(|(_, v)| *v)
            .collect();

        assert_eq!(left, vec![(1, 'a'), (3, 'c')]);
        assert_eq!(right, vec![(1, 'x'), (3, 'y')]);

        // Test with empty vectors
        let first: Vec<i32> = Vec::new();
        let second: Vec<i32> = Vec::new();

        let result = first
            .merge_ordered_ref_with_comparison_and_matching(&second, |a, b| a.cmp(b), |a, b| a == b)
            .unwrap();

        let (merged, self_common, other_common, _) = result;
        assert!(merged.is_empty());
        assert!(self_common.is_empty());
        assert!(other_common.is_empty());
    }

    // Tests for merge_ordered_ref_with_common_indices
    #[test]
    fn test_merge_with_indices_no_common() {
        let vec_a = vec![1, 3, 5];
        let vec_b = vec![2, 4, 6];
        let result = vec_a.merge_ordered_ref_with_common_indices(&vec_b);
        assert!(result.is_ok());
        let (res, common_indices_a, common_indices_b, info) = result.unwrap();
        assert_eq!(res, vec![1, 2, 3, 4, 5, 6]);
        let mut expected_a: BitVec = BitVec::with_capacity(3);
        expected_a.resize(3, false);
        let mut expected_b: BitVec = BitVec::with_capacity(3);
        expected_b.resize(3, false);
        assert_eq!(common_indices_a, expected_a);
        assert_eq!(common_indices_b, expected_b);
        let mut expected_interleaved = BitVec::with_capacity(6);
        expected_interleaved.push(true);
        expected_interleaved.push(false);
        expected_interleaved.push(true);
        expected_interleaved.push(false);
        expected_interleaved.push(true);
        expected_interleaved.push(false);
        assert_eq!(info, MergeInfo::Interleaved(expected_interleaved));
    }

    #[test]
    fn test_merge_with_indices_with_common() {
        let vec_a = vec![1, 3, 5, 7];
        let vec_b = vec![2, 3, 6, 7];
        let result = vec_a.merge_ordered_ref_with_common_indices(&vec_b);
        assert!(result.is_ok());
        let (res, common_indices_a, common_indices_b, info) = result.unwrap();
        assert_eq!(res, vec![1, 2, 5, 6]);
        let mut expected_a: BitVec = BitVec::with_capacity(4);
        expected_a.resize(4, false);
        expected_a.set(1, true);
        expected_a.set(3, true);
        let mut expected_b: BitVec = BitVec::with_capacity(4);
        expected_b.resize(4, false);
        expected_b.set(1, true);
        expected_b.set(3, true);
        assert_eq!(common_indices_a, expected_a); // 3 is at index 1, 7 is at index 3 in vec_a
        assert_eq!(common_indices_b, expected_b); // 3 is at index 1, 7 is at index 3 in vec_b

        let mut expected_interleaved = BitVec::with_capacity(4);
        expected_interleaved.push(true);
        expected_interleaved.push(false);
        expected_interleaved.push(true);
        expected_interleaved.push(false);
        assert_eq!(info, MergeInfo::Interleaved(expected_interleaved));
    }

    #[test]
    fn test_merge_with_indices_empty_vectors() {
        let vec_a: Vec<i32> = vec![];
        let vec_b = vec![2, 4, 6];
        let result = vec_a.merge_ordered_ref_with_common_indices(&vec_b);
        assert!(result.is_ok());
        let (res, common_indices_a, common_indices_b, info) = result.unwrap();
        assert_eq!(res, vec![2, 4, 6]);
        let mut expected_a: BitVec = BitVec::with_capacity(0);
        // expected_a.resize(0, false);
        let mut expected_b: BitVec = BitVec::with_capacity(3);
        expected_b.resize(3, false);
        assert_eq!(common_indices_a, expected_a);
        assert_eq!(common_indices_b, expected_b);
        assert_eq!(info, MergeInfo::SecondBeforeFirst);

        let vec_a = vec![1, 3, 5];
        let vec_b: Vec<i32> = vec![];
        let result = vec_a.merge_ordered_ref_with_common_indices(&vec_b);
        assert!(result.is_ok());
        let (res, common_indices_a, common_indices_b, info) = result.unwrap();
        assert_eq!(res, vec![1, 3, 5]);
        let mut expected_a: BitVec = BitVec::with_capacity(3);
        expected_a.resize(3, false);
        let mut expected_b: BitVec = BitVec::with_capacity(0);
        // expected_b.resize(3, false);
        assert_eq!(common_indices_a, expected_a);
        assert_eq!(common_indices_b, expected_b);
        assert_eq!(info, MergeInfo::FirstBeforeSecond);
    }

    #[test]
    fn test_merge_with_indices_error_duplicate_in_first() {
        let vec_a = vec![1, 2, 2, 3]; // Duplicate 2
        let vec_b = vec![4, 5, 6];
        let result = vec_a.merge_ordered_ref_with_common_indices(&vec_b);
        assert!(result.is_err());
        // let error = result.unwrap_err();
    }

    #[test]
    fn test_merge_with_indices_error_duplicate_in_second() {
        let vec_a = vec![1, 4, 5];
        let vec_b = vec![2, 3, 3, 6]; // Duplicate 3
        let result = vec_a.merge_ordered_ref_with_common_indices(&vec_b);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_with_indices_clean_split() {
        let vec_a = vec![1, 2, 3];
        let vec_b = vec![4, 5, 6];
        let result = vec_a.merge_ordered_ref_with_common_indices(&vec_b);
        assert!(result.is_ok());
        let (res, common_indices_a, common_indices_b, info) = result.unwrap();
        assert_eq!(res, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(common_indices_a, bitvec![0, 0, 0]);
        assert_eq!(common_indices_b, bitvec![0, 0, 0]);
        assert_eq!(info, MergeInfo::FirstBeforeSecond);

        // Common split case 2: SecondBeforeFirst
        let vec_c = vec![4, 5, 6];
        let vec_d = vec![1, 2, 3];
        let result2 = vec_c.merge_ordered_ref_with_common_indices(&vec_d);
        assert!(result2.is_ok());
        let (res2, common_indices_c, common_indices_d, info2) = result2.unwrap();
        assert_eq!(res2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(common_indices_c, bitvec![0, 0, 0]);
        assert_eq!(common_indices_d, bitvec![0, 0, 0]);
        assert_eq!(info2, MergeInfo::SecondBeforeFirst);
    }

    #[test]
    fn test_merge_with_removal_no_common() {
        let vec_a = vec![1, 3, 5];
        let vec_b = vec![2, 4, 6];
        let (res, common, info) = vec_a.merge_ordered_ref_with_common_removal(&vec_b);
        assert_eq!(res, vec![1, 2, 3, 4, 5, 6]);
        assert!(common.is_empty());
        assert_eq!(info, MergeInfo::Interleaved(bitvec![1, 0, 1, 0, 1, 0]));
    }

    #[test]
    fn test_merge_with_removal_all_common() {
        let vec_a = vec![1, 2, 3];
        let vec_b = vec![1, 2, 3];
        let (res, common, info) = vec_a.merge_ordered_ref_with_common_removal(&vec_b);
        assert!(res.is_empty());
        assert_eq!(common, vec![1, 2, 3]);
        assert_eq!(info, MergeInfo::Interleaved(BitVec::new()));
    }

    #[test]
    fn test_merge_with_removal_some_common() {
        let vec_a = vec![1, 2, 4, 5, 7];
        let vec_b = vec![2, 3, 5, 6, 7];
        let (res, common, info) = vec_a.merge_ordered_ref_with_common_removal(&vec_b);
        assert_eq!(res, vec![1, 3, 4, 6]);
        assert_eq!(common, vec![2, 5, 7]);
        assert_eq!(info, MergeInfo::Interleaved(bitvec![1, 0, 1, 0]));
    }

    #[test]
    fn test_merge_with_removal_a_empty() {
        let vec_a: Vec<i32> = vec![];
        let vec_b = vec![1, 2, 3];
        let (res, common, info) = vec_a.merge_ordered_ref_with_common_removal(&vec_b);
        assert_eq!(res, vec![1, 2, 3]);
        assert!(common.is_empty());
        assert_eq!(info, MergeInfo::SecondBeforeFirst);
    }

    #[test]
    fn test_merge_with_removal_b_empty() {
        let vec_a = vec![1, 2, 3];
        let vec_b: Vec<i32> = vec![];
        let (res, common, info) = vec_a.merge_ordered_ref_with_common_removal(&vec_b);
        assert_eq!(res, vec![1, 2, 3]);
        assert!(common.is_empty());
        assert_eq!(info, MergeInfo::FirstBeforeSecond);
    }

    #[test]
    fn test_merge_with_removal_both_empty() {
        let vec_a: Vec<i32> = vec![];
        let vec_b: Vec<i32> = vec![];
        let (res, common, info) = vec_a.merge_ordered_ref_with_common_removal(&vec_b);
        assert!(res.is_empty());
        assert!(common.is_empty());
        assert_eq!(info, MergeInfo::FirstBeforeSecond);
    }

    #[test]
    fn test_merge_clean_split_first_before_second_with_common() {
        let vec_a = vec![1, 2, 5];
        let vec_b = vec![5, 6, 7];
        let (res, common, info) = vec_a.merge_ordered_ref_with_common_removal(&vec_b);
        assert_eq!(res, vec![1, 2, 6, 7]);
        assert_eq!(common, vec![5]);
        assert_eq!(info, MergeInfo::FirstBeforeSecond);
    }

    #[test]
    fn test_merge_clean_split_second_before_first_with_common() {
        let vec_a = vec![5, 6, 7];
        let vec_b = vec![1, 2, 5];
        let (res, common, info) = vec_a.merge_ordered_ref_with_common_removal(&vec_b);
        assert_eq!(res, vec![1, 2, 6, 7]);
        assert_eq!(common, vec![5]);
        assert_eq!(info, MergeInfo::SecondBeforeFirst);
    }

    #[test]
    fn test_flag_common_no_common_items() {
        let vec_a = vec![1, 3, 5];
        let vec_b = vec![2, 4, 6];
        let (res, flags, info) = vec_a.merge_ref_flag_common_positions(&vec_b);
        assert_eq!(res, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(flags, bitvec![0, 0, 0, 0, 0, 0]);
        assert_eq!(info, MergeInfo::Interleaved(bitvec![1, 0, 1, 0, 1, 0]));
    }

    #[test]
    fn test_flag_common_all_items_equal_simple() {
        let vec_a = vec![5];
        let vec_b = vec![5];
        let (res, flags, info) = vec_a.merge_ref_flag_common_positions(&vec_b);
        assert_eq!(res, vec![5, 5]);
        assert_eq!(flags, bitvec![1, 0]);
        assert_eq!(info, MergeInfo::FirstBeforeSecond);
    }

    #[test]
    fn test_flag_common_all_items_equal_multiple() {
        let vec_a = vec![2, 2];
        let vec_b = vec![2, 2];
        let (res, flags, info) = vec_a.merge_ref_flag_common_positions(&vec_b);
        assert_eq!(res, vec![2, 2, 2, 2]);
        assert_eq!(flags, bitvec![1, 1, 0, 0]);
        assert_eq!(info, MergeInfo::FirstBeforeSecond);
    }

    #[test]
    fn test_flag_common_some_matches_interleaved() {
        let vec_a = vec![1, 2, 4, 5, 7];
        let vec_b = vec![2, 3, 5, 6, 7];
        let (res, flags, info) = vec_a.merge_ref_flag_common_positions(&vec_b);
        assert_eq!(res, vec![1, 2, 2, 3, 4, 5, 5, 6, 7, 7]);
        assert_eq!(flags, bitvec![0, 1, 0, 0, 0, 1, 0, 0, 1, 0]);
        assert_eq!(
            info,
            MergeInfo::Interleaved(bitvec![1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
        );
    }

    #[test]
    fn test_flag_common_a_empty() {
        let vec_a: Vec<i32> = vec![];
        let vec_b = vec![1, 2, 3];
        let (res, flags, info) = vec_a.merge_ref_flag_common_positions(&vec_b);
        assert_eq!(res, vec![1, 2, 3]);
        assert_eq!(flags, bitvec![0, 0, 0]);
        assert_eq!(info, MergeInfo::SecondBeforeFirst);
    }

    #[test]
    fn test_flag_common_b_empty() {
        let vec_a = vec![1, 2, 3];
        let vec_b: Vec<i32> = vec![];
        let (res, flags, info) = vec_a.merge_ref_flag_common_positions(&vec_b);
        assert_eq!(res, vec![1, 2, 3]);
        assert_eq!(flags, bitvec![0, 0, 0]);
        assert_eq!(info, MergeInfo::FirstBeforeSecond);
    }
}
