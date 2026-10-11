export apply_action, difference_polynomial_ring, differential_polynomial_ring,
  action_shift, action_derivation, action_map, action_maps
# export action_maps_commute_on

###############################################################################
#
# Candidate public API exports
#
# This is a proposed export list for a standalone package/module or for an
# Oscar integration pass.  It exports public user-facing functions only, not
# helper functions whose names start with `__`.
#
###############################################################################


export AbstractPolynomialSystem, AbstractActionSystem, AlgebraicSystem, DifferenceSystem, DifferentialSystem, #OrdinaryWindowBranch,
  leader,
  initial,
  multiplicative_variables,
  cones_check,
  is_janet_complete,
  janet_completion

# Disabled together with Difference/Janet-Difference.jl in Code/Load.jl.
# export which_elementary_symbol,
#   which_shift,
#   partition_set_difference_polynomial,
#   indexed_partition_set_difference_polynomial

export is_janet_reducible,
  is_janet_reducible_univariate,
  normal_form,
  is_passive,
  normal_form_terms
# export is_admissible # Restore when FOR LATER/Normal-Form.jl is loaded.

export leader_filter,
  solution_check,
  is_quasi_simple,
  is_quasi_simple_as_algebraic_system,
  is_simple_as_algebraic_system

# export partition_position,
#  nonnegative_compositions,
#  var_list_complete,
#  conversion_difference_algebraic,
#  conversion_differential_algebraic
#  leader_condition # Restore when its implementation is loaded.
#  shift_lower,
#  multiplicative_shift_lower,
#  partial_set_sol,
#  shift_partial_set_sol,
#  tilde_shift_partial_set_sol

export quasi_simple_splitting

export thomas_decomposition,
  branch_conversion, system_to_vectors,
  print_splitting_result, algebraic_branch_intersection, splitting_covers_system,
  is_simple
#  ordinary_local_splitting, ordinary_split, ordinary_window_data,
#  ordinary_local_conditions, quasi_simple_splitting_ordinary,  is_algorithmically_quasi_simple, is_algorithmically_quasi_simple_ordinary
