###############################################################################
#
# This solves the problem of the following example:
# Examples
@doc raw"""
```jldoctest
 julia> R, (y_2, y_1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :deglex);
 julia> f_1 = diff_action(y_2, [0, 1])^3+y_2;
 julia> f_2 = y_1^2*diff_action(y_2, [1, 0]) + diff_action(y_1, [0, 1])-1;
 julia> f_3 = diff_action(y_1, [1, 0])^4+diff_action(y_1, [0, 1])*y_2^2;
 julia> f_4 = diff_action(y_1, [0, 2])*diff_action(y_2, [1, 1])+diff_action(y_1, [0, 1])+1;
 julia> F = [f_1,f_2,f_3,f_4]
 4-element Vector{DifferencePolyRingElem{QQFieldElem}}:
   y2[0,1]^3 + y2[0,0]
   -1
   y1[1,0]^4 + y1[0,1]*y2[0,0]^2
   y1[0,2]*y2[1,1] + y1[0,1] + 1
```
"""
###############################################################################

import Oscar
import Oscar:
  __set_is_perm_up_to_date!, coeff, coefficients, data, exponents, parent, symbols

using Oscar:
  ActionPolyRingElem,
  DifferencePolyRingElem,
  DifferentialPolyRingElem

if !isdefined(@__MODULE__, :__delete_method!)
  @doc raw"""
    __delete_method!(f, argtypes::Type)

Temporarily used to redefine methods already present in OSCAR.
  """
  function __delete_method!(f, argtypes::Type)
    for m in collect(methods(f, argtypes))
      Base.delete_method(m)
    end
    return nothing
  end
end

###############################################################################
###############################################################################
#
# Replace
#
#   __set_perm_for_sort_poly!
#   __my_lt_for_vec
#
# in experimental/ActionPolyRing/src/Content.jl.
#
# The old code uses
#
#   perm = parent(dpre).permutation[1:min(end, length(exps[1]))]
#
# This takes the first entries of the parent-ring permutation. After the parent
# gets new jet variables, those first entries can be indices larger than the
# exponent vectors of an older polynomial.  Then __my_lt_for_vec reads outside
# the exponent vector, currently under @inbounds, so the order of terms becomes
# undefined.  In the example from the prompt, the constant term becomes the
# leading term, and expressify prints only `-1`.
#
###############################################################################

__delete_method!(
  Oscar.__set_perm_for_sort_poly!,
  Tuple{Union{DifferencePolyRingElem, DifferentialPolyRingElem}},
)

__delete_method!(Oscar.__my_lt_for_vec, Tuple{Vector{Int}})


function Oscar.__set_perm_for_sort_poly!(
  dpre::Union{DifferencePolyRingElem, DifferentialPolyRingElem},
)
  exps = collect(exponents(data(dpre)))
  n = length(exps)
  if n <= 1
    dpre.permutation = collect(1:n)
    return __set_is_perm_up_to_date!(dpre, true)
  end

  # I have changed the commented line with the following two:
  # perm = (parent(dpre).permutation)[1:min(end, length(exps[1]))]
  max_exp_len = maximum(length, exps)
  # The parent-ring permutation may contain jet variables added after this
  # polynomial was constructed.  Its first entries can then be larger than the
  # exponent-vector length of this polynomial, so slicing the first
  # length(exps[1]) entries can leave invalid indices and make term ordering
  # undefined. Instead, keep the relative ranking order and discard precisely
  # the indices that do not exist in this polynomial's exponent vectors.
  perm = filter(k -> k <= max_exp_len, Oscar.__perm_for_sort(parent(dpre)))

  dpre.permutation = sortperm(exps; lt = Oscar.__my_lt_for_vec(perm), rev = true)
  __set_is_perm_up_to_date!(dpre, true)
end

function Oscar.__my_lt_for_vec(perm::Vector{Int})
  return function ___my_lt_for_vec(ei::Vector{Int}, ej::Vector{Int})
    @inbounds for k in perm
      vi = k <= length(ei) ? ei[k] : 0
      vj = k <= length(ej) ? ej[k] : 0
      vi != vj && return vi < vj
    end
    return false
  end
end