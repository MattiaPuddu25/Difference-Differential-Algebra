###############################################################################
#####  Importing OSCAR + Internal Helpers
###############################################################################

import Oscar
import Oscar:
  base_ring,
  coeff, coefficient_ring,
  data, diff_action, differential_polynomial_ring,
  elem_type, exponent_vector,
  finish,
  gen, gens,
  is_zero,
  ngens, n_action_maps,
  parent, push_term!,
  terms,
  vars
using Oscar:
  DifferentialPolyRing,
  DifferentialPolyRingElem,
  MPolyBuildCtx,
  Ring

if !isdefined(@__MODULE__, :_ACTION_POLY_BASE_DERIVATIONS)
  _ACTION_POLY_BASE_DERIVATIONS = IdDict{Any,Vector{Any}}()
end

if !isdefined(@__MODULE__, :__delete_method!)
  function __delete_method!(f::Function, argtypes::Type)
    for m in collect(methods(f, argtypes))
      Base.delete_method(m)
    end
    return nothing
  end
end

function __zero_base_derivation(c)
  return zero(parent(c))
end

######################################################################################################################################
######################################################################################################################################

function __base_derivation_commutativity_test_elements(K)
  elements = Any[zero(K), one(K)]
  if applicable(gens, K)
    append!(elements, gens(K))
  elseif applicable(gen, K)
    push!(elements, gen(K))
  end
  return elements
end

function __base_derivations_commute_on(derivations, K, elements)
  elements = collect(elements)
  for i in eachindex(derivations)
    for j in (i + 1):length(derivations)
      for c in elements
        lhs = K(derivations[i](K(derivations[j](K(c)))))
        rhs = K(derivations[j](K(derivations[i](K(c)))))
        lhs == rhs || return false
      end
    end
  end
  return true
end

@doc raw"""
    __set_base_derivations!(
      dpr::DifferentialPolyRing,
      derivations::Union{AbstractVector,Tuple}
    )

Attach the coefficient-ring derivations in `derivations` to the differential polynomial ring `dpr`.

This is an internal helper used in differential_polynomial_ring(K, ..., derivations).
The vector `derivations` must have one entry for each action map of `dpr`.
Before storing them, the function checks pairwise commutativity on `0`, `1`,
  and the generators returned by `gens(coefficient_ring(dpr))` or
  `gen(coefficient_ring(dpr))`, when available.
This proves commutativity provided those elements
  generate the coefficient ring over the common field of constants.

The data are stored in the internal dictionary _ACTION_POLY_BASE_DERIVATIONS,
  and are later read by base_derivations(dpr) and by the extended diff_action.

The function returns `dpr`.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 1);
julia> __set_base_derivations!(R, [__zero_base_derivation]) === R
true
julia> base_derivations(R)
1-element Vector{Any}:
 __zero_base_derivation (generic function with 1 method)
```
"""
function __set_base_derivations!(
  dpr::DifferentialPolyRing,
  derivations::Union{AbstractVector,Tuple},
)
  derivations = collect(Any, derivations)
  length(derivations) == n_action_maps(dpr) ||
    error("The number of base-ring derivations must equal the number of action maps")

  K = coefficient_ring(dpr)
  test_elements = __base_derivation_commutativity_test_elements(K)
  __base_derivations_commute_on(derivations, K, test_elements) ||
    error("The base-ring derivations must commute pairwise")

  _ACTION_POLY_BASE_DERIVATIONS[dpr] = derivations
  return dpr
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    __apply_base_derivation(
      dpr::DifferentialPolyRing,
      c::Any,
      d::AbstractVector{<:Integer}
    )

Apply the coefficient-ring part of the differential action `d` to the element `c`.

If `dpr` has derivations D_1, ..., D_m and d=[d_1, ...,d_m], this
  helper applies D_i exactly d_i times.
It is used by the extended diff_action implementation before applying the Leibniz rule to the jet variables.

This function is internal; a user normally calls diff_action(p, d) instead.

# Examples

```jldoctest
julia> K, t = rational_function_field(QQ, "t");
julia> Dt = c -> derivative(c);
julia> R, u = differential_polynomial_ring(K, :u, [Dt]);
julia> __apply_base_derivation(R, t^2, [1])
2*t
julia> __apply_base_derivation(R, t^2, [2])
2


julia> Qt, t = polynomial_ring(QQ, "t");
julia> K = fraction_field(Qt);
julia> tK = K(t);
julia> Dt = c -> derivative(c);
julia> R, u = differential_polynomial_ring(K, :u, [Dt]);
julia> __apply_base_derivation(R, 1//tK, [1])
-1//t^2
julia> __apply_base_derivation(R, tK^5, [4])
120*t
```
"""
function __apply_base_derivation(
  dpr::DifferentialPolyRing,
  c::Any,
  d::AbstractVector{<:Integer},
)
  derivations = base_derivations(dpr)
  for i in eachindex(d)
    if !iszero(d[i])
      derivation = derivations[i]
      for _ in 1:d[i]
        c = derivation(c)
      end
    end
  end
  return c
end

###############################################################################
#####  Differential-polynomial rings whose action maps
#####  do not act as zero on the ground ring
###############################################################################
if isdefined(@__MODULE__, :base_derivations)
  __delete_method!(base_derivations, Tuple{DifferentialPolyRing})
end

@doc raw"""
    base_derivations(dpr::DifferentialPolyRing)

Given a differential polynomial ring `dpr`, return the vector of derivations acting on the coefficient ring.

# Examples

```jldoctest
julia> R, u = differential_polynomial_ring(QQ, :u, 1);
julia> collect(base_derivations(R))
1-element Vector{Any}:
 __zero_base_derivation (generic function with 1 method)


julia> K, t = rational_function_field(QQ, "t");
julia> Dt = c -> derivative(c);
julia> R, u = differential_polynomial_ring(K, :u, [__zero_base_derivation, Dt]);
julia> (a, b) = base_derivations(R);
julia> a(t)
0
julia> b(t)
1


julia> A, (x, y) = polynomial_ring(QQ, [:x, :y]);
julia> K = fraction_field(A);
julia> xK, yK = K(x), K(y);
julia> Dx = c -> derivative(c, 1);
julia> Dy = c -> derivative(c, 2);
julia> R, z = differential_polynomial_ring(K, :z, [Dx, Dy]);
julia> (a, b) = base_derivations(R);
julia> a(xK)
1
julia> a(yK)
0
julia> b(xK)
0
julia> b(yK)
1
```
"""
function base_derivations(dpr::DifferentialPolyRing)
  return get!(_ACTION_POLY_BASE_DERIVATIONS, dpr) do
    Any[__zero_base_derivation for _ in 1:n_action_maps(dpr)]
  end
end

@doc raw"""
    base_derivations_commute_on(
      R::DifferentialPolyRing,
      elements,
    ) -> Bool

Check whether the coefficient-ring derivations attached to `R` commute pairwise
  on every element in `elements`.

This is a finite diagnostic test.
If the supplied maps are known derivations and `elements` generate the coefficient ring over
  a subring contained in the kernel of all maps, equality on these generators proves pairwise
  commutativity on the whole coefficient ring.
"""
function base_derivations_commute_on(
  dpr::DifferentialPolyRing,
  elements,
)
  return __base_derivations_commute_on(
    base_derivations(dpr),
    coefficient_ring(dpr),
    elements,
  )
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    Oscar.differential_polynomial_ring(R::Ring, n_elementary_symbols::Integer,
        derivations::AbstractVector; kwargs...)
    Oscar.differential_polynomial_ring(R::Ring, elementary_symbols::AbstractVector{Symbol},
        derivations::AbstractVector; kwargs...)
    Oscar.differential_polynomial_ring(R::Ring, elementary_symbol::Symbol,
        derivations::AbstractVector; kwargs...)

Construct a differential polynomial ring over K whose action maps induce the derivations in `derivations`
  on the coefficient ring.
The number of action maps is length(derivations).
The constructor checks pairwise commutativity on `0`, `1`, and the available
  generators of the coefficient ring.

The old constructors with an explicit number of action maps are unchanged.
In that case, the action on the coefficient ring is zero.

# Examples

```jldoctest
##### OLD CONSTRUCTOR
julia> R, u = differential_polynomial_ring(QQ, :u, 1);
julia> diff_action(2*u + 1, 1)
2*u[1]


julia> K, t = rational_function_field(QQ, "t");
julia> Dt = c -> derivative(c);
julia> R, (u, v) = differential_polynomial_ring(K, [:u, :v], [Dt]);
julia> diff_action(t*u + v + t, 1)
t*u[1] + v[1] + u[0] + 1


julia> K, t = rational_function_field(QQ, "t");
julia> Dt = c -> derivative(c);
julia> R, u = differential_polynomial_ring(K, :u, [Dt]);
julia> diff_action(t^2*u + 1, [1])
t^2*u[1] + 2*t*u[0]
julia> diff_action(t^2*u + 1, [2])
t^2*u[2] + 4*t*u[1] + 2*u[0]


julia> A, (x, y) = polynomial_ring(QQ, [:x, :y]);
julia> K = fraction_field(A); xK, yK = K(x), K(y);
julia> Dx = c -> derivative(c, 1);
julia> Dy = c -> derivative(c, 2);
julia> R, u = differential_polynomial_ring(K, :u, [Dx, Dy]);
julia> f = -u^2 + xK*yK;
julia> diff_action(f, [1, 0])
-2*u[0,0]*u[1,0] + y
julia> diff_action(f, [1, 1])
-2*u[0,0]*u[1,1] - 2*u[1,0]*u[0,1] + 1


julia> A, (x, y) = polynomial_ring(QQ, [:x, :y]);
julia> K = fraction_field(A); xK, yK = K(x), K(y);
julia> Dx = c -> derivative(c, 1);
julia> Dy = c -> derivative(c, 2);
julia> R, z = differential_polynomial_ring(K, :z, [Dx, Dy]);
julia> p = xK//yK * z + xK^2//(yK + 1);
julia> diff_action(p, [1, 1])
x//y*z[1,1] - x//y^2*z[1,0] + 1//y*z[0,1] - 1//y^2*z[0,0] - (2*x)//(y^2 + 2*y + 1)
```
"""
function Oscar.differential_polynomial_ring(R::Ring, n_elementary_symbols::Integer,
  derivations::AbstractVector; kwargs...)
  dpr, xs = differential_polynomial_ring(R, Int(n_elementary_symbols), length(derivations); kwargs...)
  __set_base_derivations!(dpr, derivations)
  return dpr, xs
end

function Oscar.differential_polynomial_ring(R::Ring, elementary_symbols::AbstractVector{Symbol},
  derivations::AbstractVector; kwargs...)
  dpr, xs = differential_polynomial_ring(R, collect(Symbol, elementary_symbols), length(derivations); kwargs...)
  __set_base_derivations!(dpr, derivations)
  return dpr, xs
end

function Oscar.differential_polynomial_ring(R::Ring, elementary_symbol::Symbol,
  derivations::AbstractVector; kwargs...)
  dpr, xs = differential_polynomial_ring(R, [elementary_symbol], derivations; kwargs...)
  return dpr, xs[1]
end

######################################################################################################################################
######################################################################################################################################

__delete_method!(diff_action, Tuple{DifferentialPolyRingElem,Int})
__delete_method!(diff_action, Tuple{DifferentialPolyRingElem,Vector{Int}})
__delete_method!(diff_action, Tuple{DifferentialPolyRingElem,Integer})
__delete_method!(diff_action, Tuple{DifferentialPolyRingElem,AbstractVector{<:Integer}})
__delete_method!(diff_action, Tuple{DifferentialPolyRingElem,Tuple{Vararg{Integer}}})

@doc raw"""
    diff_action(p::DifferentialPolyRingElem, i::Integer)

Apply the `i`-th derivation to `p`, including its action on coefficients.
```

# Examples

```jldoctest
julia> K, t = rational_function_field(QQ, "t");
julia> R, u = differential_polynomial_ring(K, :u, [c -> derivative(c)]);
julia> du = diff_action(u, 1);
julia> diff_action(t*u^3, 1) == u^3 + 3*t*u^2*du
true
julia> diff_action(R(t), 1) == R(1)
true
```
"""
function Oscar.diff_action(dpre::DifferentialPolyRingElem{T}, i::Integer) where {T}
  dpr = parent(dpre)
  i in 1:n_action_maps(dpr) || error("Index out of range")
  is_zero(dpre) && return dpre

  d = fill(0, n_action_maps(dpr))
  d[i] = 1

  #For each variable in dpre, retrieve its jet index: base symbol + shift.
  #Prepare a list of the new jet variables that need to be created.
  #Retrieve the dictionary that maps each jet to its position in the universal polynomial ring.
  dpre_vars_idxs = [Oscar.__vtj(dpr)[var] for var in vars(dpre)]
  add_new_vars_idx = Tuple{Int,Vector{Int}}[]
  jtu = Oscar.__jtu_idx(dpr)

  for (sym_idx, shift) in dpre_vars_idxs
    new_idx = (sym_idx, shift + d)
    if !haskey(jtu, new_idx)
      push!(add_new_vars_idx, new_idx)
    end
  end
  if !isempty(add_new_vars_idx)
    Oscar.__add_new_jetvar!(dpr, add_new_vars_idx)
  end

  old_to_new_pos = Dict{Int,Int}()
  for j in eachindex(dpre_vars_idxs)
    sym_idx, shift = dpre_vars_idxs[j]
    old_to_new_pos[jtu[(sym_idx, shift)]] = jtu[(sym_idx, shift + d)]
  end

  upr = base_ring(dpr)
  upr_gens = gens(upr)
  n_upr_gens = ngens(upr)
  C = MPolyBuildCtx(upr)

  for term in terms(data(dpre))
    coeff_t = coeff(term, 1)
    ev0 = exponent_vector(term, 1)
    ev = vcat(ev0, fill(0, n_upr_gens - length(ev0)))

    derived_coeff_t = __apply_base_derivation(dpr, coeff_t, d)
    if !iszero(derived_coeff_t)
      push_term!(C, derived_coeff_t, copy(ev))
    end

    for var in vars(term)
      var_pos = findfirst(==(var), upr_gens)
      shifted_var_pos = old_to_new_pos[var_pos]
      new_exp_vec = copy(ev)
      new_exp_vec[var_pos] -= 1
      new_exp_vec[shifted_var_pos] += 1
      push_term!(C, coeff_t * ev[var_pos], new_exp_vec)
    end
  end

  return dpr(finish(C))
end

@doc raw"""
    diff_action(p::DifferentialPolyRingElem, multiplicities)

Apply the `i`-th derivation `multiplicities[i]` times.
The derivations are applied in increasing index order.
For a valid differential coefficient ring this order is
  immaterial because the supplied base derivations commute.

# Examples

```jldoctest
julia> A, (x, y) = polynomial_ring(QQ, [:x, :y]);
julia> K = fraction_field(A); xK, yK = K(x), K(y);
julia> R, u = differential_polynomial_ring(
           K,
           :u,
           [c -> derivative(c, 1), c -> derivative(c, 2)],
       );
julia> diff_action(u, [2, 1])
u[2,1]
julia> diff_action(u, (1, 2))
u[1,2]
```
"""
function Oscar.diff_action(dpre::DifferentialPolyRingElem{T}, d::AbstractVector{<:Integer}) where {T}
  dpr = parent(dpre)
  d = collect(Int, d)
  length(d) == n_action_maps(dpr) && all(x -> x >= 0, d) ||
    error("Invalid vector of multiplicities")

  p = dpre
  for i in eachindex(d)
    for _ in 1:d[i]
      p = Oscar.diff_action(p, i)
    end
  end
  return p
end

function Oscar.diff_action(dpre::DifferentialPolyRingElem{T}, d::Tuple{Vararg{Integer}}) where {T}
  return Oscar.diff_action(dpre, collect(Int, d))
end