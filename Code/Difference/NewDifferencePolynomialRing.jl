###############################################################################
#####  Importing OSCAR + Internal Helpers
###############################################################################

import Oscar
import Oscar:
  base_ring,
  coeff, coefficient_ring,
  data, diff_action, difference_polynomial_ring,
  elem_type, exponent_vector,
  finish,
  gen, gens,
  is_zero,
  ngens, n_action_maps,
  parent, push_term!,
  terms,
  vars
using Oscar:
  DifferencePolyRing,
  DifferencePolyRingElem,
  MPolyBuildCtx,
  Ring

if !isdefined(@__MODULE__, :_ACTION_POLY_BASE_MORPHISMS)
  _ACTION_POLY_BASE_MORPHISMS = IdDict{Any,Vector{Any}}()
end

if !isdefined(@__MODULE__, :__delete_method!)
  function __delete_method!(f::Function, argtypes::Type)
    for m in collect(methods(f, argtypes))
      Base.delete_method(m)
    end
    return nothing
  end
end

######################################################################################################################################
######################################################################################################################################

function __base_morphism_commutativity_test_elements(K)
  elements = Any[zero(K), one(K)]
  if applicable(gens, K)
    append!(elements, gens(K))
  elseif applicable(gen, K)
    push!(elements, gen(K))
  end
  return elements
end

function __base_morphisms_commute_on(morphisms, K, elements)
  elements = collect(elements)
  for i in eachindex(morphisms)
    for j in (i + 1):length(morphisms)
      for c in elements
        lhs = K(morphisms[i](K(morphisms[j](K(c)))))
        rhs = K(morphisms[j](K(morphisms[i](K(c)))))
        lhs == rhs || return false
      end
    end
  end
  return true
end

@doc raw"""
    __set_base_morphisms!(
      dpr::DifferencePolyRing,
      morphisms::Union{AbstractVector,Tuple}
    )

Attach the coefficient-ring morphisms in `morphisms` to the difference polynomial ring `dpr`.

This is an internal helper used in difference_polynomial_ring(K, ..., morphisms).
The vector `morphisms` must have one entry for each action map of `dpr`.
Before storing them, the function checks pairwise commutativity on `0`, `1`,
  and the generators returned by `gens(coefficient_ring(dpr))` or
  `gen(coefficient_ring(dpr))`, when available.
This proves commutativity provided those elements
  generate the coefficient ring over a subring fixed by all maps.

The data are stored in the internal dictionary _ACTION_POLY_BASE_MORPHISMS,
  and are later read by base_morphisms(dpr) and by the extended diff_action.

The function returns `dpr`.

# Examples

```jldoctest
julia> R, y = difference_polynomial_ring(QQ, :y, 1);
julia> __set_base_morphisms!(R, [identity]) === R
true
julia> base_morphisms(R)
1-element Vector{Any}:
 identity (generic function with 1 method)
```
"""
function __set_base_morphisms!(
  dpr::DifferencePolyRing,
  morphisms::Union{AbstractVector,Tuple},
)
  morphisms = collect(Any, morphisms)
  length(morphisms) == n_action_maps(dpr) ||
    error("The number of base-ring morphisms must equal the number of action maps")

  K = coefficient_ring(dpr)
  test_elements = __base_morphism_commutativity_test_elements(K)
  __base_morphisms_commute_on(morphisms, K, test_elements) ||
    error("The base-ring morphisms must commute pairwise")

  _ACTION_POLY_BASE_MORPHISMS[dpr] = morphisms
  return dpr
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    __apply_base_action(
      dpr::DifferencePolyRing,
      c::Any,
      d::AbstractVector{<:Integer}
    )

Apply the coefficient-ring part of the difference action `d` to the element `c`.

If `dpr` has action maps σ_1, ..., σ_m and d = [d_1, ..., d_m], this
  helper applies the stored base morphism for σ_i exactly d_i times.
It is used by the extended diff_action implementation before the monomial variables are shifted.

This function is internal; a user normally calls diff_action(p, d) instead.

# Examples

```jldoctest
julia> Qx, x = polynomial_ring(QQ, "x");
julia> K, a = number_field(x^2 - 2, "a");
julia> σ = hom(K, K, -a);
julia> R, y = difference_polynomial_ring(K, :y, [σ]);
julia> __apply_base_action(R, a, [1])
-a
julia> __apply_base_action(R, a, [2])
a


julia> K, a = quadratic_field(2);
julia> σ = hom(K, K, -a);
julia> R, y = difference_polynomial_ring(K, :y, [σ]);
julia> __apply_base_action(R, a, [1])
-sqrt(2)
julia> __apply_base_action(R, a, [2])
sqrt(2)
```
"""
function __apply_base_action(
  dpr::DifferencePolyRing,
  c::Any,
  d::AbstractVector{<:Integer},
)
  morphisms = base_morphisms(dpr)
  for i in eachindex(d)
    if !iszero(d[i])
      morphism = morphisms[i]
      for _ in 1:d[i]
        c = morphism(c)
      end
    end
  end
  return c
end

###############################################################################
#####  Difference-polynomial rings whose action maps
#####  do not act as the identity on the ground ring
###############################################################################
if isdefined(@__MODULE__, :base_morphisms)
  __delete_method!(base_morphisms, Tuple{DifferencePolyRing})
end
@doc raw"""
    base_morphisms(dpr::DifferencePolyRing)

Given a difference polynomial ring `dpr`, return the vector of morphisms acting on the coefficient ring.

# Examples

```jldoctest
julia> R, y = difference_polynomial_ring(QQ, :y, 1)
julia> collect(base_morphisms(R))
1-element Vector{Any}:
  identity (generic function with 1 method)


julia> Qx, x = polynomial_ring(QQ, "x")
julia> K, i = number_field(x^2 + 1, "i")
julia> S = automorphism_list(K)
julia> R, y = difference_polynomial_ring(K, :y, S)
julia> (a,b) = base_morphisms(R)
2-element Vector{Any}:
 Map: K -> K
 Map: K -> K
julia> a(i)
i
julia> b(i)
-i


julia> K, (x,y) = rational_function_field(QQ,[:x,:y]);
julia> R, z = difference_polynomial_ring(K, :z, [f->f(x+1,y),f->f(x,y+1)]);
julia> (a,b) = base_morphisms(R)
julia> a(x)
x + 1
julia> a(y)
y
julia> b(x)
x
julia> b(y)
y + 1
```
"""
function base_morphisms(dpr::DifferencePolyRing)
  return get!(_ACTION_POLY_BASE_MORPHISMS, dpr) do
    Any[identity for _ in 1:n_action_maps(dpr)]
  end
end

@doc raw"""
    base_morphisms_commute_on(
      R::DifferencePolyRing,
      elements,
    ) -> Bool

Check whether the coefficient-ring morphisms attached to `R` commute pairwise
  on every element in `elements`.

This is a finite diagnostic test.
If the supplied maps are known endomorphisms and `elements` generate the coefficient ring over
  a subring fixed by all maps, equality on these generators proves pairwise commutativity on
  the whole coefficient ring.
"""
function base_morphisms_commute_on(
  dpr::DifferencePolyRing,
  elements,
)
  return __base_morphisms_commute_on(
    base_morphisms(dpr),
    coefficient_ring(dpr),
    elements,
  )
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    Oscar.difference_polynomial_ring(R::Ring, n_elementary_symbols::Integer,
        morphisms::AbstractVector; kwargs...)
    Oscar.difference_polynomial_ring(R::Ring, elementary_symbols::AbstractVector{Symbol},
        morphisms::AbstractVector; kwargs...)
    Oscar.difference_polynomial_ring(R::Ring, elementary_symbol::Symbol,
        morphisms::AbstractVector; kwargs...)

Construct a difference polynomial ring over K whose action maps induce the morphisms in `morphisms`
  on the coefficient ring.
The number of action maps is length(morphisms).
The constructor checks pairwise commutativity on `0`, `1`, and the available
  generators of the coefficient ring.

The old constructors with an explicit number of action maps are unchanged.
In that case, the action on the coefficient ring is the identity.

# Examples

```jldoctest
##### OLD CONSTRUCTOR
julia> R, y = difference_polynomial_ring(QQ, :y, 1)
julia> diff_action(2*y + 1, 1)
2*y[1] + 1


julia> Qx, x = polynomial_ring(QQ, "x")
julia> K, i = number_field(x^2 + 1, "i")
julia> σ = hom(K, K, -i)
julia> R, (y, z) = difference_polynomial_ring(K, [:y, :z], [σ])
julia> diff_action(i*y + z + i, 1)
-i*y[1] + z[1] - i


julia> Qx, x = polynomial_ring(QQ, "x")
julia> K, a = number_field(x^2 - 2, "a")
julia> σ = hom(K, K, -a)
julia> R, y = difference_polynomial_ring(K, :y, [σ])
julia> diff_action(a*y + 1, [1])
-a*y[1] + 1
julia> diff_action(a*y + 1, [2])
a*y[2] + 1


julia> Qx, x = polynomial_ring(QQ, "x")
julia> K, i = number_field(x^2 + 1, "i")
julia> S = automorphism_list(K)
julia> R, y = difference_polynomial_ring(K, :y, S)
julia> f = -y^2+i
julia> f = -y^2+i
julia> diff_action(f,[1,0])
  -y[1,0]^2 + i
julia> diff_action(f,[1,1])
  -y[1,1]^2 - i


julia> K, (x,y) = rational_function_field(QQ,[:x,:y]);
julia> R, z = difference_polynomial_ring(K, :z, [f->f(x+1,y),f->f(x,y+1)]);
julia> p = x//y * z + x^2//(y + 1);
julia> diff_action(p,[1,1])
(x + 1)//(y + 1)*z[1,1] + (x^2 + 2*x + 1)//(y + 2)
```
"""
function Oscar.difference_polynomial_ring(R::Ring, n_elementary_symbols::Integer,
  morphisms::AbstractVector; kwargs...)
  dpr, xs = difference_polynomial_ring(R, Int(n_elementary_symbols), length(morphisms); kwargs...)
  __set_base_morphisms!(dpr, morphisms)
  return dpr, xs
end

function Oscar.difference_polynomial_ring(R::Ring, elementary_symbols::AbstractVector{Symbol},
  morphisms::AbstractVector; kwargs...)
  dpr, xs = difference_polynomial_ring(R, collect(Symbol, elementary_symbols), length(morphisms); kwargs...)
  __set_base_morphisms!(dpr, morphisms)
  return dpr, xs
end

function Oscar.difference_polynomial_ring(R::Ring, elementary_symbol::Symbol,
  morphisms::AbstractVector; kwargs...)
  dpr, xs = difference_polynomial_ring(R, [elementary_symbol], morphisms; kwargs...)
  return dpr, xs[1]
end

######################################################################################################################################
######################################################################################################################################

__delete_method!(diff_action, Tuple{DifferencePolyRingElem,Int})
__delete_method!(diff_action, Tuple{DifferencePolyRingElem,Vector{Int}})
__delete_method!(diff_action, Tuple{DifferencePolyRingElem,Integer})
__delete_method!(diff_action, Tuple{DifferencePolyRingElem,AbstractVector{<:Integer}})
__delete_method!(diff_action, Tuple{DifferencePolyRingElem,Tuple{Vararg{Integer}}})

function Oscar.diff_action(dpre::DifferencePolyRingElem{T}, i::Integer) where {T}
  dpr = parent(dpre)
  i in 1:n_action_maps(dpr) || error("Index out of range")
  d = fill(0, n_action_maps(dpr))
  d[i] = 1
  return Oscar.diff_action(dpre, d)
end

function Oscar.diff_action(dpre::DifferencePolyRingElem{T}, d::AbstractVector{<:Integer}) where {T}
  dpr = parent(dpre)
  d = collect(Int, d)
  length(d) == n_action_maps(dpr) && all(x -> x >= 0, d) ||
    error("Invalid vector of multiplicities")

  all(iszero, d) || is_zero(dpre) && return dpre

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
  for i in eachindex(dpre_vars_idxs)
    sym_idx, shift = dpre_vars_idxs[i]
    old_to_new_pos[jtu[(sym_idx, shift)]] = jtu[(sym_idx, shift + d)]
  end

  upr = base_ring(dpr)
  upr_gens = gens(upr)
  n_upr_gens = ngens(upr)
  C = MPolyBuildCtx(upr)

  for term in terms(data(dpre))
    coeff_t = __apply_base_action(dpr, coeff(term, 1), d)
    ev0 = exponent_vector(term, 1)
    ev = vcat(ev0, fill(0, n_upr_gens - length(ev0)))
    new_exp_vec = fill(0, length(ev))

    for var in vars(term)
      var_pos = findfirst(==(var), upr_gens)
      shifted_var_pos = old_to_new_pos[var_pos]
      new_exp_vec[shifted_var_pos] = ev[var_pos]
    end

    push_term!(C, coeff_t, new_exp_vec)
  end

  return dpr(finish(C))
end

function Oscar.diff_action(dpre::DifferencePolyRingElem{T}, d::Tuple{Vararg{Integer}}) where {T}
  return Oscar.diff_action(dpre, collect(Int, d))
end