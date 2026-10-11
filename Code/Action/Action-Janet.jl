###############################################################################
#####  Action Janet data
###############################################################################

"""
Return the internal jet data associated to the jet variable `v`.

The output is a named tuple whose fields are `symbol` and `shift`:\
`symbol` is the position of the elementary symbol, and\
`shift` is the shift vector such that v = shift(y_symbol).

An error is thrown if `v` is not a single jet variable of its parent action polynomial ring.
"""
function __jet_data(v::ActionPolyRingElem)
    d = Oscar.__vtj(parent(v))
    jet = get(d, v, nothing)
    if jet !== nothing
        symbol, shift = jet
        return (symbol=symbol, shift=shift)
    end
    error("$v is not a jet variable")
end

"""
Internal cache for computations on a fixed vector of action polynomials.

The fields store the original action polynomials (`equations`), their leaders, the elementary-symbol index and shift of each leader,
the indices grouped by elementary symbol, the multiplicative-variable vectors, and a dictionary from each polynomial to
its position in the original list.
Public functions build this object once and pass it to avoid recomputing the same data repeatedly.

Reuse computed prolongations by equation index and action multi-index.
The differential splitting also caches nonmultiplicative derivatives here for
passivity checks; Janet reduction checks admissibility before using any entry.
Rebuild the data only after changing `equations`, the ranking or the coefficient actions.
The cache is not shared with a refined branch.
"""
struct __ActionJanetData{P<:ActionPolyRingElem}
    equations::Vector{P}
    leaders::Vector{P}
    symbols::Vector{Int64}
    shifts::Vector{Vector{Int64}}
    by_symbol::Vector{Vector{Int64}}
    multiplicative_variables::Vector{BitVector}
    indices::Dict{P,Int64}
    prolongations::Dict{Tuple{Int,Tuple{Vararg{Int}}},P}
    # For every cached admissible prolongation, store its degree and initial with respect to its own leader.
    # Repeated normal-form reductions should not recompute these data
    prolongation_univariate_data::Dict{Tuple{Int,Tuple{Vararg{Int}}},Tuple{Int,P}}
end

"""
Compute the leader data of the polynomial list `F`.

Return a named tuple with fields `leaders`, `symbols`, `shifts`, and `by_symbol`, where
`leaders[i]` is leader(F[i]),\
`symbols[i]` is the elementary-symbol position of that leader,\
`shifts[i]` is its shift vector,
`by_symbol[j]` contains the indices of the polynomials whose leader belongs to the j-th elementary symbol.

The helper checks that all polynomials have the same parent and that no polynomial is constant.
Pass `R` only when the common parent has already been certified.
"""
function __action_leader_vectors(
    F::Vector{P};
    R=__require_same_parent(F),
) where {P<:ActionPolyRingElem}
    isempty(F) && return (
        leaders=P[],
        symbols=Int64[],
        shifts=Vector{Int64}[],
        by_symbol=Vector{Vector{Int64}}(),
    )
    any(is_constant(f) for f in F) && error("The given action polynomials must be non-constant")
    n = n_action_indeterminates(R)
    leaders = Vector{P}(undef, length(F))
    symbols = Vector{Int64}(undef, length(F))
    shifts = Vector{Vector{Int64}}(undef, length(F))
    by_symbol = [Int64[] for _ in 1:n]
    for (i, f) in enumerate(F)
        ld = leader(f)
        jet = __jet_data(ld)
        leaders[i] = ld
        symbols[i] = jet.symbol
        shifts[i] = copy(jet.shift)
        push!(by_symbol[jet.symbol], i)
    end
    return (leaders=leaders, symbols=symbols, shifts=shifts, by_symbol=by_symbol)
end

"""
Build the internal Janet cache for `F`.
Pass the optional keyword `R` only after validating the common parent of `F`.

The first method uses the multiplicative variables `MV` provided by the caller.
The second method computes them from the leader shifts of `F`, grouped by
elementary symbol.

When `MV` is supplied, its length must agree with the length of `F`, and each
entry of `MV` must have one component for each action map of the common parent
ring.

    # Examples

    ```jldoctest
    julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :deglex);
    julia> f_1 = y1[0, 1]^2+1;
    julia> f_2 = y1[2, 0]+y2[0, 1];
    julia> f_3 = y1[0, 1]*y2[0, 2];
    julia> f_4 = y1[1, 1]^2-y1[1, 0]^4;
    julia> F = [f_1, f_2, f_3, f_4];
    julia> D = __action_janet_data(F);
    julia> D.equations == F
    true
    julia> D.leaders
    4-element Vector{DifferencePolyRingElem{QQFieldElem}}:
     y1[0,1]
     y1[2,0]
     y2[0,2]
     y1[1,1]
    julia> D.symbols
    4-element Vector{Int64}:
     2
     2
     1
     2
    julia> D.shifts
    4-element Vector{Vector{Int64}}:
     [0, 1]
     [2, 0]
     [0, 2]
     [1, 1]
    julia> D.by_symbol
    2-element Vector{Vector{Int64}}:
     [3]
     [1, 2, 4]
    julia> D.multiplicative_variables
    4-element Vector{BitVector}:
     [0, 1]
     [1, 1]
     [1, 1]
     [0, 1]
    julia> MV = multiplicative_variables(F);
    julia> MV == D.multiplicative_variables
    true
    ```
"""
function __action_janet_data(
    F::Vector{P},
    MV::AbstractVector{<:BitVector};
    R=__require_same_parent(F),
) where {P<:ActionPolyRingElem}
    leader_data = __action_leader_vectors(F; R)
    length(MV) == length(F) || error("The polynomial and multiplicative-variable lists must have the same length")
    if !isempty(F)
        n = n_action_maps(R)
        any(v -> length(v) != n, MV) && error("Each multiplicative variable vector must have one entry per action map")
    end
    multiplicative = MV isa Vector{BitVector} ? MV : collect(BitVector, MV)
    indices = Dict{P,Int64}()
    sizehint!(indices, length(F))
    for (i, f) in enumerate(F)
        indices[f] = i
    end
    return __ActionJanetData{P}(
        F,
        leader_data.leaders,
        leader_data.symbols,
        leader_data.shifts,
        leader_data.by_symbol,
        multiplicative,
        indices,
        Dict{Tuple{Int,Tuple{Vararg{Int}}},P}(),
        Dict{Tuple{Int,Tuple{Vararg{Int}}},Tuple{Int,P}}(),
    )
end

function __action_janet_data(
    F::Vector{P};
    R=__require_same_parent(F),
) where {P<:ActionPolyRingElem}
    leader_data = __action_leader_vectors(F; R)
    MV = __action_multiplicative_masks(leader_data)
    poly_indices = Dict{P,Int64}()
    sizehint!(poly_indices, length(F))
    for (i, f) in enumerate(F)
        poly_indices[f] = i
    end
    return __ActionJanetData{P}(
        F,
        leader_data.leaders,
        leader_data.symbols,
        leader_data.shifts,
        leader_data.by_symbol,
        MV,
        poly_indices,
        Dict{Tuple{Int,Tuple{Vararg{Int}}},P}(),
        Dict{Tuple{Int,Tuple{Vararg{Int}}},Tuple{Int,P}}(),
    )
end

# Read leader shifts without padding/copying, and avoid constructing a full Janet cache
# when the caller only needs multiplicative variables.
function __action_multiplicative_masks(leader_data)
    MV = Vector{BitVector}(undef, length(leader_data.shifts))
    for indices in leader_data.by_symbol
        isempty(indices) && continue
        monomials = view(leader_data.shifts, indices)
        local_mv = __multiplicative_variable_masks_unchecked(monomials)
        for (j, idx) in enumerate(indices)
            MV[idx] = local_mv[j]
        end
    end
    return MV
end

"""
Return `true` if the shift `θ` only uses multiplicative variables encoded by the `BitVector` `mv`.
A positive component of `θ` is admissible only when the corresponding component of `mv` is nonzero.
"""
function __admissible_shift(
    θ::Union{AbstractVector{<:Integer},Tuple{Vararg{Integer}}},
    mv::BitVector
)
    length(θ) == length(mv) || error("The action-map and multiplicative-variable vectors must have the same length")
    for (shift_component, multiplicative_component) in zip(θ, mv)
        !iszero(shift_component) && iszero(multiplicative_component) && return false
    end
    return true
end

""" Return `true` if all components of the shift vector `θ` are nonnegative. """
function __nonnegative_shift(
    θ::AbstractVector{<:Integer}
)
    for x in θ
        x < 0 && return false
    end
    return true
end

"""
Internal implementation of Janet completeness for cached leader data.
The check is performed separately for the shift monomials attached to each elementary symbol.

If the optional argument `info` is set to `true`, the function prints the first obstruction found.
If the optional argument `missing` is true, it returns the first elementary-symbol index and
the first missing shift that would have to be added.
"""
function __is_janet_complete(
    data::__ActionJanetData;
    info::Bool=false,
    missing::Bool=false
)
    if length(data.leaders) != length(Set(data.leaders))
        info == true && println("There are action polynomials with the same leader")
        return false
    end
    for symbol in eachindex(data.by_symbol)
        indices = data.by_symbol[symbol]
        isempty(indices) && continue
        shifts = view(data.shifts, indices)
        # Supplied reduction masks may differ from the canonical Janet masks.
        # Recompute the latter, but do not check again or copy the known shifts.
        masks = __multiplicative_variable_masks_unchecked(shifts)
        missing_shift = __first_janet_missing_unchecked(shifts, masks)
        missing_shift === nothing && continue
        if info
            println("The leader shifts for elementary symbol $symbol " *
            "are not Janet complete: $missing_shift is missing")
        end
        return missing ? (false, symbol, collect(Int, missing_shift)) : false
    end
    return true
end

###############################################################################
#####  Action Janet multiplicative variables
###############################################################################

@doc raw"""
    multiplicative_variables(
        F::AbstractVector{<:P}
    ) where {P<:ActionPolyRingElem}

Let R be an action polynomial ring with action maps σ\_1, ..., σ\_m and action indeterminates y\_1, ..., y\_n
with a fixed ranking.

For a vector `F`=[f\_1, ..., f\_s] of action polynomials in R, return a `Vector{BitVector}` V defined as follows.
Let θ be the shift and y\_j the elementary symbol such that leader(f\_i)=θy\_j.
Let G\_j be the set of all shifts η such that ηy\_j is the leader of a polynomial in F.
The vector V[i] is equal to the set of multiplicative variables of θ as an element of G\_j: it has a 1 in position j
if σ\_j is multiplicative for f\_i, and 0 otherwise.

    # Examples

    ```jldoctest
    julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :deglex);
    julia> f_1 = y2[0,1]^3 + y2;
    julia> f_2 = y1^2*y2[1,0] + y1[0,1]-1;
    julia> f_3 = y1[1,0]^4 + y1[0,1]*y2^2;
    julia> f_4 = y1[0,2]*y2[1,1]+y1[0,1]+1;
    julia> F=[f_1, f_2, f_3, f_4];
    julia> multiplicative_variables(F)
    4-element Vector{BitVector}:
     [0, 1]
     [1, 0]
     [1, 1]
     [1, 1]

    julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :invlex);
    julia> f_1 = y2[0,1]^3+y2;
    julia> f_2 = y1^2*y2[1,0] + y1[0,1]-1;
    julia> f_3 = y1[1,0]^4+y1[0,1]*y2^2;
    julia> f_4 = y1[0,2]*y2[1,1]+y1[0,1]+1;
    julia> F=[f_1, f_2, f_3, f_4];
    julia> multiplicative_variables(F)
    4-element Vector{BitVector}:
     [1, 1]
     [1, 0]
     [1, 0]
     [1, 1]

    julia> R, u = differential_polynomial_ring(QQ, :u, 2);
    julia> E = [u[1,0], u[0,1]];
    julia> multiplicative_variables(E)
    2-element Vector{BitVector}:
     [1, 1]
     [0, 1]
    ```
"""
function multiplicative_variables(F::AbstractVector{<:P}) where {P<:ActionPolyRingElem}
    F = collect(P, F)
    return __action_multiplicative_masks(__action_leader_vectors(F))
end

###############################################################################
#####  Action Janet reducibility
###############################################################################

@doc raw"""
    is_janet_reducible(
        p::P,
        F::AbstractVector{<:P},
        MV::AbstractVector{<:BitVector}
    ) where T
    is_janet_reducible(
        p::P,
        F::AbstractVector{<:P}
    ) where {P<:ActionPolyRingElem}

Let R be an action polynomial ring with action maps σ\_1, ..., σ\_m and action indeterminates y\_1, ..., y\_n
with a fixed ranking.
Identify σ\_1^(a\_1)\*...\*σ\_m^(a\_m) with the vector [a\_1, ..., a\_m].

For an action polynomial `p` and a vector `F` of action polynomials in R, check whether `p` is Janet-reducible modulo `F`.\
If the multiplicative variables `MV` are supplied, they are used; they are computed otherwise.

If Janet-reduction is possible, return a named tuple with fields `reducible`, `alpha`, `polynomial`, `prolongation` and `multiplicative_variables`.
Here\
`reducible` is `true`;
`alpha` is the shift used to perform the reduction;\
`polynomial` is the polynomial in `F` that allows the reduction;\
`prolongation` is the actual prolongation;\
`multiplicative_variables` is the multiplicative-variable vector of that polynomial.\
Return `false` otherwise.

    # Examples

    ```jldoctest
    julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :deglex);
    julia> f_1 = y1[0, 1]^2+1;
    julia> f_2 = y1[2, 0]+y2[0, 1];
    julia> f_3 = y1[0, 1]*y2[0, 2];
    julia> f_4 = y1[1, 1]^2-y1[1, 0]^4;
    julia> F = [f_1, f_2, f_3, f_4];
    julia> MV = multiplicative_variables(F);
    julia> p = y2[1, 1]*y2[0, 2]-y1[0, 2];
    julia> is_janet_reducible(p, F, MV)
    (reducible = true, alpha = [0, 0], polynomial = y1[0,1]*y2[0,2], prolongation = y1[0,1]*y2[0,2], multiplicative_variables = Bool[1, 1])
    julia> is_janet_reducible(p, F) == is_janet_reducible(p,F,MV)
    true
    julia> p = y1[0, 1];
    julia> is_janet_reducible(p, F)
    false

    julia> R, u = differential_polynomial_ring(QQ, :u, 1);
    julia> E = [u^2 - 1];
    julia> reduction = is_janet_reducible(u[2], E);
    julia> reduction.alpha
    1-element Vector{Int64}:
     2
    julia> reduction.prolongation
    2*u[0]*u[2] + 2*u[1]^2
    ```
"""
function is_janet_reducible(
    p::P,
    F::AbstractVector{<:P},
    MV::AbstractVector{<:BitVector},
) where {P<:ActionPolyRingElem}
    R = __require_same_parent(p, F)
    data = __action_janet_data(collect(P, F), MV; R)
    return __is_janet_reducible(p, data)
end

function is_janet_reducible(
    p::P,
    F::AbstractVector{<:P},
) where {P<:ActionPolyRingElem}
    R = __require_same_parent(p, F)
    data = __action_janet_data(collect(P, F); R)
    return __is_janet_reducible(p, data)
end

@doc raw"""
    is_janet_reducible_univariate(
        p::P,
        F::AbstractVector{<:P},
        MV::AbstractVector{<:BitVector},
    ) where {P<:ActionPolyRingElem}
    is_janet_reducible_univariate(
        p::P,
        F::AbstractVector{<:P},
    ) where {P<:ActionPolyRingElem}

Let R be an action polynomial ring with action maps σ\_1, ..., σ\_m and action indeterminates y\_1, ..., y\_n
with a fixed ranking.
Identify σ\_1^(a\_1)\*...\*σ\_m^(a\_m) with the vector [a\_1, ..., a\_m].

For an action polynomial `p` and a vector `F` of action polynomials in R, check whether it is possible to perform Janet-reduction by `F`
using only the leader of `p` as reduction variable.\
If the multiplicative variables `MV` are supplied, they are used; they are computed otherwise.

If Janet-reduction is possible, return a named tuple with fields `reducible`, `alpha`, `polynomial`, `prolongation` and `multiplicative_variables`.
Here\
`reducible` is `true`;
`alpha` is the shift used to perform the reduction;\
`polynomial` is the polynomial in `F` that allows the reduction;\
`prolongation` is the actual prolongation;\
`multiplicative_variables` is the multiplicative-variable vector of that polynomial.\
Return `false` otherwise.

    # Examples

    ```jldoctest
    julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :deglex);
    julia> f_1 = y1[0,1]^2+1;
    julia> f_2 = y1[2,0]+y2[0,1];
    julia> f_3 = y1[0,1]*y2[0,2];
    julia> f_4 = y1[1, 1]^2-y1[1,0]^4;
    julia> F = [f_1, f_2, f_3, f_4];
    julia> p = y2[1,1]*y2[0,3] - y1[2,0];
    julia> is_janet_reducible_univariate(p, F)
    (reducible = true, alpha = [0, 1], polynomial = y1[0,1]*y2[0,2], prolongation = y1[0,2]*y2[0,3], multiplicative_variables = Bool[1, 1])
    julia> p = y1[0,1];
    julia> is_janet_reducible_univariate(p, F)
    false
    ```
"""
function is_janet_reducible_univariate(
    p::P,
    F::AbstractVector{<:P},
    MV::AbstractVector{<:BitVector},
) where {P<:ActionPolyRingElem}
    R = __require_same_parent(p, F)
    data = __action_janet_data(collect(P, F), MV; R)
    return __is_janet_reducible_univariate(p, data)
end

function is_janet_reducible_univariate(
    p::P,
    F::AbstractVector{<:P},
) where {P<:ActionPolyRingElem}
    R = __require_same_parent(p, F)
    data = __action_janet_data(collect(P, F); R)
    return __is_janet_reducible_univariate(p, data)
end

"""
Internal implementation of Janet-reducibility of `p` with respect to cached Janet data.

The function scans the variables of `p`.  A variable is reducible by data.equations[i] if it has the same elementary symbol as leader(data.equations[i]),
its shift differs from the leader shift by a nonnegative admissible shift, and its degree in `p` is at least the degree of the corresponding leader in data.equations[i].
"""
function __is_janet_reducible(
    p::P,
    data::__ActionJanetData{P}
) where {P<:ActionPolyRingElem}
    isempty(data.equations) && return false
    __require_parent(parent(first(data.equations)), p)
    for v in vars(p)
        reduction = __action_reduction_candidate(p,v,data)
        reduction === false || return reduction
    end
    return false
end

"""
Internal Janet-reducibility test restricted to the leader of `p`.

It is the test used in Janet-reduction and normal-form computations where
the current polynomial is viewed as a univariate polynomial in its leader.
"""
function __is_janet_reducible_univariate(
    p::P,
    data::__ActionJanetData{P},
) where {P<:ActionPolyRingElem}
    isempty(data.equations) && return false
    __require_parent(parent(first(data.equations)), p)
    is_constant(p) && return false
    return __action_reduction_candidate(p, leader(p), data)
end

function __action_reduction_candidate_detailed(
    p::P,
    v::P,
    data::__ActionJanetData{P},
) where {P<:ActionPolyRingElem}
    jet = __jet_data(v)
    indices = data.by_symbol[jet.symbol]
    isempty(indices) && return false
    variable_index = var_index(Oscar.data(v))
    polynomial_degree = degree(Oscar.data(p), variable_index)
    for i in indices
        alpha = jet.shift - data.shifts[i]
        __nonnegative_shift(alpha) || continue
        __admissible_shift(alpha, data.multiplicative_variables[i]) || continue
        # Reuse the same prolongation when successive coefficients or
        # prolongations reduce at this leader; do not reapply the action maps.
        key = (i, Tuple(alpha))
        prolongation = get!(data.prolongations, key) do
            apply_action(data.equations[i], alpha)
        end
        univariate_data = get(data.prolongation_univariate_data, key, nothing)
        if univariate_data === nothing
            leader(prolongation) == v || continue
            reducer_degree = degree(Oscar.data(prolongation), variable_index)
        else
            reducer_degree = univariate_data[1]
        end
        polynomial_degree >= reducer_degree || continue
        # Carry the degree, raw variable index and immutable cache key forward,
        # so the normal-form caller can obtain the initial without rescanning.
        return (
            reducible=true,
            index=i,
            variable=v,
            alpha=alpha,
            polynomial=data.equations[i],
            prolongation=prolongation,
            multiplicative_variables=data.multiplicative_variables[i],
            degree=reducer_degree,
            variable_index=variable_index,
            cache_key=key,
        )
    end
    return false
end

# Internal normal-form code that needs the basis index and reduction variable 
# uses the detailed helper above.
function __action_reduction_candidate(
    p::P,
    v::P,
    data::__ActionJanetData{P},
) where {P<:ActionPolyRingElem}
    reduction = __action_reduction_candidate_detailed(p, v, data)
    reduction === false && return false
    return (
        reducible=reduction.reducible,
        alpha=reduction.alpha,
        polynomial=reduction.polynomial,
        prolongation=reduction.prolongation,
        multiplicative_variables=reduction.multiplicative_variables
    )
end

function __is_janet_reducible_univariate_detailed(
    p::P,
    data::__ActionJanetData{P},
) where {P<:ActionPolyRingElem}
    isempty(data.equations) && return false
    __require_parent(parent(first(data.equations)), p)
    is_constant(p) && return false
    return __action_reduction_candidate_detailed(p, leader(p), data)
end

# Return the leading coefficient of p viewed as a univariate polynomial in v.
# Extract it directly from the underlying universal polynomial.
# Support reduction in a chosen variable while avoiding intermediate action-polynomial terms.
function __action_univariate_initial(p::P, v::P) where {P<:ActionPolyRingElem}
    R = __require_same_parent(p, v)
    is_gen(v) || throw(ArgumentError("Supply one action indeterminate"))
    iszero(p) && throw(ArgumentError("The zero polynomial has no initial"))
    raw = Oscar.data(p)
    variable_index = var_index(Oscar.data(v))
    d = degree(raw, variable_index)
    d == 0 && return deepcopy(p)
    return R(coeff(raw, [variable_index], [d]))
end

# Return the degree and initial of the candidate prolongation with respect to its leader.
# Cache them by equation index and action multi-index while the Janet basis is unchanged.
# Reuse the known degree and avoid extracting the same initial in repeated reductions.
function __action_reduction_univariate_data(
    reduction,
    data::__ActionJanetData{P},
) where {P<:ActionPolyRingElem}
    return get!(data.prolongation_univariate_data, reduction.cache_key) do
        reducer = reduction.prolongation
        d = reduction.degree
        initial = parent(reducer)(coeff(Oscar.data(reducer), [reduction.variable_index], [d]))
        (d, initial)
    end
end

###############################################################################
#####  Action Janet completeness
###############################################################################

@doc raw"""
    is_janet_complete(
        F::AbstractVector{<:P};
        info::Bool=false,
        missing::Bool=false
    ) where {P<:ActionPolyRingElem}

Let R be an action polynomial ring with action maps σ\_1, ..., σ\_m and action indeterminates y\_1, ..., y\_n
with a fixed ranking.
Identify σ\_1^(a\_1)\*...\*σ\_m^(a\_m) with the vector [a\_1, ..., a\_m].

For a vector `F` of action polynomials in `R`, return `true` if `F` is a Janet complete set, and `false` otherwise.\
If the result is `false` and the optional argument `info` is set to `true`,
a message explaining the first problem encountered is printed.\
If the optional argument `missing` is set to true, it returns the first elementary symbol position
whose corresponding set of monomials is not Janet complete, and the first missing monomial encountered.

# Examples

```jldoctest
    julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :deglex);
    julia> f_1 = y2[0,1]^3+y2;
    julia> f_2 = y1^2*y2[1,0] + y1[0,1]-1;
    julia> f_3 = y1[1,0]^4+y1[0,1]*y2^2;
    julia> f_4 = y1[0,2]*y2[1,1]+y1[0,1]+1;
    julia> F = [f_1, f_2, f_3, f_4];
    julia> is_janet_complete(F)
    true

    julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :deglex);
    julia> f_1 = y1[0,1]^2+1;
    julia> f_2 = y1[2, 0]+y2[0,1];
    julia> f_3 = y1[0,1]*y2[0,2];
    julia> f_4 = y1[1, 1]^2-y1[1,0]^4;
    julia> F = [f_1, f_2, f_3, f_4];
    julia> is_janet_complete(F)
    true

    julia> R, (y2, y1) = difference_polynomial_ring(QQ, [:y2, :y1], 2; partition=[[1, 1]], index_ordering_name = :deglex);
    julia> f_1=y1[0,1]^2+1;
    julia> f_2=y1[0,4];
    julia> is_janet_complete([f_1,f_2]; info=true, missing=true)
    The leader shifts for elementary symbol 2 are not Janet complete: [0, 2] is missing
    (false, 2, [0, 2])

    julia> R, u = differential_polynomial_ring(QQ, :u, 2);
    julia> E = [u[1,0], u[0,1]];
    julia> is_janet_complete(E)
    true
    julia> E = [u, u[2,0]];
    julia> is_janet_complete(E; missing=true)
    (false, 1, [1, 0])
    ```
"""
function is_janet_complete(
    F::AbstractVector{<:P};
    info::Bool=false,
    missing::Bool=false
) where {P<:ActionPolyRingElem}
    R = __require_same_parent(F)
    isempty(F) && return true
    F = collect(P, F)
    if any(is_constant, F)
        info && println("There are constant polynomials")
        return false
    end
    return __is_janet_complete(__action_janet_data(F; R); info, missing)
end

###############################################################################
#####  Polynomial-system methods
###############################################################################

is_janet_complete(system::AbstractActionSystem; kwargs...) = is_janet_complete(first(__action_read_system(system)); kwargs...)
multiplicative_variables(system::AbstractActionSystem) = multiplicative_variables(first(__action_read_system(system)))

# A successful reduction returns a basis polynomial. Copy the result so that
# callers can modify it, without copying every equation before the search.
function is_janet_reducible(p::ActionPolyRingElem, system::AbstractActionSystem)
    __require_parent(system.ring, p)
    return deepcopy(is_janet_reducible(p, first(__action_read_system(system))))
end

function is_janet_reducible_univariate(p::ActionPolyRingElem, system::AbstractActionSystem)
    __require_parent(system.ring, p)
    return deepcopy(is_janet_reducible_univariate(p, first(__action_read_system(system))))
end

function is_janet_reducible(
    p::ActionPolyRingElem,
    system::AbstractActionSystem,
    multiplicative::AbstractVector{<:BitVector}
)
    __require_parent(system.ring, p)
    return deepcopy(is_janet_reducible(p, first(__action_read_system(system)), multiplicative))
end
