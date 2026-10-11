###############################################################################
#####  Branch intersections
###############################################################################

@doc raw"""
    algebraic_branch_intersection(
        left::Union{AlgebraicSystem,AbstractActionSystem,Tuple},
        right::Union{AlgebraicSystem,AbstractActionSystem,Tuple},
    )

Return `true` if the supplied systems of polynomial equations and inequations
have a common algebraic solution over an algebraic closure of the coefficient field,
and `false` otherwise.

Accept each branch as an `AlgebraicSystem`, `DifferenceSystem`, `DifferentialSystem`,
or a pair `(E, I)` representing `E = 0`, `I != 0`.

For algebraic systems, return `true` exactly when their solution sets intersect:

`V(E_1 ∪ E_2) ∩ D(prod(I_1, I_2)) ≠ ∅`.

For action systems, convert both branches together to one finite algebraic
ring and treat the occurring jets as independent coordinates.
Do not impose additional shifted or differentiated equations, even when the input objects
are certified simple. Interpret the result as follows:

- `false`: the supplied algebraic constraints are incompatible, so there is
  no common difference or differential solution either.
- `true`: the supplied algebraic constraints are compatible; a common
  difference or differential solution may still not exist.

    # Examples

    ```jldoctest
    julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);
    julia> P = elem_type(R);
    julia> algebraic_branch_intersection(([x], P[]), ([x, y], P[]))
    true
    julia> algebraic_branch_intersection(([x], [y]), ([x, y], P[]))
    false

    julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);    
    julia> branches = thomas_decomposition([(x^2 - 1)*y]);
    julia> all(!algebraic_branch_intersection(branches[i], branches[j])
           for i in eachindex(branches)
           for j in (i + 1):lastindex(branches)
           )
    true
    ```

The function does not detect differential incompatibilities:
x=u[0] and y=u[1] are independent as algebraic variables in polynomial_ring(QQ,[:y,:x])

    ```jldoctest
    julia> D, u = differential_polynomial_ring(QQ, :u, 1);
    julia> A = DifferentialSystem([u]);
    julia> B = DifferentialSystem([u[1] - 1]);
    julia> algebraic_branch_intersection(A, B)
    true
    ```

Certify differential disjointness when the finite algebraic intersection is empty.
The following two equations cannot hold together, since they differ by 1.

    ```jldoctest
    julia> D, u = differential_polynomial_ring(QQ, :u, 1);
    julia> A = DifferentialSystem([u[1]-u]);
    julia> B = DifferentialSystem([u[1]-u-1]);
    julia> algebraic_branch_intersection(A, B)
    false
    ```
"""
function algebraic_branch_intersection(
    left::Union{AlgebraicSystem,AbstractActionSystem,Tuple},
    right::Union{AlgebraicSystem,AbstractActionSystem,Tuple}
)
    left_vectors, right_vectors = branch_conversion([left, right])
    all(isempty, (left_vectors..., right_vectors...)) && return true
    E1, I1 = left_vectors
    E2, I2 = right_vectors
    # Convert action pairs to one finite algebraic ring before cleanup.
    if eltype(E1) <: ActionPolyRingElem
        E1, I1, branches = __action_splitting_algebraic_conversion(E1, I1, [right_vectors])
        E2, I2 = only(branches)
    end
    # A common solution must satisfy both equation sets and every guard from
    # both branches; the helper tests non-emptiness of this locally closed set.
    return __algebraic_branch_intersection(E1, I1, E2, I2)
end

function __algebraic_branch_intersection(E1, I1, E2, I2)
    R = __require_same_parent(E1, I1, E2, I2)
    R === nothing && return true
    P = elem_type(R)
    E = P[collect(E1); collect(E2)]
    I = P[collect(I1); collect(I2)]
    clean = __algebraic_cleanup(E, I)
    clean === nothing && return false
    E_clean, I_clean = clean
    isempty(E_clean) && isempty(I_clean) && return true
    return __compatible_zero_algebraic_system(zero(R), E_clean, I_clean)
end

###############################################################################
#####  Branch unions and coverage
###############################################################################

@doc raw"""
    splitting_covers_system(
        E::AbstractVector,
        I::AbstractVector,
        branches::AbstractVector;
        return_remainder::Bool=false,
        vector_output::Bool=false
    )
    splitting_covers_system(
    E::AbstractVector,
    branches::AbstractVector;
    kwargs...
    )
    splitting_covers_system(
        system::Tuple{<:AbstractVector,<:AbstractVector},
        branches::AbstractVector;
        kwargs...
    )
    splitting_covers_system(
        system::AlgebraicSystem,
        branches::AbstractVector;
        return_remainder::Bool=false,
        vector_output::Bool=false
    )

Return `true` if the solution set of the algebraic system `E` = 0, `I` ≠ 0 is
exactly the union of the solution sets of the systems in `branches`, and `false` otherwise.

The input system may be an `AlgebraicSystem`, a pair `(E, I)`, or separate vectors.
Each branch independently may be an `AlgebraicSystem` or a pair of vectors;
mixed lists are accepted and converted with `branch_conversion`.

The number of branches is arbitrary.
The test subtracts the branches one at a time from the initial system and also checks
that no branch contributes points outside the initial system.

If `return_remainder` is set to `true`, the function returns a named tuple `(covers, missing, extra)`, where\
`missing` is the part of the initial system not covered by the branches, and\
`extra` is the part of the branches outside the initial system.\
Both remainders are vectors of `AlgebraicSystem` by default, with `quasi_simple === nothing`;
if `vector_output` is set to `true`, they are vectors of pairs `(equations, inequations)` instead.

For action polynomial vectors, compare the input and all branches in one finite algebraic ring,
treating the occurring jets as independent coordinates, without imposing additional shifted or differentiated equations.\
If the result is equal to `true`, it can be interpreted as a sufficient certificate of equality of the action-solution:
restricting equal algebraic sets to jets that respect the action maps preserves equality.\
If the result is equal to `false`, this information is inconclusive for action systems;
the algebraic sets may differ only at jet assignments that do not extend to valid solutions.
For example, compare the differential system `{u = 0}` with the single branch `{u = 0, u' = 0}`.
The finite algebraic input permits `(u, u') = (0, 1)`, which the branch excludes.
The differential solution sets are nevertheless equal, since differentiating `u = 0` already gives `u' = 0`.

Remainders use the finite ring's coordinates, so a nonempty `missing` or `extra` need not contain an action solution.
Check disjointness separately; this function checks equality with the union.

    # Examples

    ```jldoctest
    julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);
    julia> branches = quasi_simple_splitting([(x^2 - 1)*y]; factorization=true);
    julia> splitting_covers_system([(x^2 - 1)*y], branches)
    true

    julia> branch_with_extra_points = [([x], empty([x])), ([y], empty([x]))];
    julia> splitting_covers_system([x], branch_with_extra_points)
    false

    julia> D, u = differential_polynomial_ring(QQ, :u, 1);
    julia> splitting_covers_system([u], [([u, u[1]], elem_type(D)[])])
    false

    julia> D, u = differential_polynomial_ring(QQ, :u, 1);
    julia> E = [(u[1]-u)*(u[1]-u-1)];
    julia> branches = [DifferentialSystem([u[1]-u]), DifferentialSystem([u[1]-u-1])];
    julia> algebraic_branch_intersection(branches[1], branches[2])
    false
    julia> splitting_covers_system(E, branches)
    true
    ```
"""
function splitting_covers_system(
    E::AbstractVector,
    I::AbstractVector,
    branches::AbstractVector;
    return_remainder::Bool=false,
    vector_output::Bool=false,
)
    branch_vectors = branch_conversion(branches)
    R = __require_same_parent(E, I, branches)
    P = __splitting_input_element_type(R, E, I, branch_vectors)
    E = collect(P, E)
    I = collect(P, I)
    branch_vectors = Tuple{Vector{P},Vector{P}}[(collect(P, F), collect(P, G)) for (F, G) in branch_vectors]
    if P <: ActionPolyRingElem
        return __action_splitting_covers_system(E, I, branch_vectors; return_remainder, vector_output)
    end
    result = __algebraic_splitting_covers_system(E, I, branch_vectors; return_remainder)
    return return_remainder ? __splitting_remainder_output(result, R; vector_output) : result
end

function splitting_covers_system(
    E::AbstractVector,
    branches::AbstractVector;
    kwargs...
)
    return splitting_covers_system(E, eltype(E)[], branches; kwargs...)
end

function splitting_covers_system(
    system::Tuple{<:AbstractVector,<:AbstractVector},
    branches::AbstractVector;
    kwargs...
)
    return splitting_covers_system(system..., branches; kwargs...)
end

function splitting_covers_system(
    system::AlgebraicSystem,
    branches::AbstractVector;
    return_remainder::Bool=false,
    vector_output::Bool=false
)
    __require_parent(system.ring, branches)
    result = splitting_covers_system(__algebraic_copy_system(system)..., branches; return_remainder, vector_output=true)
    return return_remainder ? __splitting_remainder_output(result, system.ring; vector_output) : result
end

function __algebraic_splitting_covers_system(
    E::Vector{T},
    I::Vector{T},
    branches::Vector{Tuple{Vector{T},Vector{T}}};
    return_remainder::Bool
) where {T<:MPolyRingElem}
    initial = __algebraic_normalized_nonempty([(E, I)])
    branches = __algebraic_normalized_nonempty(branches)
    # What remains after removing every output branch consists of lost solutions.
    missing = __algebraic_subtract_branch_list(initial, branches)
    # Conversely, subtract the initial system from every branch to detect
    # solutions introduced by the purported splitting.
    extra = Tuple{Vector{T},Vector{T}}[]
    for branch in branches
        append!(extra, __algebraic_subtract_branch_list(
            Tuple{Vector{T},Vector{T}}[branch],
            initial
        ))
        extra = __algebraic_deduplicate_branch_list(extra)
    end
    # Equality holds precisely when both relative complements are empty.
    covers = isempty(missing) && isempty(extra)
    result = (covers=covers, missing=missing, extra=extra)
    return return_remainder ? result : covers
end

"""
Start from the union of the branches in `sources`, and remove every point belonging to any branch in `subtractors`.
Return a vector of branches describing the remaining points.
"""
function __algebraic_subtract_branch_list(
    sources::Vector{Tuple{Vector{T},Vector{T}}},
    subtractors::Vector{Tuple{Vector{T},Vector{T}}},
) where {T<:MPolyRingElem}
    current = copy(sources)
    for subtractor in subtractors
        next_current = Tuple{Vector{T},Vector{T}}[]
        for source in current
            append!(next_current, __algebraic_subtract_branch(source, subtractor))
        end
        current = __algebraic_deduplicate_branch_list(next_current)
        isempty(current) && break
    end
    return current
end

"""
Start from one branch `source = (E, I)` and remove the solutions of one branch `subtractor = (F, G)`.
Require both branches to be normalized and known to have solutions,
with their common parent already checked by the caller.
Return normalized, nonempty branches of this new set of solutions.
"""
function __algebraic_subtract_branch(
    source::Tuple{Vector{T},Vector{T}},
    subtractor::Tuple{Vector{T},Vector{T}},
) where {T<:MPolyRingElem}
    current = Tuple{Vector{T},Vector{T}}[source]
    F, G = subtractor
    outside = Tuple{Vector{T},Vector{T}}[]
    for (equation, polynomials) in ((true, F), (false, G)), p in polynomials
        next_current = Tuple{Vector{T},Vector{T}}[]
        for (E, I) in current
            zero = __algebraic_normalized_nonempty([([E; p], I)])
            nonzero = __algebraic_normalized_nonempty([(E, [I; p])])
            append!(outside, equation ? nonzero : zero)
            append!(next_current, equation ? zero : nonzero)
        end
        current = __algebraic_deduplicate_branch_list(next_current)
        isempty(current) && return __algebraic_deduplicate_branch_list(outside)
    end
    return __algebraic_deduplicate_branch_list(outside)
end

function __algebraic_normalized_nonempty(systems::Vector{Tuple{Vector{T},Vector{T}}}) where {T<:MPolyRingElem}
    out = __algebraic_normalized(systems)
    filter!(branch -> __algebraic_branch_has_solution(branch...), out)
    return __algebraic_deduplicate_branch_list(out)
end

function __algebraic_branch_has_solution(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    isempty(E) && isempty(I) && return true
    p = isempty(E) ? first(I) : first(E)
    return __compatible_zero_algebraic_system(zero(p), E, I)
end

function __algebraic_deduplicate_branch_list(branches::Vector{Tuple{Vector{T},Vector{T}}}) where {T<:MPolyRingElem}
    return unique(branch -> __algebraic_key(branch...), branches)
end

function __splitting_remainder_output(result::NamedTuple, R; vector_output::Bool)
    vector_output && return deepcopy(result)
    if R === nothing
        isempty(result.missing) && isempty(result.extra) &&
            return (covers=result.covers, missing=AlgebraicSystem[], extra=AlgebraicSystem[])
        throw(ArgumentError("Empty vectors do not determine a parent; pass AlgebraicSystem(R) or vector_output=true"))
    end
    S = AlgebraicSystem{typeof(R),elem_type(R)}
    return (covers=result.covers,
        missing=S[AlgebraicSystem(R, E, I) for (E, I) in result.missing],
        extra=S[AlgebraicSystem(R, E, I) for (E, I) in result.extra])
end

function __splitting_input_element_type(
    R,
    E::AbstractVector,
    I::AbstractVector,
    branches::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}}
)
    polynomial_lists = Iterators.flatten(((E, I), Iterators.flatten(branches)))
    if R !== nothing
        P = elem_type(R)
        P <: Union{MPolyRingElem,ActionPolyRingElem} ||
            error("Every equation and guard must be a polynomial")
        all(p -> p isa P, Iterators.flatten(polynomial_lists)) ||
            error("All equations and guards must have the same polynomial element type")
        return P
    end
    # Empty lists can still record their polynomial element type.
    for polys in polynomial_lists
        P = eltype(polys)
        P <: Union{MPolyRingElem,ActionPolyRingElem} && return P
    end
    return MPolyRingElem
end

###############################################################################
#####  Difference/Differential systems
###############################################################################

function __action_splitting_algebraic_conversion(
    E::AbstractVector{P},
    I::AbstractVector{P},
    branches::AbstractVector{<:Tuple{<:AbstractVector{P},<:AbstractVector{P}}},
) where {P<:ActionPolyRingElem}
    total_length = length(E) + length(I) + sum(branch -> length(branch[1]) + length(branch[2]), branches; init=0)
    system = P[]
    sizehint!(system, total_length)
    append!(system, E)
    append!(system, I)
    for (F, G) in branches
        append!(system, F)
        append!(system, G)
    end
    isempty(system) && return nothing
    conversion = Oscar.__algebraic_conversion_data_from_polys(parent(first(system)), system)
    Q = elem_type(conversion.mpr)
    E_alg = Q[conversion(p) for p in E]
    I_alg = Q[conversion(q) for q in I]
    branches_alg = Tuple{Vector{Q},Vector{Q}}[
        (Q[conversion(p) for p in F], Q[conversion(q) for q in G])
        for (F, G) in branches
    ]
    return E_alg, I_alg, branches_alg
end

function __action_splitting_covers_system(
    E::Vector{P},
    I::Vector{P},
    branches::Vector{Tuple{Vector{P},Vector{P}}};
    return_remainder::Bool,
    vector_output::Bool,
) where {P<:ActionPolyRingElem}
    converted = __action_splitting_algebraic_conversion(E,I,branches)
    if converted === nothing
        covers = !isempty(branches)
        empty_branches = Tuple{Vector{P},Vector{P}}[]
        result = (covers=covers, missing = covers ? empty_branches : [(E,I)], extra=empty_branches)
        return return_remainder ? __splitting_remainder_output(result, nothing; vector_output) : covers
    end
    E_alg, I_alg, branches_alg = converted
    result = __algebraic_splitting_covers_system(E_alg, I_alg, branches_alg; return_remainder)
    return_remainder || return result
    # All converted polynomials share a parent, even when E_alg and I_alg are empty.
    polynomial_lists = Iterators.flatten(((E_alg, I_alg), branches_alg))
    R = parent(first(Iterators.flatten(polynomial_lists)))
    return __splitting_remainder_output(result, R; vector_output)
end