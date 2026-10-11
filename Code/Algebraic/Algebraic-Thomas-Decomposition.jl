###############################################################################
#####  Branch conversion
###############################################################################

@doc raw"""
    branch_conversion(branches::AbstractVector)

Return a vector of pairs `(equations, inequations)` containing independent polynomial copies.
Each input branch may independently be an `AlgebraicSystem`, a `DifferenceSystem`,
a `DifferentialSystem`, or a pair of polynomial vectors; the same list may contain
objects and pairs. All branches must have the same parent.

The input systems and their certificates are unchanged. Empty pairs do not carry a parent.

    # Examples

    ```jldoctest
    julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);
    julia> branch_conversion([AlgebraicSystem([x^2*y-x], [y-x]), ([x^2-y^3],[y])])
    2-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
     ([x^2*y - x], [-x + y])
     ([x^2 - y^3], [y])
    julia> branch_conversion([AlgebraicSystem([x^2*y-x], [y-x]), AlgebraicSystem([x^2-y^3], [y])])
    2-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
     ([x^2*y - x], [-x + y])
     ([x^2 - y^3], [y])
    ```
"""
function branch_conversion(branches::AbstractVector)
    for branch in branches
        branch isa Union{AlgebraicSystem,AbstractActionSystem} && continue
        branch isa Tuple{AbstractVector,AbstractVector} ||
            throw(ArgumentError("Each branch must be an algebraic or action system, or a pair of polynomial vectors"))
        all(p -> p isa Union{MPolyRingElem,ActionPolyRingElem}, Iterators.flatten(branch)) ||
            throw(ArgumentError("Every equation and inequation must be a polynomial"))
    end
    R = __require_same_parent(branches)
    # Only empty pairs (or an empty branch list) can have no parent.
    R === nothing && return Tuple{Vector,Vector}[(collect(E), collect(I)) for (E, I) in branches]
    P = elem_type(R)
    return Tuple{Vector{P},Vector{P}}[
        branch isa Union{AlgebraicSystem,AbstractActionSystem} ? system_to_vectors(branch) :
            (P[deepcopy(p) for p in branch[1]], P[deepcopy(q) for q in branch[2]])
        for branch in branches
    ]
end

# Preserve the element type even for an empty vector of concrete systems.
function branch_conversion(branches::AbstractVector{<:AlgebraicSystem{R,P}}) where {R,P}
    __require_same_parent(branches)
    return Tuple{Vector{P},Vector{P}}[system_to_vectors(branch) for branch in branches]
end

###############################################################################
#####  Algebraic splitting
###############################################################################

@doc raw"""
    quasi_simple_splitting(
        E::AbstractVector{<:MPolyRingElem},
        I::AbstractVector{<:MPolyRingElem};
        factorization::Bool=true,
        print_output::Bool=false,
        vector_output::Bool=false
    )
    quasi_simple_splitting(
        E::AbstractVector{<:MPolyRingElem};
        factorization::Bool=true,
        print_output::Bool=false,
        vector_output::Bool=false
    )
    quasi_simple_splitting(system::AlgebraicSystem; kwargs...)

Split the finite algebraic system `E` = 0, `I` ≠ 0 into quasi-simple algebraic branches.\
When only `E` is supplied, the guard list `I` is taken to be empty.

The algorithm repairs repeated leaders and initials that may vanish, and it discards inconsistent branches.
During branch cleanup, equations and guards are replaced by equivalent reduced forms when lower equations or
already nonzero guards make this possible.

A completely inconsistent input returns an empty vector of branches.\
A single branch with no equations or inequations represents the whole space.

All input forms return a vector of `AlgebraicSystem` branches by default.
Each returned branch has `quasi_simple === true`; the input is unchanged.
New intermediate systems start with `quasi_simple === nothing` and are certified when processed.
Pass `AlgebraicSystem(R)` for an empty system, since empty vectors do not determine a parent.

# Keyword arguments

- `factorization`:\
If `true`, equations are split into disjoint branches defined by their
nonconstant factors and the necessary complementary guards.
Guards are replaced by the simultaneous nonvanishing of their nonconstant factors.\
If `false`, the algorithm only applies the ordinary algebraic splitting repairs.

- `vector_output`:\
If `true`, return pairs `(equations, inequations)` of independent polynomial vectors using `branch_conversion`.\
The default `false` returns systems.

- `print_output`:\
If `true`, the final branches are printed with print\_splitting\_result.
To avoid repeated outputs, a semicolon should be written before launching the command.\
If `false`, no extra formatted output is printed.\
In both cases, the returned vector of branches is unchanged.

    # Examples

    ```jldoctest
    julia> R, (x, c, b, a) = polynomial_ring(QQ, [:x, :c, :b, :a]);
    julia> p = a*x^2 + b*x + c;
    julia> quasi_simple_splitting([p])
    3-element Vector{AlgebraicSystem{QQMPolyRing, QQMPolyRingElem}}:
     AlgebraicSystem(3 equations, 0 inequations; quasi-simple = true)
     AlgebraicSystem(2 equations, 1 inequations; quasi-simple = true)
     AlgebraicSystem(1 equations, 1 inequations; quasi-simple = true)
    julia> quasi_simple_splitting([p]; vector_output=true)
    3-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
     ([c, a, b], [])
     ([x*b + c, a], [b])
     ([x^2*a + x*b + c], [a])
    julia> quasi_simple_splitting([p]; print_output=true);
    Branch 1
      E = QQMPolyRingElem[c, a, b]
      I = QQMPolyRingElem[]
    Branch 2
      E = QQMPolyRingElem[x*b + c, a]
      I = QQMPolyRingElem[b]
    Branch 3
      E = QQMPolyRingElem[x^2*a + x*b + c]
      I = QQMPolyRingElem[a]

    julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);
    julia> quasi_simple_splitting([(x^2 - 1)*y]; factorization=false, print_output=true);
    Branch 1
      E = QQMPolyRingElem[y]
      I = QQMPolyRingElem[]
    Branch 2
      E = QQMPolyRingElem[(x + 1)*(x - 1)]
      I = QQMPolyRingElem[y]
    julia> quasi_simple_splitting([(x^2 - 1)*y]; factorization=true, print_output=true);
    Branch 1
      E = QQMPolyRingElem[y]
      I = QQMPolyRingElem[(x + 1)*(x - 1)]
    Branch 2
      E = QQMPolyRingElem[x - 1]
      I = QQMPolyRingElem[]
    Branch 3
      E = QQMPolyRingElem[x + 1]
      I = QQMPolyRingElem[]
    ```
"""
function quasi_simple_splitting(system::AlgebraicSystem; kwargs...)
    return __quasi_simple_splitting(system; kwargs..., simple=false)
end

function quasi_simple_splitting(
    E::AbstractVector,
    I::AbstractVector=eltype(E)[];
    kwargs...
)
    return __quasi_simple_splitting(E, I; kwargs..., simple=false)
end

###############################################################################
#####  Discriminant refinement - Algebraic Thomas decomposition
###############################################################################

@doc raw"""
    thomas_decomposition(
        E::AbstractVector{<:MPolyRingElem},
        I::AbstractVector{<:MPolyRingElem};
        factorization::Bool=true,
        print_output::Bool=false,
        vector_output::Bool=false
    )
    thomas_decomposition(
        E::AbstractVector{<:MPolyRingElem};
        factorization::Bool=true,
        print_output::Bool=false,
        vector_output::Bool=false
    )
    thomas_decomposition(system::AlgebraicSystem; kwargs...)

Split the finite algebraic system `E` = 0, `I` ≠ 0 into disjoint simple algebraic branches.

Thomas decomposition requires a coefficient field of characteristic zero.\
On a discriminant-zero branch, quadratics are replaced by their derivative with respect to the leader variable.\
For higher degrees, first try coefficient reduction by lower equations followed by removal of repeated factors.
Use the result only when its degree in the leader decreases and every pseudo multiplier is certified nonzero.
With `factorization=true`, also try one factor of the discriminant at a time when reduction on its
whole zero locus does not lower the degree. Keep both the zero and nonzero cases of that factor.
Otherwise, split by principal subresultant coefficients and replace the polynomial on each branch by its
square-free part, retaining every equation and inequations involving the lower variables that define that branch. 
The same replacement preserves both its zero and nonzero loci; no radical or lexicographic Groebner basis is needed for this step.

For every polynomial `r` in `E` or in `I`, let v = ld(r), d = deg\_v(r), and

disc(r) = (-1)^(d(d-1)/2) * res(r, dr/dv, v) / init(r).

The extra condition requires disc(r) = 0 to have no solution in the partial system made of the equations and guards
whose leaders are at most ld(r), including the constraint defined by r itself.
Use the same discriminant condition as `is_simple`.

When only `E` is supplied, the guard list `I` is taken to be empty.

All input forms return a vector of `AlgebraicSystem` branches by default.
Each returned branch has `quasi_simple === true` and `simple === true`; the input is unchanged.
Start each intermediate system with both certificates equal to `nothing` and certify it when processed.
Pass `AlgebraicSystem(R)` for an empty system, since empty vectors do not determine a parent.

# Keyword arguments

- `factorization`:\
If `true`, equations are split into disjoint branches defined by their
nonconstant factors and the necessary complementary guards.
Guards are replaced by the simultaneous nonvanishing of their nonconstant factors.\
If `false`, the algorithm only applies the ordinary algebraic splitting repairs.

- `vector_output`:\
If `true`, return pairs `(equations, inequations)` of independent
polynomial vectors using `branch_conversion`.\
The default `false` returns systems.

- `print_output`:\
If `true`, the final branches are printed with print\_splitting\_result.
To avoid repeated outputs, a semicolon should be written before launching the command.\
If `false`, no extra formatted output is printed.\
In both cases, the returned vector of branches is unchanged.

    # Examples

    ```jldoctest
    julia> R, (x,) = polynomial_ring(QQ, [:x]);
    julia> thomas_decomposition(AlgebraicSystem([x^2+x]); vector_output=true)
    2-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
     ([x + 1], [])
     ([x], [])
    julia> thomas_decomposition(AlgebraicSystem([x^2+x]); vector_output=true, factorization=false)
    1-element Vector{Tuple{Vector{QQMPolyRingElem}, Vector{QQMPolyRingElem}}}:
     ([x^2 + x], [])

    julia> R, (x, c, b, a) = polynomial_ring(QQ, [:x, :c, :b, :a]);
    julia> p = a*x^2 + b*x + c;
    julia> thomas_decomposition([p])
    4-element Vector{AlgebraicSystem{QQMPolyRing, QQMPolyRingElem}}:
     AlgebraicSystem(3 equations, 0 inequations; simple = true)
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(1 equations, 2 inequations; simple = true)
    julia> thomas_decomposition([p]; print_output=true);
    Branch 1
      E = QQMPolyRingElem[c, a, b]
      I = QQMPolyRingElem[]
    Branch 2
      E = QQMPolyRingElem[x*b + c, a]
      I = QQMPolyRingElem[b]
    Branch 3
      E = QQMPolyRingElem[2*x*a + b, -4*c*a + b^2]
      I = QQMPolyRingElem[a]
    Branch 4
      E = QQMPolyRingElem[x^2*a + x*b + c]
      I = QQMPolyRingElem[a, -4*c*a + b^2]

    julia> R, (x, y) = polynomial_ring(QQ, [:x, :y]);
    julia> system = AlgebraicSystem([x^2-y]);
    julia> quasi = quasi_simple_splitting(system; print_output=true);
    Branch 1
      E = QQMPolyRingElem[x^2 - y]
      I = QQMPolyRingElem[]
    julia> length(quasi), is_simple(only(quasi))
    (1, false)
    julia> branches = thomas_decomposition(system; print_output=true);
    Branch 1
      E = QQMPolyRingElem[x, y]
      I = QQMPolyRingElem[]
    Branch 2
      E = QQMPolyRingElem[x^2 - y]
      I = QQMPolyRingElem[y]

    julia> R, (x,y,z) = polynomial_ring(QQ, [:x,:y,:z]);
    julia> steiner = x^2*y^2 + x^2*z^2 + y^2*z^2 - x*y*z;
    julia> thomas_decomposition(AlgebraicSystem([steiner]))
    11-element Vector{AlgebraicSystem{QQMPolyRing, QQMPolyRingElem}}:
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(3 equations, 0 inequations; simple = true)
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(3 equations, 0 inequations; simple = true)
     AlgebraicSystem(3 equations, 0 inequations; simple = true)
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(2 equations, 1 inequations; simple = true)
     AlgebraicSystem(1 equations, 2 inequations; simple = true)
    ```
"""
function thomas_decomposition(
    E::AbstractVector,
    I::AbstractVector=eltype(E)[];
    kwargs...
)
    return __quasi_simple_splitting(E, I; kwargs..., simple=true)
end

function thomas_decomposition(system::AlgebraicSystem; kwargs...)
    return __quasi_simple_splitting(system; kwargs..., simple=true)
end

function __algebraic_discriminant_split(
    E::Vector{T},
    I::Vector{T},
    issue::NamedTuple;
    factorize::Bool=true
) where {T<:MPolyRingElem}
    i = issue.index
    disc = __algebraic_squarefree_part(issue.discriminant)
    is_equation = issue.kind === :equation
    p = is_equation ? E[i] : I[i]
    k = var_index(__algebraic_leader(p))
    lower_guards = T[q for q in I if !is_constant(q) && var_index(__algebraic_leader(q)) > k]
    # These guards remain in both branches, so their factors cannot make the discriminant vanish.
    disc = __algebraic_strip_guard_factors(disc, lower_guards)
    if degree(p, k) > 2
        replacement = __algebraic_specialized_squarefree_part(p, [issue.E_partial; disc], I, k)
        if replacement !== nothing
            return __algebraic_split_zero(E, I, issue.kind, i, disc, replacement)
        end
        if factorize
            conditions = __algebraic_nonconstant_factors(disc)
            if length(conditions) > 1
                lower_E, lower_I = __algebraic_subresultant_lower_context(E, I, k)
                for h in conditions
                    replacement = __algebraic_specialized_squarefree_part(p, [issue.E_partial; h], I, k)
                    replacement === nothing && continue
                    __subresultant_condition_status(h, lower_E, lower_I) === true && continue
                    return __algebraic_split_zero(E, I, issue.kind, i, h, replacement)
                end
            end
        end
        return __algebraic_subresultant_squarefree_step(E, I, issue, disc, k)
    end
    # A valid discriminant issue has degree at least two. Higher degrees were
    # handled above; on the quadratic discriminant-zero locus use the derivative.
    return __algebraic_split_zero(E, I, issue.kind, i, disc, derivative(p, k))
end

"""
Work over a coefficient field of characteristic zero, with `idx` the leader index of `p`.
Try to simplify `p` on the locus `E = 0`, `I ≠ 0` without computing an ideal or a Groebner basis.
Use only equations and guards with leaders strictly below that of `p`.
In particular, never use `p ≠ 0` itself to justify cancelling a factor of its replacement.

Reduce in the leaders of the lower equations, allowing changes to every coefficient of `p`.
Use exact cancellation when possible; otherwise multiply by an initial only if its nonvanishing
follows from the retained lower guards. Strip factors certified nonzero and remove repeated factors.
The result has the same zero and nonzero loci as `p` on this locus.
Return it only if its degree in the original leader decreases; return `nothing` otherwise.
This strict decrease permits another splitting step without revisiting the same discriminant issue.
"""
function __algebraic_specialized_squarefree_part(
    p::T, E::Vector{T}, I::Vector{T}, idx::Int
) where {T<:MPolyRingElem}
    guards = T[q for q in I if !is_constant(q) && var_index(__algebraic_leader(q)) > idx]
    equations = T[f for f in E if !is_constant(f) && var_index(__algebraic_leader(f)) > idx]
    sort!(equations; by=f -> var_index(__algebraic_leader(f)))
    reduced = p
    for equation in equations
        # Removing a factor known to be nonzero preserves this lower equation.
        g = __algebraic_strip_guard_factors(equation, guards)
        is_constant(g) && continue
        v = __algebraic_leader(g)
        j = var_index(v)
        dg = degree(g, j)
        dr = degree(reduced, j)
        dr < dg && continue
        h = leading_coefficient(g, j)
        initial_is_one = isone(h)
        safe_initial = nothing
        while dr >= dg
            a = leading_coefficient(reduced, j)
            division = initial_is_one ? (divides=true, quotient=a) : __checked_divides(h, a)
            if !division.divides
                if safe_initial === nothing
                    safe_initial = is_constant(__algebraic_strip_guard_factors(h, guards))
                end
                safe_initial || break
            end
            shift = dr - dg
            shifted_g = iszero(shift) ? g : v^shift * g
            reduced = division.divides ? reduced - division.quotient * shifted_g :
                                        h * reduced - a * shifted_g
            dr = degree(reduced, j)
        end
        is_constant(reduced) && return reduced
    end
    reduced = __algebraic_strip_guard_factors(reduced, guards)
    is_constant(reduced) && return reduced
    reduced = __algebraic_squarefree_part(reduced)
    return degree(reduced, idx) < degree(p, idx) ? reduced : nothing
end

###############################################################################
#####  Splitting loop and termination
###############################################################################

"""
    Run the shared algebraic splitting algorithm. Set `simple=true` for a Thomas decomposition
        over a field of characteristic zero. Keep discriminant checks disabled otherwise.
    Preserve the input and return certified systems, or detached vector pairs with `vector_output=true`.
"""
function __quasi_simple_splitting(
    input::S;
    simple::Bool=false,
    factorization::Bool=true,
    print_output::Bool=false,
    vector_output::Bool=false
) where {S<:AlgebraicSystem}
    if simple && !iszero(characteristic(coefficient_ring(input.ring))) &&
        throw(ArgumentError("Discriminant checks require characteristic zero"))
    end
    # Store the result of __algebraic_splitting_loop_cached, reusing cached computations when available.
    result::Vector{S} = __with_algebraic_cache() do
        __algebraic_splitting_loop_cached(input; check_discriminants=simple, factorize=factorization)
    end
    __algebraic_finalize_result(result)
    print_output && print_splitting_result(result)
    return vector_output ? branch_conversion(result) : result
end

function __quasi_simple_splitting(
    E::AbstractVector,
    I::AbstractVector=eltype(E)[];
    kwargs...
)
    return __quasi_simple_splitting(AlgebraicSystem(E, I); kwargs...)
end

function __algebraic_splitting_loop_cached(
    input::AlgebraicSystem;
    check_discriminants::Bool,
    factorize::Bool
)
    initial_branch = __algebraic_cleanup(input)
    result = typeof(input)[]
    initial_branch === nothing && return result
    queue = typeof(input)[initial_branch]
    seen = Set{__PolynomialSystemKey}()
    while !isempty(queue)
        # Process the most recently appended system
        current = pop!(queue)
        key = __algebraic_key(current)
        key in seen && continue
        push!(seen, key)
        target_issue = __algebraic_target_issue(current; check_discriminants, factorize)
        if target_issue === nothing
            push!(result, current)
            continue
        end
        next_systems = __algebraic_step(current; check_discriminants, factorize, target_issue)
        append!(queue, next_systems)
    end
    return result
end

# The issue is valid only for this unchanged system and is treated in the next step.
function __algebraic_target_issue(
    system::AlgebraicSystem;
    check_discriminants::Bool,
    factorize::Bool=true
)
    issue = __algebraic_simplicity_issue(system)
    issue === nothing || return issue
    (factorize || check_discriminants) || return nothing
    E, I = __algebraic_read_system(system)
    if factorize
        issue = __algebraic_factorization_target(E, I)
        issue === nothing || return (kind=:factorization, data=issue)
    end
    if check_discriminants
        return __algebraic_simplicity_issue(system; simple=true)
    end
    return nothing
end

function __algebraic_finalize_result(result::AbstractVector{<:AlgebraicSystem})
    if isempty(result)
        println("INCONSISTENT SYSTEM")
    elseif length(result) == 1 && all(isempty, __algebraic_read_system(result[1]))
        println("SYSTEM ALWAYS VERIFIED")
    end
    return result
end

###############################################################################
#####  Splitting steps and redundant inequations
###############################################################################

function __algebraic_step(
    system::AlgebraicSystem;
    check_discriminants::Bool=false,
    factorize::Bool=true,
    target_issue::Union{Nothing,NamedTuple}=nothing
)
    E, I = __algebraic_read_system(system)
    simplification = __algebraic_inequation_simplification(E, I)
    # Three cases: :nothing, : inconsistent, :remove_inequation
    if simplification !== nothing
        simplification.kind === :inconsistent && return typeof(system)[]
        I_new = eltype(I)[I[j] for j in eachindex(I) if j != simplification.index]
        return __algebraic_step_systems(system, __algebraic_normalized([(E, I_new)]))
    end
    if factorize
        if target_issue !== nothing && target_issue.kind === :factorization
            factorization = target_issue.data
        elseif target_issue !== nothing && target_issue.kind === :discriminant
            # Already checked by __algebraic_target_issue before the discriminant.
            factorization = nothing
        else
            factorization = __algebraic_factorization_target(E, I)
        end
        factorization !== nothing &&
            return __algebraic_step_systems(system, __algebraic_factorization_split(E, I, factorization))
    end
    issue = target_issue === nothing ?
        __algebraic_simplicity_issue(E, I; simple=check_discriminants, quasi_simple=system.quasi_simple === true) :
        target_issue
    issue === nothing && return nothing
    branches = if issue.kind === :discriminant
        # The discriminant may vanish on the corresponding lower subsystem.
        __algebraic_discriminant_split(E, I, issue.data; factorize)
    elseif issue.kind === :quasi_simple
        if issue.reason === :leaders
            # Two polynomials have the same leader
            duplicate = issue.data
            repair = duplicate.kind === :equations ? __algebraic_same_leader_equations :
                     duplicate.kind === :inequations ? __algebraic_same_leader_inequations :
                     __algebraic_same_leader_equation_inequation
            repair(E, I, duplicate.first_index, duplicate.second_index)
        elseif issue.reason === :initial
            # An initial may vanish on the corresponding lower subsystem.
            initial = issue.data
            repair = initial.kind === :equation ?
                __algebraic_initial_equation :
                __algebraic_initial_inequation
            repair(E, I, initial.index; initial=initial.initial)
        elseif issue.reason === :constant
        # A constant equation or inequation must be removed or may make the system inconsistent.
            __algebraic_normalized([(E, I)])
        else
            return nothing
        end
    else
        return nothing
    end
    return __algebraic_step_systems(system, branches)
end

# Every nonempty branch list supplied here has passed through __algebraic_normalized.
function __algebraic_step_systems(system::S, branches) where {S<:AlgebraicSystem}
    return S[AlgebraicSystem(Val(:normalized), system.ring, E, I) for (E, I) in branches]
end

# Compatibility adapter; preserve detached vector output and the empty-input case.
function __algebraic_step(
    E::Vector{T},
    I::Vector{T};
    check_discriminants::Bool=false,
    factorize::Bool=true,
    target_issue::Union{Nothing,NamedTuple}=nothing
) where {T<:MPolyRingElem}
    isempty(E) && isempty(I) && return nothing
    branches = __algebraic_step(AlgebraicSystem(E, I); check_discriminants, factorize, target_issue)
    branches === nothing && return nothing
    return Tuple{Vector{T},Vector{T}}[system_to_vectors(branch) for branch in branches]
end

function __algebraic_inequation_simplification(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    for i in eachindex(I)
        issue = __algebraic_inequation_simplification(E, I, i)
        issue === nothing || return issue
    end
    return nothing
end

# Check only I[i] using the other lower-leader constraints as context
function __algebraic_inequation_simplification(E::Vector{T}, I::Vector{T}, i::Integer;
        exact::Bool=true) where {T<:MPolyRingElem}
    q = I[i]
    iszero(q) && return (kind=:inconsistent, index=i)
    is_constant(q) && return (kind=:remove_inequation, index=i)
    v = __algebraic_leader(q)
    E_partial = [p for p in E if !is_constant(p) && __algebraic_leader(p) <= v]
    I_partial = [p for (j, p) in pairs(I)
                  if j != i && !is_constant(p) && __algebraic_leader(p) <= v]
    if coefficient_ring(parent(q)) isa Field
        status = __subresultant_condition_status(q, E_partial, I_partial)
        status === true && return (kind=:remove_inequation, index=i)
        status === false && return (kind=:inconsistent, index=i)
    end
    # With exact=false, keep q ≠ 0 when the above checks are inconclusive,
    # avoiding other compatibility tests that may require expensive computations.
    exact || return nothing
    !__compatible_zero_algebraic_system(q, E_partial, I_partial) &&
        return (kind=:remove_inequation, index=i)
    __algebraic_vanishes_on_system(q, E_partial, I_partial) &&
        return (kind=:inconsistent, index=i)
    return nothing
end

###############################################################################
#####  Factorization steps
###############################################################################

function __algebraic_factorization_target(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    for i in eachindex(E)
        factors = __algebraic_nonconstant_factors(E[i])
        __algebraic_nontrivial_factorization(E[i], factors) &&
            return (kind=:equation, index=i, factors=factors)
    end
    for i in eachindex(I)
        factors = __algebraic_nonconstant_factors(I[i])
        grouped_factors = __algebraic_group_factors_by_leader(factors)
        __algebraic_nontrivial_factorization(I[i], grouped_factors) &&
            return (kind=:inequation, index=i, factors=grouped_factors)
    end
    return nothing
end

function __algebraic_nontrivial_factorization(p::T, factors::Vector{T}) where {T<:MPolyRingElem}
    p = __algebraic_remove_constant_factor(p)
    return length(factors) > 1 || (length(factors) == 1 && first(factors) != p)
end

function __algebraic_group_factors_by_leader(factors::Vector{T}) where {T<:MPolyRingElem}
    grouped = T[]
    leaders = T[]
    sizehint!(grouped, length(factors))
    sizehint!(leaders, length(factors))
    for f in factors
        v = __algebraic_leader(f)
        index = findfirst(==(v), leaders)
        if index === nothing
            push!(leaders, v)
            push!(grouped, f)
        else
            grouped[index] = __algebraic_remove_constant_factor(grouped[index] * f)
        end
    end
    return grouped
end

function __algebraic_factorization_split(
    E::Vector{T},
    I::Vector{T},
    issue::NamedTuple
) where {T<:MPolyRingElem}
    i = issue.index
    factors = collect(T, issue.factors)
    if issue.kind === :equation
        __algebraic_order_factors_for_disjoint_split!(factors)
        systems = Tuple{Vector{T},Vector{T}}[]
        previous_factors = T[]
        sizehint!(systems, length(factors))
        sizehint!(previous_factors, length(factors))
        for f in factors
            E_new = copy(E)
            E_new[i] = f
            push!(systems, (E_new, [I; previous_factors]))
            push!(previous_factors, f)
        end
        return __algebraic_normalized(systems)
    else
        I_new = [I[j] for j in eachindex(I) if j != i]
        append!(I_new, factors)
        return __algebraic_normalized([(E, I_new)])
    end
end

function __algebraic_order_factors_for_disjoint_split!(
    factors::Vector{T}
) where {T<:MPolyRingElem}
    sort!(factors; by=f -> begin
            i = var_index(__algebraic_leader(f))
            (i, -degree(f, i), string(f))
        end
    )
    return factors
end

###############################################################################
#####  Repeated leaders
###############################################################################

"""
Repair two equations with the same leader. If the smaller leader degree is one,
pseudo-divide and retain both cases for the divisor initial. Otherwise call the
shared conditional-GCD step directly, deferring the first undecided zero condition.
Retain all parameter conditions. Discard degree-zero GCD cases; remove equations
that vanish identically on a case during cleanup.
"""
function __algebraic_same_leader_equations(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    f, g = E[i], E[j]
    v = __algebraic_leader(f)
    reduce_index, divisor_index = i, j
    k = var_index(v)
    if degree(f, k) < degree(g, k)
        reduce_index, divisor_index = j, i
        f, g = g, f
    end
    if degree(g, k) > 1
        return __algebraic_subresultant_pair_step(E, I, i, j, k, :equations)
    end
    h = leading_coefficient(g, k)
    _, r = __algebraic_pseudo_divrem(f, g; compute_quotient=false)
    E_nonzero = copy(E)
    E_nonzero[reduce_index] = r
    E_zero = copy(E)
    E_zero[divisor_index] = __algebraic_tail(g)
    push!(E_zero, h)
    return __algebraic_normalized([(E_nonzero, [I; h]), (E_zero, I)])
end

"""
Replace two inequations with the same leader by their polynomial LCM. Remove
their ambient GCD before multiplying, without splitting parameter cases merely
to obtain a minimal specialized LCM. Preserve the nonzero locus even when the
specialized GCD grows. Leave any required initial and discriminant cases to
the usual simplicity checks.
"""
function __algebraic_same_leader_inequations(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    f, g = I[i], I[j]
    q = divexact(f, gcd(f, g))*g
    new_I = T[I[k] for k in eachindex(I) if k != j]
    new_I[i - (j < i)] = q
    return __algebraic_normalized([(E, new_I)])
end

"""
First reduce the equation and inequation modulo lower equations. If either
changes, return the normalized replacement system for further processing.
Otherwise call the shared conditional-GCD step directly for `f = 0, q != 0`.
For coprime inputs with nonzero `f`, drop the inequation. Otherwise replace the
equation by its certified pseudo-cofactor and retain the inequation, repeating
in the splitting loop if necessary. Discard cases where every root of `f` is a
root of `q`, or where `q` vanishes identically. If `f` vanishes identically,
retain the specialized inequation. Keep all parameter conditions.
"""
function __algebraic_same_leader_equation_inequation(E::Vector{T}, I::Vector{T}, i::Integer, j::Integer) where {T<:MPolyRingElem}
    # Reduce both constraints modulo lower equations before building their subresultants.
    f, q = E[i], I[j]
    v = __algebraic_leader(f)
    k = var_index(v)
    E_lower = [p for p in E if !is_constant(p) && __algebraic_leader(p) < v]
    f_reduced, q_reduced = __algebraic_normal_form_mod_equations([f, q], E_lower)
    if f_reduced != f || q_reduced != q
        E_reduced = copy(E)
        I_reduced = copy(I)
        E_reduced[i] = f_reduced
        I_reduced[j] = q_reduced
        return __algebraic_normalized([(E_reduced, I_reduced)])
    end
    return __algebraic_subresultant_pair_step(E, I, i, j, k, :mixed)
end

###############################################################################
#####  Initial conditions
###############################################################################

function __algebraic_split_zero(
    E::Vector{T}, I::Vector{T}, kind::Symbol, i::Integer, h::T, replacement::T
) where {T<:MPolyRingElem}
    E_zero, I_zero = copy(E), kind === :inequation ? copy(I) : I
    target = kind === :equation ? E_zero : I_zero
    target[i] = replacement
    push!(E_zero, h)
    return __algebraic_normalized([(E, [I; h]), (E_zero, I_zero)])
end

function __algebraic_initial_equation(
    E::Vector{T}, I::Vector{T}, i::Integer; initial::T=__algebraic_initial(E[i])
) where {T<:MPolyRingElem}
    return __algebraic_split_zero(E, I, :equation, i, initial, __algebraic_tail(E[i]))
end

function __algebraic_initial_inequation(
    E::Vector{T}, I::Vector{T}, i::Integer; initial::T=__algebraic_initial(I[i])
) where {T<:MPolyRingElem}
    return __algebraic_split_zero(E, I, :inequation, i, initial, __algebraic_tail(I[i]))
end

###############################################################################
#####  Other helpers used throughout multiple functions above
###############################################################################

# System keys, copies and branch vectors
# --------------------------------------

function __algebraic_key(R::Union{Nothing,Ring}, E::AbstractVector{P}, I::AbstractVector{P}) where {P<:MPolyRingElem}
    return __polynomial_system_key(R, E, I; normalize=__algebraic_remove_constant_factor)
end

__algebraic_key(system::AlgebraicSystem) = getfield(system, :_key)

function __algebraic_key(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    R = !isempty(E) ? parent(first(E)) : !isempty(I) ? parent(first(I)) : nothing
    return __algebraic_key(R, E, I)
end

__algebraic_copy_system(system::AlgebraicSystem) = (system.equations, system.inequations)
__algebraic_read_system(system::AlgebraicSystem) = (getfield(system, :equations), getfield(system, :inequations))

function __algebraic_copy_system(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    __require_same_parent(E, I)
    return deepcopy.(E), deepcopy.(I)
end

# Branch cleanup and normalization
# --------------------------------

function __algebraic_cleanup(system::AlgebraicSystem)
    branch = __algebraic_cleanup(__algebraic_read_system(system)...)
    branch === nothing && return nothing
    return AlgebraicSystem(Val(:normalized), system.ring, branch...)
end

function __algebraic_cleanup(E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    E_clean = T[]
    I_clean = T[]
    sizehint!(E_clean, length(E))
    sizehint!(I_clean, length(I))
    for f in E
        is_zero(f) && continue
        is_constant(f) && return nothing
        push!(E_clean, __algebraic_remove_constant_factor(f))
    end
    for q in I
        is_zero(q) && return nothing
        is_constant(q) && continue
        push!(I_clean, __algebraic_remove_constant_factor(q))
    end
    unique!(E_clean)
    unique!(I_clean)
    reduced = __algebraic_autoreduce_branch(E_clean, I_clean)
    reduced === nothing && return nothing
    E_clean, I_clean = reduced
    unique!(E_clean)
    unique!(I_clean)
    any(f -> f in I_clean, E_clean) && return nothing
    return E_clean, I_clean
end

function __algebraic_autoreduce_branch(
    E::Vector{T},
    I::Vector{T}
) where {T<:MPolyRingElem}
    # Pseudo-reduction lowers the degree in the current leader, while deleting a
    # vanishing leading coefficient removes its entire highest-degree term.
    # Cancelling a certified nonzero nonconstant factor lowers total degree,
    # and deleting a redundant constraint reduces the number of constraints.
    # Unit-linear substitutions replace a variable by lower variables, but
    # they may increase the number of terms.
    while true
        changed = false
        for equation in (true, false)
            polynomials = equation ? E : I
            for i in eachindex(polynomials)
                p = polynomials[i]
                v = __algebraic_leader(p)
                if equation
                    E_context = [f for f in E if !is_constant(f) && __algebraic_leader(f) < v]
                    I_context = I
                else
                    E_context = [f for f in E
                                  if !is_constant(f) && __algebraic_leader(f) <= v]
                    I_context = [q for (j, q) in pairs(I)
                                  if j != i && __algebraic_leader(q) <= v]
                    p = __algebraic_substitute_linear_equations(p, E_context)
                end
                reduced = __algebraic_reduce_polynomial(p, E_context, I_context)
                if is_constant(reduced)
                    is_zero(reduced) == equation || return nothing
                    deleteat!(polynomials, i)
                elseif reduced != polynomials[i]
                    polynomials[i] = reduced
                else
                    continue
                end
                changed = true
                break
            end
            changed && break
        end
        changed || return E, I
    end
end

# Simplify all terms of an inequation using equations h*v+b=0 with unit h.
# Each replacement uses only variables below v; the surrounding cleanup loop
# handles substitutions exposed by equations visited earlier in this scan.
function __algebraic_substitute_linear_equations(p::T, equations::Vector{T}) where {T<:MPolyRingElem}
    for f in equations
        is_constant(p) && break
        v = __algebraic_leader(f)
        k = var_index(v)
        iszero(degree(p, k)) && continue
        degree(f, k) == 1 || continue
        h = leading_coefficient(f, k)
        is_unit(h) || continue
        replacement = divexact(h*v - f, h)
        p = evaluate(p, [k], [replacement])
    end
    return p
end

function __algebraic_reduce_polynomial(
    f::T,
    E_context::Vector{T},
    I_context::Vector{T}
) where {T<:MPolyRingElem}
    reduced = __algebraic_remove_constant_factor(
        __algebraic_reduce(f, E_context, I_context)
    )
    is_constant(reduced) && return reduced
    # With no equations or guards, the nonzero-factor test cannot remove any
    # nonconstant factor. Preserve the polynomial without factoring it.
    isempty(E_context) && isempty(I_context) && return reduced
    factors = __algebraic_nonconstant_factors(reduced)
    n = length(factors)
    # The factor helper returns a detached vector; compact it in place.
    filter!(f -> !__algebraic_factor_known_nonzero(f, E_context, I_context), factors)
    length(factors) == n && return reduced
    isempty(factors) && return one(reduced)
    return __algebraic_remove_constant_factor(prod(factors))
end

# Simplify p on E_context = 0, I_context != 0 without computing a Gröbner basis.
# Pseudo-reduce by an equation with the same leader and suitable degree only
# after certifying that its initial is nonzero on the branch.
# If no such reducer exists, recursively reduce the initial using lower-leader equations.
# If it reduces to zero, remove the entire highest-degree term and repeat;
# otherwise return the current polynomial.
# Preserve its zero and nonzero loci on the branch, even during cleanup before
# quasi-simplicity is established. Do not fully reduce tails or coefficients: for y < x,
# the equation y = 0 removes y*x^2 from y*x^2+x+1, but leaves x+y unchanged; the goal here
# is to reduce the degree in the leader or lower the leader variable.
# This is a conservative adaptation of Algorithm 2.6
# in https://arxiv.org/pdf/1108.0817.
function __algebraic_reduce(
    p::T,
    E_context::Vector{T},
    I_context::Vector{T}
) where {T<:MPolyRingElem}
    __require_same_parent(p, E_context, I_context)
    is_constant(p) && return p
    isempty(E_context) && return p
    reducers = __algebraic_reduction_data(E_context)
    return __algebraic_reduce_prepared(p, reducers, I_context)
end

# Choose a usable equation with leader v; In a simple system there is at most one such equation.
# During cleanup repeated leaders may still be present, so prefer the usable equation of smallest degree in v.
function __algebraic_reduction_equation(
    q::T,
    E_context::Vector{T},
    I_context::Vector{T}
) where {T<:MPolyRingElem}
    is_constant(q) && return nothing
    reducers = __algebraic_reduction_data(E_context)
    k = var_index(__algebraic_leader(q))
    reducer = __algebraic_choose_reducer(q, k, reducers, I_context)
    return reducer === nothing ? nothing : reducer.polynomial
end

# Certify that the initial of g is nonzero on the current branch before using g for pseudo-reduction.
# Pseudo-division multiplies by a power of this initial, so this certificate is what makes replacement
# by a pseudo-remainder solution-preserving for both equations and inequations.
function __algebraic_reduction_initial_is_nonzero(
    g::T,
    E_context::Vector{T},
    I_context::Vector{T};
    v::T=__algebraic_leader(g)
) where {T<:MPolyRingElem}
    reducers = __algebraic_reduction_data(E_context)
    k = var_index(v)
    h = leading_coefficient(g, k)
    return __algebraic_prepared_initial_is_nonzero(h, k, reducers, I_context)
end

# Return true when h ≠ 0 follows immediately from the current guards.
# If h divides a guard q and q ≠ 0, then h ≠ 0.
# This is only a cheap sufficient criterion; failing it does not mean that h can vanish.
function __algebraic_guard_certifies_nonzero(
    h::T,
    I_context::Vector{T}
) where {T<:MPolyRingElem}
    is_constant(h) && return !is_zero(h)
    h_terms = length(h)
    h_terms > 8 && return false

    for q in I_context
        is_zero(q) && continue
        is_constant(q) && continue

        # This is only a sufficient certificate. Avoid expensive exact
        # divisibility tests on large polynomial pairs; failure to test
        # merely means that nonvanishing is not certified here.
        length(q) > 8 && continue

        __checked_divides(h, q).divides && return true
    end

    return false
end

function __algebraic_normalized(
    systems::Vector{Tuple{Vector{T},Vector{T}}}
) where {T<:MPolyRingElem}
    out = Tuple{Vector{T},Vector{T}}[]
    sizehint!(out, length(systems))
    for (E, I) in systems
        clean = __algebraic_cleanup(E, I)
        clean === nothing || push!(out, clean)
    end
    return out
end

# Polynomial normalization and factors
# ------------------------------------

function __algebraic_remove_constant_factor(p::T) where {T<:MPolyRingElem}
    is_zero(p) && return p
    c = content(p)
    q = isone(c) ? p : divexact(p, c)
    is_constant(q) && return q
    lc = leading_coefficient(q)
    # Give opposite constraints the same structural key over QQ and ZZ.
    return lc isa Union{QQFieldElem,ZZRingElem} && lc < zero(lc) ? -q : q
end

function __algebraic_nonconstant_factors(p::T) where {T<:MPolyRingElem}
    p = __algebraic_remove_constant_factor(p)
    is_constant(p) && return [p]
    factors = __algebraic_cached(:factors, parent(p), [p]) do
        result = T[]
        for (f, _) in factor(p)
            push!(result, __algebraic_remove_constant_factor(f))
        end
        unique!(result)
        return filter!(f -> !is_constant(f), result)
    end
    return deepcopy(factors)
end

function __algebraic_squarefree_part(p::T) where {T<:MPolyRingElem}
    is_constant(p) && return p
    result = one(p)
    for (f, _) in factor_squarefree(p)
        result *= f
    end
    return __algebraic_remove_constant_factor(result)
end

# Partial systems and polynomial reduction
# ----------------------------------------

function __algebraic_vanishes_on_system(q::T, E::Vector{T}, I::Vector{T}) where {T<:MPolyRingElem}
    R = __require_same_parent(q, E, I)
    is_zero(q) && return true
    # q vanishes everywhere iff the same system with q ≠ 0 is empty.
    # Equivalently, prod(I)*q belongs to radical(ideal(E))
    return !__compatible_zero_algebraic_system(zero(R), E, [I; q])
end

function __algebraic_normal_form_mod_equations(
    p::T,
    E_lower::Vector{T}
) where {T<:MPolyRingElem}
    isempty(E_lower) && return p
    return only(__algebraic_normal_form_mod_equations([p], E_lower))
end

function __algebraic_normal_form_mod_equations(
    polynomials::Vector{T},
    E_lower::Vector{T}
) where {T<:MPolyRingElem}
    R = __require_same_parent(polynomials, E_lower)
    (isempty(polynomials) || isempty(E_lower)) && return copy(polynomials)
    equations = unique!(filter(!iszero, E_lower))
    isempty(equations) && return copy(polynomials)
    if __algebraic_coprime_leading_monomials(R, equations)
        return Oscar.reduce(
            polynomials,
            equations;
            ordering=lex(R),
            complete_reduction=true
        )
    end
    return Oscar.normal_form(
        polynomials,
        __algebraic_equation_ideal(R, equations);
        ordering=lex(R)
    )
end

# OSCAR.tail removes only one monomial term.
function __algebraic_tail(p::T) where {T<:MPolyRingElem}
    is_constant(p) && return zero(p)
    v = __algebraic_leader(p)
    k = var_index(v)
    return p - leading_coefficient(p, k) * v^degree(p, k)
end

# Return the same pseudo-remainder with compute_quotient=false, replacing the unused quotient by nothing.
function __algebraic_pseudo_divrem(f::T, g::T; compute_quotient::Bool=true, reducer_data=nothing) where {T<:MPolyRingElem}
    R = __require_same_parent(f, g)
    q = compute_quotient ? R(0) : nothing
    is_constant(f) && return q, f
    idx = reducer_data === nothing ? var_index(__algebraic_leader(g)) : reducer_data.variable_index
    v = gen(R, idx)
    dg = reducer_data === nothing ? degree(g, idx) : reducer_data.degree
    hg = reducer_data === nothing ? leading_coefficient(g, idx) : __algebraic_reducer_initial!(reducer_data)
    initial_is_one = isone(hg)
    r = f
    while !is_constant(r) && __algebraic_leader(r) == v
        dr = degree(r, idx)
        dr >= dg || break
        hr = leading_coefficient(r, idx)
        t = hr * v^(dr - dg)
        q, r = __polynomial_pseudo_step(q, r, g, hg, t, initial_is_one)
    end
    return q, r
end

function __algebraic_factor_known_nonzero(
    f::T,
    E_context::Vector{T},
    I_context::Vector{T}
) where {T<:MPolyRingElem}
    # Directly certified by an inequation.
    __algebraic_guard_certifies_nonzero(f, I_context) && return true
    # Try __algebraic_reduce. Since every pseudo-division used
    # here has a certified nonvanishing initial, f and its remainder have the
    # same vanishing behaviour on the current branch.
    reduced = __algebraic_reduce(f, E_context, I_context)
    is_constant(reduced) && return !is_zero(reduced)
    return __algebraic_guard_certifies_nonzero(reduced, I_context)
end

###############################################################################
#####  SUBRESULTANTS MODIFICATION
###############################################################################

"""
Collect constraints strictly below the chosen leader. Use them as parameter
conditions for a univariate subresultant computation. Keep constraints at or
above this leader in the enclosing branch, without using them to cancel factors.
"""
function __algebraic_subresultant_lower_context(E::Vector{P}, I::Vector{P}, idx::Int) where {P<:MPolyRingElem}
    lower_E = P[f for f in E if is_constant(f) || var_index(__algebraic_leader(f)) > idx]
    lower_I = P[q for q in I if is_constant(q) || var_index(__algebraic_leader(q)) > idx]
    return lower_E, lower_I
end

"""
Perform one GCD step for two equations (`kind=:equations`) or for the mixed case
equation+inequation (`kind=:mixed`). Keep the nonzero case of the first undecided
coefficient and return the original pair with that coefficient zero as a separate branch.
After cancelling certified nonzero factors, return the normalized replacement branch
if a polynomial becomes constant or changes leader. Otherwise continue with the updated
pair and unchanged lower context, handling equal polynomials and linear equation pairs directly.
In differential splitting, return that lower equation to S_Q before asking for the next GCD degree.
"""
function __algebraic_subresultant_pair_step(
    E::Vector{P}, I::Vector{P}, i::Integer, j::Integer, idx::Int, kind::Symbol
) where {P<:MPolyRingElem}
    lower_E, lower_I = __algebraic_subresultant_lower_context(E, I, idx)
    g = kind === :equations ? E[j] : I[j]
    # OPTIMIZATION: Use I to certify nonzero monomial factors and remove them from equations before
    # constructing the sequence. For I[j], use only unchanged inequations with strictly
    # lower leaders: using I[j] itself could turn x != 0 into 1 != 0 and admit x = 0
    nonzero_variables = __algebraic_nonzero_monomial_variables(I)
    left = __algebraic_strip_monomial_variables(E[i], nonzero_variables)
    right_variables = kind === :equations ? nonzero_variables :
        __algebraic_nonzero_monomial_variables(lower_I)
    right = __algebraic_strip_monomial_variables(g, right_variables)
    # OPTIMIZATION: Try cached factors when monomial cancellation changes neither input
    if left == E[i] && right == g
        left = __algebraic_strip_cached_guard_factors(E[i], I)
        right = __algebraic_strip_cached_guard_factors(g, kind === :equations ? I : lower_I)
    end
    if left != E[i] || right != g
        new_E, new_I = copy(E), copy(I)
        new_E[i] = left
        if kind === :equations
            new_E[j] = right
        else
            new_I[j] = right
        end
        # A constant or a lower leader requires a new choice of constraints.
        # In differential splitting, new lower equations must return to S_Q.
        if is_constant(left) || is_constant(right) ||
           var_index(__algebraic_leader(left)) != idx || var_index(__algebraic_leader(right)) != idx
            return __algebraic_normalized([(new_E, new_I)])
        end
        E, I, g = new_E, new_I, right
        if left == right
            kind === :mixed && return Tuple{Vector{P},Vector{P}}[]
            deleteat!(E, j)
            return __algebraic_normalized([(E, I)])
        end
        if kind === :equations && min(degree(left, idx), degree(right, idx)) == 1
            return __algebraic_same_leader_equations(E, I, i, j)
        end
        # Only constraints at idx changed: the strictly lower context is still valid
    end
    step = __subresultant_gcd_step(E[i], g, idx, lower_E, lower_I)
    branches = Tuple{Vector{P},Vector{P}}[]
    step.kind === :inconsistent && return branches
    if step.case === nothing
        push!(branches, (E, [I; step.condition]))
    else
        c = step.case
        branch = __algebraic_subresultant_pair_replacement(E, I, i, j, kind, c)
        branch === nothing || push!(branches, branch)
    end
    if step.kind === :split
        push!(branches, ([E; step.condition], I))
    end
    return __algebraic_normalized(branches)
end

# Apply the certified GCD case while retaining every condition used in its certificate.
function __algebraic_subresultant_pair_replacement(
    E::Vector{P}, I::Vector{P}, i::Integer, j::Integer, kind::Symbol, c::__SubresultantGCDCase{P}
) where {P<:MPolyRingElem}
    if kind === :equations
        c.degree == 0 && return nothing
        new_E = P[E[k] for k in eachindex(E) if k != j]
        new_E[i - (j < i)] = __algebraic_strip_guard_factors(c.gcd, c.inequations)
        return [new_E; c.equations], [I; c.inequations]
    end
    iszero(c.right) && return nothing
    df = degree(c.left, c.variable_index)
    df == 0 && return nothing
    c.degree > 0 && c.degree == df && return nothing
    new_E, new_I = copy(E), copy(I)
    if iszero(c.left)
        new_E[i], new_I[j] = c.left, c.right
    elseif c.degree == 0
        new_E[i] = c.left
        deleteat!(new_I, j)
    else
        quotient = __algebraic_subresultant_cofactor(c; quotient_only=true)
        new_E[i] = __algebraic_strip_guard_factors(quotient, c.inequations)
        new_I[j] = c.right
    end
    return [new_E; c.equations], [new_I; c.inequations]
end

"""
Retain the discriminant-nonzero branch. On its zero locus, make only the first
undecided GCD decision for `p` and its derivative. Replace `p` by the certified
cofactor in the decided case; keep `p` and the new zero condition in the deferred
case. This avoids constructing every square-free specialization in one call.
"""
function __algebraic_subresultant_squarefree_step(
    E::Vector{P},
    I::Vector{P},
    issue::NamedTuple,
    disc::P,
    idx::Int
) where {P<:MPolyRingElem}
    is_equation = issue.kind === :equation
    p = is_equation ? E[issue.index] : I[issue.index]
    lower_E, lower_I = __algebraic_subresultant_lower_context(E, I, idx)
    # Discriminant-zero branch
    push!(lower_E, disc)
    # Complementary discriminant-nonzero branch
    branches = Tuple{Vector{P},Vector{P}}[(E, [I; disc])]
    data = get(issue, :subresultants, nothing)
    # Since init(p) is already certified nonzero and disc(p)=0 on this branch,
    # res(p, dp/dx)=0. Hence gcd degree 0 is impossible and PSC_0 need not be constructed
    step = __subresultant_gcd_step(p, derivative(p, idx), idx, lower_E, lower_I;
                data=data, first_index=1)
    if step.kind === :inconsistent
        return __algebraic_normalized(branches)
    end
    E_zero = [E; disc]
    if step.case === nothing
        push!(branches, (E_zero, [I; step.condition]))
    else
        c = step.case
        # Keep the certified pseudo-cofactor itself. Any multiplier introduced
        # by pseudo-division is nonzero on the GCD case, so no additional
        # guard-factor stripping is required here.
        replacement = if iszero(c.left)
            zero(p)
        elseif degree(c.left, idx) == 0
            one(p)
        else
            __algebraic_subresultant_cofactor(c; quotient_only=true)
        end
        new_E, new_I = copy(E_zero), copy(I)
        (is_equation ? new_E : new_I)[issue.index] = replacement
        push!(branches, ([new_E; c.equations], [new_I; c.inequations]))
    end
    if step.kind === :split
        push!(branches, ([E_zero; step.condition], I))
    end
    return __algebraic_normalized(branches)
end

"""Reduce heads and vanishing initials, reusing data for this unchanged equation set"""
function __algebraic_reduce_prepared(p, reducers, guards, first_reducer::Int=1)
    q = p
    while !is_constant(q)
        v = __algebraic_leader(q)
        k = var_index(v)
        reducer = __algebraic_choose_reducer(q, k, reducers, guards, first_reducer)
        if reducer !== nothing
            _, remainder = __algebraic_pseudo_divrem(q, reducer.polynomial;
                compute_quotient=false, reducer_data=reducer)
            q = __algebraic_remove_constant_factor(remainder)
            continue
        end
        lower = __algebraic_lower_reducer_start(reducers, k, first_reducer)
        lower > length(reducers) && return q
        h = leading_coefficient(q, k)
        is_zero(__algebraic_reduce_prepared(h, reducers, guards, lower)) || return q
        q -= h * v^degree(q, k)
    end
    return q
end

"""Find the first strictly lower leader in the sorted reducer data, without copying a subvector"""
function __algebraic_lower_reducer_start(reducers, k::Int, first_reducer::Int=1)
    i = first_reducer
    while i <= length(reducers) && reducers[i].variable_index <= k
        i += 1
    end
    return i
end

"""Choose the usable same-leader reducer of least degree"""
function __algebraic_choose_reducer(q, k::Int, reducers, guards, first_reducer::Int=1)
    dq = degree(q, k)
    best = nothing
    best_degree = typemax(Int)
    for i in first_reducer:length(reducers)
        reducer = reducers[i]
        reducer.variable_index < k && continue
        reducer.variable_index > k && break
        reducer.degree <= dq && reducer.degree < best_degree || continue
        h = __algebraic_reducer_initial!(reducer)
        __algebraic_prepared_initial_is_nonzero(h, k, reducers, guards, i + 1) || continue
        best, best_degree = reducer, reducer.degree
    end
    return best
end

"""Certify that an initial is nonzero using the guards and equations with strictly lower leaders.
Return false if these checks do not establish the nonvanishing condition"""
function __algebraic_prepared_initial_is_nonzero(h, k::Int, reducers, guards, first_reducer::Int=1)
    is_constant(h) && return !is_zero(h)
    __algebraic_guard_certifies_nonzero(h, guards) && return true
    lower = __algebraic_lower_reducer_start(reducers, k, first_reducer)
    lower > length(reducers) && return false
    reduced = __algebraic_reduce_prepared(h, reducers, guards, lower)
    is_constant(reduced) && return !is_zero(reduced)
    return __algebraic_guard_certifies_nonzero(reduced, guards)
end

"""
Remove general factors using existing factorizations and literal equality only.
Read the cache without populating it: a missing entry leaves p unchanged.
The supplied guards must remain in the branch independently of this replacement.
Cached factors omit multiplicities, as in __algebraic_reduce_polynomial; their
remaining product preserves the zero and nonzero loci, not the original multiplicities
"""
function __algebraic_strip_cached_guard_factors(p::P, guards::Vector{P}) where {P<:MPolyRingElem}
    (isempty(guards) || is_constant(p)) && return p
    cache = get(task_local_storage(), :__algebraic_splitting, nothing)
    cache === nothing && return p
    ring_id = objectid(parent(p))
    factors = get(cache, (:factors, ring_id, (p,), ()), nothing)
    factors === nothing && return p
    nonzero_factors = Set(guards)
    for q in guards
        guard_factors = get(cache, (:factors, ring_id, (q,), ()), nothing)
        guard_factors === nothing || union!(nonzero_factors, guard_factors)
    end
    remaining = P[f for f in factors if !(f in nonzero_factors)]
    length(remaining) == length(factors) && return p
    return isempty(remaining) ? one(p) : prod(remaining)
end