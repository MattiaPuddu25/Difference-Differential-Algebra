import Oscar: is_simple
###############################################################################
#####  Leader filtering
###############################################################################

@doc raw"""
    leader_filter(V::AbstractVector{<:MPolyRingElem{T}}, p::MPolyRingElem{T}) where T

For a vector of polynomials `V` and a polynomial `p` (in the same polynomial ring,
with a fixed ordering of the variables), return a vector consisting of all polynomials
`f` in `V` such that `__algebraic_leader(f) ≤ __algebraic_leader(p)`.\
Even if such a polynomial `f` appears several times in `V`, it only appears one time in the output vector.

    # Examples

    ```jldoctest
    julia> R, (x3, x2, x1) = polynomial_ring(QQ, [:x3, :x2, :x1]);
    julia> V = [x1^2 + 1, x2 + x1, x3 + x1, x1^2+1];
    julia> leader_filter(V, x2^2 + x1)
    2-element Vector{QQMPolyRingElem}:
     x1^2 + 1
     x2 + x1
    ```
"""
function leader_filter(V::AbstractVector{<:MPolyRingElem{T}}, p::MPolyRingElem{T}) where T
    return __leader_filter(V, (__algebraic_leader(f) for f in V), __algebraic_leader(p))
end

"""
Internal version of leader_filter using supplied leaders.

Return the polynomials in `V` whose supplied leader is less than or equal to `max_leader`.
When the leaders are precomputed, this avoids recomputing the leaders of
the same equation and inequation lists during quasi-simplicity checks.
Supply exactly one leader per polynomial, in the same order; internal callers guarantee this alignment.\
Pass the already computed leader of the reference polynomial as `max_leader`.\
Set `check_parent=false` only when the caller has validated the polynomials
and supplied leaders against the same parent.
"""
function __leader_filter(
    V::AbstractVector{<:MPolyRingElem{T}},
    leaders,
    max_leader::MPolyRingElem{T};
    check_parent::Bool=true
) where T
    R = check_parent ? __require_same_parent(V, max_leader) : parent(max_leader)
    S = eltype(V)[]
    sizehint!(S, length(V))
    for (f, f_leader) in zip(V, leaders)
        check_parent && __require_parent(R, f_leader)
        f_leader <= max_leader && push!(S, f)
    end
    return unique!(S)
end

###############################################################################
#####  Solution checks
###############################################################################

@doc raw"""
    solution_check(r::MPolyRingElem{T}, E::AbstractVector{<:MPolyRingElem{T}}) where T
    solution_check(
        r::MPolyRingElem{T},
        E::AbstractVector{<:MPolyRingElem{T}},
        I::AbstractVector{<:MPolyRingElem{T}}
    ) where T

For a polynomial `r` and a vector of polynomials `E` (in the same polynomial ring over a field K,
with a fixed ordering of the variables), return `true` if the system

`r` = 0,\
`f` = 0 for `f` in `E` with __algebraic_leader(`f`) ≤ __algebraic_leader(`r`)

has solutions in a fixed algebraic closure of K, and `false` otherwise.

With the third input `I`, also impose

q ≠ 0 for q in `I` with __algebraic_leader(q) ≤ __algebraic_leader(`r`).

The function uses Hilbert Nullstellensatz.
In the equation-only case, it checks whether the equation ideal is the whole ring;
in the equation-and-inequation case, it checks radical membership of the product of the relevant inequations.

    # Examples

    ```jldoctest
    julia> R, (x1, x2, x3) = polynomial_ring(QQ, [:x1, :x2, :x3]);
    julia> r = x2^2+1;
    julia> E = [x1^2+x2^2-1, x3^2-x1];
    julia> solution_check(r, E)
    true

    julia> R, (x1, x2, x3) = polynomial_ring(QQ, [:x1, :x2, :x3]);
    julia> r = x2^2-1;
    julia> E = [x2^2-2];
    julia> solution_check(r, E)
    false

    julia> R, (x1, x2, x3) = polynomial_ring(QQ, [:x1, :x2, :x3]);
    julia> r = x2^2-1;
    julia> E = [x1^3*x2+x1];
    julia> I = [x1+x2, x1^2];
    julia> solution_check(r, E, I)
    true

    julia> R, (x4, x1, x2, x3) = polynomial_ring(QQ, [:x4, :x1, :x2, :x3]);
    julia> r = x1^2-x1*x2-x1*x3-5;
    julia> E = [x2^2-x2*x3-x1*x2+4, x3^2-x1*x3-x2*x3+7];
    julia> solution_check(r, E)
    true
    julia> I = [x1^2-25];
    julia> solution_check(r, E, I)
    false
    ```
"""
function solution_check(r::MPolyRingElem{T}, E::AbstractVector{<:MPolyRingElem{T}}) where T
    return __solution_check(r, E, [__algebraic_leader(f) for f in E])
end

function solution_check(
    r::MPolyRingElem{T},
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}}
) where T
    E_leaders = [__algebraic_leader(f) for f in E]
    I_leaders = [__algebraic_leader(q) for q in I]
    return __solution_check(r, E, E_leaders, I, I_leaders)
end

"""
Internal solution check using precomputed leaders.\
Pass the keyword `R` only when the caller has validated all polynomials and
supplied leaders against that parent.
Otherwise, validate them on entry.

The first method tests whether the algebraic system consisting of `r` = 0 and
the relevant equations from `E` has a solution in a fixed algebraic closure.
The second method also imposes the relevant inequations from `I`.
"""
function __solution_check(
    r::MPolyRingElem{T},
    E::AbstractVector{<:MPolyRingElem{T}},
    E_leaders::AbstractVector;
    R=__require_same_parent(r, E, E_leaders)
) where T
    Eq = __leader_filter(E, E_leaders, __algebraic_leader(r); check_parent=false)
    return __compatible_zero_algebraic_system(R, r, Eq, typeof(r)[])
end

function __solution_check(
    r::MPolyRingElem{T},
    E::AbstractVector{<:MPolyRingElem{T}},
    E_leaders::AbstractVector,
    I::AbstractVector{<:MPolyRingElem{T}},
    I_leaders::AbstractVector;
    R=__require_same_parent(r, E, E_leaders, I, I_leaders)
) where T
    max_leader = __algebraic_leader(r)
    Eq = __leader_filter(E, E_leaders, max_leader; check_parent=false)
    Ineq = __leader_filter(I, I_leaders, max_leader; check_parent=false)
    return __compatible_zero_algebraic_system(R, r, Eq, Ineq)
end

"""
Return `true` if `r = 0` has a solution compatible with `E = 0` and `I ≠ 0`.

This tests whether the product of the inequations is not in the radical of the ideal generated by `E` and `r`.
With no inequations, it only tests whether that ideal is proper.
Over a coefficient field, first remove highest-leader constraints when every solution of the
remaining constraints extends to a solution of the removed ones over the algebraic closure.
This only simplifies the existence check; it does not remove constraints from a splitting branch.

The internal overload `__compatible_zero_algebraic_system(R, r, E, I)` skips parent checks;
supply `R` only after validating all inputs against it.
"""
function __compatible_zero_algebraic_system(
    r::MPolyRingElem{T},
    E::AbstractVector{<:MPolyRingElem{T}}
) where T
    return __compatible_zero_algebraic_system(r, E, typeof(r)[])
end

function __compatible_zero_algebraic_system(
    r::MPolyRingElem{T},
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}}
) where T
    R = __require_same_parent(r, E, I)
    return __compatible_zero_algebraic_system(R, r, E, I)
end

function __compatible_zero_algebraic_system(
    R::Ring,
    r::MPolyRingElem{T},
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}}
) where T
    is_constant(r) && !is_zero(r) && return false
    Eq = unique!(filter(!iszero, [E; r]))
    any(is_constant, Eq) && return false
    any(iszero, I) && return false
    guards = unique!(filter(q -> !is_constant(q), I))
    isempty(Eq) && return true
    any(q -> q in Eq, guards) && return false
    if coefficient_ring(R) isa Field
        reduced = __subresultant_compatibility_constraints(Eq, guards)
        reduced === nothing && return false
        Eq, guards = reduced
        Eq, guards = __algebraic_compatibility_projection(Eq, guards)
        isempty(Eq) && return true
    end
    # Cache only the projected query, without retaining copies of eliminated polynomials
    return __algebraic_cached(:compatible, R, Eq, guards) do
        principal = __algebraic_principal_generator(Eq)
        if principal !== nothing
            is_constant(principal) && return false
            # Remove every component excluded by a guard, using gcds only.
            remaining = __algebraic_strip_guard_factors(principal, guards)
            return !is_constant(remaining)
        end
        isempty(guards) && __algebraic_coprime_leading_monomials(R, Eq) && return true
        J = __algebraic_equation_ideal(R, Eq)
        isempty(guards) && return !is_one(J)
        return !radical_membership(prod(guards; init=one(R)), J)
    end
end

"""
Simplify a compatibility test, not considering equations and inequations having
certainly a solution that can be obtained after specializing lower variables.

Consider the highest remaining leader. If exactly one equation has this leader
and no inequation has it, omit the equation when its initial is certified
nonzero using only conditions with lower leaders. For every solution of the remaining
conditions, the equation then becomes a nonconstant univariate polynomial after specialization,
and therefore has a root over the algebraic closure.

If no equation and one or more inequations have the highest leader, remove all
of these inequations when their initials are certified nonzero using only lower
conditions. They exclude only finitely many values of that variable, so a value
satisfying all of them still exists over the algebraic closure.

Repeat until neither elimination applies. The returned equations and
inequations are used only to decide compatibility: they need not define the
same solution set as the original system.

For example, with `x > y > z`, the existence of solutions of `x^2-y = 0, y^2-z = 0, z ≠ 0` reduces to that of `z ≠ 0`.
In contrast, with `x > a`, the system `a*x-1 = 0, a = 0` cannot be reduced in
this way, since the initial `a` cannot be certified nonzero.
"""
function __algebraic_compatibility_projection(E::AbstractVector{<:MPolyRingElem}, I::AbstractVector{<:MPolyRingElem})
    isempty(E) && return E, I
    E_leaders = [var_index(__algebraic_leader(f)) for f in E]
    I_leaders = [var_index(__algebraic_leader(q)) for q in I]
    while !isempty(E)
        idx = min(minimum(E_leaders), minimum(I_leaders; init=typemax(Int)))
        n_equations = count(==(idx), E_leaders)
        n_guards = count(==(idx), I_leaders)
        (n_equations > 1 || (n_equations == 1 && n_guards > 0)) && break
        polynomials, indices = n_equations == 1 ? (E, E_leaders) : (I, I_leaders)
        for i in eachindex(polynomials)
            indices[i] == idx || continue
            h = leading_coefficient(polynomials[i], idx)
            is_constant(h) && continue
            lower_guards = (q for (q, j) in zip(I, I_leaders) if j > idx)
            lower_E = typeof(h)[f for (f, j) in zip(E, E_leaders) if j > idx]
            lower_I = typeof(h)[q for q in lower_guards]
            __subresultant_condition_status(h, lower_E, lower_I) === true || return E, I
        end
        if n_equations == 1
            keep = E_leaders .!= idx
            E, E_leaders = E[keep], E_leaders[keep]
        else
            keep = I_leaders .!= idx
            I, I_leaders = I[keep], I_leaders[keep]
        end
    end
    return E, I
end

###############################################################################
#####  Quasi-simple systems
###############################################################################

@doc raw"""
    is_quasi_simple(
        system::AlgebraicSystem;
        info::Bool=false
    )
    is_quasi_simple(
        E::AbstractVector{<:MPolyRingElem{T}};
        info::Bool=false
    ) where T
    is_quasi_simple(
        E::AbstractVector{<:MPolyRingElem{T}},
        I::AbstractVector{<:MPolyRingElem{T}};
        info::Bool=false
    ) where T

Work in a polynomial ring over a field with a fixed ordering of the variables.
Return `true` if an `AlgebraicSystem` is quasi-simple as an algebraic system, and `false` otherwise.\
The input may be also given as a vector `E` of polynomials, representing the system

`f = 0` for `f` in `E`

and possibly another vector `I` to include the inequations

q ≠ 0 for q in `I`

to the system.

If the result is `false` and the optional argument `info` is set to `true`,
a message explaining the first problem encountered is printed.

    # Examples

    ```jldoctest
    julia> R, (x3, x2, x1) = polynomial_ring(QQ, [:x3, :x2, :x1]);
    julia> E = [x1^2+x2^2-1, x3^2-x1];
    julia> is_quasi_simple(E)
    true

    julia> R, (x1, x2, x3, x4, x5) = polynomial_ring(QQ, [:x1, :x2, :x3, :x4, :x5]);
    julia> E = [(x5+1)*x1^2+x2-x3, (x4+x5-1)*x2^2+x3-x5, (x3+x4*x5-1)*x3^2+x4^2-x5];
    julia> I = [x5*x4^2-1];
    julia> is_quasi_simple(E; info=true)
    The initial of Equation 1 has solutions in the corresponding subsystem
    false

    julia> R, (x3, x2, x1) = polynomial_ring(QQ, [:x3, :x2, :x1]);
    julia> E = [x1^2*x3-1, x1*x2^2+x2];
    julia> I = [x1];
    julia> is_quasi_simple(E, I)
    true

    julia> R, (x3, x2, x1) = polynomial_ring(QQ, [:x3, :x2, :x1]);
    julia> E = [3*x1^2*x2-5*x3+2, -x1^3+7*x2^2+4];
    julia> I = [x1^2-1];
    julia> is_quasi_simple(E, I)
    true

    julia> R, (x1, x2, x3, x4, x5) = polynomial_ring(QQ, [:x1, :x2, :x3, :x4, :x5]);
    julia> E = [x2^2+x3-x5, x3^2+x4^2-x5];
    julia> I = [(x5+1)*x1^2+x2-x3];
    julia> is_quasi_simple(E, I; info=true)
    The initial of Inequation 1 has solutions in the corresponding subsystem
    false
    ```
"""
function is_quasi_simple(system::AlgebraicSystem; info::Bool=false)
    cached = system.quasi_simple
    cached !== nothing && !(info && !cached) && return cached
    return __algebraic_simplicity_issue(system; info) === nothing
end

function is_quasi_simple(
    E::AbstractVector{<:MPolyRingElem{T}};
    info::Bool=false
) where T
    return __is_quasi_simple(E, empty(E); info=info)
end

function is_quasi_simple(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}};
    info::Bool=false
) where T
    return __is_quasi_simple(E, I; info=info)
end

function __is_quasi_simple(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}};
    info::Bool=false,
    simple::Bool=false
) where T
    return __algebraic_simplicity_issue(E, I; info, simple) === nothing
end

"""
Return the first failed quasi-simplicity condition, or `nothing` if all conditions hold.
With `simple=true`, also check discriminants in the subsystems up to the original
polynomial's leader in order to certify simplicity.
Work over a field of characteristic 0 in this case.

Set `quasi_simple=true` only when the unchanged input is already certified quasi-simple
(useful for the differential splitting, in order to certify algebraic (quasi-)simplicity).

Return the indices and computed polynomials needed by the next splitting step.
"""
function __algebraic_simplicity_issue(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}};
    info::Bool=false,
    simple::Bool=false,
    quasi_simple::Bool=false,
) where T
    isempty(E) && isempty(I) && return nothing
    R = __require_same_parent(E, I)
    simple && !iszero(characteristic(coefficient_ring(R))) &&
        throw(ArgumentError("Discriminant checks require characteristic 0"))
    if !quasi_simple
        if any(is_constant, E) || any(is_constant, I)
            info && println("There are constant polynomials")
            return (kind=:quasi_simple, reason=:constant)
        end
    end
    E_leaders = [__algebraic_leader(p) for p in E]
    I_leaders = [__algebraic_leader(q) for q in I]
    if !quasi_simple
        issue = __algebraic_leader_issue(E, I, E_leaders, I_leaders)
        if issue !== nothing
            info && println("There are polynomials with the same leader")
            return (kind=:quasi_simple, reason=:leaders, data=issue)
        end
        issue = __algebraic_initial_issue(E, I, E_leaders, I_leaders; R)
        if issue !== nothing
            label = issue.kind === :equation ? "Equation" : "Inequation"
            info && println("The initial of $label $(issue.index) has solutions in the corresponding subsystem")
            return (kind=:quasi_simple, reason=:initial, data=issue)
        end
    end
    if simple
        issue = __algebraic_discriminant_issue(E, I, E_leaders, I_leaders; R)
        if issue !== nothing
            label = issue.kind === :equation ? "Equation" : "Inequation"
            info && println("The discriminant of $label $(issue.index) has solutions in the corresponding subsystem")
            return (kind=:discriminant, data=issue)
        end
    end
    return nothing
end

# Keep certificates separate and pass a failed condition directly to the splitting step.
function __algebraic_simplicity_issue(
    system::AlgebraicSystem;
    info::Bool=false,
    simple::Bool=false
)
    (simple ? system.simple : system.quasi_simple) === true && return nothing
    E, I = __algebraic_read_system(system)
    issue = __algebraic_simplicity_issue(E, I; info, simple, quasi_simple=system.quasi_simple === true)
    quasi_simple = issue === nothing || issue.kind === :discriminant
    getfield(system, :quasi_simple)[] = quasi_simple
    (simple || !quasi_simple) && (getfield(system, :simple)[] = issue === nothing)
    return issue
end

###############################################################################
#####  Simple systems
###############################################################################

@doc raw"""
    is_simple(
        E::AbstractVector{<:MPolyRingElem{T}};
        info::Bool=false
    ) where T
    is_simple(
        E::AbstractVector{<:MPolyRingElem{T}},
        I::AbstractVector{<:MPolyRingElem{T}};
        info::Bool=false
    ) where T

Work in a polynomial ring over a field with a fixed ordering of the variables.
Return `true` if an `AlgebraicSystem` is simple as an algebraic system, and `false` otherwise.\
The input may be also given as a vector `E` of polynomials, representing the system

`f = 0` for `f` in `E`

and possibly another vector `I` to include the inequations

q ≠ 0 for q in `I`

to the system.\
Check every quasi-simplicity condition and, for every `r` in `E` or `I`, require `disc(r) = 0`
to have no solution in the subsystem containing the equations and inequations
with leader at most `__algebraic_leader(r)`.\
Use the leader of `r`, not the leader of its discriminant, to select this subsystem.

If the result is `false` and the optional argument `info` is set to `true`,
a message explaining the first problem encountered is printed.

    # Examples

    ```jldoctest
    julia> R, (x3, x2, x1) = polynomial_ring(QQ, [:x3, :x2, :x1]);
    julia> E = [x1^2+x2^2-1, x3^2-x1];
    julia> is_simple(E; info=true)
    The discriminant of Equation 1 has solutions in the corresponding subsystem
    false

    julia> R, (x3, x2, x1) = polynomial_ring(QQ, [:x3, :x2, :x1]);
    julia> E = [x1^2*x3-1, x1*x2^2+x2];
    julia> I = [x1];
    julia> is_simple(E, I)
    true

    julia> R, (x3, x2, x1) = polynomial_ring(QQ, [:x3, :x2, :x1]);
    julia> E = [3*x1^2*x2-5*x3+2];
    julia> I = [x1^2-1, -x1^3+7*x2^2+4];
    julia> is_simple(E, I; info=true)
    The discriminant of Inequation 2 has solutions in the corresponding subsystem
    false
    ```
"""
function is_simple(system::AlgebraicSystem; info::Bool=false)
    iszero(characteristic(coefficient_ring(system.ring))) ||
        throw(ArgumentError("Discriminant checks require characteristic zero"))
    cached = system.simple
    cached !== nothing && !(info && !cached) && return cached
    return __algebraic_simplicity_issue(system; info, simple=true) === nothing
end

function is_simple(
    E::AbstractVector{<:MPolyRingElem{T}};
    info::Bool=false
) where T
    return __is_quasi_simple(E, empty(E); info=info, simple=true)
end

function is_simple(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}};
    info::Bool=false
) where T
    return __is_quasi_simple(E, I; info=info, simple=true)
end

###############################################################################
#####  Shared internal helpers
###############################################################################

"""
Return the first pair of equations, inequations, or mixed constraints with the same leader.
Reuse `E_leaders` and `I_leaders` when supplied; otherwise compute them through
the default argument expressions `[__algebraic_leader(f) for f in E]` and `[__algebraic_leader(q) for q in I]`.\
Return `nothing` if all leaders differ.
"""
function __algebraic_leader_issue(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}},
    E_leaders=[__algebraic_leader(f) for f in E], I_leaders=[__algebraic_leader(q) for q in I]
) where T
    indices = Dict{eltype(E_leaders),Int}()
    sizehint!(indices, max(length(E_leaders), length(I_leaders)))
    for (kind, leaders) in ((:equations, E_leaders), (:inequations, I_leaders))
        empty!(indices)
        issue = nothing
        # Scan backwards to retain the lexicographically first pair (i, j),
        # matching the previous nested-loop order, not the first repeated j.
        for i in reverse(eachindex(leaders))
            v = leaders[i]
            j = get(indices, v, nothing)
            j === nothing || (issue = (kind=kind, first_index=i, second_index=j))
            indices[v] = i
        end
        issue === nothing || return issue
    end
    # The dictionary now contains the first index of each inequation leader.
    for i in eachindex(E_leaders)
        j = get(indices, E_leaders[i], nothing)
        j === nothing || return (kind=:equation_inequation, first_index=i, second_index=j)
    end
    return nothing
end

"""
Return the first initial that can vanish in the subsystem up to its own leader.
Supply nonconstant polynomials and optionally reuse their precomputed leaders.
Keep the initial in the returned issue for the splitting step; return `nothing` if none fails.
Pass `R` only for inputs already certified against that parent.
Compute each initial from its supplied polynomial leader,
but filter its subsystem using the leader of the initial itself.
"""
function __algebraic_initial_issue(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}},
    E_leaders=[__algebraic_leader(f) for f in E], I_leaders=[__algebraic_leader(q) for q in I];
    R=__require_same_parent(E, I, E_leaders, I_leaders)
    ) where T
    for (kind, polynomials) in ((:equation, E), (:inequation, I)), i in eachindex(polynomials)
        p = polynomials[i]
        v = kind === :equation ? E_leaders[i] : I_leaders[i]
        h = leading_coefficient(p, var_index(v))
        is_constant(h) && !iszero(h) && continue
        __solution_check(h, E, E_leaders, I, I_leaders; R) && return (kind=kind, index=i, initial=h)
    end
    return nothing
end

"""
Return the first discriminant that can vanish in the subsystem up to the original polynomial's leader.
Reuse supplied leaders when available and keep the discriminant and selected equations for the splitting step.
Pass `R` only for inputs already validated against that parent.\
Return `nothing` if every discriminant condition holds.

Do not factor the discriminant here: only its zero set matters to the compatibility check.
"""
function __algebraic_discriminant_issue(
    E::AbstractVector{<:MPolyRingElem{T}},
    I::AbstractVector{<:MPolyRingElem{T}},
    E_leaders=[__algebraic_leader(f) for f in E],
    I_leaders=[__algebraic_leader(q) for q in I];
    R=__require_same_parent(E, I, E_leaders, I_leaders)
    ) where T
    for (kind, polynomials) in ((:equation, E), (:inequation, I)), i in eachindex(polynomials)
        p = polynomials[i]
        max_leader = kind === :equation ? E_leaders[i] : I_leaders[i]
        disc = is_constant(p) ? one(p) : __algebraic_discriminant(p, var_index(max_leader))
        is_constant(disc) && !iszero(disc) && continue
        E_partial = __leader_filter(E, E_leaders, max_leader; check_parent=false)
        I_partial = __leader_filter(I, I_leaders, max_leader; check_parent=false)
        if __compatible_zero_algebraic_system(R, disc, E_partial, I_partial)
            return (kind=kind, index=i, discriminant=disc, E_partial=E_partial)
        end
    end
    return nothing
end

function __with_algebraic_cache(f::F) where F
    key = :__algebraic_splitting
    haskey(task_local_storage(), key) && return f()
    cache = Dict{Any,Any}()
    sizehint!(cache, 256)
    return task_local_storage(f, key, cache)
end

"""
Work in a polynomial ring `R`. Evaluate `f()` and return its result, reusing an existing result
when a cache is activated by `__with_algebraic_cache`. Otherwise, evaluate `f()` directly.

Look up the current polynomials without copying them.
On a cache miss, create independent polynomial snapshots for the stored key before evaluating `f()`.
Clear the cache before inserting a new result if it already contains 256 entries.

Return cached results directly, including `false` and `nothing`.
"""
function __algebraic_cached(f::F, operation::Symbol, R, E, I=elem_type(R)[]) where F
    cache = get(task_local_storage(), :__algebraic_splitting, nothing)
    cache === nothing && return f()
    key = (operation, objectid(R), Tuple(E), Tuple(I))
    return get(cache, key) do
        # Only stored keys need detached snapshots; never insert the lookup key.
        stored_key = (operation, objectid(R), Tuple(deepcopy(p) for p in E), Tuple(deepcopy(q) for q in I))
        result = f()
        length(cache) >= 256 && empty!(cache)
        cache[stored_key] = result
        return result
    end
end

"""
Return the ideal generated by `E` in `R`.
If an algebraic cache is active, reuse a previously computed value when available.
"""
function __algebraic_equation_ideal(R, E)
    return __algebraic_cached(:ideal, R, E) do
        ideal(R, deepcopy(E))
    end
end

"""
Return `true` if the leading monomials of the polynomials in `E`, with respect to the lexicographic ordering of `R`,
are pairwise coprime; equivalently, if no variable occurs in two distinct leading monomials.\
Use the algebraic cache when available.
"""
function __algebraic_coprime_leading_monomials(R, E)
    return __algebraic_cached(:lex_product_criterion, R, E) do
        seen = falses(nvars(R))
        ord = lex(R)
        for f in E
            exponents = exponent_vector(leading_monomial(f; ordering=ord), 1)
            @inbounds for i in eachindex(exponents)
                iszero(exponents[i]) && continue
                seen[i] && return false
                seen[i] = true
            end
        end
        return true
    end
end

"""
Return a generator of the ideal generated by `E` when this can be certified cheaply,
and `nothing` otherwise.

This does not test principality in general: it only recognizes the cases where
`E` consists of a single polynomial, all polynomials involve at most one common variable,
or a polynomial of minimal total degree divides every element of `E`.
"""
function __algebraic_principal_generator(E)
    length(E) == 1 && return Base.first(E)
    active_variable = 0
    multivariate = false
    for f in E
        ds = degrees(f)
        @inbounds for i in eachindex(ds)
            ds[i] > 0 || continue
            if active_variable == 0
                active_variable = i
            elseif active_variable != i
                multivariate = true
                break
            end
        end
        multivariate && break
    end
    !multivariate && return foldl(gcd, E)
    i_min = firstindex(E)
    d_min = total_degree(E[i_min])
    for i in eachindex(E)
        i == i_min && continue
        d = total_degree(E[i])
        if d < d_min
            i_min = i
            d_min = d
        end
    end
    f = E[i_min]
    for g in E
        __checked_divides(f, g).divides || return nothing
    end
    return f
end

"""
Return a named tuple with fields `divides` and `quotient`.

The field `divides` is `true` exactly when `a*q == b` for the returned `quotient`;
otherwise `divides` is `false` and `quotient` is `nothing`.
"""
function __checked_divides(a, b)
    __require_same_parent(a, b)
    divides, q = Oscar.divides(b, a)
    divides || return (divides=false, quotient=nothing)
    return (divides=true, quotient=q)
end

"""
Detect whether `p` contains only even/odd powers of the variable indexed by `idx`,
and compress these powers by a factor of two. Let `x` denote the chosen variable.
If only even powers occur, construct `g` such that p(x) = g(x^2).
If only odd powers occur, instead, construct `g` such that p(x) = x*g(x^2).

In these two cases, the computation of the discriminant can be made significantly
cheaper, since

degree(g, idx) = (degree(p, idx) - parity) / 2

where `parity = 0` in the first case and `parity = 1` in the second. If `n = degree(g,idx)`, then

disc_x(p) = (-1)^n * 2^(2n) * lc_x(g) * g(0)^(2*parity + 1) * disc_x(g)^2

This helper is used for this reason in `__algebraic_discriminant`.
Return `(polynomial=g, parity=parity)`, or `nothing` if both even and odd powers
occur or if `p` has degree smaller than two in the chosen variable.
"""
function __algebraic_parity_compression(p::P, idx::Int) where {P<:MPolyRingElem}
    d = degree(p, idx)
    d >= 2 || return nothing
    coefficients = AbstractAlgebra.coefficients(p, idx)
    even_only = true
    odd_only = true
    for j in eachindex(coefficients)
        iszero(coefficients[j]) && continue
        isodd(j) || (even_only = false)
        iseven(j)  || (odd_only = false)
    end
    (even_only || odd_only) || return nothing
    parity = even_only ? 0 : 1
    x = gen(parent(p), idx)
    g = zero(p)
    compressed_degree = 0
    for exponent in parity:2:d
        c = coefficients[exponent + 1]
        if !iszero(c)
            g += c * x^compressed_degree
        end
        compressed_degree += 1
    end
    return (polynomial=g, parity=parity)
end

"""
Return the discriminant of `p` as a polynomial in its leader.
Return one for polynomials of degree at most one, including constant polynomials.
If the index `idx` of the leader is supplied, use it directly instead of recomputing the leader.
Prefer OSCAR's native discriminant implementation when available; otherwise, the discriminant is computed from the
signed resultant of `p` and its derivative, divided exactly by the initial.
When possible, reduce the cost of the computation by detecting polynomial structures (even or odd polynomials)
and recursively computing the discriminant of the corresponding compressed polynomial.

If a factorization of `p` is already available, its factors may be supplied to avoid computing the discriminant of `p` directly
and recovering it from the discriminants of its factors.
Let `x` be the chosen variable and suppose

    p = c * prod(f_i),

where `c` is independent of `x` and

    d = degree(p, idx) = sum(degree(f_i, idx)).

Then

    disc_x(p) = c^(2d-2) * prod(disc_x(f_i)) * prod(res_x(f_i, f_j)^2, i < j)

The supplied factors must have full multiplicities and may omit only factors
independent of the chosen variable.
If these conditions cannot be verified, fall back to the direct discriminant computation.
"""
function __algebraic_discriminant(p::T) where {T<:MPolyRingElem}
    is_constant(p) && return one(p)
    return __algebraic_discriminant(p, var_index(__algebraic_leader(p)))
end

# Supply the index of the known leader of p.
function __algebraic_discriminant(p::T, idx::Int) where {T<:MPolyRingElem}
    R = parent(p)
    d = degree(p, idx)
    d <= 1 && return one(p)
    # Prefer the native implementation when this polynomial type supports it.
    applicable(discriminant, p, idx) && return discriminant(p, idx)
    # The univariate case is handled differently
    if nvars(R) == 1
        K = coefficient_ring(R)
        S, z = polynomial_ring(K, :z)
        f = S(0)
        for (c, e) in zip(coefficients(p), AbstractAlgebra.exponent_vectors(p))
            f += c * z^e[1]
        end
        return R(discriminant(f))
    end
    dp = derivative(p, idx)
    is_zero(dp) && return zero(p)
    compression = __algebraic_parity_compression(p, idx)
    if compression !== nothing
        g = compression.polynomial
        n = degree(g, idx)
        if n >= 1
            disc_g = __algebraic_discriminant(g, idx)
            a = leading_coefficient(g, idx)
            c = first(AbstractAlgebra.coefficients(g, idx))
            disc = (-one(parent(p)))^n * parent(p)(4)^n * a * disc_g^2
            disc *= compression.parity == 0 ? c : c^3
            return disc
        end
    end
    disc = resultant(p, dp, idx)
    signed_disc = isodd(div(d * (d - 1), 2)) ? -disc : disc
    return divexact(signed_disc, leading_coefficient(p, idx))
end

function __algebraic_discriminant(p::T, idx::Int, factors::AbstractVector{T}) where {T<:MPolyRingElem}
    d = degree(p, idx)
    (d <= 1 || length(factors) <= 1) && return __algebraic_discriminant(p, idx)
    total_degree_in_leader = 0
    for f in factors
        df = degree(f, idx)
        df <= 0 && return __algebraic_discriminant(p, idx)
        total_degree_in_leader += df
    end
    total_degree_in_leader == d || return __algebraic_discriminant(p, idx)
    result = one(p)
    for i in eachindex(factors)
        result *= __algebraic_discriminant(factors[i], idx)
        iszero(result) && return result
        for j in firstindex(factors):(i-1)
            result *= resultant(factors[j], factors[i], idx)^2
            iszero(result) && return result
        end
    end
    scale = divexact(leading_coefficient(p), prod(leading_coefficient, factors))
    return isone(scale) ? result : scale^(2*d-2) * result
end

###############################################################################
#####  SUBRESULTANTS APPLICATION
###############################################################################

"""
Simplify equations and inequations before proceeding with a compatibility test.
Reduce every equation using the other equations and the inequations, then reduce
every inequation using the simplified equations and the other inequations.
Also remove zero equations, nonzero constant inequations and duplicates.

Return `nothing` if some equation reduces to a nonzero constant or an inequation reduces
to zero (the corresponding system has no solutions); otherwise, return the simplified
equations and inequations.
"""
function __subresultant_compatibility_constraints(E::AbstractVector{<:MPolyRingElem}, I::AbstractVector{<:MPolyRingElem})
    isempty(E) && return copy(E), copy(I)
    P = typeof(first(E))
    equations = P[f for f in E]
    guards = P[q for q in I]
    for i in eachindex(equations)
        others = P[equations[j] for j in eachindex(equations) if j != i && !iszero(equations[j])]
        p = __subresultant_reduce_condition(equations[i], others, guards)
        is_constant(p) && !iszero(p) && return nothing
        equations[i] = p
    end
    filter!(!iszero, equations)
    unique!(equations)
    for i in eachindex(guards)
        others = P[guards[j] for j in eachindex(guards) if j != i]
        q = __subresultant_reduce_condition(guards[i], equations, others)
        iszero(q) && return nothing
        guards[i] = q
    end
    filter!(q -> !is_constant(q), guards)
    unique!(guards)
    return equations, guards
end