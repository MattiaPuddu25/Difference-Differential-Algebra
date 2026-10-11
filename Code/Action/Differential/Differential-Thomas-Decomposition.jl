###############################################################################
#####  Quasi-simple splitting
###############################################################################

@doc raw"""
    quasi_simple_splitting(
        system::DifferentialSystem;
        factorization=true,
        vector_output=false
    )
    quasi_simple_splitting(equations, inequations; ...)
    quasi_simple_splitting(equations; ...)

Split a differential system into quasi-simple branches
(over a coefficient field of characteristic zero).\
The splitting uses polynomial factorization by default: this can be deactivated
by setting factorization to false.
The output is a vector of `DifferentialSystem` objects; setting `vector_output`
to `true`, the output is a vector of pairs, each of which consists of a vector
of equations and a vector of inequations of a branch.

Even when `factorization` is `false`, equation discriminants are checked,
splitting into branches accordingly; this establishes the behavior of the
separant of the equation under consideration. It is important, since every
proper derivative of the equation has that separant as its initial.

The algorithm follows the same algorithm described in https://arxiv.org/abs/1108.0817
(§2.2 and §3.4, Algorithms 2.25, 3.2, 3.3 and 3.6). See also
https://doi.org/10.1007/978-3-319-11445-3.

Print `INCONSISTENT SYSTEM` or `SYSTEM ALWAYS VERIFIED` if the original system
satisfies these properties. This does not change the returned output: in the
first case, it is an empty vector; in the second case, a vector consisting of a
single branch without equations and inequations.

Examples

Inconsistencies are also detected without factorization:
the equation u^2 = 0 implies that u = 0, hence u[1] = 0, which contradicts u[1] = 1.

    ```jldoctest
    julia> R, u = differential_polynomial_ring(QQ, :u, 1);
    julia> quasi_simple_splitting([u^2, u[1]]; factorization=false, vector_output=true)
    1-element Vector{Tuple{Vector{DifferentialPolyRingElem{QQFieldElem}}, Vector{DifferentialPolyRingElem{QQFieldElem}}}}:
     ([u[0]], [])
    julia> quasi_simple_splitting([u^2, u[1] - 1]; factorization=false)
    INCONSISTENT SYSTEM
    DifferentialSystem[]
    ```

Quasi-simple branches need not satisfy the discriminant condition for inequations:

    ```jldoctest
    julia> R, (x, a) = differential_polynomial_ring(QQ, [:x, :a], 1; partition_name=:pot);
    julia> q = x^2-a;
    julia> discriminant(q)
    4*a[0]
    julia> source = DifferentialSystem(R, elem_type(R)[], [q]);
    julia> branch = only(quasi_simple_splitting(source; factorization=false))
    DifferentialSystem(0 equations, 1 inequations; quasi_simple = true)
    julia> is_quasi_simple(branch), is_simple(branch)
    (true, false)
    ```
"""
function quasi_simple_splitting(
    system::DifferentialSystem;
    factorization::Bool=true,
    vector_output::Bool=false
)
    branches = __differential_splitting(system; check_discriminants=false, factorization)
    __differential_finalize_result(branches)
    return vector_output ? __action_branch_vectors(system, branches) : branches
end

function quasi_simple_splitting(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    kwargs...
) where {T}
    R = __require_same_parent(equations, inequations)
    R isa DifferentialPolyRing || throw(ArgumentError("Supply a differential polynomial system"))
    system = DifferentialSystem(R, equations, inequations)
    return quasi_simple_splitting(system; kwargs...)
end

function quasi_simple_splitting(equations::AbstractVector{<:DifferentialPolyRingElem{T}}; kwargs...) where {T}
    R = __require_same_parent(equations)
    R isa DifferentialPolyRing || throw(ArgumentError("Supply a nonempty vector of differential equations"))
    return quasi_simple_splitting(DifferentialSystem(R, equations, elem_type(R)[]); kwargs...)
end

###############################################################################
#####  Differential Thomas decomposition
###############################################################################

@doc raw"""
    thomas_decomposition(
        system::DifferentialSystem;
        factorization=true,
        print_output=false,
        vector_output=false,
        factorized_output=true
    )
    thomas_decomposition(equations, inequations; ...)
    thomas_decomposition(equations; ...)

Split a differential system into simple differential branches
(over a coefficient field of characteristic zero).
The algorithm is the same as that used in quasi_simple_splitting: the only
difference is that, in this case, the discriminant condition is also checked
for inequations.

The output is normalized so that the scalar coefficient of the main term of each
polynomial is equal to 1; if `factorized_output` is set to true, the inequations in
the output are factorized.

If `print_output` is set to true, the final branches are printed with
print_splitting_result. This is only useful for improving readability: the output
is still determined by vector_output.

Print INCONSISTENT SYSTEM or SYSTEM ALWAYS VERIFIED if the original system satisfies
these properties. This does not change the returned output: in the first case, it is
an empty vector; in the second case, a vector consisting of a single branch without
equations and inequations.

# Examples

    ```jldoctest
    julia> R, u = differential_polynomial_ring(QQ, :u, 1; partition_name=:top);
    julia> E = [u*u[2]-u[1]-1];
    julia> source = DifferentialSystem(R, E, elem_type(R)[]);
    julia> is_simple(source)
    false
    julia> branch = only(thomas_decomposition(source; vector_output=true))
    (DifferentialPolyRingElem{QQFieldElem}[u[0]*u[2] - u[1] - 1], DifferentialPolyRingElem{QQFieldElem}[u[0]])
    ```

Represent `x(t) + t*y''(t) = 1` over `QQ(t)`, with differentiation with
respect to `t` acting on the coefficients as well. Here `x` and `y` are
differential indeterminates and `y[2]` represents `y''(t)`.
Since `t` is invertible in `QQ(t)`, this equation gives one simple branch
without a separate case for `t = 0`.

    ```jldoctest
    julia> K, t = rational_function_field(QQ, "t");
    julia> delta = action_derivation(map_from_func(K, K, derivative));
    julia> R, (x, y) = differential_polynomial_ring(K, [:x, :y], [delta]; partition_name=:top);
    julia> E = [(t^2*y+x)*x[1] - t*y, x*y[1] - y^2];
    julia> branches = thomas_decomposition(E; vector_output=true)
    3-element Vector{Tuple{Vector{DifferentialPolyRingElem{AbstractAlgebra.Generic.RationalFunctionFieldElem{QQFieldElem, QQPolyRingElem}}}, Vector{DifferentialPolyRingElem{AbstractAlgebra.Generic.RationalFunctionFieldElem{QQFieldElem, QQPolyRingElem}}}}}:
     ([y[0], x[0]], [])
     ([x[1], y[0]], [x[0]])
     ([x[0]*y[1] - y[0]^2, (x[0] + t^2*y[0])*x[1] - t*y[0]], [x[0]^2 + t^2*y[0]*x[0], y[0]])
    ```

KdV equation, restricted to solutions satisfying the Wronskian condition W_x(u,u_t)=0

    ```jldoctest
    julia> R, u = differential_polynomial_ring(QQ, :u, 2);
    julia> ut, ux, uxx = u[1,0], u[0,1], u[0,2];
    julia> utx, uxxx = u[1,1], u[0,3];
    julia> E = [ut - 6*u*ux + uxxx, u*utx - ut*ux];
    julia> is_janet_complete(E)
    true
    julia> q = ut * uxx;
    julia> normal_form(q,E)
    (nf = -u[0,2]*u[0,3] + 6*u[0,2]*u[0,1]*u[0,0], multiplier = 1)
    julia> is_simple(E)
    false
    julia> (A,Ai), (B,Bi), (C,Ci) = thomas_decomposition(E; factorization=true, vector_output=true);
    julia> normal_form(q,A)
    (nf = 0, multiplier = 1)
    julia> normal_form(q,B)
    (nf = 0, multiplier = 1)
    julia> normal_form(q,C)
    (nf = 0, multiplier = 1)
    ```
"""
function thomas_decomposition(
    system::DifferentialSystem;
    factorization::Bool=true,
    print_output::Bool=false,
    vector_output::Bool=false
)
    branches = __differential_splitting(system; check_discriminants=true, factorization)
    __differential_finalize_result(branches)
    # Factor only the printed presentation
    print_output && print_splitting_result(branches)
    return vector_output ? __action_branch_vectors(system, branches) : branches
end

function thomas_decomposition(
    equations::AbstractVector{<:DifferentialPolyRingElem{T}},
    inequations::AbstractVector{<:DifferentialPolyRingElem{T}};
    kwargs...
) where {T}
    R = __require_same_parent(equations, inequations)
    R isa DifferentialPolyRing || throw(ArgumentError("Supply a differential polynomial system"))
    system = DifferentialSystem(R, equations, inequations)
    return thomas_decomposition(system; kwargs...)
end

function thomas_decomposition(equations::AbstractVector{<:DifferentialPolyRingElem{T}}; kwargs...) where {T}
    R = __require_same_parent(equations)
    R isa DifferentialPolyRing || throw(ArgumentError("Supply a nonempty vector of differential equations"))
    return thomas_decomposition(DifferentialSystem(R, equations, elem_type(R)[]); kwargs...)
end

function __differential_finalize_result(result::AbstractVector{<:DifferentialSystem})
    if isempty(result)
        println("INCONSISTENT SYSTEM")
    elseif length(result) == 1
        stored = getfield(only(result), :_data)
        if isempty(stored.equations) && isempty(stored.inequations)
            println("SYSTEM ALWAYS VERIFIED")
        end
    end
    return result
end

###############################################################################
#####  (S_T,S_Q) processing
###############################################################################

mutable struct __DifferentialThomasState{
    R,
    P<:DifferentialPolyRingElem
}
    ring::R
    T_equations::Vector{P}
    T_inequations::Vector{P}
    Q_equations::Vector{P}
    Q_inequations::Vector{P}
    added_prolongations::Set{Tuple{P,Int}}
    # Cached Janet data for S_T
    janet_data::Union{Nothing,__ActionJanetData{P}}
    # The normal-form cache, valid only until the Janet data is (both
    # depend only on T_equations. Reusing it in consecutive reductions
    # avoids rebuilding the recursive coefficient cache for every element of S_Q.
    normal_form_cache::Dict{P,Tuple{P,P}}
    # Certificates that survive changes of the Janet basis. An entry is
    # reused only when its old equations and multiplicative shits are also
    # available in the new basis.
    # Unlike normal_form_cache, this Dict is not cleared by __differential_invalidate_janet!
    normal_form_certificate_cache::Dict{P,__NormalFormCertificate{P}}
    # Proofs that a polynomial reduces to zero, obtained by a shortcut
    # that does not compute the multiplier returned by normal_form.
    zero_inspection_certificates::Dict{P,__PassivityCertificate{P}}
    # Constraints already checked for contradictions
    safe_prechecked_equations::Set{P}
    safe_prechecked_inequations::Set{P}
    safe_precheck_reducers::Set{P}
    # Fingerprints invalidate optional pending reductions without retaining old polynomials
    pending_short_signature::Union{Nothing,Tuple{UInt,UInt}}
    # Hashes of large pending constraints that are already simplified using
    # short equations. These avoid retaining the large polynomials themselves.
    pending_short_checked::Set{UInt}
    # Request an early search for contradictions at the start and after a split.
    # Otherwise, reduce each selected equation or inequation using the processed
    # equations and their allowed derivatives. Discard the branch if an equation
    # becomes a nonzero constant or an inequation becomes zero. This avoids
    # repeating the broader search after every step that produces only one child.
    needs_safe_precheck::Bool
end

"""
Create the initial state, with all input equations and inequations in S_Q
and with S_T empty
"""
function __differential_state(system)
    R = system.ring
    P = elem_type(R)
    E, I = __system_vectors(system)
    return __DifferentialThomasState{
        typeof(R),
        P,
    }(
        R,
        P[],
        P[],
        E,
        I,
        Set{Tuple{P,Int}}(),
        nothing,
        Dict{P,Tuple{P,P}}(),
        Dict{P,__NormalFormCertificate{P}}(),
        Dict{P,__PassivityCertificate{P}}(),
        Set{P}(),
        Set{P}(),
        Set{P}(),
        nothing,
        Set{UInt}(),
        true
    )
end

__DifferentialThomasState(system) = __differential_state(system)

function Base.copy(state::__DifferentialThomasState{R,P}) where {R,P<:DifferentialPolyRingElem}
    return __DifferentialThomasState{R,P}(
        state.ring,
        copy(state.T_equations),
        copy(state.T_inequations),
        copy(state.Q_equations),
        copy(state.Q_inequations),
        copy(state.added_prolongations),
        state.janet_data,
        copy(state.normal_form_cache),
        copy(state.normal_form_certificate_cache),
        copy(state.zero_inspection_certificates),
        copy(state.safe_prechecked_equations),
        copy(state.safe_prechecked_inequations),
        copy(state.safe_precheck_reducers),
        state.pending_short_signature,
        copy(state.pending_short_checked),
        state.needs_safe_precheck
    )
end

function __differential_processed_system(state::__DifferentialThomasState)
    return DifferentialSystem(state.ring, state.T_equations, state.T_inequations)
end

"""
Differential splitting process behind the quasi-simple and simple splittings.
It is based on a decomposition S = (S_T, S_Q) of processed and pending constraints.
To begin with, every input constraint starts in S_Q.
After selecting one of them, reduce it modulo the already processed equations and
perform algebraic refinements. Then some operation is made so that some missing
condition is repaired.
For the Thomas decomposition, the inequation discriminants are also checked by
setting check_discriminants to true.
Every information that can be reused in the process (or in a part of it) is cached.
Eventually, the result is added to S_T.
A branch is certified terminal if and only if S_Q is empty.
"""
function __differential_splitting(
    system::DifferentialSystem;
    check_discriminants::Bool,
    factorization::Bool=true
)
    K = coefficient_ring(system.ring)
    K isa Field && iszero(characteristic(K)) ||
        throw(ArgumentError("Differential splitting requires a coefficient field of characteristic zero"))
    initial_state = __differential_state(system)
    pending = typeof(initial_state)[initial_state]
    finished = DifferentialSystem[]

    while !isempty(pending)
        state = pop!(pending)
        # Remove trivial constraints and detect exact contradictions before
        # normal forms and algebraic splitting
        if state.needs_safe_precheck
            __differential_simplify_trivial_pending!(state) && continue
        end
        # Exploit new short equations throughout the pending queues,
        # before any expensive reduction or algebraic case distinction.
        __differential_simplify_large_pending!(state) && continue
        # Discard a pending inequation with zero normal form before checks
        pending_inequation_contradiction =
            __differential_pending_inequation_contradiction(state)      # OPTIMIZATION
        if pending_inequation_contradiction
            continue
        end
        # Check for contradictions before processing the input and after a split
        # produces several systems. Use exact substitutions from equations linear
        # in their leader with a nonzero constant initial to check the inequations.
        # Otherwise, let the reduction of each selected constraint detect contradictions
        # without repeating this scan of the equations and inequations.
        if state.needs_safe_precheck
            # Known nonzero factors may hide a low-leader pending equation.
            # Cancel them before the next selection; keep the guards that
            # justify this equivalence and do not change the Janet basis.
            __differential_cancel_pending_guard_factors!(state)     # OPTIMIZATION
            inconsistent =
                __differential_state_has_safe_contradiction(state; scan_equations=false)
            state.needs_safe_precheck = false
            inconsistent && continue
        end
        __differential_remove_processed_duplicates!(state)
        if isempty(state.Q_equations) && isempty(state.Q_inequations)
            if check_discriminants
                __differential_autoreduce_terminal!(state)
            end
            terminal = __differential_certify_terminal!(state; check_discriminants)
            if terminal === nothing
                push!(pending, state)
                continue
            end
            push!(finished, terminal)
            continue
        end
        selected = __differential_select_pending!(state)
        children =
            __differential_process_selected_constraint(
                state, selected;
                check_discriminants, factorization
            )
        # Discard obvious contradictions before storing new branches
        for child in children
            if child.needs_safe_precheck && __differential_simplify_trivial_pending!(child)
                continue
            end
            push!(pending, child)
        end
    end
    return unique(finished)
end

function __differential_certify_terminal!(
    state::__DifferentialThomasState{R,P},
    terminal::Union{Nothing,DifferentialSystem}=nothing;
    check_discriminants::Bool
) where {R,P<:DifferentialPolyRingElem}
    if !isempty(state.T_equations)
        janet = __differential_janet_data!(state)
        __differential_cached_passivity!(state, janet)
        for i in reverse(eachindex(state.T_inequations))
            q = state.T_inequations[i]
            __is_janet_reducible(q, janet) === false && continue
            __differential_push_unique!(state.Q_inequations, q)
            deleteat!(state.T_inequations, i)
        end
    end
    isempty(state.Q_equations) && isempty(state.Q_inequations) || return nothing
    # Copy the terminal system only after checking that no pending work remains.
    # Keep the existing return convention for callers supplying their own system.
    supplied_terminal = terminal !== nothing
    terminal === nothing && (terminal = __differential_processed_system(state))
    certificates = getfield(terminal, :_data).certificates
    certificates[:quasi_simple] = true
    check_discriminants && (certificates[:simple] = true)
    return supplied_terminal ? nothing : terminal
end

function __differential_cached_passivity!(
    state::__DifferentialThomasState{R,P},
    data::__ActionJanetData{P}
) where {R,P<:DifferentialPolyRingElem}
    for equation_index in reverse(eachindex(data.equations))
        multiplicative = data.multiplicative_variables[equation_index]
        for action_index in eachindex(multiplicative)
            multiplicative[action_index] && continue
            prolongation = __differential_cached_prolongation(data, equation_index, action_index)
            normal_form = __normal_form_cached(
                prolongation,
                data,
                state.normal_form_cache,
                state.normal_form_certificate_cache,
            ).nf
            # Requeue the nonzero remainder: the branch still needs this compatibility equation
            is_zero(normal_form) || __differential_push_unique!(state.Q_equations, normal_form)
        end
    end
    return nothing
end

###############################################################################
#####  S_T / S_Q process helpers
###############################################################################

function __differential_select_pending!(state::__DifferentialThomasState)
    guards = Iterators.flatten((state.T_inequations, state.Q_inequations))
    if !isempty(state.Q_equations)
        i = __differential_smallest_constraint_index(state.Q_equations, guards)
        p = state.Q_equations[i]
        deleteat!(state.Q_equations, i)
        return (equation=true, polynomial=p)
    end
    isempty(state.Q_inequations) && return nothing
    i = __differential_smallest_constraint_index(state.Q_inequations, guards)
    p = state.Q_inequations[i]
    deleteat!(state.Q_inequations, i)
    return (equation=false, polynomial=p)
end

function __differential_smallest_constraint_index(queue::AbstractVector{P}, guards=()) where {P<:DifferentialPolyRingElem}
    best = firstindex(queue)
    length(queue) == 1 && return best
    is_constant(queue[best]) && return best
    best_leader = nothing
    tied_indices = nothing
    for i in Iterators.drop(eachindex(queue), 1)
        p = queue[i]
        is_constant(p) && return i
        v = leader(p)
        best_leader === nothing && (best_leader = leader(queue[best]))
        if v < best_leader
            best = i
            best_leader = v
            tied_indices = nothing
        elseif v == best_leader
            if tied_indices === nothing
                tied_indices = [best, i]
            else
                push!(tied_indices, i)
            end 
        end
    end
    tied_indices === nothing && return best
    best_data = nothing
    best_coefficient_cost = nothing
    for i in Iterators.drop(tied_indices, 1)
        p = queue[i]
        p == queue[best] && continue
        candidate_data = __differential_selection_data(p, best_leader, guards)
        best_data === nothing && (best_data = __differential_selection_data(queue[best], best_leader, guards))
        comparison = __differential_compare_selection_data(candidate_data, best_data)
        candidate_cost = nothing
        if comparison == 0
            best_coefficient_cost === nothing &&
                (best_coefficient_cost = __differential_constraint_coefficient_cost(queue[best]))
            candidate_cost = __differential_constraint_coefficient_cost(p)
            comparison = cmp(candidate_cost, best_coefficient_cost)
        end
        if comparison < 0
            best = i
            best_data = candidate_data
            best_coefficient_cost = candidate_cost
        end
    end
    return best
end

###############################################################################
#####  Terminal autoreduction and certification
###############################################################################

"""
Reduce lower terms and coefficients of any terminal equation, preserving their
leaders and degrees.
As an optimization, every factor that is certified nonzero by mens of a terminal
inequation (within the limits of __algebraic_strip_guard_factors) is removed.
We skip expensive cancellations, since this only improve the representation but
are not strictly required.
The scalar leading coefficients of the equations are normalized to 1.
"""
function __differential_autoreduce_terminal!(state::__DifferentialThomasState{R,P}) where {R,P<:DifferentialPolyRingElem}
    coefficient_ring(state.ring) isa Field || return state
    isempty(state.T_equations) && return state
    full_data = __differential_janet_data!(state)
    order = sortperm(full_data.leaders)
    lower_equations = P[]
    lower_masks = BitVector[]
    sizehint!(lower_equations, length(order))
    sizehint!(lower_masks, length(order))
    changed = false
    for i in order
        p = state.T_equations[i]
        reduced = if isempty(lower_equations)
            p
        else
            lower_data = __action_janet_data(lower_equations, lower_masks; R=state.ring)
            result = __normal_form_coefficient_cached(p, lower_data, Dict{P,Tuple{P,P}}())
            result.nf
        end
        reduced = __algebraic_strip_guard_factors(reduced, state.T_inequations; max_variables=4)
        scalar = leading_coefficient(reduced)
        if !isone(scalar)
            reduced = divexact(reduced, state.ring(scalar))
        end
        if reduced != p
            state.T_equations[i] = reduced
            changed = true
        end
        push!(lower_equations, reduced)
        push!(lower_masks, full_data.multiplicative_variables[i])
    end
    changed && __differential_invalidate_janet!(state)
    return state
end

function __differential_janet_data!(state::__DifferentialThomasState{R,P}) where {R,P<:DifferentialPolyRingElem}
    isempty(state.T_equations) && return nothing
    if state.janet_data === nothing
        state.janet_data = __action_janet_data(copy(state.T_equations); R=state.ring)
    end
    return state.janet_data
end

function __differential_invalidate_janet!(state::__DifferentialThomasState)
    state.janet_data = nothing
    empty!(state.normal_form_cache)
    empty!(state.added_prolongations)
    return state
end

###############################################################################
#####  Processing constraints
###############################################################################

function __differential_process_selected_constraint(
    state::__DifferentialThomasState{R,P},
    selected;
    check_discriminants::Bool,
    factorization::Bool
) where {R,P<:DifferentialPolyRingElem}
    reduced = __differential_reduce_pending(selected.polynomial, state)
    if is_zero(reduced)
        return selected.equation ?
            __DifferentialThomasState{R,P}[state] :
            __DifferentialThomasState{R,P}[]
    end
    if is_constant(reduced)
        return selected.equation ?
            __DifferentialThomasState{R,P}[] :
            __DifferentialThomasState{R,P}[state]
    end
    x = leader(reduced)
    lower_inequation_context = selected.equation ?
        __differential_lower_inequation_context(state, x) :
        P[]
    local_equations = copy(state.T_equations)
    local_inequations = [state.T_inequations; lower_inequation_context;]
    if selected.equation
        push!(local_equations, reduced)
    else
        push!(local_inequations, reduced)
    end
    algebraic_branches = __differential_algebraic_selected_branches(
            state.ring, local_equations, local_inequations, state, x;
            check_discriminants, factorization)
    children = __DifferentialThomasState{R,P}[]
    number_of_branches = length(algebraic_branches)
    sizehint!(children, number_of_branches)
    for (branch_index, branch) in enumerate(algebraic_branches)
        # Copy earlier children, then reuse the original state for the last one.
        # No later child needs the original contents after this point.
        child = __differential_state_from_algebraic_branch(
                state, branch.system, x;
                algebraic_context_inequations=lower_inequation_context,
                copy_state=branch_index != number_of_branches,
                deferred=branch.deferred
            )
        child === nothing && continue
        if number_of_branches != 1
            child.needs_safe_precheck = true
        end
        push!(children, child)
    end
    return children
end

function __differential_reduce_pending(
    p::P,
    state::__DifferentialThomasState{R,P}
) where {R,P<:DifferentialPolyRingElem}
    isempty(state.T_equations) && return p
    data = __differential_janet_data!(state)
    data === nothing && return p
    # A known zero is enough to discard this pending constraint.
    # The inspection for nonzero cached representatives is a separate
    # sufficient zero test and need not return that same representative
    cached = get(state.normal_form_cache, p, nothing)
    cached !== nothing && is_zero(cached[1]) && return cached[1]
    certified = __normal_form_certificate_lookup(p, data, state.normal_form_certificate_cache)
    if certified !== nothing
        state.normal_form_cache[p] = (certified.nf, certified.multiplier)
        return certified.nf
    end
    if length(p) >= __DIFFERENTIAL_ZERO_INSPECTION_MINIMUM_TERMS # OPTIMIZATION
        zero_inspection = __differential_fast_zero_inspection(p, data; certificate_cache=state.zero_inspection_certificates)
        zero_inspection && return parent(p)(0)
    end
    reduced = __normal_form_cached(p, data, state.normal_form_cache, state.normal_form_certificate_cache)
    return reduced.nf
end

function __differential_lower_inequation_context(state::__DifferentialThomasState{R,P}, x) where {R,P<:DifferentialPolyRingElem}
    result = P[]
    for q in state.Q_inequations
        if is_constant(q) || leader(q) <= x
            push!(result, q)
        end
    end
    return result
end

function __differential_algebraic_selected_branches(
    local_system::DifferentialSystem, state,
    selected_leader; check_discriminants::Bool, factorization::Bool)
    stored = getfield(local_system, :_data)
    return __differential_algebraic_selected_branches(
        local_system.ring, stored.equations, stored.inequations, state,
        selected_leader; check_discriminants, factorization)
end

# Read the local constraints directly, avoiding an intermediate DifferentialSystem
# and its polynomial copies/key. Converted output systems still own their data.
function __differential_algebraic_selected_branches(
    R, E, I, state, selected_leader;
    check_discriminants::Bool, factorization::Bool)
    isempty(E) && isempty(I) && return [(system=DifferentialSystem(R, E, I), deferred=false)]
    data = __differential_algebraic_splitting_data(R, E, I)
    conversion = data.conversion
    selected_algebraic = __differential_algebraic_image_if_available(selected_leader, conversion)
    algebraic_leader = __algebraic_leader(selected_algebraic)
    # The input vectors contain the processed constraints first, and
    # they have therefore already been converted as the leading blocks of `data.system`; we
    # reuse those images instead of converting all of S_T another time
    E_algebraic, I_algebraic = __algebraic_read_system(data.system)
    protected_equations = view(E_algebraic, 1:length(state.T_equations))
    protected_inequations = view(I_algebraic, 1:length(state.T_inequations))
    deferred = Set{Int}()
    algebraic_branches = __with_algebraic_cache() do
        __algebraic_selected_leader_loop(
            data.system,
            algebraic_leader;
            protected_equations,
            protected_inequations,
            check_discriminants,
            factorize=factorization,
            deferred
        )
    end
    return [
        (system=__lift_differential_algebraic_system(branch, conversion), deferred=i in deferred)
        for (i, branch) in enumerate(algebraic_branches)
    ]
end

# Convert a differential polynomial only when its jets belong to the
# support represented by the current finite conversion. Return nothing for unsupported
# jets, so the caller can handle them without attempting an invalid conversion
function __differential_algebraic_image_if_available(
    p::P,
    conversion::Oscar.__AlgebraicConversionData
) where {P<:DifferentialPolyRingElem}
    all(v -> v in conversion.supported_vars, vars(p; sorted=false)) || return nothing
    return conversion(p)
end

"""
Convert `system` to an algebraic system, retaining what is needed
to lift the algebraic branches back to the differential ring.
The empty differential system is not passed to this helper
"""
function __differential_algebraic_splitting_data(system::DifferentialSystem)
    # Conversion only reads the owned polynomials; the output constructor
    # takes its own copies of the converted constraints.
    stored = getfield(system, :_data)
    return __differential_algebraic_splitting_data(system.ring, stored.equations, stored.inequations)
end

# Share the same finite conversion for system data and internal read-only vectors
function __differential_algebraic_splitting_data(R, E, I)
    polynomials = [E; I]
    conversion = Oscar.__algebraic_conversion_data_from_polys(R, polynomials)
    P = elem_type(conversion.mpr)
    E_algebraic = P[conversion(p) for p in E]
    I_algebraic = P[conversion(q) for q in I]
    return (conversion=conversion, system=AlgebraicSystem(conversion.mpr, E_algebraic, I_algebraic))
end

function __lift_differential_algebraic_system(system::AlgebraicSystem, conversion::Oscar.__AlgebraicConversionData)
    E, I = __algebraic_read_system(system)
    P = elem_type(conversion.apr)
    E_differential = P[conversion(p) for p in E]
    I_differential = P[conversion(q) for q in I]
    return DifferentialSystem(conversion.apr, E_differential, I_differential)
end

function __differential_state_from_algebraic_branch(
    base_state::__DifferentialThomasState{R,P},
    branch::DifferentialSystem,
    selected_leader;
    algebraic_context_inequations::AbstractVector{P}=P[],
    copy_state::Bool=true,
    deferred::Bool=false
) where {R,P<:DifferentialPolyRingElem}
    branch_equations, branch_inequations = __system_vectors(branch)
    child = copy_state ? copy(base_state) : base_state
    # Remove pending inequations replaced by the algebraic splitting;
    # their replacements occur in branch_inequations and are handled below
    removed_inequation = nothing
    removed_inequations = nothing
    if !isempty(child.Q_inequations)
        for q in algebraic_context_inequations
            q in branch_inequations && continue
            if removed_inequation === nothing
                removed_inequation = q
            elseif removed_inequations === nothing
                removed_inequations = P[removed_inequation, q]
            else
                push!(removed_inequations, q)
            end
        end
        # Allocate a removal list only for multiple replacements
        if removed_inequations !== nothing
            filter!(!in(removed_inequations), child.Q_inequations)
        elseif removed_inequation !== nothing
            filter!(!=(removed_inequation), child.Q_inequations)
        end
    end
    __differential_requeue_missing_processed!(child, branch_equations, branch_inequations, selected_leader)
    # A new lower equation must be processed before certifying the selected
    # constraint. In particular, repeated selected leaders may still be present.
    deferred && return __differential_defer_algebraic_constraints!(child, branch_equations, branch_inequations)
    processed_equation = nothing
    processed_inequation = nothing
    for p in branch_equations
        is_constant(p) && continue
        leader(p) == selected_leader || continue
        processed_equation = p
        break
    end
    for p in branch_inequations
        is_constant(p) && continue
        leader(p) == selected_leader || continue
        processed_inequation = p
        break
    end
    # Reduce the selected constraint again before adding it to S_T; if
    # its leader decreases, put it back into S_Q instead.
    # Return nothing if this reduction proves a contradiction
    if processed_equation !== nothing && !(processed_equation in child.T_equations)
        __differential_add_processed_equation!(
            child,
            processed_equation,
            selected_leader
        ) || return nothing
    elseif processed_inequation !== nothing && !(processed_inequation in child.T_inequations)
        __differential_add_processed_inequation!(
            child,
            processed_inequation,
            selected_leader
        ) || return nothing
    end
    # Put the remaining new equations and inequations into S_Q;
    # skip the selected constraint handled above and constraints already in S_T.
    for p in branch_equations
        is_constant(p) && continue
        processed_equation !== nothing && p == processed_equation && continue
        p in child.T_equations && continue
        __differential_push_unique!(child.Q_equations, p)
    end
    for p in branch_inequations
        is_constant(p) && continue
        processed_inequation !== nothing && p == processed_inequation && continue
        p in child.T_inequations && continue
        __differential_push_unique!(child.Q_inequations, p)
    end
    return child
end

function __differential_push_unique!(queue::Vector{P}, polynomial::P) where {P<:DifferentialPolyRingElem}
    polynomial in queue || push!(queue, polynomial)
    return nothing
end

# Remove processed constraints that do not survive an algebraic branch. They
# are returned to Q rather than discarded; the next differential reduction
# decides whether they have become redundant under the branch conditions
function __differential_requeue_missing_processed!(
    state::__DifferentialThomasState{R,P},
    branch_equations::AbstractVector{P},
    branch_inequations::AbstractVector{P},
    selected_leader
) where {R,P<:DifferentialPolyRingElem}
    equations_changed = false
    for i in reverse(eachindex(state.T_equations))
        p = state.T_equations[i]
        p in branch_equations && continue
        if is_constant(p) || leader(p) != selected_leader
            __differential_push_unique!(state.Q_equations, p)
        end
        deleteat!(state.T_equations, i)
        equations_changed = true
    end
    equations_changed && __differential_invalidate_janet!(state)
    for i in reverse(eachindex(state.T_inequations))
        q = state.T_inequations[i]
        q in branch_inequations && continue
        if is_constant(q) || leader(q) != selected_leader
            __differential_push_unique!(state.Q_inequations, q)
        end
        deleteat!(state.T_inequations, i)
    end
    return state
end

function __differential_add_processed_equation!(
    state::__DifferentialThomasState{R,P},
    p::P,
    selected_leader
) where {R,P<:DifferentialPolyRingElem}
    reduced = __differential_reduce_pending(p, state)
    is_zero(reduced) && return true
    if is_constant(reduced)
        return false
    end
    if leader(reduced) != selected_leader
        __differential_push_unique!(state.Q_equations, reduced)
        return true
    end
    __differential_insert_equation!(state, reduced)
    return true
end

function __differential_add_processed_inequation!(
    state::__DifferentialThomasState{R,P},
    p::P,
    selected_leader
) where {R,P<:DifferentialPolyRingElem}
    reduced = __differential_reduce_pending(p, state)
    is_zero(reduced) && return false
    is_constant(reduced) && return true
    if leader(reduced) != selected_leader
        __differential_push_unique!(state.Q_inequations, reduced)
        return true
    end
    for i in reverse(eachindex(state.T_inequations))
        q = state.T_inequations[i]
        !is_constant(q) && leader(q) == selected_leader || continue
        __differential_push_unique!(state.Q_inequations, q)
        deleteat!(state.T_inequations, i)
    end
    push!(state.T_inequations, reduced)
    return true
end

"""
Insert `p` into `S_T` and update every constraint that is affected by this equation.
Also move processed inequations back to `S_Q` when their leaders belong to the cone
of `ld(p)` or any of their variables is Janet-reducible modulo the updated equations.
Check all variables, including those in coefficients and lower terms.
Keep these constraints in `S_Q` for reduction.
"""
function __differential_insert_equation!(state::__DifferentialThomasState{R,P}, p::P) where {R,P<:DifferentialPolyRingElem}
    p_data = (isempty(state.T_equations) && isempty(state.T_inequations)) ?
        nothing : __jet_data(leader(p))
    # Move old processed equations whose leaders lie in the cone
    # generated by ld(p) back to S_Q
    for i in reverse(eachindex(state.T_equations))
        q = state.T_equations[i]
        __differential_leader_is_derivative_of(q, p_data) || continue
        __differential_push_unique!(state.Q_equations, q)
        deleteat!(state.T_equations, i)
    end
    push!(state.T_equations, p)
    __differential_invalidate_janet!(state)
    data = __differential_janet_data!(state)
    __differential_append_nonmultiplicative_prolongations!(state, data)
    # Match the terminal irreducibility check, which scans every variable.
    # Keep the leader-cone condition as well: even a same-leader guard of
    # smaller degree must be reconciled algebraically with the new equation.
    for i in reverse(eachindex(state.T_inequations))
        q = state.T_inequations[i]
        (__differential_leader_is_derivative_of(q, p_data) ||
         __is_janet_reducible(q, data) !== false) || continue
        __differential_push_unique!(state.Q_inequations, q)
        deleteat!(state.T_inequations, i)
    end
    return state
end

"""
Return `true` when `ld(q)` belongs to the cone generated by `ld(p)`.
This is required when inserting a new equation into `S_T` and deciding which
old processed constraints have to be moved back to S_Q.
"""
function __differential_leader_is_derivative_of(q::P, p::P) where {P<:DifferentialPolyRingElem}
    is_constant(q) && return false
    is_constant(p) && return false
    return __differential_leader_is_derivative_of(q, __jet_data(leader(p)))
end

function __differential_leader_is_derivative_of(q::DifferentialPolyRingElem, p_data::NamedTuple)
    is_constant(q) && return false
    q_data = __jet_data(leader(q))
    q_data.symbol == p_data.symbol || return false
    length(q_data.shift) == length(p_data.shift) || return false
    @inbounds for i in eachindex(q_data.shift, p_data.shift)
        q_data.shift[i] >= p_data.shift[i] || return false
    end
    return true
end

function __differential_append_nonmultiplicative_prolongations!(
    state::__DifferentialThomasState{R,P},
    data=__differential_janet_data!(state)
) where {R,P<:DifferentialPolyRingElem}
    data === nothing && return nothing
    for equation_index in eachindex(data.equations)
        equation = data.equations[equation_index]
        multiplicative = data.multiplicative_variables[equation_index]
        for action_index in eachindex(multiplicative)
            multiplicative[action_index] && continue
            key = (equation, action_index)
            key in state.added_prolongations && continue
            prolongation = __differential_cached_prolongation(data, equation_index, action_index)
            haskey(data.indices, prolongation) || __differential_push_unique!(state.Q_equations, prolongation)
            push!(state.added_prolongations, key)
        end
    end
    return nothing
end

###############################################################################
#####  Selected-leader algebraic process
###############################################################################

function __algebraic_selected_leader_loop(
    input::AlgebraicSystem,
    selected_leader;
    protected_equations,
    protected_inequations,
    check_discriminants::Bool,
    factorize::Bool,
    deferred::Union{Nothing,Set{Int}}=nothing
)
    initial = __algebraic_cleanup(input)
    initial === nothing && return typeof(input)[]
    queue = typeof(input)[initial]
    result = typeof(input)[]
    seen = Set{__PolynomialSystemKey}()
    while !isempty(queue)
        current = popfirst!(queue)
        if deferred !== nothing && __algebraic_subresultant_has_new_lower_equation(
            current, protected_equations, selected_leader)
            push!(result, current)
            push!(deferred, length(result))
            continue
        end
        key = __algebraic_key(current)
        key in seen && continue
        push!(seen, key)
        next_systems = __algebraic_selected_leader_step(
            current,
            selected_leader;
            protected_equations,
            protected_inequations,
            check_discriminants,
            factorize
        )
        if next_systems === nothing
            push!(result, current)
        else
            append!(queue, next_systems)
        end
    end
    return result
end

function __algebraic_selected_leader_step(
    system::AlgebraicSystem,
    selected_leader;
    protected_equations,
    protected_inequations,
    check_discriminants::Bool,
    factorize::Bool
)
    E, I = __algebraic_read_system(system)
    # Keep the first indices only; return as soon as the preferred pair is known
    equation_index = 0
    for i in eachindex(E)
        is_constant(E[i]) && continue
        __algebraic_leader(E[i]) == selected_leader || continue
        if equation_index != 0
            branches = __algebraic_same_leader_equations(E, I, equation_index, i)
            return __algebraic_step_systems(system, branches)
        end
        equation_index = i
    end
    inequation_index = 0
    for i in eachindex(I)
        is_constant(I[i]) && continue
        __algebraic_leader(I[i]) == selected_leader || continue
        if equation_index != 0
            branches = __algebraic_same_leader_equation_inequation(E, I, equation_index, i)
            return __algebraic_step_systems(system, branches)
        elseif inequation_index != 0
            branches = __algebraic_same_leader_inequations(E, I, inequation_index, i)
            return __algebraic_step_systems(system, branches)
        end
        inequation_index = i
    end
    # No selected-leader constraint remains: the selected constraint was
    # eliminated or lowered by the previous refinement
    equation_index == 0 && inequation_index == 0 && return nothing
    is_equation = equation_index != 0
    index = is_equation ? equation_index : inequation_index
    p = is_equation ? E[index] : I[index]
    # If the selected-leader constraint is exactly the old processed constraint,
    # the newly selected constraint has already been considered
    if (is_equation && p in protected_equations) || (!is_equation && p in protected_inequations)
        return nothing
    end
    # For a selected inequation, perform only its zero/redundancy test
    if !is_equation
        simplification = __algebraic_inequation_simplification(E, I, index; exact=false)
        if simplification !== nothing
            simplification.kind === :inconsistent && return typeof(system)[]
            I_new = eltype(I)[I[j] for j in eachindex(I) if j != index]
            return __algebraic_step_systems(system, __algebraic_normalized([(E, I_new)]))
        end
    end
    # Factor only the selected constraint
    if factorize
        factors = __algebraic_nonconstant_factors(p)
        if is_equation
            if __algebraic_nontrivial_factorization(p, factors)
                issue = (kind=:equation, index=index, factors=factors)
                return __algebraic_step_systems(system, __algebraic_factorization_split(E, I, issue))
            end
        else
            grouped = __algebraic_group_factors_by_leader(factors)
            if __algebraic_nontrivial_factorization(p, grouped)
                issue = (kind=:inequation, index=index, factors=grouped)
                return __algebraic_step_systems(system, __algebraic_factorization_split(E, I, issue))
            end
        end
    end
    # Check only the initial of the selected constraint
    max_leader = selected_leader
    leader_index = var_index(max_leader)
    h = leading_coefficient(p, leader_index)
    lower_E, lower_I = __algebraic_subresultant_lower_context(E, I, leader_index)
    if !is_constant(h) && __subresultant_condition_status(h, lower_E, lower_I) !== true
        branches = is_equation ?
            __algebraic_initial_equation(E, I, index; initial=h) :
            __algebraic_initial_inequation(E, I, index; initial=h)
        return __algebraic_step_systems(system, branches)
    end
    # Every processed equation may be differentiated later. Its separant
    # must be nonzero for those pseudo-reductions to preserve solutions
    if is_equation || check_discriminants
        disc = factorize ?
            __algebraic_discriminant(p, leader_index, factors) :
            __algebraic_discriminant(p, leader_index)
        if !(is_constant(disc) && !iszero(disc))
            if __subresultant_condition_status(disc, lower_E, lower_I) !== true
                E_partial = __leader_filter(E, (__algebraic_leader(f) for f in E), max_leader; check_parent=false)
                issue = (
                    kind=is_equation ? :equation : :inequation,
                    index=index,
                    discriminant=disc,
                    E_partial=E_partial,
                )
                return __algebraic_step_systems(
                    system,
                    __algebraic_discriminant_split(E, I, issue; factorize)
                )
            end
        end
    end
    return nothing
end

###############################################################################
#####  SUBRESULTANTS MODIFICATION
###############################################################################

function __algebraic_subresultant_has_new_lower_equation(system::AlgebraicSystem, processed, selected_leader)
    E, _ = __algebraic_read_system(system)
    return any(p -> !is_constant(p) && __algebraic_leader(p) < selected_leader && !(p in processed), E)
end

function __differential_defer_algebraic_constraints!(
    state::__DifferentialThomasState{R,P}, E::Vector{P}, I::Vector{P}
) where {R,P<:DifferentialPolyRingElem}
    for p in E
        if is_constant(p)
            iszero(p) || return nothing
        elseif !(p in state.T_equations)
            __differential_push_unique!(state.Q_equations, p)
        end
    end
    for q in I
        iszero(q) && return nothing
        if !is_constant(q) && !(q in state.T_inequations)
            __differential_push_unique!(state.Q_inequations, q)
        end
    end
    state.needs_safe_precheck = true
    return state
end

###############################################################################
###############################################################################

# Simplify new or changed large pending constraints with short equations.
function __differential_simplify_large_pending!(
    state::__DifferentialThomasState{R,P}; minimum_terms::Int=__PENDING_LARGE_REDUCTION
) where {R,P<:DifferentialPolyRingElem}
    queues = ((true, state.Q_equations), (false, state.Q_inequations))
    any(p -> length(p) >= minimum_terms,
        Iterators.flatten((state.Q_equations, state.Q_inequations))) || begin
        empty!(state.pending_short_checked)
        return false
    end
    while true
        sources = P[f for f in Iterators.flatten((state.T_equations, state.Q_equations))
            if !is_constant(f) && length(f) <= 2]
        unique!(sources)
        guards = P[q for q in state.T_inequations
            if !is_constant(q) && length(q) <= __DIFFERENTIAL_SHORT_GUARD_TERM_LIMIT]
        append!(guards, (q for q in state.Q_inequations if !is_constant(q) &&
            length(q) <= __DIFFERENTIAL_SHORT_GUARD_TERM_LIMIT && length(q) < minimum_terms))
        unique!(guards)
        signature = (__differential_constraint_fingerprint(sources),
                     __differential_constraint_fingerprint(guards))
        if signature != state.pending_short_signature
            state.pending_short_signature = signature
            empty!(state.pending_short_checked)
        end
        # Long equations also restrict the short reducers' Janet directions.
        # Include their leaders in the fingerprints without retaining any
        # long polynomial.
        equations = P[f for f in Iterators.flatten((state.T_equations, state.Q_equations))
            if !is_constant(f)]
        leader_signature = hash(Set(leader(f) for f in equations))
        has_unchecked = any(queues) do (is_equation, queue)
            any(queue) do p
                length(p) >= minimum_terms && !(is_equation && p in sources) &&
                    !(hash((is_equation, p, leader_signature)) in state.pending_short_checked)
            end
        end
        has_unchecked || return false
        nonzero_variables = __algebraic_nonzero_monomial_variables(guards)
        # Prepare Janet reducers only if an unchecked target can actually use them
        source_jets = [__jet_data(leader(f)) for f in sources]
        applicable = Set{UInt}()
        for (is_equation, queue) in queues, p in queue
            length(p) >= minimum_terms || continue
            is_equation && p in sources && continue
            key = hash((is_equation, p, leader_signature))
            key in state.pending_short_checked && continue
            if __differential_pending_may_simplify(p, source_jets, nonzero_variables)
                push!(applicable, key)
            else
                push!(state.pending_short_checked, key)
            end
        end
        isempty(applicable) && return false
        data = isempty(sources) ? nothing :
            __differential_short_pending_data(equations, guards, state.ring)
        if data === nothing && isempty(nonzero_variables)
            union!(state.pending_short_checked, applicable)
            return false
        end
        replacements = Dict{Tuple{Int,Tuple{Vararg{Int}}},Union{Nothing,P}}()
        new_short_equation = false
        for (is_equation, queue) in queues
            changed = false
            for i in reverse(eachindex(queue))
                p = queue[i]
                length(p) >= minimum_terms || continue
                is_equation && p in sources && continue
                key = hash((is_equation, p, leader_signature))
                key in state.pending_short_checked && continue
                reduced = __algebraic_strip_monomial_variables(p, nonzero_variables)
                if data !== nothing
                    if data.substitution !== nothing
                        reduced = __differential_short_pending_substitutions(
                            reduced, data.substitution, replacements)
                    end
                    if data.pseudo !== nothing && !is_constant(reduced)
                        before_pseudo = reduced
                        reduced = __differential_short_pending_pseudoreductions(
                            reduced, data.pseudo, data.pseudo_initials)
                        # Cancel powers introduced by pseudo-multipliers, then
                        # reuse any exact substitutions exposed by the result
                        reduced = __algebraic_strip_monomial_variables(reduced, nonzero_variables)
                        # Repeat substitutions only if pseudo-reduction or factor cancellation changed the result
                        if data.substitution !== nothing && !is_constant(reduced) && reduced != before_pseudo
                            reduced = __differential_short_pending_substitutions(
                                reduced, data.substitution, replacements)
                        end
                    end
                end
                if is_constant(reduced)
                    is_zero(reduced) == is_equation || return true
                    deleteat!(queue, i)
                    changed = true
                elseif reduced != p
                    queue[i] = reduced
                    changed = true
                    if is_equation && length(reduced) <= 2
                        new_short_equation = true
                    elseif length(reduced) >= minimum_terms
                        push!(state.pending_short_checked, hash((is_equation, reduced, leader_signature)))
                    end
                else
                    push!(state.pending_short_checked, key)
                end
            end
            changed && unique!(queue)
        end
        # A new short equation may simplify constraints already checked above
        new_short_equation || return false
    end
end

# Prepare short linear equations, separating scalar initials from initials that are
# certified nonzero by retained guards. The masks come from all equations:
# omitting other leaders could incorrectly enlarge their Janet cones. The data is used
# only during this queue pass and is not stored as an S_T reduction certificate
function __differential_short_pending_data(
    equations::Vector{P}, guards::Vector{P}, R
) where {P<:DifferentialPolyRingElem}
    substitution_reducers = P[]
    substitution_indices = Int[]
    pseudo_reducers = P[]
    pseudo_indices = Int[]
    pseudo_initials = P[]
    for (i, f) in pairs(equations)
        length(f) <= 2 || continue
        v = leader(f)
        idx = var_index(Oscar.data(v))
        degree(Oscar.data(f), idx) == 1 || continue
        h = R(coeff(Oscar.data(f), [idx], [1]))
        iszero(h) && continue
        if is_constant(h)
            push!(substitution_reducers, isone(h) ? f : divexact(f, h))
            push!(substitution_indices, i)
        elseif __differential_short_guard_certifies_nonzero(h, guards)
            push!(pseudo_reducers, f)
            push!(pseudo_indices, i)
            push!(pseudo_initials, h)
        end
    end
    isempty(substitution_reducers) && isempty(pseudo_reducers) && return nothing
    masks = __action_multiplicative_masks(__action_leader_vectors(equations; R))
    substitution_data = isempty(substitution_reducers) ? nothing :
        __action_janet_data(substitution_reducers, masks[substitution_indices]; R)
    pseudo_data = isempty(pseudo_reducers) ? nothing :
        __action_janet_data(pseudo_reducers, masks[pseudo_indices]; R)
    return (
        substitution=substitution_data,
        pseudo=pseudo_data,
        pseudo_initials=pseudo_initials,
    )
end

function __differential_short_pending_substitutions(
    p::P, data::__ActionJanetData{P},
    replacements=Dict{Tuple{Int,Tuple{Vararg{Int}}},Union{Nothing,P}}()
) where {P<:DifferentialPolyRingElem}
    p = __differential_drop_zero_jet_terms(p, data)
    while !is_constant(p)
        changed = false
        for v in vars(p)
            jet = __jet_data(v)
            for i in data.by_symbol[jet.symbol]
                __differential_short_shift_is_admissible(jet.shift, data.shifts[i], data.multiplicative_variables[i]) || continue
                alpha = jet.shift - data.shifts[i]
                replacement = get!(replacements, (i, Tuple(alpha))) do
                    __differential_short_pending_replacement(data, i, alpha)
                end
                replacement === nothing && continue
                p = __differential_substitute_jet(p, v, replacement)
                changed = true
                break
            end
            changed && break
        end
        changed || return p
    end
    return p
end

function __differential_short_shift_is_admissible(target, source, multiplicative)
    for j in eachindex(target, source, multiplicative)
        target[j] < source[j] && return false
        !multiplicative[j] && target[j] != source[j] && return false
    end
    return true
end

# Differentiate the normalized right-hand side, including its coefficients.
# Reuse zero-jet equations to simplify the result, and accept only zero or one
# term. Return nothing to leave any potentially expansive substitution to the
# ordinary reduction algorithm; the budget of eight terms is a conservative
# cost cutoff, not a condition needed for correctness.
function __differential_short_pending_replacement(
    data::__ActionJanetData{P}, i::Int, alpha::Vector{Int}; maximum_terms::Int=8
) where {P<:DifferentialPolyRingElem}
    replacement = data.leaders[i] - data.equations[i]
    is_zero(replacement) && return replacement
    __differential_short_prolongation_fits(replacement, alpha, maximum_terms) || return nothing
    if any(!iszero, alpha)
        replacement = apply_action(replacement, alpha)
    end
    replacement = __differential_drop_zero_jet_terms(replacement, data)
    return length(replacement) <= 1 ? replacement : nothing
end

# Remove zero equations, nonzero constant inequations, and exact queue duplicates
# before costly reductions. Detect p = 0 together with the identical p != 0,
# even for nonlinear p that has not entered S_T yet. Use only literal identities:
# no division, initial assumptions, or differential-consequence test is involved.
# Check new branches before storing them and flagged states before processing them.
# Keep the broader safe precheck enabled: these literal checks do not replace it.
# Return true when an exact contradiction makes the branch inconsistent.
function __differential_simplify_trivial_pending!(state::__DifferentialThomasState)
    any(p -> is_constant(p) && !is_zero(p), state.Q_equations) && return true
    any(is_zero, state.Q_inequations) && return true
    filter!(!is_zero, state.Q_equations)
    filter!(!is_constant, state.Q_inequations)
    unique!(state.Q_equations)
    unique!(state.Q_inequations)
    (isempty(state.T_inequations) && isempty(state.Q_inequations)) && return false
    (isempty(state.T_equations) && isempty(state.Q_equations)) && return false
    equations = Set(Iterators.flatten((state.T_equations, state.Q_equations)))
    return any(q -> q in equations, Iterators.flatten((state.T_inequations, state.Q_inequations)))
end

function __differential_cached_prolongation(
    data::__ActionJanetData{P}, equation_index::Int, action_index::Int
) where {P<:DifferentialPolyRingElem}
    equation = data.equations[equation_index]
    shift = ntuple(i -> Int(i == action_index), n_action_maps(parent(equation)))
    return get!(data.prolongations, (equation_index, shift)) do
        apply_action(equation, action_index)
    end
end

function __differential_selection_data(p::P, v::P, guards) where {P<:DifferentialPolyRingElem}
    raw = Oscar.data(p)
    variable_index = var_index(Oscar.data(v))
    d = degree(raw, variable_index)
    h = parent(p)(coeff(raw, [variable_index], [d]))
    initial_class = is_constant(h) ? 0 : (h in guards ? 1 : (length(h) == 1 ? 2 : 3))
    initial_leaders = P[]
    while !is_constant(h)
        h_leader = leader(h)
        push!(initial_leaders, h_leader)
        h = __action_univariate_initial(h, h_leader)
    end
    return (initial_class=initial_class, initial_leaders=initial_leaders, degree=d, terms=length(p))
end

function __differential_compare_selection_data(left, right)
    comparison = cmp(left.initial_class, right.initial_class)
    iszero(comparison) || return comparison
    for (a, b) in zip(left.initial_leaders, right.initial_leaders)
        a == b || return a < b ? -1 : 1
    end
    comparison = cmp(length(left.initial_leaders), length(right.initial_leaders))
    iszero(comparison) || return comparison
    comparison = cmp(left.degree, right.degree)
    iszero(comparison) || return comparison
    return cmp(left.terms, right.terms)
end

function __differential_constraint_coefficient_cost(p::DifferentialPolyRingElem)
    return sum(__differential_coefficient_cost(c) for c in coefficients(Oscar.data(p)); init=0)
end

# Heuristic estimation of rational and rational-function coefficient size without converting to
# strings: count integer bits and polynomial storage, including denominators.
function __differential_coefficient_cost(c)
    if c isa ZZRingElem
        return max(1, Oscar.Nemo.nbits(c))
    elseif c isa Integer
        return ndigits(c; base=2)
    elseif c isa Union{Rational,QQFieldElem,Oscar.AbstractAlgebra.FracElem,Oscar.AbstractAlgebra.Generic.RationalFunctionFieldElem}
        return __differential_coefficient_cost(numerator(c)) + __differential_coefficient_cost(denominator(c))
    elseif c isa Union{PolyRingElem,MPolyRingElem}
        return length(c) + sum(__differential_coefficient_cost(a) for a in coefficients(c); init=0)
    end
    return 1
end

# Remove only exact copies of equations still present in S_T, saving queue steps
# and normal-form computations. Reuse the Janet dictionary when current, without
# rebuilding Janet data. Keep proper derivatives and displaced equations in S_Q:
# their leaders alone do not prove that the required checks have been completed
function __differential_remove_processed_duplicates!(state::__DifferentialThomasState)
    (isempty(state.Q_equations) || isempty(state.T_equations)) && return state
    data = state.janet_data
    processed = data === nothing ? Set(state.T_equations) : keys(data.indices)
    filter!(p -> !(p in processed), state.Q_equations)
    return state
end

const __DIFFERENTIAL_SHORT_GUARD_TERM_LIMIT = 8

# Use a small retained guard as a cheap sufficient certificate for an initial.
function __differential_short_guard_certifies_nonzero(
    h::P, guards::AbstractVector{P}
) where {P<:DifferentialPolyRingElem}
    is_constant(h) && return !iszero(h)
    length(h) <= __DIFFERENTIAL_SHORT_GUARD_TERM_LIMIT || return false

    for q in guards
        iszero(q) && continue
        is_constant(q) && continue
        length(q) <= __DIFFERENTIAL_SHORT_GUARD_TERM_LIMIT || continue

        __checked_divides(h, q).divides && return true
    end

    return false
end

# Bound optional prolongations and pseudo-products before constructing them.
# A rejected estimate only postpones the work to the ordinary reduction path;
# it never discards an equation, an inequation or a possible solution.
function __differential_short_pending_pseudoreductions(
    p::P,
    data::__ActionJanetData{P},
    initials::Vector{P};
    maximum_reducer_terms::Int=8
) where {P<:DifferentialPolyRingElem}
    while !is_constant(p)
        changed = false

        for v in vars(p)
            jet = __jet_data(v)

            for i in data.by_symbol[jet.symbol]
                __differential_short_shift_is_admissible(jet.shift, data.shifts[i], data.multiplicative_variables[i]) || continue
                alpha = jet.shift - data.shifts[i]

                key = (i, Tuple(alpha))
                reducer = get(data.prolongations, key, nothing)
                if reducer === nothing
                    __differential_short_prolongation_fits(
                        data.equations[i], alpha, maximum_reducer_terms) || continue
                    reducer = apply_action(data.equations[i], alpha)
                    data.prolongations[key] = reducer
                end

                leader(reducer) == v || continue
                length(reducer) <= maximum_reducer_terms || continue

                h = __action_univariate_initial(reducer, v)

                # For differential prolongations of h*v + b, the coefficient
                # of the prolonged leader must remain the certified initial h.
                h == initials[i] || continue

                __differential_short_pseudoreduction_fits(p, reducer, v, h) || continue

                candidate = __action_linear_pseudoreduce(
                    p, reducer, v; reducer_initial=h
                )

                # This is only an optimization: accept the pseudo-reduction
                # when it actually decreases the polynomial.
                length(candidate) < length(p) || continue

                p = candidate
                changed = true
                break
            end

            changed && break
        end

        changed || return p
    end

    return p
end

# Estimate jet-term growth before applying repeated derivations. This bounds
# the number of jet monomials, not the complexity of their field coefficients.
# Share the same cutoff with exact substitutions and short pseudo-reductions.
function __differential_short_prolongation_fits(p, alpha, maximum_terms::Int)
    iszero(p) && return true
    bound = length(p)
    bound <= maximum_terms || return false
    growth = total_degree(Oscar.data(p)) + 1
    for _ in 1:sum(alpha)
        bound > div(maximum_terms, growth) && return false
        bound *= growth
    end
    return true
end

# Avoid optional Horner pseudo-evaluation when powers of a multi-term initial
# or tail can expand far beyond the input. Exact cancellations may make this
# upper bound pessimistic; skipping this shortcut then remains harmless.
function __differential_short_pseudoreduction_fits(p, reducer, v, h)
    idx = var_index(Oscar.data(v))
    d = degree(Oscar.data(p), idx)
    tail = parent(p)(coeff(Oscar.data(reducer), [idx], [0]))
    iszero(tail) && return true
    growth = max(length(h), length(tail))
    growth <= 1 && return true
    # Each term of p creates at most growth^d terms before like terms cancel.
    # Permit up to four times its current term count in this optional path.
    allowance = 4
    for _ in 1:d
        growth > allowance && return false
        allowance = div(allowance, growth)
    end
    return true
end
