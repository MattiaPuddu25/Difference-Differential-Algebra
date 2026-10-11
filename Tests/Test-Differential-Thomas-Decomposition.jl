using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Internal conversion reads vectors and owns its output" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    P = elem_type(R)
    E, I = [u^2-1], [deepcopy(u)]
    source = DifferentialSystem(R, E, I)
    snapshot = deepcopy((E, I))
    direct = __differential_algebraic_splitting_data(R, E, I)
    wrapped = __differential_algebraic_splitting_data(source)
    @test [direct.conversion(p) for p in direct.system.equations] ==
          [wrapped.conversion(p) for p in wrapped.system.equations]
    @test [direct.conversion(p) for p in direct.system.inequations] ==
          [wrapped.conversion(p) for p in wrapped.system.inequations]
    state = __differential_state(DifferentialSystem(R, P[], P[]))
    direct_branches = __differential_algebraic_selected_branches(R, E, I, state, leader(u);
        check_discriminants=true, factorization=true)
    wrapped_branches = __differential_algebraic_selected_branches(source, state, leader(u);
        check_discriminants=true, factorization=true)
    @test direct_branches == wrapped_branches
    @test (E, I) == snapshot
    zero!(E[1]); zero!(I[1])
    @test direct.conversion.(direct.system.equations) == snapshot[1]
    @test direct.conversion.(direct.system.inequations) == snapshot[2]
end

@testset "Reuse only the last algebraic child's state containers" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    P = elem_type(R)
    state = __differential_state(DifferentialSystem(R, P[], P[]))
    children = __differential_process_selected_constraint(state,
        (polynomial=u*(u-1), equation=true); check_discriminants=true, factorization=true)
    @test length(children) == 2
    @test last(children) === state
    @test first(children) !== state
    @test first(children).T_equations !== state.T_equations
    @test first(children).Q_equations !== state.Q_equations
    @test first(children).normal_form_cache !== state.normal_form_cache
    push!(first(children).Q_equations, u[2])
    @test !(u[2] in state.Q_equations)
end

@testset "Build terminal snapshots only after all checks succeed" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    P = elem_type(R)
    state = __differential_state(DifferentialSystem(R, P[], P[]))
    push!(state.T_equations, deepcopy(u))
    terminal = __differential_certify_terminal!(state; check_discriminants=true)
    @test terminal isa DifferentialSystem
    @test terminal.simple === true
    @test terminal.equations == [u]
    zero!(state.T_equations[1])
    @test terminal.equations == [u]
    state = __differential_state(DifferentialSystem(R, P[], P[]))
    push!(state.T_equations, u)
    push!(state.T_inequations, u[1])
    @test __differential_certify_terminal!(state; check_discriminants=true) === nothing
    @test state.Q_inequations == [u[1]]
end

@testset "Check short-reducer shifts before allocating their differences" begin
    for a in 0:2, b in 0:2, c in 0:2, d in 0:2,
        m1 in (false, true), m2 in (false, true)
        target, source, mask = [a, b], [c, d], BitVector([m1, m2])
        alpha = target-source
        expected = __nonnegative_shift(alpha) && __admissible_shift(alpha, mask)
        @test __differential_short_shift_is_admissible(target, source, mask) == expected
    end
end

@testset "Selected-leader scans preserve pair priority and order" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    P = elem_type(R)
    for (E, I, repair, i, j) in (
        ([x^2-y, x-y, x+y], [x-1], __algebraic_same_leader_equations, 1, 2),
        ([x^2-y], [x-y, x+y], __algebraic_same_leader_equation_inequation, 1, 1),
        (P[], [x-y, x+y, x-1], __algebraic_same_leader_inequations, 1, 2),
    )
        system = AlgebraicSystem(R, E, I)
        stored_E, stored_I = __algebraic_read_system(system)
        expected = __algebraic_step_systems(system, repair(stored_E, stored_I, i, j))
        actual = __algebraic_selected_leader_step(system, x;
            protected_equations=P[], protected_inequations=P[],
            check_discriminants=true, factorize=false)
        @test actual == expected
    end
end

@testset "Requeue differential inequations with reducible lower variables" begin
    R, (w, z, x) = differential_polynomial_ring(QQ, [:w, :z, :x], 1; partition_name=:pot)
    P = elem_type(R)
    state = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    append!(state.T_inequations, [z + x, z + x[1], z + 1])
    __differential_insert_equation!(state, x)
    @test state.T_inequations == [z + 1]
    @test Set(state.Q_inequations) == Set([z + x, z + x[1]])
    @test all(q -> __is_janet_reducible(q, __differential_janet_data!(state)) === false,
              state.T_inequations)

    # Same-leader rule even below the reducer's degree.
    same_leader = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    push!(same_leader.T_inequations, x)
    __differential_insert_equation!(same_leader, x^2 + 1)
    @test isempty(same_leader.T_inequations)
    @test same_leader.Q_inequations == [x]

    source = DifferentialSystem(R, P[], [z + x, x*w + 1])
    original = (source.equations, source.inequations)
    for split in (quasi_simple_splitting, thomas_decomposition)
        branches = split(source)
        @test length(branches) == 2
        @test all(b -> __differential_system_status(b; check_discriminants=true).valid, branches)
        zero_branches = filter(b -> b.equations == [x], branches)
        @test length(zero_branches) == 1
        if length(zero_branches) == 1
            @test only(zero_branches).inequations == [z]
        end
        @test (source.equations, source.inequations) == original
    end
end

@testset "Certify equation separants independently of factorization" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    for split in (quasi_simple_splitting, thomas_decomposition), factorization in (false, true)
        @test isempty(split([u^2, u[1] - 1]; factorization))
        @test isempty(split([u^2], [u[1]]; factorization))
        consistent = split([u^2, u[1]]; factorization)
        @test length(consistent) == 1
        if length(consistent) == 1
            @test only(consistent).equations == [u]
            @test isempty(only(consistent).inequations)
        end
    end
    D, (v, a) = differential_polynomial_ring(QQ, [:v, :a], 1; partition_name=:pot)
    # The polynomial is square-free before specialization, but not on a=0.
    @test isempty(quasi_simple_splitting([a, v^2 - a, v[1] - 1]; factorization=false))
    specialized = quasi_simple_splitting([a, v^2 - a]; factorization=false)
    @test length(specialized) == 1
    if length(specialized) == 1
        @test Set(only(specialized).equations) == Set([a, v])
    end
    # Quasi-simple mode still leaves inequation discriminants unchecked.
    guards = quasi_simple_splitting(DifferentialSystem(D, elem_type(D)[], [v^2 - a]); factorization=false)
    @test length(guards) == 1
    if length(guards) == 1
        @test is_quasi_simple(only(guards))
        @test !is_simple(only(guards))
    end
end

@testset "Reject unsupported differential coefficient domains at entry" begin
    for K in (GF(2), GF(7), ZZ)
        R, u = differential_polynomial_ring(K, :u, 1)
        source = DifferentialSystem([u^2])
        for split in (quasi_simple_splitting, thomas_decomposition)
            @test_throws ArgumentError split(source; factorization=false)
            @test_throws ArgumentError split(DifferentialSystem(R))
        end
        @test source.equations == [u^2]
    end
end

@testset "Differential reduction before algebraic splitting" begin
    @testset "Discard differential contradictions before factoring" begin
        R, (v, u, w) = differential_polynomial_ring(QQ, [:v, :u, :w], 2; partition_name=:pot, index_ordering_name=:degrevlex)
        # u_y=0 implies u_xy=0 and hence v=0. There is no need to split
        # w^2-w or the other algebraic constraints to contradict v != 0.
        E = [u[0, 1], v - u[1, 1], w^2 - w]
        source = DifferentialSystem(R, E, [v])
        @test source.equations == E
        @test source.inequations == [v]
        @test isempty(thomas_decomposition(source))
        @test isempty(quasi_simple_splitting(source))
    end

    @testset "Leave uncertified initials and separants to splitting" begin
        R, (u, v) = differential_polynomial_ring(QQ, [:u, :v], 1; partition_name=:pot)
        # Dividing by v would lose the valid branch v=0, u' != 0.
        source = DifferentialSystem([v*u[1]], [u[1]])
        branches = thomas_decomposition(source)
        @test !isempty(branches)
        @test all(is_simple, branches)
        @test all(b -> is_zero(normal_form(v, b.equations).nf), branches)
        repeated_root = DifferentialSystem([u^2])
        @test isempty(thomas_decomposition(DifferentialSystem([u^2], [u[1]])))
        Z, z = differential_polynomial_ring(ZZ, :z, 1)
        integral = DifferentialSystem([2*z[1]])
    end
end

@testset "OSCAR conversion in the differential algebraic bridge" begin
    K, t = rational_function_field(QQ, "t")
    delta = action_derivation(map_from_func(K, K, derivative))
    R, (u, v) = differential_polynomial_ring(K, [:u, :v], [delta])
    p = u[2]^2 + t*v
    source = DifferentialSystem([p], [u])
    data = __differential_algebraic_splitting_data(source)
    conversion = data.conversion
    @test conversion isa Oscar.__AlgebraicConversionData
    @test coefficient_ring(conversion.mpr) === K
    @test conversion.supported_vars == Set([u[2], v, u])
    @test ngens(conversion.mpr) == 3
    @test conversion(leader(p)) == gen(conversion.mpr, 1)
    @test conversion(conversion(p)) == p
    @test __lift_differential_algebraic_system(data.system, conversion) == source
    # Unsupported jets must not be silently mapped to zero.
    missing = u[1]
    @test __differential_algebraic_image_if_available(missing, conversion) === nothing
    # Regenerate conversion data for a later step after the action ring grows.
    u[20]
    @test !Oscar.__is_data_up_to_date(conversion)
    @test_throws ArgumentError conversion(p)
    refreshed = __differential_algebraic_splitting_data(source)
    @test __lift_differential_algebraic_system(refreshed.system, refreshed.conversion) == source
    constant_system = DifferentialSystem(R, [R(t)], elem_type(R)[])
    constants = __differential_algebraic_splitting_data(constant_system)
    @test ngens(constants.conversion.mpr) == 0
    @test __lift_differential_algebraic_system(constants.system, constants.conversion) == constant_system
end

@testset "Differential Thomas initial state" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    du = apply_action(u, 1)
    system = DifferentialSystem(R, [du], [u + 1])
    state = __DifferentialThomasState(system)
    @test isempty(state.T_equations)
    @test isempty(state.T_inequations)
    @test state.Q_equations == [du]
    @test state.Q_inequations == [u + 1]
end

@testset "Differential InsertEquation moves old constraints back to Q" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    du = apply_action(u, 1)
    d2u = apply_action(u, [2])
    empty_system = DifferentialSystem(R, elem_type(R)[], elem_type(R)[])
    state = __DifferentialThomasState(empty_system)
    push!(state.T_equations, du)
    push!(state.T_inequations, d2u + 1)
    __differential_insert_equation!(state, u)
    @test state.T_equations == [u]
    @test isempty(state.T_inequations)
    @test du in state.Q_equations
    @test d2u + 1 in state.Q_inequations
end

@testset "InsertEquation creates Janet prolongations in Q" begin
    R, u = differential_polynomial_ring(QQ, :u, 2)
    ux = apply_action(u, [1, 0])
    uy = apply_action(u, [0, 1])
    state = __DifferentialThomasState(DifferentialSystem(R, elem_type(R)[], elem_type(R)[]))
    __differential_insert_equation!(state, ux)
    __differential_insert_equation!(state, uy)
    data = __action_janet_data(state.T_equations)
    for i in eachindex(data.equations)
        for j in eachindex(data.multiplicative_variables[i])
            data.multiplicative_variables[i][j] &&
                continue

            action = zeros(Int, n_action_maps(R))
            action[j] = 1

            @test apply_action(data.equations[i], action) in state.Q_equations
        end
    end
end

@testset "Thomas selection uses equations-first ranking" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    du = apply_action(u, 1)
    d2u = apply_action(u, [2])
    state = __DifferentialThomasState(DifferentialSystem(R, [d2u, u], [du + 1]))
    selected = __differential_select_pending!(state)
    @test selected !== nothing
    @test selected.equation
    @test selected.polynomial == u
    @test state.Q_equations == [d2u]
    @test state.Q_inequations == [du + 1]
end

@testset "Thomas selection resolves equal leaders without changing ranking priority" begin
    R, (u, a, b) = differential_polynomial_ring(QQ, [:u, :a, :b], 1; partition_name=:pot)
    # Initials come before degree: a scalar initial requires no additional split.
    @test __differential_smallest_constraint_index([a*u + 1, u^2 + 1]) == 2
    @test __differential_smallest_constraint_index([a*u + 1, b*u + 1]) == 2
    # Compare successive initials when their first leaders coincide.
    @test __differential_smallest_constraint_index([a*b*u + 1, a*u + 1]) == 2
    # An explicit nonvanishing condition can also make an initial preferable.
    @test __differential_smallest_constraint_index([b*u + 1, a*u + 1], [a]) == 2
    state = __DifferentialThomasState(DifferentialSystem(R, [b*u + 1, a*u + 1], [a]))
    @test __differential_select_pending!(state).polynomial == a*u + 1
    @test state.Q_inequations == [a]
    # Prefer a monomial initial even when its successive-leader list is longer.
    monomial_initial = a*b*u + 1
    sum_initial = (a^2 + b)*u + 1
    @test __differential_smallest_constraint_index([sum_initial, monomial_initial]) == 2
    @test __differential_smallest_constraint_index([monomial_initial, sum_initial]) == 1
    # Scalar and explicitly certified initials still take precedence.
    @test __differential_smallest_constraint_index([monomial_initial, u^2 + 1]) == 2
    @test __differential_smallest_constraint_index([monomial_initial, sum_initial], [a^2 + b]) == 2
    # Then compare degree, term count, and coefficient size, in that order.
    @test __differential_smallest_constraint_index([u^2 + 1, u + 1]) == 2
    @test __differential_smallest_constraint_index([u + a + b, u + a]) == 2
    @test __differential_smallest_constraint_index([100*u + a, u + a]) == 2
    @test __differential_smallest_constraint_index([u/100 + a, u + a]) == 2
    # Retain input order when all criteria tie, including exact duplicates.
    @test __differential_smallest_constraint_index([u + a, u + b]) == 1
    @test __differential_smallest_constraint_index([u + b, u + a]) == 1
    @test __differential_smallest_constraint_index([u + a, u + a]) == 1
    @test __differential_smallest_constraint_index([u + 1, a^2 + 1]) == 2
    @test __differential_smallest_constraint_index([u, a, zero(R)]) == 3
    @test __differential_smallest_constraint_index([u, one(R)]) == 2
    # Discard ties at higher leaders before preparing their initial data.
    queue = [u + a, u + b, a^2 + 1, a + 1]
    snapshot = deepcopy(queue)
    @test __differential_smallest_constraint_index(queue) == 4
    @test queue == snapshot

    K, t = rational_function_field(QQ, :t)
    S, y = differential_polynomial_ring(K, :y, 1)
    @test __differential_smallest_constraint_index([(t^3 + t + 1)*y + 1, t*y + 1]) == 2
    @test __differential_smallest_constraint_index([y/(t^3 + t + 1) + 1, y/t + 1]) == 2
    A, s = polynomial_ring(QQ, [:s])
    L = fraction_field(A)
    V, z = differential_polynomial_ring(L, :z, 1)
    c = L(s[1])
    @test __differential_smallest_constraint_index([(c^2 + c + 1)*z + 1, c*z + 1]) == 2
end

@testset "Pending equations discard only exact processed duplicates" begin
    R, u = differential_polynomial_ring(QQ, :u, 2)
    state = __DifferentialThomasState(DifferentialSystem(R, [u, u[1, 0], 2*u, u + 1], [u]))
    push!(state.T_equations, u)
    __differential_remove_processed_duplicates!(state)
    @test state.Q_equations == [u[1, 0], 2*u, u + 1]
    @test state.Q_inequations == [u]  # Keep the inequation: it is a contradiction.
    @test state.T_equations == [u]
    @test state.janet_data === nothing
    data = __differential_janet_data!(state)
    push!(state.Q_equations, deepcopy(u))
    __differential_remove_processed_duplicates!(state)
    @test state.Q_equations == [u[1, 0], 2*u, u + 1]
    @test state.janet_data === data
    # Removing a processed equation makes its pending copy necessary again.
    empty!(state.T_equations)
    __differential_invalidate_janet!(state)
    push!(state.Q_equations, u)
    __differential_remove_processed_duplicates!(state)
    @test u in state.Q_equations

    P = elem_type(R)
    completed = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    append!(completed.T_equations, [u[0, 1], u[1, 1], u[2, 0]])
    __differential_append_nonmultiplicative_prolongations!(completed)
    @test !(u[1, 1] in completed.Q_equations)
    @test u[1, 1] in completed.T_equations
    @test (u[0, 1], 1) in completed.added_prolongations
    data = __differential_janet_data!(completed)
    for i in eachindex(data.equations), j in eachindex(data.multiplicative_variables[i])
        data.multiplicative_variables[i][j] && continue
        prolongation = apply_action(data.equations[i], j)
        @test prolongation in completed.T_equations || prolongation in completed.Q_equations
    end
end

@testset "Differential Reduce uses Janet prolongations" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    du = apply_action(u, 1)

    state = __DifferentialThomasState(DifferentialSystem(R, elem_type(R)[], elem_type(R)[]))
    __differential_insert_equation!(state, u)
    @test is_zero(__differential_reduce_pending(du, state))
end

@testset "Trivial pending constraints are simplified without differential assumptions" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    P = elem_type(R)
    p = u^2 + 1
    source = DifferentialSystem(R, [zero(R), p, p, u[1]], [one(R), p + 1, p + 1])
    original = system_to_vectors(source)
    state = __DifferentialThomasState(source)
    @test !__differential_simplify_trivial_pending!(state)
    @test state.Q_equations == [p, u[1]]
    @test state.Q_inequations == [p + 1]
    @test system_to_vectors(source) == original
    for equation_processed in (false, true), inequation_processed in (false, true)
        contradictory = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
        push!(equation_processed ? contradictory.T_equations : contradictory.Q_equations, p)
        push!(inequation_processed ? contradictory.T_inequations : contradictory.Q_inequations, deepcopy(p))
        @test __differential_simplify_trivial_pending!(contradictory)
    end
    constant_equation = __DifferentialThomasState(DifferentialSystem(R, [R(2)], P[]))
    zero_inequation = __DifferentialThomasState(DifferentialSystem(R, P[], [zero(R)]))
    @test __differential_simplify_trivial_pending!(constant_equation)
    @test __differential_simplify_trivial_pending!(zero_inequation)
    trivial = __DifferentialThomasState(DifferentialSystem(R, [zero(R), zero(R)], [one(R), R(2)]))
    @test !__differential_simplify_trivial_pending!(trivial)
    @test isempty(trivial.Q_equations) && isempty(trivial.Q_inequations)
    # This inexpensive check must leave differential consequences to Janet reduction.
    differential = __DifferentialThomasState(DifferentialSystem(R, [u], [u[1]]))
    @test !__differential_simplify_trivial_pending!(differential)
    @test differential.Q_equations == [u]
    @test differential.Q_inequations == [u[1]]
end

@testset "Elementary derivative cache preserves coefficient actions and Janet admissibility" begin
    K, t = rational_function_field(QQ, :t)
    delta = action_derivation(map_from_func(K, K, derivative))
    R, y = differential_polynomial_ring(K, :y, [delta])
    data = __action_janet_data([t*y])
    dy = __differential_cached_prolongation(data, 1, 1)
    @test dy == y + t*y[1]
    @test __differential_cached_prolongation(data, 1, 1) === dy
    changed = __action_janet_data([(t + 1)*y])
    @test isempty(changed.prolongations)
    @test __differential_cached_prolongation(changed, 1, 1) == y + (t + 1)*y[1]
    @test data.prolongations[(1, (1,))] == y + t*y[1]

    S, u = differential_polynomial_ring(QQ, :u, 2)
    incomplete = __action_janet_data([u[0, 1], u[2, 0]])
    @test !incomplete.multiplicative_variables[1][1]
    @test __differential_cached_prolongation(incomplete, 1, 1) == u[1, 1]
    @test __is_janet_reducible_univariate(u[1, 1], incomplete) === false
    P = elem_type(S)
    state = __DifferentialThomasState(DifferentialSystem(S, P[], P[]))
    __differential_insert_equation!(state, u[1, 0])
    __differential_insert_equation!(state, u[0, 1])
    janet = __differential_janet_data!(state)
    key = (janet.indices[u[0, 1]], (1, 0))
    cached = janet.prolongations[key]
    @test cached == u[1, 1]
    @test isnothing(__differential_cached_passivity!(state, janet))
    @test janet.prolongations[key] === cached
    __differential_invalidate_janet!(state)
    @test isempty(__differential_janet_data!(state).prolongations)
end

@testset "Short pending equations simplify large constraints without premature certification" begin
    R, (w, u, v, a) = differential_polynomial_ring(QQ, [:w, :u, :v, :a], 1; partition_name=:pot)
    P = elem_type(R)
    dense = sum(v^j for j in 1:40)
    p = w + u*dense
    original = deepcopy(p)
    state = __DifferentialThomasState(DifferentialSystem(R, [p, u], P[]))
    @test !__differential_simplify_large_pending!(state; minimum_terms=32)
    @test state.Q_equations == [w, u]
    @test p == original
    @test isempty(state.T_equations)
    @test isempty(state.normal_form_cache)
    @test state.janet_data === nothing
    # The normal size threshold avoids preparing extra data for small targets.
    small = __DifferentialThomasState(DifferentialSystem([u, w + u]))
    @test !__differential_simplify_large_pending!(small; minimum_terms=32)
    @test small.Q_equations == [u, w + u]
    nonunit = __DifferentialThomasState(DifferentialSystem([a*u - v, p]))
    @test !__differential_simplify_large_pending!(nonunit; minimum_terms=32)
    @test nonunit.Q_equations == [a*u - v, p]
    # Keep only immutable fingerprints; branch-local checked sets remain independent.
    @test state.pending_short_signature ==
        (__differential_constraint_fingerprint([w, u]), __differential_constraint_fingerprint(P[]))
    child = copy(state)
    @test child.pending_short_signature === state.pending_short_signature
    push!(child.pending_short_checked, UInt(42))
    @test !(UInt(42) in state.pending_short_checked)

    # Simplify inequations even before S_T contains any equation.
    guarded = __DifferentialThomasState(DifferentialSystem([u], [p]))
    @test !__differential_simplify_large_pending!(guarded; minimum_terms=32)
    @test guarded.Q_inequations == [w]
    @test guarded.Q_equations == [u]
    contradictory = __DifferentialThomasState(DifferentialSystem([u], [u*dense]))
    @test __differential_simplify_large_pending!(contradictory; minimum_terms=32)
    constant_equation = __DifferentialThomasState(DifferentialSystem([u, 1 + u*dense]))
    @test __differential_simplify_large_pending!(constant_equation; minimum_terms=32)

    # Leave the snapshot unchanged when no new short equation is available.
    repeated = __DifferentialThomasState(DifferentialSystem([a*u - v, p]))
    @test !__differential_simplify_large_pending!(repeated; minimum_terms=32)
    snapshot = repeated.pending_short_signature
    @test !__differential_simplify_large_pending!(repeated; minimum_terms=32)
    @test repeated.pending_short_signature === snapshot
    push!(repeated.Q_equations, u)
    @test !__differential_simplify_large_pending!(repeated; minimum_terms=32)
    @test w in repeated.Q_equations
    @test u in repeated.Q_equations

    K, t = rational_function_field(QQ, :t)
    delta = action_derivation(map_from_func(K, K, derivative))
    S, (z, x, y) = differential_polynomial_ring(K, [:z, :x, :y], [delta]; partition_name=:pot)
    relation = t*x - 2*y*y[1]
    linear_data = __differential_short_pending_data([relation], elem_type(S)[], S)
linear = linear_data.substitution
@test linear !== nothing
@test linear_data.pseudo === nothing
@test __differential_short_pending_substitutions(z + x^2, linear) ==
    z + 4*y^2*y[1]^2/t^2
@test relation == t*x - 2*y*y[1]

scalar_data = __differential_short_pending_data([t*x - 1], elem_type(S)[], S)
scalar = scalar_data.substitution
@test scalar !== nothing
@test scalar_data.pseudo === nothing
@test __differential_short_pending_substitutions(z + x[1], scalar) ==
    z - 1/t^2

# Do not expand a derivative with a multi-term right-hand side.
@test __differential_short_pending_substitutions(z + x[1], linear) == z + x[1]
@test __differential_short_pending_substitutions(z + x[3], linear) == z + x[3]
end

@testset "Pending substitutions retain nonmultiplicative Janet prolongations" begin
    R, (w, u) = differential_polynomial_ring(QQ, [:w, :u], 2; partition_name=:pot)
    P = elem_type(R)
    equations = [u[0, 1], u[2, 0]]
    data_info = __differential_short_pending_data(equations, P[], R)
    data = data_info.substitution
    @test data !== nothing
    @test data_info.pseudo === nothing
    # The x direction of u_y is nonmultiplicative in the full equation set.
    @test __differential_short_pending_substitutions(w + u[1, 1], data) == w + u[1, 1]
    @test __differential_short_pending_substitutions(w + u[0, 2], data) == w
    @test equations == [u[0, 1], u[2, 0]]
    # A long equation excluded from the reducers still restricts their masks.
    long = u[2, 0]^2 + u[1, 0] + u
    restricted_info = __differential_short_pending_data([u[0, 1], long], P[], R)
    restricted = restricted_info.substitution
    @test restricted !== nothing
    @test restricted_info.pseudo === nothing
    @test __differential_short_pending_substitutions(u[1, 1], restricted) == u[1, 1]
    # Pure zero-jet consequences need no expanded derivative, even at high order.
    zero_jet_info = __differential_short_pending_data([u], P[], R)
    zero_jet = zero_jet_info.substitution
    @test zero_jet !== nothing
    @test zero_jet_info.pseudo === nothing
    @test __differential_short_pending_substitutions(w + u[11, 13], zero_jet) == w
end

@testset "The S_T/S_Q process returns a minimal differential equation set" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    du = apply_action(u, 1)
    system = DifferentialSystem(R, [u, du], elem_type(R)[])
    branches = quasi_simple_splitting(system; factorization=false)
    @test length(branches) == 1
    branch = only(branches)
    @test branch.equations == [u]
    @test isempty(branch.inequations)
    @test branch.quasi_simple === true
    @test is_quasi_simple(branch)
end

@testset "Algebraic bridge absorbs replaced same-leader constraints" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    P = elem_type(R)
    state = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    old_inequation = u + 1
    replacement = 2*u + 2
    push!(state.T_inequations, old_inequation)
    __differential_requeue_missing_processed!(state, P[], P[replacement], leader(replacement))
    # The local algebraic branch has replaced/normalized the old constraint at
    # the selected leader.  It must be removed from S_T without being sent back
    # to S_Q, otherwise the two representatives can alternate indefinitely.
    @test isempty(state.T_inequations)
    @test isempty(state.Q_inequations)
end

@testset "Branch-local Janet data are cached" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    state = __DifferentialThomasState(DifferentialSystem(R, elem_type(R)[], elem_type(R)[]))
    __differential_insert_equation!(state, u)
    data_1 = __differential_janet_data!(state)
    data_2 = __differential_janet_data!(state)
    @test data_1 === data_2
    @test data_1 !== nothing
    @test state.janet_data === data_1
    __differential_invalidate_janet!(state)
    @test state.janet_data === nothing
    data_3 = __differential_janet_data!(state)
    @test data_3 !== nothing
    @test state.janet_data === data_3
    @test data_3 !== data_1
end

@testset "Selected-leader bridge factorizes only the selected constraint" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    source = DifferentialSystem(R, [u * (u - 1)], elem_type(R)[])
    branches = quasi_simple_splitting(source; factorization=true)
    @test length(branches) == 2
    @test all(is_quasi_simple, branches)
end

@testset "Branch-local normal-form cache follows Janet lifetime" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    state = __DifferentialThomasState(DifferentialSystem(R, elem_type(R)[], elem_type(R)[]))
    __differential_insert_equation!(state, u)
    du = apply_action(u, 1)
    @test is_zero(__differential_reduce_pending(du, state))
    @test !isempty(state.normal_form_cache)
    cache_size = length(state.normal_form_cache)
    @test is_zero(__differential_reduce_pending(du, state))
    @test length(state.normal_form_cache) == cache_size
    __differential_invalidate_janet!(state)
    @test isempty(state.normal_form_cache)
end

@testset "Terminal certification reuses current normal-form cache" begin
    R, u = differential_polynomial_ring(QQ, :u, 2)
    ux = apply_action(u, [1, 0])
    uy = apply_action(u, [0, 1])
    state = __DifferentialThomasState(DifferentialSystem(R, elem_type(R)[], elem_type(R)[]))
    __differential_insert_equation!(state, ux)
    __differential_insert_equation!(state, uy)
    while !isempty(state.Q_equations)
        selected = __differential_select_pending!(state)
        @test selected.equation
        @test is_zero(__differential_reduce_pending(selected.polynomial, state))
    end
    @test isempty(state.Q_inequations)
    cache_size = length(state.normal_form_cache)
    @test cache_size > 0
    terminal = __differential_processed_system(state)
    @test isnothing(__differential_certify_terminal!(state, terminal; check_discriminants=true))
    # All current nonmultiplicative prolongations were already reduced while
    # draining S_Q, so terminal passivity certification should be cache-only.
    @test length(state.normal_form_cache) == cache_size
    @test terminal.quasi_simple === true
    @test terminal.simple === true
end

@testset "Reconsider prolongations after changing the processed equation set" begin
    R, (u, v) = differential_polynomial_ring(QQ, [:u, :v], 2)
    P = elem_type(R)
    state = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    __differential_insert_equation!(state, u[0, 1])
    __differential_insert_equation!(state, u[2, 0])
    @test u[1, 1] in state.Q_equations
    @test (u[0, 1], 1) in state.added_prolongations
    # Model a prolongation already removed from Q under an earlier equation set.
    # A new equation set must reconsider it rather than trusting the historical set.
    empty!(state.Q_equations)
    sibling = copy(state)
    old_data = __differential_janet_data!(sibling)
    __differential_insert_equation!(state, v)
    @test u[1, 1] in state.Q_equations
    @test old_data.equations == sibling.T_equations
    @test !(v in old_data.equations)
    queued = copy(state.Q_equations)
    __differential_append_nonmultiplicative_prolongations!(state)
    @test state.Q_equations == queued
end

@testset "Recompose a simple system with a necessary Janet completion" begin
    R, u = differential_polynomial_ring(QQ, :u, 2; partition_name=:top, index_ordering_name=:degrevlex)
    P = elem_type(R)
    E = [u[0, 1], u[1, 1], u[2, 0]]
    source = DifferentialSystem(R, E, P[])
    @test is_simple(source)
    @test !is_simple(DifferentialSystem(R, [u[0, 1], u[2, 0]], P[]))
    # Input u_xy is first reduced by u_y alone. It must be restored once
    # u_xx changes the Janet division, including on a second decomposition.
    first = thomas_decomposition(source)
    @test length(first) == 1
    if length(first) == 1
        @test Set(only(first).equations) == Set(E)
        second = thomas_decomposition(only(first))
        @test length(second) == 1
        if length(second) == 1
            @test Set(only(second).equations) == Set(E)
            @test __differential_system_status(only(second); check_discriminants=true).valid
        end
    end
end

@testset "Terminal autoreduction keeps leaders and certified nonzero factors" begin
    R, (w, u, a) = differential_polynomial_ring(QQ, [:w, :u, :a], 1; partition_name=:pot)
    P = elem_type(R)
    equations = [w + a*u^3, a*u^3 - 1]
    source = DifferentialSystem(R, equations, [a])
    original = deepcopy(equations)
    branches = thomas_decomposition(source; factorization=false)
    @test length(branches) == 1
    if length(branches) == 1
        E = only(branches).equations
        @test w + 1 in E
        @test Set(leader.(E)) == Set([w, u])
        @test source.equations == original
        @test __differential_system_status(only(branches); check_discriminants=true).valid
    end
    # Only cancel factors certified by the explicit guards. Here h is
    # implicitly nonzero on a^2+1=0, but it has no explicit guard.
    state = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    h = a + 1
    append!(state.T_equations, [a^2 + 1, h*w + h])
    __differential_autoreduce_terminal!(state)
    @test state.T_equations == [a^2 + 1, h*w + h]
    # A known nonzero factor can be cancelled. No equation, including a
    # necessary Janet-completion equation, is removed by the terminal helper.
    state = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    push!(state.T_equations, h*w + h)
    push!(state.T_inequations, h)
    __differential_autoreduce_terminal!(state)
    @test state.T_equations == [w + 1]
    @test state.T_inequations == [h]
    D, v = differential_polynomial_ring(QQ, :v, 1)
    # Even quasi-simple mode certifies equation separants before installing
    # reducers. This refinement is independent of the factorization option.
    quasi = quasi_simple_splitting(DifferentialSystem([v^2]); factorization=false)
    @test length(quasi) == 1
    @test only(quasi).equations == [v]
end

__nf_zero(p, system::DifferentialSystem) = is_zero(normal_form(p, system).nf)

@testset "Literature: Cole-Hopf (BGLR 2012, Example 3.8)" begin
    # F = Q(x,t), derivations ordered (x,t).
    A, (x, t) = polynomial_ring(QQ, [:x, :t])
    K = fraction_field(A)
    dx(c) = derivative(c, 1)
    dt(c) = derivative(c, 2)
    # Order zeta before eta, and use an orderly degree-lex ranking with x > t.
    # In particular zeta_x > eta_t, as required in the paper.
    R, (zeta, eta) = differential_polynomial_ring(
        K,
        [:zeta, :eta],
        [action_derivation(map_from_func(K, K, dx)),
         action_derivation(map_from_func(K, K, dt))];
        index_ordering_name=:deglex
    )
    zeta_x  = zeta[1, 0]
    zeta_t  = zeta[0, 1]
    zeta_xx = zeta[2, 0]
    eta_x   = eta[1, 0]
    eta_t   = eta[0, 1]
    eta_xx  = eta[2, 0]
    heat = eta_t + eta_xx
    coupling = eta*zeta - eta_x
    input = DifferentialSystem(R, [heat, coupling], [eta])
    # Single simple system displayed in the paper.
    e1 = eta_x - eta*zeta
    e2 = eta*zeta_x + eta_t + eta*zeta^2
    paper_branch = DifferentialSystem(R, [e1, e2], [eta])
    @test leader(e2) == zeta_x
    @test is_janet_complete(paper_branch.equations)
    @test is_passive(paper_branch.equations)
    @test is_quasi_simple(paper_branch)
    @test is_simple(paper_branch)
    burgers = zeta_t + zeta_xx + 2zeta_x*zeta
    @test __nf_zero(burgers, paper_branch)
    T = thomas_decomposition(input; factorization=true)
    @test length(T) == 1
    @test all(is_simple, T)
    @test __nf_zero(burgers, only(T))
end

@testset "Literature: singular ODE branches (Robertz 2018, Example 3.27)" begin
    K, x = rational_function_field(QQ, "x")
    R, u = differential_polynomial_ring(K, :u, [action_derivation(map_from_func(K, K, derivative))])
    ux = u[1]
    p = ux^3 - 4x*u*ux + 8u^2
    singular = (27u - 4x^3)*u
    input = DifferentialSystem(R, [p], elem_type(R)[])
    # The two systems displayed in the paper when the factorization of
    # the singular branch is not performed.
    regular_branch = DifferentialSystem(R, [p], [singular])
    singular_branch = DifferentialSystem(R, [singular], elem_type(R)[])
    @test is_simple(regular_branch)
    @test is_simple(singular_branch)
    T_nofact = thomas_decomposition(input; factorization=false, vector_output=false)
    @test length(T_nofact) == 3
    @test all(branch -> is_simple(branch), T_nofact)
    # The singular branch can be split into
    # u = 0 and 27u - 4x^3 = 0.
    zero_branch = DifferentialSystem(R, [u], elem_type(R)[])
    envelope_branch = DifferentialSystem(R, [27u - 4x^3], elem_type(R)[])
    @test is_simple(zero_branch)
    @test is_simple(envelope_branch)
    T_fact = thomas_decomposition(input; factorization=true)
    @test all(is_simple, T_fact)
    # This is a particularly useful regression test for the semantics of
    # factorization=true.  The paper explicitly permits these three branches.
    @test length(T_fact) == 3
end

@testset "Literature: nonlinear PDE decomposition (Robertz 2018, Example 3.28)" begin
    # Derivation order: (x, y), degree-reverse lexicographic ranking with dx > dy.
    R, u = differential_polynomial_ring(QQ, :u, 2; index_ordering_name=:degrevlex)
    ux  = u[1, 0]
    uy  = u[0, 1]
    uxx = u[2, 0]
    uyy = u[0, 2]
    p1 = uxx - uyy
    p2 = ux - u^2
    input = DifferentialSystem(R, [p1, p2], elem_type(R)[])
    # After differential pseudo-division the paper reaches:
    E0 = [ux - u^2, uyy - 2u^3]
    @test is_janet_complete(E0)
    @test !is_passive(E0)
    obstruction_data = is_passive(E0; missing=true)
    @test obstruction_data !== true
    obstruction = obstruction_data[2]
    expected_obstruction = 2*(uy^2 - u^4)
    @test obstruction == expected_obstruction || obstruction == -expected_obstruction
    # The two simple systems displayed in the paper.
    branch_plus = DifferentialSystem(R, [ux - u^2, uy + u^2], elem_type(R)[])
    branch_minus = DifferentialSystem(R, [ux - u^2, uy - u^2], [u])
    @test is_simple(branch_plus)
    @test is_simple(branch_minus)
    @test is_janet_complete(branch_plus.equations)
    @test is_janet_complete(branch_minus.equations)
    @test is_passive(branch_plus.equations)
    @test is_passive(branch_minus.equations)
    Q = quasi_simple_splitting(input; factorization=true)
    @test !isempty(Q)
    @test all(is_quasi_simple, Q)
    T = thomas_decomposition(input; factorization=true)
    @test length(T) == 2
    @test all(is_simple, T)
    # Do not require the output equations to be textually identical to the
    # paper: the implementation may retain redundant differential consequences.
    @test count(B -> __nf_zero(uy + u^2, B), T) == 1
    @test count(B -> __nf_zero(uy - u^2, B), T) == 1
    @test sort([length(B.inequations) for B in T]) == [0, 1]
end

@testset "Literature: KdV + Wronskian system (Robertz 2018, Example 3.1 + exercise after 3.28)" begin
    # Paper ranking: d_t u > d_x u.
    R, u = differential_polynomial_ring(QQ, :u, 2; index_ordering_name=:degrevlex)
    ut   = u[1, 0]
    ux   = u[0, 1]
    utx  = u[1, 1]
    uxx  = u[0, 2]
    uxxx = u[0, 3]
    p1 = ut - 6u*ux + uxxx
    p2 = u*utx - ut*ux
    input = DifferentialSystem(R, [p1, p2], elem_type(R)[])
    # The three simple systems displayed in the 2018 notes.
    B1 = DifferentialSystem(R, [u], elem_type(R)[])
    B2 = DifferentialSystem(R, [ut - 6u*ux, uxx], [u])
    B3 = DifferentialSystem(R, [ut, uxxx - 6u*ux], [uxx, u])
    @test is_simple(B1)
    @test is_simple(B2)
    @test is_simple(B3)
    T = thomas_decomposition(input; factorization=true)
    @test length(T) == 3
    @test all(is_simple, T)
    @test count(B -> __nf_zero(u, B), T) == 1
    @test count(
        B ->
            isempty(B.inequations) == false &&
            length(B.inequations) == 1 &&
            __nf_zero(ut - 6u*ux, B) &&
            __nf_zero(uxx, B),
        T
    ) == 1
    @test count(
        B ->
        length(B.inequations) == 2 &&
        __nf_zero(ut, B) &&
        __nf_zero(uxxx - 6u*ux, B),
        T,
    ) == 1
end

@testset "Literature: Navier-Stokes simple system (Robertz 2018, Example 3.30)" begin
    # Coefficient mu is a differential constant.
    K, mu = rational_function_field(QQ, "mu")
    zero_derivations = [action_derivation(K) for _ in 1:4]
    # Derivation order (t,x,y,z), differential indeterminates v1 > v2 > v3 > p.
    R, (v1, v2, v3, p) = differential_polynomial_ring(
        K,
        [:v1, :v2, :v3, :p],
        zero_derivations;
        index_ordering_name=:degrevlex,
    )
    # Short jet aliases.
    v1t, v1x, v1y, v1z = v1[1,0,0,0], v1[0,1,0,0], v1[0,0,1,0], v1[0,0,0,1]
    v2t, v2x, v2y, v2z = v2[1,0,0,0], v2[0,1,0,0], v2[0,0,1,0], v2[0,0,0,1]
    v3t, v3x, v3y, v3z = v3[1,0,0,0], v3[0,1,0,0], v3[0,0,1,0], v3[0,0,0,1]
    px, py, pz = p[0,1,0,0], p[0,0,1,0], p[0,0,0,1]
    v1xx, v1yy, v1zz = v1[0,2,0,0], v1[0,0,2,0], v1[0,0,0,2]
    v2xx, v2yy, v2zz = v2[0,2,0,0], v2[0,0,2,0], v2[0,0,0,2]
    v3xx, v3yy, v3zz = v3[0,2,0,0], v3[0,0,2,0], v3[0,0,0,2]
    v2xy = v2[0,1,1,0]
    v3xz = v3[0,1,0,1]
    pxx, pyy, pzz = p[0,2,0,0], p[0,0,2,0], p[0,0,0,2]
    continuity = v1x + v2y + v3z
    A1 = mu*(v1xx + v1yy + v1zz) - v1*v1x - v2*v1y - v3*v1z - px - v1t
    A2 = mu*(v2xx + v2yy + v2zz) - v1*v2x - v2*v2y - v3*v2z - py - v2t
    A3 = mu*(v3xx + v3yy + v3zz) - v1*v3x - v2*v3y - v3*v3z - pz - v3t
    # v1-equation printed in Example 3.30.
    A1_reduced =
        mu*v1yy + mu*v1zz - mu*v2xy - mu*v3xz +
        v1*v2y + v1*v3z - v2*v1y - v3*v1z - px - v1t
    pressure =
        pxx + pyy + pzz +
        2v1y*v2x + 2v1z*v3x +
        2v2y^2 + 2v2y*v3z + 2v2z*v3y + 2v3z^2
    paper_system = DifferentialSystem(R, [continuity, A2, A1_reduced, A3, pressure], elem_type(R)[])
    @test is_janet_complete(paper_system.equations)
    @test is_passive(paper_system.equations)
    @test is_quasi_simple(paper_system)
    @test is_simple(paper_system)
    input = DifferentialSystem(R, [A1, A2, A3, continuity], elem_type(R)[])
    T = thomas_decomposition(input; factorization=true)
    @test length(T) == 1
    @test all(is_simple, T)
    B = only(T)
    @test __nf_zero(continuity, B)
    @test __nf_zero(A2, B)
    @test __nf_zero(A1_reduced, B)
    @test __nf_zero(A3, B)
    @test __nf_zero(pressure, B)
end

@testset "Literature (Gerdt-Robertz 2018, Example 4.6)" begin
    R, u = differential_polynomial_ring(QQ, :u, 3; index_ordering_name=:degrevlex)
    uxx = u[2,0,0]
    uxy = u[1,1,0]
    uxz = u[1,0,1]
    uyy = u[0,2,0]
    uyz = u[0,1,1]
    uzz = u[0,0,2]
    H = matrix(R, [
    u[2,0,0]  u[1,1,0]  u[1,0,1];
    u[1,1,0]  u[0,2,0]  u[0,1,1];
    u[1,0,1]  u[0,1,1]  u[0,0,2]
    ])
    p = det(H)
    S = DifferentialSystem([p])
    T = thomas_decomposition(S; vector_output=true)
    h = uyy*uzz - uyz^2
    @test leader(p) == uxx
    @test initial(p) == h
    paper_T2_equations = [-uyz^2*uxx + 2*uyz*uxz*uxy - uyy*uxz^2, uzz]
    uxzz = u[1,0,2]
    # The paper's T2 omits this differential consequence. Our definition of simple system
    # requires Janet completeness: the variable x is non-multiplicative for uzz because
    # the other leader is uxx.
    @test !is_janet_complete(paper_T2_equations)
    @test is_janet_complete([paper_T2_equations; uxzz])
    expected = [
        ([p], [h, uzz]),                        # T1
        ([paper_T2_equations; uxzz], [uyz]),    # T2 (+ uxzz=0)
        ([uzz*uxy - uyz*uxz, h], [uzz]),        # T3
        ([uxz, uyz, uzz], [uyy]),               # T4
        ([uyy, uyz, uzz], elem_type(R)[]),      # T5
    ]
    normalize_polynomial(f) = iszero(f) ? f : divexact(f, R(leading_coefficient(f)))
    signature(E, I) = (
        length(E), length(I),
        Set(normalize_polynomial(f) for f in E),
        Set(normalize_polynomial(q) for q in I),
    )
    actual_signatures = [signature(E, I) for (E, I) in T]
    @test length(T) == length(expected)
    for (i, (E, I)) in enumerate(expected)
        @testset "Paper branch T$i with required Janet completion" begin
            expected_signature = signature(E, I)
            @test count(s -> s == expected_signature, actual_signatures) == 1
        end
    end
    @test all(is_simple(E, I) for (E, I) in T)
end

@testset "Literature (Gerdt-Robertz 2018, Example 6.2)" begin
    R, (u, v, w) = differential_polynomial_ring(QQ, [:u, :v, :w], 2; partition_name=:top, index_ordering_name=:lex)
    ut, ux, utt, uxx, utx = u[1,0], u[0,1], u[2,0], u[0,2], u[1,1]
    vt, vtt, vtx, vtxx = v[1,0], v[2,0], v[1,1], v[1,2]
    wx, wtt, wtx, wxx = w[0,1], w[2,0], w[1,1], w[0,2]
    @test w < v < u < wx < v[0,1] < ux
    @test u[0,5] < w[1,0] < vt < ut < wtx < vtx < utx
    # Euler-Lagrange equations of L = ut^2/2 + u*vt^2*wx + wt*(ut + vx).
    equations = [
        utt + wtt - vt^2*wx,
        2*u*wx*vtt + 2*ut*vt*wx + 2*u*vt*wtx + wtx,
        utt + vtx + 2*u*vt*vtx + ux*vt^2
    ]
    @test leader.(equations) == [utt, vtt, utt]
    h = 2*u*vt + 1
    paper_branches = [
        ([utt + h*vtx + ux*vt^2,
          2*u*wx*vtt + h*wtx + 2*wx*vt*ut,
          h*vtx - wtt + (ux + wx)*vt^2], [wx, u]),      # T1
        ([utt + h*vtx + ux*vt^2,
          h*vtxx + 2*u*vtx^2 + 4*ux*vt*vtx + vt^2*uxx,
          h*vtx + ux*vt^2 - wtt, wx], [u, h]),          # T2
        ([u, vtx, wtt, wx], elem_type(R)[]),            # T3
        ([utt, ux, h, wtt, wx], [u]),                   # T4
        ([u, vt, wtt, wtx], [wx]),                      # T5
        ([u, vtx, wtt - wx*vt^2, wtx, wxx], [wx, vt]),  # T6
    ]
    # With our definition of simple system, some new equations (obtained by taking
    # derivatives of other equations already in the system) must be added in T2,T3,T4
    completion_equations = [elem_type(R)[], [wtx], [wtx], [utx, wtx], elem_type(R)[], elem_type(R)[]]
    @test [is_janet_complete(E) for (E, I) in paper_branches] == [true, false, false, false, true, true]
    expected = [
        ([E; completion_equations[i]], I)
        for (i, (E, I)) in enumerate(paper_branches)
    ]
    @test all(is_janet_complete(E) for (E, I) in expected)
    T = thomas_decomposition(DifferentialSystem(equations); vector_output=true)
    normalize_polynomial(f) = iszero(f) ? f : divexact(f, R(leading_coefficient(f)))
    signature(E, I) = (
        length(E), length(I),
        Set(normalize_polynomial(f) for f in E),
        Set(normalize_polynomial(q) for q in I),
    )
    actual_signatures = [signature(E, I) for (E, I) in T]
    @test length(T) == length(expected)
    for (i, (E, I)) in enumerate(expected)
        @testset "Paper branch T$i with required Janet completion" begin
            expected_signature = signature(E, I)
            @test count(s -> s == expected_signature, actual_signatures) == 1
        end
    end
    @test all(is_simple(E, I) for (E, I) in T)
end

@testset "Literature (Gerdt-Robertz 2018, Example 6.3)" begin
    R, (q1, q2) = differential_polynomial_ring(QQ, [:q1, :q2], 1; partition_name=:top, index_ordering_name=:lex)
    q1t, q2t, q1tt, q2tt = q1[1], q2[1], q1[2], q2[2]
    @test q2 < q1 < q2t < q1t < q2tt < q1tt
    # Euler-Lagrange equations (15), pp. 16-17.
    equations = [
        4*q2*q2t*q1t + 2*q2^2*q1tt + 2*q1*q2*q2tt - 2*q1,
        4*q1*q2t*q1t + 2*q1^2*q2tt + 2*q1*q2*q1tt - 2*q2,
    ]
    @test leader.(equations) == [q1tt, q1tt]
    @test q2*equations[2] - q1*equations[1] == 2*(q1^2 - q2^2)
    expected = [
        ([2*q2*q2tt + 2*q2t^2 - 1, q1 - q2], [q2]),     # T1
        ([2*q2*q2tt + 2*q2t^2 - 1, q1 + q2], [q2]),     # T2
        ([q1, q2], elem_type(R)[])                      # T3
    ]
    @test all(is_janet_complete(E) for (E, I) in expected)
    T = thomas_decomposition(DifferentialSystem(equations); vector_output=true)
    normalize_polynomial(f) = iszero(f) ? f : divexact(f, R(leading_coefficient(f)))
    signature(E, I) = (
        length(E), length(I),
        Set(normalize_polynomial(f) for f in E),
        Set(normalize_polynomial(q) for q in I),
    )
    actual_signatures = [signature(E, I) for (E, I) in T]
    @test length(T) == length(expected)
    for (i, (E, I)) in enumerate(expected)
        @testset "Paper branch T$i" begin
            expected_signature = signature(E, I)
            @test count(s -> s == expected_signature, actual_signatures) == 1
        end
    end
    @test all(is_simple(E, I) for (E, I) in T)
end

@testset "Literature:  (Gerdt-Robertz 2018, Example 6.4)" begin
    A, (kA, lambdaA, muA) = polynomial_ring(QQ, [:k, :lambda, :mu]);
    K = fraction_field(A);
    k, lambda, mu = K(kA), K(lambdaA), K(muA);
    R, (q1, q2) = differential_polynomial_ring(
        K,
        [:q1, :q2],
        [action_derivation(K)];
        partition_name=:top,
        index_ordering_name=:lex
    );
    q1t, q2t, q1tt, q2tt = q1[1], q2[1], q1[2], q2[2];
    @test q2 < q1 < q2t < q1t < q2tt < q1tt
    @test all(iszero(apply_action(R(c), 1)) for c in (k, lambda, mu))
    h = q1^2*q2t^2 + q1t^2 - k;
    c = q1t*q1tt + q1*q1t*q2t^2 + q1^2*q2t*q2tt;
    equations = [
        2*c*q1t + h*(q1tt - q1*q2t^2) - mu*q1 + lambda*q1^3,
        2*c*q1^2*q2t + h*(2*q1*q1t*q2t + q1^2*q2tt)
    ];
    @test leader.(equations) == [q1tt, q1tt]
    # Prefer the equation with monomial initial 2*q1^2*q1t*q2t in either input order.
    monomial_initial_equation = 2*c*q1^2*q2t + h*(2*q1*q1t*q2t + q1^2*q2tt)
    @test equations[__differential_smallest_constraint_index(equations)] == monomial_initial_equation
    T = thomas_decomposition(DifferentialSystem(equations); vector_output=true);
    @test !isempty(T)
    equations_reversed = [
        2*c*q1^2*q2t + h*(2*q1*q1t*q2t + q1^2*q2tt),
        2*c*q1t + h*(q1tt - q1*q2t^2) - mu*q1 + lambda*q1^3
    ]
    @test equations_reversed[__differential_smallest_constraint_index(equations_reversed)] == monomial_initial_equation
    U = thomas_decomposition(DifferentialSystem(equations_reversed); vector_output=true);
    @test !isempty(U)
    @test T == U
end

__mf_nf_zero(p, system::DifferentialSystem) = is_zero(normal_form(p, system).nf)
@testset "Differential systems from my notes on Modular Forms" begin
    @testset "Ramanujan-Serre system" begin
        # pp. 296-298: D = q*d/dq and the Ramanujan identities.
        R, (E2, E4, E6) = differential_polynomial_ring(QQ, [:E2, :E4, :E6], 1; index_ordering_name=:deglex)
        DE2 = E2[1]
        DE4 = E4[1]
        DE6 = E6[1]
        # p. 297, Theorem 6.9.7:
        f2 = 12*DE2 - E2^2 + E4
        f4 = 3*DE4 - E2*E4 + E6
        f6 = 2*DE6 - E2*E6 + E4^2
        equations = [f2, f4, f6]
        system = DifferentialSystem(R, equations, elem_type(R)[])
        # p. 297: each Ramanujan identity solves for the corresponding first derivative.
        @test leader(f2) == DE2
        # p. 297: each Ramanujan identity solves for the corresponding first derivative.
        @test leader(f4) == DE4
        # p. 297: each Ramanujan identity solves for the corresponding first derivative.
        @test leader(f6) == DE6
        # p. 297: the three Ramanujan identities define the differential system under study
        @test is_janet_complete(equations)
        @test is_passive(equations)
        @test is_quasi_simple(system)
        @test is_simple(system)
        T = thomas_decomposition(system; factorization=false)
        @test length(T) == 1
        branch = only(T)
        @test is_simple(branch)
        # p. 297: no non-vanishing condition occurs in the Ramanujan identities.
        @test isempty(branch.inequations)
        # p. 297, Theorem 6.9.7: all three Ramanujan identities must vanish modulo the branch.
        @test all(f -> __mf_nf_zero(f, branch), equations)
        # p. 298, Corollary 6.9.8:
        #   dΔ/dz = 2*pi*i*E2*Δ,
        # hence D(Δ) = E2*Δ for D = q*d/dq.
        # The factor 1728 in Δ is irrelevant for this differential identity.
        Δ = E4^3 - E6^2
        delta_relation = apply_action(Δ, 1) - E2*Δ
        # p. 298, Corollary 6.9.8: D(Δ) = E2*Δ.
        @test __mf_nf_zero(delta_relation, branch)
        # p. 297, Theorem 6.9.7(1): partial_R(E4) = 8E6.
        serre_E4 = -24*DE4 + 8*E2*E4 - 8*E6
        # p. 297, Theorem 6.9.7(1): partial_R(E4) = 8E6.
        @test __mf_nf_zero(serre_E4, branch)
        # p. 297, Theorem 6.9.7(2): partial_R(E6) = 12E4^2.
        serre_E6 = -24*DE6 + 12*E2*E6 - 12*E4^2
        # p. 297, Theorem 6.9.7(2): partial_R(E6) = 12E4^2.
        @test __mf_nf_zero(serre_E6, branch)
    end

    @testset "Weierstrass differential equation" begin
        # p. 642, Equation (12.4):
        #   wp'^2 = 4wp^3 - 60G4*wp - 140G6.
        P, (g4, g6) = polynomial_ring(QQ, [:G4, :G6])
        K = fraction_field(P)
        G4 = K(g4)
        G6 = K(g6)
        # G4 and G6 are lattice invariants, hence constants for d/dz.
        R, u = differential_polynomial_ring(K, :u, [action_derivation(K)])
        u0 = u[0]
        u1 = u[1]
        u2 = u[2]
        p = u1^2 - 4*u0^3 + 60*G4*u0 + 140*G6
        system = DifferentialSystem(R, [p], elem_type(R)[])
        # p. 642, Equation (12.4): the equation is quadratic in wp'.
        @test leader(p) == u1
        # p. 642, Equation (12.4): the equation has a nontrivial singular locus at wp' = 0.
        @test is_quasi_simple(system)
        # p. 642, Equation (12.4): the quadratic equation in wp' is not globally simple without splitting its singular locus.
        @test !is_simple(system)
        T = thomas_decomposition(system; factorization=false)
        # p. 642, Equation (12.4): Thomas decomposition must return simple differential branches.
        @test all(is_simple, T)
        # p. 642, Equation (12.4): the separant 2wp' yields a regular and a singular case.
        @test length(T) == 2
        singular = only(filter(B -> __mf_nf_zero(u1, B), T))
        regular = only(filter(B -> !__mf_nf_zero(u1, B), T))
        cubic = 4*u0^3 - 60*G4*u0 - 140*G6
        # p. 642, Equation (12.4): on the singular branch wp' = 0.
        @test __mf_nf_zero(u1, singular)
        # p. 642, Equation (12.4): setting wp' = 0 leaves 4wp^3 - 60G4*wp - 140G6 = 0.
        @test __mf_nf_zero(cubic, singular)
        # p. 642, Equation (12.4): the regular branch still satisfies the Weierstrass equation.
        @test __mf_nf_zero(p, regular)
        differentiated_relation = 2*u1*(u2 - 6*u0^2 + 30*G4)
        # p. 642, Equation (12.4): differentiating the Weierstrass equation gives
        # 2wp'*(wp'' - 6wp^2 + 30G4) = 0.
        @test __mf_nf_zero(differentiated_relation, regular)
        # p. 653, Lemma 12.3.12: non-singularity of a Weierstrass cubic is controlled by A^3 - 27B^2.
        A = 60*G4
        B = 140*G6
        discriminant_parameter = A^3 - 27*B^2
        # p. 653, Lemma 12.3.12: for generic independent G4,G6 this discriminant parameter is nonzero.
        @test !is_zero(discriminant_parameter)
    end

    @testset "Hypergeometric equation and Wronskian" begin
        # p. 681, Proposition 12.4.20, specialized on p. 714 to (a,b;c) = (1/2,1/2;1):
        #   z(1-z)y'' + (1-2z)y' - y/4 = 0.
        K, z = rational_function_field(QQ, "z")
        R, (y1, y2) = differential_polynomial_ring(K, [:y1, :y2], [action_derivation(map_from_func(K, K, derivative))]; index_ordering_name=:deglex)
        h1 = 4*z*(1-z)*y1[2] + 4*(1-2*z)*y1[1] - y1[0]
        h2 = 4*z*(1-z)*y2[2] + 4*(1-2*z)*y2[1] - y2[0]
        pair_system = DifferentialSystem(R, [h1, h2], elem_type(R)[])
        # p. 714: y1, y2 satisfy the specialized hypergeometric equation
        @test __mf_nf_zero(h1, pair_system)
        @test __mf_nf_zero(h2, pair_system)

        @test is_janet_complete(pair_system.equations)
        @test is_passive(pair_system.equations)
        @test is_simple(pair_system)

        W = y1[0]*y2[1] - y1[1]*y2[0]
        # p. 690, Equation (12.13): W = y1*y2' - y1'*y2.
        wronskian_relation = z*(1-z)*apply_action(W, 1) + (1-2*z)*W
        # p. 715, Proposition 13.2.9:
        # W' = -(1-2z)/(z(1-z))*W.
        @test __mf_nf_zero(wronskian_relation, pair_system)
        conserved_wronskian = apply_action(z*(1-z)*W, 1)
        # p. 715, Proposition 13.2.9: delta(z(1-z)W) = 0.
        @test __mf_nf_zero(conserved_wronskian, pair_system)
        independent_system = DifferentialSystem(R, [h1, h2], [y2[0], W])
        # p. 713: locally y2 is nonzero and y1,y2 are linearly independent.
        # p. 690: W != 0 is equivalent to linear independence over constants.
        @test is_simple(independent_system)
        T = thomas_decomposition(independent_system; factorization=false)
        # pp. 690 and 713-715: the chosen non-vanishing conditions already describe the independent-solution branch.
        @test length(T) == 1
        branch = only(T)
        # pp. 690 and 713-715: the independent-solution branch should remain simple.
        @test is_simple(branch)
        # p. 715, Proposition 13.2.9: the Wronskian differential relation must hold on the branch.
        @test __mf_nf_zero(wronskian_relation, branch)
    end
end
###############################################################################
#####  SUBRESULTANTS MODIFICATION
###############################################################################

@testset "Exceptional lower equations leave the selected-leader loop unfinished" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    input = AlgebraicSystem([x^2-a, x^2-b], [a])
    deferred = Set{Int}()
    branches = __algebraic_selected_leader_loop(
        input, x; protected_equations=[x^2-a], protected_inequations=[a],
        check_discriminants=true, factorize=false, deferred)
    @test !isempty(branches)
    @test !isempty(deferred)
    @test all(i -> __algebraic_subresultant_has_new_lower_equation(branches[i], [x^2-a], x), deferred)
    @test splitting_covers_system(input, branches)
end

@testset "Factor cancellation returns a lower equation to the differential queue" begin
    R, (x, a) = polynomial_ring(QQ, [:x, :a])
    P = elem_type(R)
    guard = x*sum(a^k for k in 0:8)
    input = AlgebraicSystem([a*x^2], [guard])
    original = deepcopy(input)
    deferred = Set{Int}()
    # The nine-term guard leaves x^2 in the equation during initial cleanup.
    # Cancelling it in the mixed pair exposes a = 0, below the selected leader x.
    branches = __algebraic_selected_leader_loop(
        input, x; protected_equations=P[], protected_inequations=[guard],
        check_discriminants=true, factorize=false, deferred)
    @test length(branches) == 1
    @test deferred == Set([1])
    @test only(branches).equations == [a]
    @test only(branches).inequations == [x]
    @test splitting_covers_system(input, branches)
    @test input.equations == original.equations
    @test input.inequations == original.inequations
end

@testset "Deferred constraints enter S_Q before a selected equation enters S_T" begin
    R, (u, a) = differential_polynomial_ring(QQ, [:u, :a], 1; partition_name=:top)
    P = elem_type(R)
    @test a < u
    state = __differential_state(DifferentialSystem(R, P[], P[]))
    branch = DifferentialSystem([u^2-a, u^3-a, a], [u+1])
    child = __differential_state_from_algebraic_branch(
        state, branch, leader(u); copy_state=false, deferred=true)
    @test child === state
    @test isempty(state.T_equations)
    @test isempty(state.T_inequations)
    @test Set(state.Q_equations) == Set(branch.equations)
    @test state.Q_inequations == [u+1]
    @test __differential_select_pending!(state).polynomial == a
    @test state.needs_safe_precheck
end

@testset "Progressive differential GCD splitting keeps generic and exceptional solutions" begin
    R, (u, a) = differential_polynomial_ring(QQ, [:u, :a], 1; partition_name=:top)
    input = [u^2-a, u^3-a]
    for factorization in (false, true)
        branches = thomas_decomposition(input; factorization)
        @test length(branches) == 2
        @test all(is_simple, branches)
        @test all(D -> all(p -> iszero(normal_form(p, D).nf), input), branches)
        # Identify both constant solution families through the certified normal forms.
        @test count(D -> iszero(normal_form(u, D).nf) && iszero(normal_form(a, D).nf), branches) == 1
        @test count(D -> iszero(normal_form(u-1, D).nf) && iszero(normal_form(a-1, D).nf), branches) == 1
    end
end
@testset "Factorized printing preserves each certified constraint" begin
    R, (x, y) = differential_polynomial_ring(QQ, [:x, :y], 1; partition_name=:pot)
    branch = DifferentialSystem([y^2+1], [x*y])
    original = system_to_vectors(branch)
    # Splitting x*y != 0 into x != 0 and y != 0 would duplicate the equation's leader.
    @test is_simple(branch)
    @test !is_simple(DifferentialSystem([y^2+1], [x, y]))
    io = IOBuffer()
    print_splitting_result([branch]; io)
    @test occursin("Branch 1", String(take!(io)))
    @test system_to_vectors(branch) == original
    A, (a, b) = polynomial_ring(QQ, [:a, :b])
    text = sprint(io -> print_splitting_result([([a^2-b^2], [(a+1)*(b+1)])]; io))
    @test occursin("a + b", text) && occursin("a - b", text)
    @test occursin("a + 1", text) && occursin("b + 1", text)
    branches = thomas_decomposition(branch)
    vectors = thomas_decomposition(branch; vector_output=true)
    @test Set(DifferentialSystem(R, E, I) for (E, I) in vectors) == Set(branches)
    @test all(is_simple(DifferentialSystem(R, E, I)) for (E, I) in vectors)
end

@testset "Incremental pending simplification tracks new targets and changed Janet cones" begin
    R, (w, u, v) = differential_polynomial_ring(QQ, [:w, :u, :v], 2; partition_name=:pot)
    dense = sum(v^j for j in 1:40)
    state = __DifferentialThomasState(DifferentialSystem([u, w + dense]))
    @test !__differential_simplify_large_pending!(state)
    push!(state.Q_equations, v + u*dense)
    @test !__differential_simplify_large_pending!(state)
    @test v in state.Q_equations
    @test w in state.Q_equations
    @test all(p -> length(p) <= 2, state.Q_equations)

    blocker = u[2, 0] + dense
    target = w + u[1, 1]*dense
    restricted = __DifferentialThomasState(DifferentialSystem([u[0, 1], blocker, target]))
    @test !__differential_short_pending_data(restricted.Q_equations, elem_type(R)[], R).substitution.multiplicative_variables[1][1]
    @test !__differential_simplify_large_pending!(restricted)
    @test target in restricted.Q_equations
    # The short reducers did not change, but their x direction is now allowed.
    filter!(!=(blocker), restricted.Q_equations)
    @test __differential_short_pending_data(restricted.Q_equations, elem_type(R)[], R).substitution.multiplicative_variables[1][1]
    @test !__differential_simplify_large_pending!(restricted)
    @test w in restricted.Q_equations
    @test !(target in restricted.Q_equations)
end

@testset "Pending monomial cancellation retains its certifying inequations" begin
    R, (w, u, v, a) = differential_polynomial_ring(QQ, [:w, :u, :v, :a], 1; partition_name=:pot)
    dense = w + sum(v^j for j in 1:40)
    state = __DifferentialThomasState(DifferentialSystem([a^9*dense], [a*(u + 1)]))
    @test !__differential_simplify_large_pending!(state)
    @test state.Q_equations == [dense]
    @test state.Q_inequations == [a*(u + 1)]
    # Lowering the cutoff must not make a guard certify its own removal.
    guard = a*u + a*v + a
    self_guard = __DifferentialThomasState(DifferentialSystem(R, elem_type(R)[], [guard]))
    @test !__differential_simplify_large_pending!(self_guard; minimum_terms=3)
    @test self_guard.Q_inequations == [guard]
    nonunit = __DifferentialThomasState(DifferentialSystem([a*u - 1], [guard]))
    @test !__differential_simplify_large_pending!(nonunit; minimum_terms=3)
    @test nonunit.Q_inequations == [guard]

    # Short retained guards allow cancellation of pseudo-multiplier powers.
    relation = a*u - v
    p = w + sum(u^j*v^(40-j) for j in 1:40)
    guarded = __DifferentialThomasState(DifferentialSystem([relation, p], [a]))
    @test !__differential_simplify_large_pending!(guarded)
    @test relation in guarded.Q_equations
    @test guarded.Q_inequations == [a]
    @test p == w + sum(u^j*v^(40-j) for j in 1:40)
end

@testset "Bound optional pseudo-reduction growth before expanding" begin
    R, (u, a, b, c) = differential_polynomial_ring(QQ, [:u, :a, :b, :c], 1; partition_name=:pot)
    reducer = a*u + b + c
    @test !__differential_short_pseudoreduction_fits(u^8 + 1, reducer, u, a)
    @test __differential_short_pseudoreduction_fits(u^8 + 1, a*u + b, u, a)
    @test __differential_short_pseudoreduction_fits(u^8 + 1, a*u, u, a)
    @test __differential_short_prolongation_fits(u, [0], 8)
    @test !__differential_short_prolongation_fits(a*u + b, [3], 8)
end


@testset "Minimal-leader selection is retained for large equations" begin
    R, (w, u, v) = differential_polynomial_ring(QQ, [:w, :u, :v], 1; partition_name=:pot)
    large = u + sum(v^j for j in 1:40)
    # This short higher-leader equation can reduce a different large pending.
    # Its potential impact must not override the smaller leader of large.
    state = __DifferentialThomasState(DifferentialSystem([large, w, w*sum(v^j for j in 1:40)]))
    @test leader(large) < leader(w)
    @test __differential_select_pending!(state).polynomial == large
end

@testset "Terminal certification returns missing work instead of a false certificate" begin
    K, t = rational_function_field(QQ, :t)
    dx = action_derivation(map_from_func(K, K, derivative))
    dy = action_derivation(K)
    R, u = differential_polynomial_ring(K, :u, [dx, dy])
    P = elem_type(R)
    state = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    append!(state.T_equations, [u[1, 0], u[0, 1]-t])
    terminal = __differential_processed_system(state)
    @test isnothing(__differential_certify_terminal!(state, terminal; check_discriminants=true))
    @test any(p -> is_constant(p) && !iszero(p), state.Q_equations)
    @test terminal.simple === nothing
    @test isempty(thomas_decomposition([u[1, 0], u[0, 1]-t]))

    state = __DifferentialThomasState(DifferentialSystem(R, P[], P[]))
    push!(state.T_equations, u)
    push!(state.T_inequations, u[1, 0])
    terminal = __differential_processed_system(state)
    @test isnothing(__differential_certify_terminal!(state, terminal; check_discriminants=true))
    @test isempty(state.T_inequations)
    @test state.Q_inequations == [u[1, 0]]
    @test terminal.simple === nothing
end
