using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Raw arithmetic for differential zero inspections" begin
    A, (x, y) = polynomial_ring(QQ, [:x, :y])
    K = fraction_field(A)
    R, (u, w) = differential_polynomial_ring(K, [:u, :w], 1)
    v = u[1]
    P = elem_type(R)

    @testset "Fused products preserve fraction coefficients" begin
        # These coefficients are polynomials, including nonmonomial numerators.
        polynomial_inputs = (
            (K(x + y)*u + 1, v + w, u + w, K(x)*v - 1),
            (zero(R), v + w, u + 1, v - w),
            (u + w, v + 1, u + w, v + 1),
        )
        for inputs in polynomial_inputs
            original = deepcopy(inputs)
            a, b, c, d = inputs
            expected = a*b - c*d
            @test __action_try_unit_denominator_mul_submul(inputs...) == expected
            @test __action_try_monomial_fraction_mul_submul(inputs...) == expected
            @test inputs == original
        end

        # Force the monomial-denominator path and distinct denominator shifts
        # in the two products. Build a later jet after the inputs exist.
        fraction_inputs = (
            (u/K(x) + w/K(y), v/K(y^2) + 1,
             u/K(x^2), v/K(x*y) - w),
            (u/K(x) + w/K(y), v/K(y^2),
             u/K(x) + w/K(y), v/K(y^2)),
        )
        unused = w[7]
        for inputs in fraction_inputs
            original = deepcopy(inputs)
            a, b, c, d = inputs
            @test __action_try_unit_denominator_mul_submul(inputs...) === nothing
            @test __action_try_monomial_fraction_mul_submul(inputs...) == a*b - c*d
            @test inputs == original
        end

        # A nonmonomial denominator must use the general arithmetic fallback.
        inputs = (u/K(x + y), v + 1, w, u + v)
        @test __action_try_monomial_fraction_mul_submul(inputs...) === nothing
        @test __action_fast_mul_submul(inputs...) ==
            inputs[1]*inputs[2] - inputs[3]*inputs[4]
    end

    @testset "Coefficient scans, reconstruction, and head deletion" begin
        polynomials = P[
            v^4/K(x) + w*v^2/K(y^2) + u/K(x*y),
            w*v^3 + R(K(x)/K(y)),
            R(K(x)/K(y)),
            zero(R),
        ]
        originals = deepcopy(polynomials)
        unused = u[9]
        set_ranking!(R; partition_name=:pot, index_ordering_name=:deglex)
        for p in polynomials
            coefficients = __action_univariate_coefficients_by_terms(p, v)
            @test coefficients == univariate_coefficients(p, v)
            @test __action_rebuild_univariate_by_terms(coefficients, v) == p
            @test __action_drop_univariate_initial_by_terms(p, v) ==
                (iszero(p) ? zero(R) :
                 p - univariate_leading_coefficient(p, v)*v^degree(p, v))
        end
        @test polynomials == originals
        @test __action_rebuild_univariate_by_terms(P[], v) == zero(R)
    end

    @testset "One nonlinear pseudo step retains its exact scale" begin
        h = w
        reducer = h*v^2 + h - 1
        cases = (
            (v^3 + v + 1, reducer, v + h),
            (v^2 + v + 1, reducer, h*v + 1),
            (v^2*reducer, reducer, zero(R)),
        )
        for (p, g, expected) in cases
            originals = deepcopy((p, g))
            result = __action_coefficientwise_pseudo(p, g, v)
            @test result.remainder == expected
            @test result.remainder ==
                univariate_leading_coefficient(g, v)*p -
                univariate_leading_coefficient(p, v)*v^(degree(p, v)-degree(g, v))*g
            @test (p, g) == originals
        end
        p = v^4/K(x) + w*v^2/K(y^2) + u/K(x*y)
        g = v^2/K(y) + w*v/K(x^2) + 1
        @test __action_coefficientwise_pseudo(p, g, v).remainder ==
            p/K(y) - v^2*g/K(x)
    end
end

@testset "Linear pseudo-reduction preserves its scaling conventions" begin
    A, (x, y) = polynomial_ring(QQ, [:x, :y])
    K = fraction_field(A)
    R, (u, w) = differential_polynomial_ring(K, [:u, :w], 1)
    v = u[1]
    h = w
    cases = (
        # The degree-one branch retains the pseudo multiplier h even when
        # the reducer can be made monic on the locus h != 0.
        (v + w, h*(v + w + 1), -h),
        (h*v + w, h*(v + w + 1), w - h*(w + 1)),
        # Higher degrees use the monic factor directly when h divides its tail.
        (v^2 + w, h*(v + w + 1), (w + 1)^2 + w),
        (v + w, h*v, h*w),
        (h*v + w, h*v, w),
        (v^2 + w, h*v, w),
        # Nonmonicizable initials exercise exact, pseudo, and zero transitions.
        (v^2 + v + 1, h*v + 1, h^2 - h + 1),
        (v^3 + v^2 + v + 1, h*v + 1, h^3 - h^2 + h - 1),
        (v^4 + v^3 + v^2 + v + 1, h*v + 1,
         h^4 - h^3 + h^2 - h + 1),
        (h*v^2 + v + 1, h*v + 1, one(R)),
        (v^2/K(x) + w/K(y), v - w, w^2/K(x) + w/K(y)),
    )
    prepared_reducers = Dict{typeof(v),__PreparedActionLinearReducer{typeof(v)}}()
    for (p, g, expected) in cases
        originals = deepcopy((p, g))
        @test __action_linear_pseudoreduce(p, g, v) == expected
        prepared = get!(prepared_reducers, g) do
            tail = g - __action_univariate_initial(g, v)*v
            __PreparedActionLinearReducer{typeof(v)}(tail, nothing, false)
        end
        @test __action_linear_pseudoreduce(p, g, v; prepared) == expected
        @test (p, g) == originals
    end
end

@testset "Zero-jet prefilter respects Janet directions" begin
    R, (u, v) = differential_polynomial_ring(QQ, [:u, :v], 2; partition_name=:pot)
    P = elem_type(R)
    equation = u[1, 0]
    data = __action_janet_data([equation], [BitVector([true, false])])
    p = u[2, 0]*v^2 + equation*(v + 1) + v + 3
    original = deepcopy(p)
    reductions = Tuple{P,Vector{Int}}[]
    @test __differential_drop_zero_jet_terms(p, data, reductions) == v + 3
    @test p == original
    @test Set(reductions) == Set([(equation, [1, 0]), (equation, [0, 0])])

    # This differential consequence remains necessary outside the Janet cone.
    missing = u[1, 1]
    @test __differential_drop_zero_jet_terms(missing, data) == missing
    @test !__differential_fast_zero_inspection(missing, data; minimum_terms=1)

    for f in (equation^2, v*equation)
        other_data = __action_janet_data([f])
        @test __differential_drop_zero_jet_terms(equation, other_data) == equation
    end

    # The raw parent can grow after an input polynomial was constructed.
    old_p = equation + v
    unused = u[7, 3]
    @test __differential_drop_zero_jet_terms(old_p, data) == v
    @test !iszero(unused)
end

@testset "Zero-inspection certificates depend on used reductive prolongations" begin
    R, (u, v) = differential_polynomial_ring(QQ, [:u, :v], 2; partition_name=:pot)
    P = elem_type(R)
    equation = u[1, 0]
    data = __action_janet_data([equation], [BitVector([true, false])])
    p = u[2, 0]^2 + equation*v
    cache = Dict{P,__PassivityCertificate{P}}()
    @test __differential_fast_zero_inspection(p, data; minimum_terms=1, certificate_cache=cache)
    proof = cache[p]

    extended = __action_janet_data([equation, v[0, 1] - v])
    @test __passivity_certificate_valid(proof, extended)
    @test __differential_fast_zero_inspection(p, extended; minimum_terms=1, certificate_cache=cache)
    @test cache[p] === proof
    @test !__differential_fast_zero_inspection(p + 1, extended; minimum_terms=1, certificate_cache=cache)
    @test !haskey(cache, p + 1)

    # Losing a used direction invalidates the proof, even with the same equation.
    restricted = __action_janet_data([equation], [BitVector([false, true])])
    @test !__passivity_certificate_valid(proof, restricted)
    @test !__differential_fast_zero_inspection(p, restricted; minimum_terms=1, certificate_cache=cache)
    @test !haskey(cache, p)

    cache[p] = proof
    removed = __action_janet_data([v[0, 1] - v])
    @test !__passivity_certificate_valid(proof, removed)
    @test !__differential_fast_zero_inspection(p, removed; minimum_terms=1, certificate_cache=cache)
    @test !haskey(cache, p)

    # Branch dictionaries are independent; certificates themselves are read-only.
    state = __differential_state(DifferentialSystem(R, [p]))
    state.zero_inspection_certificates[p] = proof
    child = copy(state)
    empty!(child.zero_inspection_certificates)
    @test haskey(state.zero_inspection_certificates, p)
    __differential_invalidate_janet!(state)
    @test state.zero_inspection_certificates[p] === proof

    S, z = differential_polynomial_ring(QQ, :z, 2)
    @test_throws ArgumentError __differential_fast_zero_inspection(z, data; minimum_terms=1)
end

@testset "Zero inspections certify consequences without changing the input" begin
    R, (w, u) = differential_polynomial_ring(
        QQ, [:w, :u], 1; partition_name=:pot,
    )
    equations = [u[1] - u]
    data = __action_janet_data(equations)
    p = (u[2] - u)*w^2 + (u[3] - u)*w + u[1] - u
    originals = deepcopy((p, equations))
    @test __differential_fast_zero_inspection(p, data; minimum_terms=1)
    @test !__differential_fast_zero_inspection(p + 1, data; minimum_terms=1)
    @test !__differential_fast_zero_inspection(u[1] - u, data)
    proof = __janet_zero_reduce_with_reductions(p, data, nothing)
    @test proof.certificate_safe && iszero(proof.nf)
    @test (p, equations) == originals

    # Here the coefficient derivation contributes the essential extra u.
    K, t = rational_function_field(QQ, "t")
    D, z = differential_polynomial_ring(
        K, :z, [action_derivation(map_from_func(K, K, derivative))],
    )
    nonconstant_data = __action_janet_data([z[1] - t*z])
    consequence = z[2] - (t^2 + 1)*z
    @test __differential_fast_zero_inspection(consequence, nonconstant_data; minimum_terms=1)
    coefficient_cache = Dict{typeof(z),__PassivityCertificate{typeof(z)}}()
    @test __differential_fast_zero_inspection(
        consequence, nonconstant_data; minimum_terms=1, certificate_cache=coefficient_cache,
    )
    @test __passivity_certificate_valid(coefficient_cache[consequence], nonconstant_data)
    @test !__differential_fast_zero_inspection(z[2] - t^2*z, nonconstant_data; minimum_terms=1)

    # Differentiating z^2-t produces 2*z*z'-1, so z' cannot be certified zero.
    nonlinear_data = __action_janet_data([z^2 - t])
    nonzero = __janet_zero_reduce_with_reductions(z[1], nonlinear_data, nothing)
    @test nonzero.certificate_safe
    @test nonzero.nf == one(D)
    @test !__differential_fast_zero_inspection(z[1], nonlinear_data; minimum_terms=1)
end

@testset "Pending guards simplify equations before Thomas selection" begin
    R, (u, v, w) = differential_polynomial_ring(
        QQ, [:u, :v, :w], 1; partition_name=:pot,
    )
    g = u[1] + v
    system = DifferentialSystem(R, [g^2*w, v + w], [g])
    originals = deepcopy((system.equations, system.inequations))
    state = __differential_state(system)
    __differential_insert_equation!(state, u[1])
    data = __differential_janet_data!(state)
    processed = copy(state.T_equations)
    guards = copy(state.Q_inequations)

    __differential_cancel_pending_guard_factors!(state)
    @test Set(state.Q_equations) == Set([w, v + w])
    @test state.Q_inequations == guards
    @test state.T_equations == processed
    @test state.janet_data === data
    @test data !== nothing
    @test __differential_select_pending!(state).polynomial == w
    @test (system.equations, system.inequations) == originals

    # Each size bound skips an optional optimization, leaving the input intact.
    for limits in ((maximum_equation_terms=2, maximum_guard_terms=32),
                   (maximum_equation_terms=32, maximum_guard_terms=1))
        skipped = __differential_state(system)
        before = copy(skipped.Q_equations)
        __differential_cancel_pending_guard_factors!(skipped; limits...)
        @test skipped.Q_equations == before
        @test skipped.Q_inequations == guards
    end

    # Cancel factors of a guard, not only the whole guard, with full multiplicity.
    inconsistent = __differential_state(DifferentialSystem(R, [w^3], [w*g]))
    __differential_cancel_pending_guard_factors!(inconsistent)
    @test isone(only(inconsistent.Q_equations))
    @test inconsistent.Q_inequations == [w*g]
    @test __differential_state_has_safe_contradiction(inconsistent; scan_equations=false)

    unguarded = __differential_state(DifferentialSystem(R, [g*w]))
    __differential_cancel_pending_guard_factors!(unguarded)
    @test unguarded.Q_equations == [g*w]
end

@testset "Keep reduced pending guards visible to the algebraic bridge" begin
    R, (u, v, w) = differential_polynomial_ring(
        QQ, [:u, :v, :w], 1; partition_name=:pot,
    )
    P = elem_type(R)
    for factorization in (false, true)
        state = __differential_state(
            DifferentialSystem(R, [v*w], [u[1] + v, u[1] + 1]),
        )
        __differential_insert_equation!(state, u[1])
        @test !__differential_pending_inequation_contradiction(state)
        @test state.Q_inequations == [v]
        @test __differential_lower_inequation_context(state, v) == [v]
        selected = __differential_select_pending!(state)
        children = __differential_process_selected_constraint(
            state, selected; check_discriminants=true, factorization,
        )
        @test length(children) == 1
        if length(children) == 1
            child = only(children)
            @test w in [child.T_equations; child.Q_equations]
            @test v in [child.T_inequations; child.Q_inequations]
        end
    end

    state = __differential_state(
        DifferentialSystem(R, P[], [w, u[1] + v]),
    )
    __differential_insert_equation!(state, w*u[1] - 1)
    @test !__differential_pending_inequation_contradiction(state)
    @test Set(state.Q_inequations) == Set([w, 1 + w*v])
    @test state.T_equations == [w*u[1] - 1]

    inconsistent = __differential_state(DifferentialSystem(R, P[], [u[1]]))
    __differential_insert_equation!(inconsistent, u[1])
    @test __differential_pending_inequation_contradiction(inconsistent)
end

@testset "Fall back from fraction kernels for rational coefficients" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    inputs = (one(R), u, R(QQ(1, 2)), u[1])
    original = deepcopy(inputs)
    @test __action_try_unit_denominator_mul_submul(inputs...) === nothing
    @test __action_try_monomial_fraction_mul_submul(inputs...) === nothing
    @test __action_fast_mul_submul(inputs...) == u - QQ(1, 2)*u[1]
    @test inputs == original

    # Reach the same fallback through a zero inspection above its normal threshold.
    # Reduction by 2*u'-1 produces the monic tail -1/2.
    f = 2*u[1] - 1
    p = f*sum(u^i for i in 0:64) + u[1]^2 - QQ(1, 4)
    snapshot = deepcopy(p)
    data = __action_janet_data([f])
    @test length(p) >= 64
    @test __differential_fast_zero_inspection(p, data)
    @test !__differential_fast_zero_inspection(p + 1, data)
    @test p == snapshot
    branches = thomas_decomposition([f, p]; factorization=false)
    @test length(branches) == 1
    if length(branches) == 1
        @test iszero(normal_form(f, only(branches).equations).nf)
    end

    # Univariate rational-function numerators also lack the multivariate
    # representation expected by the specialized flattening kernel.
    K, t = rational_function_field(QQ, "t")
    D, z = differential_polynomial_ring(K, :z, 1)
    fractions = (one(D), z, D(1/t), z[1])
    @test __action_try_monomial_fraction_mul_submul(fractions...) === nothing
    @test __action_fast_mul_submul(fractions...) == z - z[1]/t
end

@testset "Discard visible differential contradictions before splitting" begin
    R, u =
        differential_polynomial_ring(
            QQ,
            :u,
            1,
        )

    du = apply_action(u, 1)

    # The inequation u_x != 0 is impossible under u = 0.  The new S_T/S_Q
    # safe contradiction check must detect this from the full current branch
    # without entering the algebraic/factorization bridge.
    source =
        DifferentialSystem(
            R,
            [u],
            [du],
        )

    state =
        __DifferentialThomasState(
            source,
        )

    @test __differential_state_has_safe_contradiction(state)

    @test isempty(
        thomas_decomposition(
            source,
        ),
    )

    @test isempty(
        quasi_simple_splitting(
            source,
        ),
    )
end

@testset "Large differential zero inspection" begin
    R, u =
        differential_polynomial_ring(
            QQ,
            :u,
            1,
        )

    du = apply_action(u, 1)
    equation = du - u

    state =
        __DifferentialThomasState(
            DifferentialSystem(
                R,
                elem_type(R)[],
                elem_type(R)[],
            ),
        )

    __differential_insert_equation!(
        state,
        equation,
    )

    factor = sum(u^k for k in 0:64)
    consequence = equation * factor

    data = __differential_janet_data!(state)

    @test __differential_fast_zero_inspection(
        consequence,
        data;
        minimum_terms=1,
    )

    @test !__differential_fast_zero_inspection(
        consequence + 1,
        data;
        minimum_terms=1,
    )

    @test is_zero(
        __differential_reduce_pending(
            consequence,
            state,
        ),
    )
end

@testset "Pending inequation contradiction is pruned before equations" begin
    R, u =
        differential_polynomial_ring(
            QQ,
            :u,
            1,
        )

    du = apply_action(u, 1)
    d2u = apply_action(u, [2])

    state =
        __DifferentialThomasState(
            DifferentialSystem(
                R,
                elem_type(R)[],
                elem_type(R)[],
            ),
        )

    __differential_insert_equation!(
        state,
        u,
    )

    # Keep an equation pending to ensure that the public selection strategy is
    # still equations-first.  Nevertheless du != 0 is already impossible
    # modulo the processed equation u = 0 and can safely kill the branch.
    push!(state.Q_equations, d2u + 1)
    push!(state.Q_inequations, du)

    @test __differential_pending_inequation_contradiction(state)

    selected = __differential_select_pending!(state)
    @test selected.equation
end

@testset "Pending nonzero inequation precheck reuses normal-form cache" begin
    R, u =
        differential_polynomial_ring(
            QQ,
            :u,
            1,
        )

    du = apply_action(u, 1)

    state =
        __DifferentialThomasState(
            DifferentialSystem(
                R,
                elem_type(R)[],
                elem_type(R)[],
            ),
        )

    __differential_insert_equation!(state, u)
    push!(state.Q_inequations, du + 1)

    @test !__differential_pending_inequation_contradiction(state)
    cache_size = length(state.normal_form_cache)
    @test cache_size > 0

    @test !__differential_pending_inequation_contradiction(state)
    @test length(state.normal_form_cache) == cache_size
end

@testset "Safe contradiction precheck is incremental" begin
    R, u =
        differential_polynomial_ring(
            QQ,
            :u,
            1,
        )

    P = elem_type(R)

    state =
        __differential_state(
            DifferentialSystem(
                R,
                P[u + 1],
                P[],
            ),
        )

    # First certification sees the complete initial branch.
    @test !__differential_state_has_safe_contradiction(state)
    @test u + 1 in state.safe_prechecked_equations
    @test length(state.safe_precheck_reducers) == 1

    # No new unit-linear reducer appears.  The added nonlinear consequence is
    # the only constraint that needs a new safe check.
    consequence = (u + 1)^2
    push!(state.Q_equations, consequence)

    @test !__differential_state_has_safe_contradiction(state)
    @test consequence in state.safe_prechecked_equations
    @test length(state.safe_precheck_reducers) == 1

    # A genuinely new unit-linear reducer must invalidate the incremental
    # shortcut and rescan the old equations.  u = 0 contradicts u + 1 = 0.
    push!(state.Q_equations, u)

    @test __differential_state_has_safe_contradiction(state)
end

@testset "Safe contradiction checks retain unchecked equations and full Janet masks" begin
    R, (u, v) = differential_polynomial_ring(QQ, [:u, :v], 1)
    P = elem_type(R)

    # Skipping equations must not hide their contradiction in a later full check,
    # either after an early return or after preparing and using the reducers.
    for I in (P[], P[one(R)], P[v])
        state = __differential_state(DifferentialSystem(R, P[u, u + 1], I))
        @test !__differential_state_has_safe_contradiction(state; scan_equations=false)
        @test isempty(state.safe_prechecked_equations)
        @test __differential_state_has_safe_contradiction(state; scan_equations=true)
    end
    constant_equation = __differential_state(DifferentialSystem(R, P[one(R)], P[]))
    @test __differential_state_has_safe_contradiction(constant_equation; scan_equations=false)
    zero_guard = __differential_state(DifferentialSystem(R, P[], P[zero(R)]))
    @test __differential_state_has_safe_contradiction(zero_guard; scan_equations=false)

    # The nonlinear equation still affects the Janet directions of the linear one.
    E = P[u, u[1]^2 + 1, v]
    sources = [1, 3]
    reducers = E[sources]
    full_masks = __action_multiplicative_masks(__action_leader_vectors(E; R))
    expected = __action_janet_data(reducers, full_masks[sources]; R)
    data = __differential_safe_substitution_data(reducers, leader.(E), sources)
    @test data.shifts == expected.shifts
    @test data.by_symbol == expected.by_symbol
    @test data.multiplicative_variables == expected.multiplicative_variables
    @test __differential_reduce_unit_linear(u[1], data; janet_head=true) == u[1]
    @test iszero(__differential_reduce_unit_linear(u[1], data))

    unrestricted = __differential_safe_substitution_data(reducers, leader.(reducers), nothing)
    @test unrestricted.multiplicative_variables === nothing
    @test unrestricted.shifts == expected.shifts
    @test unrestricted.by_symbol == expected.by_symbol
    @test iszero(__differential_reduce_unit_linear(u[1] + v, unrestricted))
end

@testset "Driver safe precheck may skip equation scan" begin
    R, u =
        differential_polynomial_ring(
            QQ,
            :u,
            1,
        )

    du = apply_action(u, 1)

    # The driver-only fast path must still detect the important pending
    # inequation contradiction without scanning all equations.
    source =
        DifferentialSystem(
            R,
            [u],
            [du],
        )

    state = __DifferentialThomasState(source)

    @test __differential_state_has_safe_contradiction(
        state;
        scan_equations=false,
    )
end


@testset "Cheap pending filter and compact optional-cache signatures" begin
    R, (u, v, w) = differential_polynomial_ring(QQ, [:u, :v, :w], 2; partition_name=:pot)
    sources = [__jet_data(u[1, 0])]
    @test __differential_pending_may_simplify(u[2, 1]+v, sources, BitSet())
    @test !__differential_pending_may_simplify(u[0, 2]+v, sources, BitSet())
    @test !__differential_pending_may_simplify(v^2+w, sources, BitSet())
    nonzero = BitSet([var_index(Oscar.data(v))])
    @test __differential_pending_may_simplify(v^2+w, sources, nonzero)
    @test __differential_constraint_fingerprint([u, v]) == __differential_constraint_fingerprint([v, u])
    @test __differential_constraint_fingerprint([u, v]) != __differential_constraint_fingerprint([u, w])
    target = sum(v^j for j in 1:40)
    state = __differential_state(DifferentialSystem([u[1, 0], target]))
    @test !__differential_simplify_large_pending!(state)
    checked = copy(state.pending_short_checked)
    @test !isempty(checked)
    @test !__differential_simplify_large_pending!(state)
    @test state.pending_short_checked == checked
    @test state.Q_equations == [u[1, 0], target]
    @test state.janet_data === nothing
end


@testset "Repeated shifts share a detached polynomial snapshot" begin
    R, u = differential_polynomial_ring(QQ, :u, 2)
    equation = u[1, 0]+u^2
    original = deepcopy(equation)
    reductions = [(equation, [1, 0]), (equation, [0, 1])]
    key, proof = __passivity_certificate_snapshot(equation, reductions)
    @test key === proof.reductions[1][1] === proof.reductions[2][1]
    @test key !== equation
    reductions[1][2][1] = 9
    zero!(equation)
    @test key == original
    @test proof.reductions[1][2] == [1, 0]
    @test proof.reductions[2][2] == [0, 1]
end
