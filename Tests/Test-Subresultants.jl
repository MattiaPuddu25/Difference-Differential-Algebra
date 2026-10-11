using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Regular subresultants preserve resultant signs and coefficient factors" begin
    K, t = rational_function_field(QQ, :t)
    for field in (QQ, GF(7), K)
        R, (x, a, b) = polynomial_ring(field, [:x, :a, :b])
        for (f, g) in ((x^2-a, x-b), (a*x^3+b, b*x^2+a),
                       (a*x^4+b, a*x^2+b), (a*x^3, b*x^2),
                       (x^5+a*x+b, x^3+a), (a*x^2+b*x+1, b*x^2+a*x+1))
            original = deepcopy((f, g))
            for (left, right) in ((f, g), (g, f))
                data = __algebraic_subresultant_data(left, right, 1)
                @test __algebraic_subresultant_resultant(data) == resultant(left, right, 1)
                for j in keys(data.regular), input in (1, 2)
                    certificate = __algebraic_subresultant_cofactor(data, j; polynomial=input)
                    source = input == 1 ? left : right
                    @test certificate.multiplier*source ==
                          certificate.quotient*certificate.divisor + certificate.remainder
                end
            end
            @test (f, g) == original
        end
    end
end

@testset "Truncated sequences distinguish uncomputed coefficients from zero" begin
    R, (x, a) = polynomial_ring(QQ, [:x, :a])
    f, g = x^3+x+1, 3*x^2+1
    full = __algebraic_subresultant_data(f, g, 1)
    truncated = __algebraic_subresultant_data(f, g, 1; minimum_degree=1)
    @test __algebraic_subresultant_resultant(full) == R(31)
    @test truncated.computed_minimum == 1
    @test __algebraic_principal_subresultant(truncated, 1) ==
          __algebraic_principal_subresultant(full, 1)
    # The terminal input degree is a valid starting degree when lower PSCs vanish.
    terminal = __subresultant_gcd_step(x^2, x, 1, elem_type(R)[], elem_type(R)[]; first_index=1)
    @test terminal.kind === :gcd
    @test terminal.case.degree == 1
    # An actual zero tail remains distinguishable from a truncated, unknown tail.
    common = __algebraic_subresultant_data(x^3, x^2, 1; minimum_degree=1)
    @test common.computed_minimum == 0
    @test iszero(__algebraic_subresultant_resultant(common))
end

@testset "Conditional GCDs cover degree drops and exceptional parameters" begin
    for field in (QQ, GF(7))
        R, (x, a, b) = polynomial_ring(field, [:x, :a, :b])
        U, z = polynomial_ring(field, :z)
        for (f, g) in ((x^2-a, x-b), (a*x^3+b*x+1, b*x^2+a),
                       (a*x^4+b, a*x^2+b), (a*x^3, b*x^2),
                       (a*x+b, a*x+b), (R(0), a*x+b), (a, b))
            snapshot = deepcopy((f, g))
            cases = __algebraic_subresultant_gcd_cases(f, g, 1)
            for av in -1:2, bv in -1:2
                point = field.([0, av, bv])
                active = filter(c -> all(e -> iszero(evaluate(e, point)), c.equations) &&
                                     all(q -> !iszero(evaluate(q, point)), c.inequations), cases)
                @test length(active) == 1
                length(active) == 1 || continue
                c = only(active)
                F = U([evaluate(h, point) for h in AbstractAlgebra.coefficients(f, 1)])
                G = U([evaluate(h, point) for h in AbstractAlgebra.coefficients(g, 1)])
                D = U([evaluate(h, point) for h in AbstractAlgebra.coefficients(c.gcd, 1)])
                expected = gcd(F, G)
                @test degree(D) == degree(expected) == c.degree
                if !iszero(expected)
                    @test divexact(D, leading_coefficient(D)) == divexact(expected, leading_coefficient(expected))
                    for input in (1, 2)
                        cert = __algebraic_subresultant_cofactor(c; polynomial=input)
                        @test !iszero(evaluate(cert.multiplier, point))
                        @test all(h -> iszero(evaluate(h, point)), AbstractAlgebra.coefficients(cert.remainder, 1))
                    end
                end
            end
            @test (f, g) == snapshot
        end
    end
end

@testset "Cofactors can return only an independent quotient" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    P = elem_type(R)
    f, g = x^2-a, x-b
    data = __algebraic_subresultant_data(f, g, 1)
    for j in keys(data.regular), input in (1, 2)
        certificate = __algebraic_subresultant_cofactor(data, j; polynomial=input)
        q = __algebraic_subresultant_cofactor(data, j; polynomial=input, quotient_only=true)
        @test q == certificate.quotient
        zero!(q)
        @test __algebraic_subresultant_cofactor(data, j; polynomial=input).quotient == certificate.quotient
    end
    for (E, I) in (([a-b^2], P[]), (P[], [b^2-a]))
        c = __subresultant_gcd_step(f, g, 1, E, I).case
        for input in (1, 2)
            certificate = __algebraic_subresultant_cofactor(c; polynomial=input)
            q = __algebraic_subresultant_cofactor(c; polynomial=input, quotient_only=true)
            @test q == certificate.quotient
            source = input == 1 ? c.left : c.right
            @test certificate.multiplier*source == certificate.quotient*c.gcd + certificate.remainder
            snapshot = deepcopy(source)
            zero!(q)
            @test source == snapshot
        end
    end
end

@testset "Prepared lower reductions retain safe multipliers" begin
    R, (x, a, b, c) = polynomial_ring(QQ, [:x, :a, :b, :c])
    P = elem_type(R)
    E, I = [a*x+1, a-b, b-c], [c]
    snapshot = deepcopy((E, I))
    cache = Ref{Union{Nothing,Vector{__AlgebraicReductionData{P}}}}(nothing)
    @test __subresultant_reduce_condition(x, E, I; reduction_data=cache) == -one(R)
    prepared = cache[]
    @test prepared !== nothing
    @test iszero(__subresultant_reduce_condition(a-c, E, I; reduction_data=cache))
    @test cache[] === prepared
    @test (E, I) == snapshot
    @test __subresultant_reduce_condition(x, [a*x], P[]) == x
    @test iszero(__subresultant_reduce_condition(x, [a*x], [a]))
    @test __subresultant_guard_nonzero(a*b, [a, b]) === true
    @test __subresultant_guard_nonzero(a, [zero(R)]) === nothing
end

@testset "Conditional square-free parts and LCMs specialize correctly" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    U, z = polynomial_ring(QQ, :z)
    for p in (x^3+a*x+b, a*x^3+b*x^2, (x-a)^2*(x-b), R(0), a)
        cases = __algebraic_subresultant_squarefree_cases(p, 1)
        for av in -1:1, bv in -1:1
            point = QQ.([0, av, bv])
            active = filter(c -> all(e -> iszero(evaluate(e, point)), c.equations) &&
                                 all(q -> !iszero(evaluate(q, point)), c.inequations), cases)
            @test length(active) == 1
            length(active) == 1 || continue
            actual = U([evaluate(h, point) for h in AbstractAlgebra.coefficients(only(active).polynomial, 1)])
            source = U([evaluate(h, point) for h in AbstractAlgebra.coefficients(p, 1)])
            if iszero(source)
                @test iszero(actual)
            else
                expected = divexact(source, gcd(source, derivative(source)))
                @test divexact(actual, leading_coefficient(actual)) == divexact(expected, leading_coefficient(expected))
            end
        end
    end
    for (f, g) in (((x-a)^2, (x-b)^3), (a*x+b, b*x+a), (R(0), x+a))
        cases = __algebraic_subresultant_lcm_cases(f, g, 1)
        for av in -1:1, bv in -1:1
            point = QQ.([0, av, bv])
            active = filter(c -> all(e -> iszero(evaluate(e, point)), c.equations) &&
                                 all(q -> !iszero(evaluate(q, point)), c.inequations), cases)
            @test length(active) == 1
            length(active) == 1 || continue
            actual = U([evaluate(h, point) for h in AbstractAlgebra.coefficients(only(active).polynomial, 1)])
            F = U([evaluate(h, point) for h in AbstractAlgebra.coefficients(f, 1)])
            G = U([evaluate(h, point) for h in AbstractAlgebra.coefficients(g, 1)])
            if iszero(F) || iszero(G)
                @test iszero(actual)
            else
                expected = lcm(F, G)
                @test divexact(actual, leading_coefficient(actual)) == divexact(expected, leading_coefficient(expected))
            end
        end
    end
end

@testset "Splitting retains all subresultant conditions" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    P = elem_type(R)
    cases = [([a*x^2+b, x^2-a], P[], :equations),
             (P[], [x-a, x-b], :inequations),
             ([x^2-a], [x-b], :mixed), ([x^2*(x-1)], [x], :mixed),
             ([a*x], [x], :mixed), ([a*x+1], [a*x], :mixed)]
    for (E, I, kind) in cases
        original = deepcopy((E, I))
        branches = kind === :equations ? __algebraic_same_leader_equations(E, I, 1, 2) :
                   kind === :inequations ? __algebraic_same_leader_inequations(E, I, 1, 2) :
                                           __algebraic_same_leader_equation_inequation(E, I, 1, 1)
        @test splitting_covers_system(AlgebraicSystem(R, E, I), branches)
        @test all(!algebraic_branch_intersection(AlgebraicSystem(R, branches[i]...),
                                               AlgebraicSystem(R, branches[j]...))
                  for i in eachindex(branches) for j in 1:i-1)
        @test (E, I) == original
    end
    p = x^3+a*x+b
    disc = __algebraic_discriminant(p, 1)
    @test disc == -4*a^3-27*b^2
    issue = (kind=:equation, index=1, discriminant=disc, E_partial=[p])
    __with_algebraic_cache() do
        branches = __algebraic_discriminant_split([p], P[], issue; factorize=false)
        @test all(key -> first(key) !== :radical_lex, keys(task_local_storage()[:__algebraic_splitting]))
        @test splitting_covers_system(AlgebraicSystem([p]), branches)
    end
end

@testset "Differential splitting uses conditional square-free parts" begin
    R, (u, a, b) = differential_polynomial_ring(QQ, [:u, :a, :b], 1; partition_name=:top)
    system = DifferentialSystem([u^3+a*u+b, a[1], b[1]])
    branches = thomas_decomposition(system; factorization=false)
    @test !isempty(branches)
    @test all(is_simple, branches)
    # Check each returned algebraic branch against the original equations;
    # general differential covering cannot be decided by finite conversion alone.
    @test all(D -> all(f -> iszero(normal_form(f, D).nf), system.equations), branches)
end

@testset "Lower conditions require certified pseudo multipliers" begin
    R, (x, a, b, c) = polynomial_ring(QQ, [:x, :a, :b, :c])
    P = elem_type(R)
    E, I = [a-b, b-c], P[]
    original = deepcopy((E, I))
    @test __subresultant_condition_status(a-c, E, I) === false
    @test __subresultant_condition_status(a-c+1, E, I) === true
    @test __subresultant_condition_status(a+c, E, I) === nothing
    @test __subresultant_condition_status(a*b, P[], [a, b]) === true
    @test (E, I) == original
    # Without a != 0, a*x = 0 does not imply x = 0.
    @test __subresultant_reduce_condition(x, [a*x], P[]) == x
    @test iszero(__subresultant_reduce_condition(x, [a*x], [a]))
    @test __subresultant_reduce_condition(a*x+1, [a*x], P[]) == 1
    # In particular, the guard being simplified cannot certify its own multiplier.
    @test __compatible_zero_algebraic_system(zero(R), [a*x], [x])
    @test !__compatible_zero_algebraic_system(zero(R), [a*x], [a, x])
    @test !__compatible_zero_algebraic_system(zero(R), [a-b, b-c], [a-c])
    @test __compatible_zero_algebraic_system(zero(R), [x^2-a, a-b], [b])
end

@testset "One GCD step defers the first undecided zero condition" begin
    R, (x, a, b, c) = polynomial_ring(QQ, [:x, :a, :b, :c])
    P = elem_type(R)
    snapshot = (x^2-a, x-b)
    decision = __subresultant_gcd_step(snapshot..., 1, P[], P[])
    @test decision.kind === :split
    @test decision.case.degree == 0
    @test decision.condition == b^2-a
    decided = __subresultant_gcd_step(x^2-a, x-b, 1, [a-b^2], P[])
    @test decided.kind === :gcd
    @test decided.case.degree == 1
    @test decided.case.gcd == x-b
    initial = __subresultant_gcd_step(a*x^2+b, x+c, 1, P[], P[])
    @test initial.kind === :split
    @test initial.case === nothing
    @test initial.condition == a
    E, I = [(x-a)*(x-b)], [x-c]
    original = deepcopy((E, I))
    branches = __algebraic_same_leader_equation_inequation(E, I, 1, 1)
    @test length(branches) <= 2
    @test splitting_covers_system(AlgebraicSystem(E, I), branches)
    @test all(!algebraic_branch_intersection(AlgebraicSystem(R, branches[i]...),
                                           AlgebraicSystem(R, branches[j]...))
              for i in eachindex(branches) for j in 1:i-1)
    @test (E, I) == original
end

@testset "Combining guards does not split additional specialized GCDs" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    P = elem_type(R)
    branches = __algebraic_same_leader_inequations(P[], [x-a, x-b], 1, 2)
    @test length(branches) == 1
    E, I = only(branches)
    @test isempty(E)
    @test splitting_covers_system(AlgebraicSystem(R, P[], [x-a, x-b]), branches)
    # The same replacement remains valid on a=b, where the specialized GCD grows.
    for xv in -1:1, av in -1:1, bv in -1:1
        point = QQ.([xv, av, bv])
        @test all(q -> !iszero(evaluate(q, point)), I) == (xv != av && xv != bv)
    end
end
println("AA")
@testset "Guard-factor cost filters preserve polynomial and jet supports" begin
    A, algebraic_variables = polynomial_ring(QQ, [:x, :y, :z, :w])
    D, differential_variables = differential_polynomial_ring(QQ, [:x, :y, :z, :w], 1; partition_name=:top)
    for (R, variables) in ((A, algebraic_variables), (D, differential_variables))
        x, y, z, w = variables
        f = x^2*y
        guards = [x, y]
        snapshot = deepcopy((f, guards))
        @test __algebraic_strip_guard_factors(f, guards; max_variables=1) == f
        @test __algebraic_strip_guard_factors(f, guards; max_variables=2) == one(R)
        @test __algebraic_strip_guard_factors(f, guards) == one(R)
        @test (f, guards) == snapshot
        # After removing x, the pair (y, y*z) has only two active variables.
        @test __algebraic_strip_guard_factors(x*y, [x, y*z]; max_variables=2) == one(R)
        @test __algebraic_strip_guard_factors(x*y, [z]; max_variables=3) == x*y
        @test __algebraic_strip_guard_factors(x*y*z*w, [x]; max_variables=4) == y*z*w
        dense = (x+1)*sum(y^i for i in 0:40)
        @test __algebraic_strip_guard_factors(dense, [x+1]; max_variables=2) == dense
        @test __algebraic_strip_guard_factors(dense, [x+1]) == sum(y^i for i in 0:40)
    end
    x, y, z, w = differential_variables
    f = x*y
    # Track a new jet after constructing f: underlying degree vectors must
    # still use the same indices, independently of the differential ranking.
    later_jet = y[4]
    @test __algebraic_strip_guard_factors(f, [x, y*later_jet]; max_variables=2) == one(D)
    @test f == x*y
end

@testset "Truncated subresultants keep uncomputed degrees unknown for action polynomials" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    x = u[1]

    f = x^3 + x + 1
    g = 3*x^2 + 1

    # Use the same finite algebraic conversion as differential Thomas splitting.
    source = DifferentialSystem(R, [f, g], elem_type(R)[])
    conversion_data = __differential_algebraic_splitting_data(source)

    E, I = __algebraic_read_system(conversion_data.system)
    @test isempty(I)

    F, G = E

    # Check that these really are the algebraic images of the action polynomials.
    @test F == conversion_data.conversion(f)
    @test G == conversion_data.conversion(g)
    @test __algebraic_leader(F) == __algebraic_leader(G)

    idx = var_index(__algebraic_leader(F))

    # Stop as soon as degree 1 has been computed.
    truncated =
        __algebraic_subresultant_data(
            F,
            G,
            idx;
            minimum_degree=1
        )

    @test truncated.computed_minimum == 1
    @test haskey(truncated.regular, 1)
    @test !haskey(truncated.regular, 0)

    # Degree 1 is known and may be queried.
    s1_truncated = __algebraic_subresultant(truncated, 1)
    psc1_truncated = __algebraic_principal_subresultant(truncated, 1)

    # Degree 0 was not computed: internal callers must not query it on truncated data.

    # Compute the complete sequence independently.
    full =
        __algebraic_subresultant_data(
            F,
            G,
            idx;
            minimum_degree=0
        )

    @test full.computed_minimum == 0

    # Everything already computed by the truncated sequence agrees
    # with the complete one.
    @test s1_truncated == __algebraic_subresultant(full, 1)
    @test psc1_truncated == __algebraic_principal_subresultant(full, 1)

    # The missing degree-zero entry was genuinely nonzero:
    # Res(x^3 + x + 1, 3*x^2 + 1) = 31.
    @test __algebraic_subresultant_resultant(full) == parent(F)(31)
end
@testset "Cofactor-only division defers multiplier construction" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    data = __algebraic_subresultant_data(x^3+b, a*x+1, 1)
    quotient = __algebraic_subresultant_cofactor(data, 1; quotient_only=true)
    cached = data.cofactors[(1, 1)]
    @test cached.multiplier === nothing
    cert = __algebraic_subresultant_cofactor(data, 1)
    @test cached.multiplier !== nothing
    @test cert.quotient == quotient
    @test cert.multiplier*(x^3+b) == quotient*cert.divisor + cert.remainder
    U, t = polynomial_ring(R, :t)
    f, g = t^3+U(b), U(a)*t+1
    q, r, m = __subresultant_pseudo_certificate(f, g)
    q_only, r_only, omitted = __subresultant_pseudo_certificate(f, g; compute_multiplier=false)
    @test (q_only, r_only) == (q, r)
    @test omitted === nothing
    @test m*f == q*g+r
end

@testset "Prepared reducer data cache initials without changing the inputs" begin
    R, (x, a, b) = polynomial_ring(QQ, [:x, :a, :b])
    E = [a*x-b]
    snapshot = deepcopy(E)
    data = __algebraic_reduction_data(E)
    @test only(data).initial === nothing
    @test __subresultant_reduce_prepared(a*x+b, data, [a]) == 2*b
    h = only(data).initial
    @test h == a
    @test __subresultant_reduce_prepared(a*x-b, data, [a]) == 0
    @test only(data).initial === h
    @test E == snapshot
end

@testset "Cancel certified monomial factors without a polynomial GCD" begin
    A, variables = polynomial_ring(QQ, [:x, :a, :b])
    D, jets = differential_polynomial_ring(QQ, [:x, :a, :b], 1; partition_name=:pot)
    for (R, (x, a, b)) in ((A, variables), (D, jets))
        guards = [a*(b + 1)]
        p = a^7*b^3*(x^4 + x + b)
        snapshot = deepcopy((p, guards))
        nonzero = __algebraic_nonzero_monomial_variables(guards)
        @test __algebraic_strip_monomial_variables(p, nonzero) == b^3*(x^4 + x + b)
        @test __algebraic_strip_monomial_variables(p, BitSet()) === p
        @test __algebraic_strip_monomial_variables(zero(R), nonzero) == zero(R)
        @test (p, guards) == snapshot
        # b+1 != 0 does not allow removal of b.
        @test __algebraic_strip_monomial_variables(b^3, nonzero) == b^3
    end
    x, a, b = jets
    p = a^7*(x^4 + b)
    raw = AbstractAlgebra.data(Oscar.data(p))
    old_parent = parent(raw)
    later = b[8]
    nonzero = __algebraic_nonzero_monomial_variables([a*later])
    @test __algebraic_strip_monomial_variables(p, nonzero) == x^4 + b
    @test AbstractAlgebra.data(Oscar.data(p)) === raw
    @test parent(raw) === old_parent
    @test p == a^7*(x^4 + b)

    x, a, b = variables
    E, I = [a^7*(x^2 + b)], [a^3*(x + a), a*(b + 1)]
    original = deepcopy((E, I))
    branches = __algebraic_subresultant_pair_step(E, I, 1, 1, 1, :mixed)
    # Both leaders stay equal to x: continue with the already-cancelled pair,
    # including its generic and exceptional resultant conditions.
    expected = __algebraic_subresultant_pair_step(
        [x^2+b], [x+a, a*(b+1)], 1, 1, 1, :mixed)
    @test branches == expected
    @test all(branch -> a*(b+1) in last(branch), branches)
    @test splitting_covers_system(AlgebraicSystem(E, I), branches)
    @test (E, I) == original
    # The mixed inequation x+a must not cancel itself when no lower guard exists.
    variables_from_lower_guards = __algebraic_nonzero_monomial_variables(elem_type(A)[])
    @test __algebraic_strip_monomial_variables(x + a, variables_from_lower_guards) == x + a
end
