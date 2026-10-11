using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Passivity reads system data without exposing it" begin
    R, (u, a) = differential_polynomial_ring(QQ, [:u, :a], 1; index_ordering_name=:deglex)
    E = [a, a*u^2, apply_action(a*u^2, 1)+1]
    system = DifferentialSystem(E)
    key_hash = hash(system)
    @test !is_passive(system)
    for result in (is_passive(system; missing=true),
                   is_passive(system, multiplicative_variables(system); missing=true))
        @test result == (false, -one(R), 2, 1)
        zero!(result[2])
        @test system.equations == E
        @test hash(system) == key_hash
    end
end

@testset "OSCAR dev coefficient actions" begin
    K, t = rational_function_field(QQ, "t")
    A = parent(numerator(t))
    phi = hom(A, K, t + 1)
    sigma(c) = phi(numerator(c)) // phi(denominator(c))
    R, x = difference_polynomial_ring(K, :x, [action_shift(map_from_func(K, K, sigma))])
    p = x/t^2
    @test apply_action(p, 1) == x[1]/(t + 1)^2
    @test apply_action(p, [2]) == x[2]/(t + 2)^2
    @test apply_action(R(t), 1) == R(t + 1)
    @test p == x/t^2
    @test apply_action(p, [0]) == p
    @test_throws ArgumentError apply_action(p, [-1])
    @test_throws ArgumentError apply_action(p, 2)
    @test_throws ArgumentError apply_action(p, [1, 0])
    D, u = differential_polynomial_ring(K, :u, [action_derivation(map_from_func(K, K, derivative))])
    @test apply_action(t*u^2, 1) == u^2 + 2*t*u*u[1]
    @test apply_action(D(t), 1) == D(1)
    @test apply_action(D(t), [2]) == D(0)
    @test normal_form(apply_action(u, 1), [u^2-t]).nf == D(1)
    @test normal_form(apply_action(u, 1), [u^2-t]).multiplier == 2*u
    S, y = difference_polynomial_ring(K, :y, 1)
    @test apply_action(t*y, 1) == Oscar.apply_action(t*y, 1)
    @test apply_action(t*y, 1) == t*y[1]
    @test action_map(S, 1)(t) == t
    @test domain(action_map(S, 1)) === K
    @test codomain(action_map(S, 1)) === K
    @test_throws ArgumentError normal_form(x, [y])
    @test_throws ArgumentError Oscar.__algebraic_conversion_data_from_polys(R, [x, y])
    Qz, z = polynomial_ring(QQ, :z)
    L, a = number_field(z^2-2, "a")
    T, w = difference_polynomial_ring(L, :w, [action_shift(hom(L, L, -a))])
    @test apply_action(a*w, [2]) == a*w[2]
    @test apply_action(a*w, 1) == -a*w[1]
end

@testset "Normal form certificates" begin
    R, y = difference_polynomial_ring(QQ, :y, 1)
    F = [y[1]-y]
    p = y[3]^2-y^2
    q, certificate, remainder = normal_form_terms(p, F)
    @test (nf=remainder, multiplier=q) == normal_form(p, F)
    @test q*p-remainder == sum(a*apply_action(f, shift) for (a, shift, f) in certificate; init=R(0))
    @test iszero(remainder)
    S, z = difference_polynomial_ring(QQ, :z, 1)
    @test_throws ArgumentError __normal_form_scale_terms!([(y, [1], z)], y)
    K, t = rational_function_field(QQ, "t")
    A = parent(numerator(t)); phi = hom(A, K, t+1)
    sigma(c) = phi(numerator(c)) // phi(denominator(c))
    D, x = difference_polynomial_ring(K, :x, [action_shift(map_from_func(K, K, sigma))])
    F = [x[1]-t*x]
    p = apply_action(F[1], [2])
    q, certificate, remainder = normal_form_terms(p, F)
    @test (nf=remainder, multiplier=q) == normal_form(p, F)
    @test q*p-remainder == sum(a*apply_action(f, shift) for (a, shift, f) in certificate; init=D(0))
    @test iszero(remainder)
end

@testset "Recorded differential normal forms" begin
    K, t = rational_function_field(QQ, "t")
    R, (v, u) = differential_polynomial_ring(
        K, [:v, :u], [action_derivation(map_from_func(K, K, derivative))],
    )
    f = u^2 - t
    q, terms, r = normal_form_terms(u[1], [f])
    @test (nf=r, multiplier=q) == normal_form(u[1], [f])
    @test q == 2*u
    @test r == one(R)
    @test q*u[1] - r == sum(c*apply_action(g, alpha) for (c, alpha, g) in terms; init=zero(R))
    @test normal_form_terms(u[1], DifferentialSystem([f])) == (q, terms, r)

    equations = [u[1] - t*u]
    p = u[2] - (t^2 + 1)*u
    q, terms, r = normal_form_terms(p, equations)
    @test (nf=r, multiplier=q) == normal_form(p, equations)
    @test iszero(r)
    @test q*p - r == sum(c*apply_action(g, alpha) for (c, alpha, g) in terms; init=zero(R))

    # Record all coefficient reductions, including their common multiplier.
    p = u[1]*v[2]^2 + (u[1] + u)*v[2]
    q, terms, r = normal_form_terms(p, [u*u[1] - 1])
    @test (nf=r, multiplier=q) == normal_form(p, [u*u[1] - 1])
    @test q*p - r == sum(c*apply_action(g, alpha) for (c, alpha, g) in terms; init=zero(R))
    @test normal_form(r, [u*u[1] - 1]).nf == r

    q, terms, r = normal_form_terms(p, elem_type(R)[])
    @test isone(q) && isempty(terms) && r == p
    S, z = differential_polynomial_ring(K, :z, 1)
    @test_throws ArgumentError normal_form_terms(u, [z])
end

@testset "Recorded reductions share the normal-form algorithm" begin
    R, (v, u, a) = differential_polynomial_ring(QQ, [:v, :u, :a], 1)
    P = elem_type(R)
    cases = [
        (R(7), P[]),
        (v + 1, P[]),
        (R(0), [u[1] - 1]),
        (u[1] - 1, [2*u[1] - 2]),
        (apply_action(u^2 - a, 1), [u^2 - a]),
        (u[1]^2 + u, [a*u[1] - 1]),
        (u^3 + u, [a*u^2 + u + a]),
        (u[1]*(v[2]^2 + v[2] + 1), [a^2 - 1, a*u[1] - 1]),
        (u[1]*(v[2]^2 + v[2]), [u[1]]),
    ]
    for (p, equations) in cases
        before = deepcopy(equations)
        q, terms, r = normal_form_terms(p, equations)
        @test (nf=r, multiplier=q) == normal_form(p, equations)
        @test q*p - r == sum(
            c*apply_action(f, alpha) for (c, alpha, f) in terms; init=zero(R),
        )
        @test equations == before
    end

    # Repeated coefficients reuse complete identities. After reconstruction,
    # reduce the products of their multipliers by the lower equation a^2=1.
    p = u[1]*(v[2]^2 + v[2] + 1)
    q, terms, r = normal_form_terms(p, [a^2 - 1, a*u[1] - 1])
    @test q == a^3
    @test r == v[2]^2 + v[2] + 1

    # A coefficient with zero normal form can still contribute to the identity.
    p = u[1]*(v[2]^2 + v[2])
    q, terms, r = normal_form_terms(p, [u[1]])
    @test iszero(r) && isone(q)
    @test !isempty(terms)
end

@testset "Passivity agrees with public normal forms" begin
    for partition_name in (:top, :pot)
        R, (u, a) = differential_polynomial_ring(
            QQ, [:u, :a], 1; partition_name, index_ordering_name=:deglex,
        )
        f = a*u^2
        prolongation = apply_action(f, 1)
        equations = [a, f, prolongation + 1]
        masks = multiplicative_variables(equations)
        @test is_janet_complete(equations)
        @test masks == [BitVector([1]), BitVector([0]), BitVector([1])]

        # Exact cancellation leaves -1. An unnecessary pseudo-step would
        # multiply it by 2*a*u and then erase it using the equation a=0.
        @test normal_form(prolongation, equations) == (nf=-one(R), multiplier=one(R))
        q, terms, r = normal_form_terms(prolongation, equations)
        @test (nf=r, multiplier=q) == (nf=-one(R), multiplier=one(R))
        @test q*prolongation - r == sum(
            c*apply_action(f, alpha) for (c, alpha, f) in terms; init=zero(R),
        )
        expected = (false, -one(R), 2, 1)
        @test !is_passive(equations)
        @test !is_passive(equations, masks)
        @test is_passive(equations; missing=true) == expected
        @test is_passive(equations, masks; missing=true) == expected

        system = DifferentialSystem(equations)
        @test !is_passive(system)
        @test is_passive(system; missing=true) == expected
        @test is_passive(system, masks; missing=true) == expected

        # Keep zero prolongations passive and do not reuse results from a
        # different equation basis in a later call.
        compatible = [a, f, prolongation]
        @test iszero(normal_form(prolongation, compatible).nf)
        @test is_passive(compatible)
        @test is_passive(compatible; missing=true) === true
        @test !is_passive(equations)
    end
end

@testset "Persistent normal-form certificate across basis extension" begin
    R, (v, u) = differential_polynomial_ring(
        QQ,
        [:v, :u],
        1,
    )

    du = apply_action(u, 1)
    ddu = apply_action(u, [2])
    dv = apply_action(v, 1)

    f = du - u
    g = dv - v
    p = ddu - u

    data_small = __action_janet_data([f])
    data_large = __action_janet_data([f, g])

    local_cache = Dict{typeof(p),Tuple{typeof(p),typeof(p)}}()
    certificate_cache =
        Dict{
            typeof(p),
            __NormalFormCertificate{typeof(p)},
        }()

    first_nf = __normal_form_cached(
        p,
        data_small,
        local_cache,
        certificate_cache,
    )

    @test is_zero(first_nf.nf)
    @test haskey(certificate_cache, p)

    empty!(local_cache)

    certificate = __normal_form_certificate_lookup(
        p,
        data_large,
        certificate_cache,
    )

    @test certificate !== nothing
    @test is_zero(certificate.nf)

    second_nf = __normal_form_cached(
        p,
        data_large,
        local_cache,
        certificate_cache,
    )

    @test is_zero(second_nf.nf)
    @test haskey(local_cache, p)
end

@testset "Persistent normal-form certificate rejects removed reducers" begin
    R, (v, u) = differential_polynomial_ring(
        QQ,
        [:v, :u],
        1,
    )

    du = apply_action(u, 1)
    ddu = apply_action(u, [2])
    dv = apply_action(v, 1)

    f = du - u
    g = dv - v
    p = ddu - u

    data_with_f = __action_janet_data([f, g])
    data_without_f = __action_janet_data([g])

    local_cache = Dict{typeof(p),Tuple{typeof(p),typeof(p)}}()
    certificate_cache =
        Dict{
            typeof(p),
            __NormalFormCertificate{typeof(p)},
        }()

    __normal_form_cached(
        p,
        data_with_f,
        local_cache,
        certificate_cache,
    )

    @test __normal_form_certificate_lookup(
        p,
        data_without_f,
        certificate_cache,
    ) === nothing
end

@testset "Irreducible coefficient normal form avoids recursive reduction" begin
    R, (v, u) = differential_polynomial_ring(
        QQ,
        [:v, :u],
        1,
    )

    du = apply_action(u, 1)
    reducer = du - u
    data = __action_janet_data([reducer])

    coefficient = v + 1
    P = typeof(coefficient)
    cache = Dict{P,Tuple{P,P}}()

    result = __normal_form_coefficient_cached(
        coefficient,
        data,
        cache,
        nothing,
    )

    @test result.nf == coefficient
    @test isone(result.multiplier)
    @test haskey(cache, coefficient)
    @test cache[coefficient] == (coefficient, R(1))
end

@testset "Reducible coefficient fast path agrees with full normal form" begin
    R, u = differential_polynomial_ring(
        QQ,
        :u,
        1,
    )

    du = apply_action(u, 1)
    ddu = apply_action(u, [2])
    reducer = du - u
    coefficient = ddu - u
    data = __action_janet_data([reducer])

    P = typeof(coefficient)
    coefficient_cache = Dict{P,Tuple{P,P}}()
    full_cache = Dict{P,Tuple{P,P}}()

    fast = __normal_form_coefficient_cached(
        coefficient,
        data,
        coefficient_cache,
        nothing,
    )

    full = __normal_form_cached(
        coefficient,
        data,
        full_cache,
        nothing,
    )

    @test fast.nf == full.nf
    @test fast.multiplier == full.multiplier
end

@testset "Univariate coefficient scans in action rings" begin
    K, t = rational_function_field(QQ, "t")
    for make_ring in (differential_polynomial_ring, difference_polynomial_ring)
        R, (u, v) = make_ring(K, [:u, :v], 2; index_ordering_name=:degrevlex)
        uy = u[0, 1]
        vx = v[1, 0]
        p = (t + 1)/t*uy^3*vx + (vx + t)*uy + v/t^2
        original = deepcopy(p)
        equations = [uy^2 - vx]
        old_zero = zero(R)

        # Grow the universal parent after p was built. Raw variable positions
        # must not be confused with their changing positions in the ranking.
        unused = u[5, 4]
        set_ranking!(R; partition_name=:pot, index_ordering_name=:deglex)

        for variable in (uy, vx, v, unused)
            coefficients = __action_univariate_coefficients_by_terms(p, variable)
            @test coefficients == univariate_coefficients(p, variable)
            @test sum(c*variable^(i-1) for (i, c) in enumerate(coefficients)) == p
            @test __action_univariate_initial(p, variable) ==
                univariate_leading_coefficient(p, variable)
        end
        @test p == original
        @test isempty(__action_univariate_coefficients_by_terms(zero(R), uy))
        @test isempty(__action_univariate_coefficients_by_terms(old_zero, unused))
        @test __action_univariate_coefficients_by_terms(R(t), uy) == [R(t)]
        @test __action_univariate_initial(R(t), uy) == R(t)
        @test_throws ArgumentError __action_univariate_initial(zero(R), uy)
        @test_throws ArgumentError __action_univariate_initial(old_zero, unused)

        # A degree-one input must not use an unshifted degree-two reducer,
        # even after the underlying universal parent and the ranking change.
        data = __action_janet_data(equations)
        @test __action_reduction_candidate_detailed(uy, uy, data) === false
        candidate = __action_reduction_candidate_detailed(p, uy, data)
        @test candidate !== false
        if candidate !== false
            @test candidate.prolongation == only(equations)
            @test __action_reduction_univariate_data(candidate, data) == (2, R(1))
        end

        S, other = make_ring(K, :other, 2)
        @test_throws ArgumentError __action_univariate_coefficients_by_terms(p, other)
    end
end

@testset "Coefficient extraction preserves normal forms and multipliers" begin
    R, (w, u, v) = differential_polynomial_ring(
        QQ, [:w, :u, :v], 1;
        partition_name=:pot, index_ordering_name=:degrevlex,
    )
    P = elem_type(R)
    examples = [
        # Exact cancellation by a monic leader-linear equation.
        (u[1]^3 + (v + 1)*u[1], [u[1] - v], v^3 + v^2 + v, R(1)),
        # A nonconstant initial still requires the original pseudo multiplier.
        (u[1]^2, [v*u[1] - 1], R(1), v^2),
        # Recurse into coefficients of the irreducible highest variable w.
        ((u[2] - v)*w^2 + (v[1] - v)*w + u[1]^2,
         [u[1] - v, v[1] - v], v^2, R(1)),
        # Retain monic nonlinear reduction semantics as well.
        (u^3 + u, [u^2 - v], u*v + u, R(1)),
    ]

    for (p, equations, expected_nf, expected_multiplier) in examples
        original = deepcopy(p)
        original_equations = deepcopy(equations)
        data = __action_janet_data(equations)
        cache = Dict{P,Tuple{P,P}}()
        ordinary = __normal_form_cached(p, data, cache, nothing)
        @test ordinary == (nf=expected_nf, multiplier=expected_multiplier)
        @test __normal_form_cached(p, data, cache, nothing) == ordinary
        @test p == original
        @test equations == original_equations
    end

    K, t = rational_function_field(QQ, "t")
    D, y = differential_polynomial_ring(
        K, :y, [action_derivation(map_from_func(K, K, derivative))],
    )
    @test normal_form(y[2], [y[1] - t*y]) == (nf=(t^2 + 1)*y, multiplier=D(1))
    @test normal_form(y[1], [y^2 - t]) == (nf=D(1), multiplier=2*y)
end

@testset "Zero certificates respect changing Janet cones" begin
    R, u = differential_polynomial_ring(QQ, :u, 2)
    P = elem_type(R)
    uy, uxy, uxx = u[0, 1], u[1, 1], u[2, 0]

    certificates = Dict{P,__NormalFormCertificate{P}}()
    cache = Dict{P,Tuple{P,P}}()
    data = __action_janet_data([uy])
    first = __normal_form_cached(uxy, data, cache, certificates)
    @test iszero(first.nf)

    # Inserting u_xx makes x nonmultiplicative for u_y, so u_xy
    # becomes a missing Janet leader even though it stays a consequence.
    extended = __action_janet_data([uy, uxx])
    @test __normal_form_certificate_lookup(uxy, extended, certificates) === nothing
    empty!(cache)
    second = __normal_form_cached(uxy, extended, cache, certificates)
    @test second == (nf=uxy, multiplier=one(R))
    @test !__is_janet_complete(extended)
    @test __is_janet_complete(__action_janet_data([uy, uxy, uxx]))

    # Explicitly supplied masks must also participate in cache validity,
    # even when the equation list itself is unchanged.
    restricted = __action_janet_data([uy], [BitVector([false, true])])
    @test __normal_form_certificate_lookup(uxy, restricted, certificates) === nothing

    # A certificate owns its masks rather than aliasing mutable Janet data.
    data.multiplicative_variables[1][1] = false
    @test certificates[uxy].multiplicative_variables[1][1]
end

@testset "Native coefficient actions survive system conversion" begin
    K, t = rational_function_field(QQ, "t")
    A = parent(numerator(t))
    forward = hom(A, K, t+1)
    backward = hom(A, K, t-1)
    sigma(c) = forward(numerator(c)) // forward(denominator(c))
    inverse(c) = backward(numerator(c)) // backward(denominator(c))
    R, y = difference_polynomial_ring(K, :y, [action_shift(map_from_func(K, K, sigma))])
    system = DifferenceSystem([y[1]-t*y])
    @test normal_form(y[2], system).nf == t*(t+1)*y
    D, u = differential_polynomial_ring(K, :u, [action_derivation(map_from_func(K, K, derivative))])
    differential = DifferentialSystem([u[1]-t*u])
    @test normal_form(u[2], differential).nf == (1+t^2)*u
    @test is_simple(differential)
end
