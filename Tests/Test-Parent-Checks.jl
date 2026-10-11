using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Reuse the checked parent with empty inputs" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached=false)
    S, (u, v) = polynomial_ring(QQ, [:x, :y]; cached=false)
    empty_polynomials = elem_type(R)[]
    @test __require_same_parent(empty_polynomials, (empty_polynomials,)) === nothing
    @test __require_same_parent(empty_polynomials, x, empty_polynomials) === R
    @test __require_same_parent(empty_polynomials, [x]) === R
    @test __require_same_parent(AlgebraicSystem(R)) === R
    @test_throws ArgumentError __require_same_parent(empty_polynomials, x, [u])
    @test isempty(multiplicative_variables(empty_polynomials))
    @test is_janet_complete(empty_polynomials)
    @test isempty(janet_completion(empty_polynomials))
    @test __algebraic_copy_system(empty_polynomials, empty_polynomials) ==
        (empty_polynomials, empty_polynomials)
    @test __algebraic_branch_intersection(empty_polynomials, empty_polynomials,
                                         empty_polynomials, empty_polynomials)
    for (constructor, cache) in (
        (difference_polynomial_ring, __action_janet_data),
        (differential_polynomial_ring, __action_janet_data),
    )
        D, z = constructor(QQ, :z, 1)
        empty_equations = elem_type(D)[]
        @test __require_same_parent(empty_equations) === nothing
        @test __require_same_parent(empty_equations, z) === D
        @test isempty(cache(empty_equations, BitVector[]).multiplicative_variables)
        @test normal_form(z + 1, empty_equations) == (nf=z + 1, multiplier=one(D))
        conversion = Oscar.__algebraic_conversion_data_from_polys(D, [z])
        @test isempty(conversion(empty_equations))
        converted = conversion([z])
        A = conversion.mpr
        @test converted == [only(gens(A))] && parent(only(converted)) === A
        empty_conversion = Oscar.__algebraic_conversion_data_from_polys(D, empty_equations)
        @test ngens(empty_conversion.mpr) == 0
        @test empty_conversion(empty_conversion(D(1))) == D(1)

    end
end

@testset "Delegated algebraic input validation" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached=false)
    S, (u, v) = polynomial_ring(QQ, [:x, :y]; cached=false)
    P = elem_type(R)

    # Check the full input before filtering leaders or returning on constants.
    for check in (
        () -> leader_filter([u], y),
        () -> solution_check(y, [u]),
        () -> solution_check(R(1), P[], [u]),
        () -> __solution_check(y, [x], [u]),
        () -> __solution_check(y, [x], [x], [y], [v]),
        () -> __compatible_zero_algebraic_system(R(1), [u]),
        () -> is_simple([R(1), u]),
        () -> AlgebraicSystem([R(1), u]),
        () -> AlgebraicSystem(P[], [x, u]),
        () -> __algebraic_step([R(1)], [u]),
        () -> __algebraic_step([u], [R(0)]),
        () -> __algebraic_key([R(0)], [u]),
        () -> __algebraic_branch_has_solution([R(1)], [u]),
        () -> __algebraic_normal_form_mod_equations(R(1), [u]),
        () -> algebraic_branch_intersection(([R(1)], P[]), ([u], P[])),
        () -> algebraic_branch_intersection(AlgebraicSystem(R), AlgebraicSystem(S)),
        () -> splitting_covers_system([R(1)], P[], [([u], P[])]),
        () -> splitting_covers_system(AlgebraicSystem(R), [AlgebraicSystem(S)]),
    )
        @test_throws ArgumentError check()
    end

    # Inferring an element type must not silently coerce nonpolynomial inputs.
    @test_throws ArgumentError AlgebraicSystem([1])
    @test_throws ErrorException splitting_covers_system(Any[1], Any[], [])
    @test_throws ErrorException splitting_covers_system(Any[ZZ(1)], Any[], [])
    @test_throws MethodError __algebraic_branch_intersection([1], [], [], [])
    @test splitting_covers_system(Any[x], Any[], [(Any[x], Any[])])
    @test splitting_covers_system(Any[], Any[], [AlgebraicSystem(R)])
    @test AlgebraicSystem(P[], [x]).ring === R
    @test __algebraic_step(P[], P[]) === nothing
end

@testset "Algebraic parent identity" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached=false)
    S, (u, v) = polynomial_ring(QQ, [:x, :y]; cached=false)
    @test typeof(x) === typeof(u)
    @test R !== S
    @test leader_filter([x, y], x) == [x, y]
    @test_throws ArgumentError __leader_filter([u], [x], x)
    @test_throws ArgumentError __leader_filter([x], [u], x)
    for check in [() -> leader_filter([u], R(1)),
                  () -> solution_check(R(1), [u]),
                  () -> is_quasi_simple([R(1), u]),
                  () -> is_quasi_simple([x], [v]),
                  () -> multiplicative_variables([x, u]),
                  () -> cones_check(x, [u]),
                  () -> cones_check(x, [x], [[u]]),
                  () -> is_janet_complete([x], [[u]]),
                  () -> janet_completion([x, u]),
                  () -> quasi_simple_splitting([R(1), u]),
                  () -> thomas_decomposition([x], [u]),
                  () -> __algebraic_pseudo_divrem(R(1), u),
                  () -> __checked_divides(x, u),
        ]
        @test_throws ArgumentError check()
    end
    @test length(quasi_simple_splitting([x^2 - 1])) == 2
    @test is_quasi_simple([x - y])
end

@testset "Action polynomial parents" begin
    for constructor in (difference_polynomial_ring, differential_polynomial_ring)
        R, x = constructor(QQ, :x, 1)
        S, y = constructor(QQ, :other, 1)
        @test typeof(x) === typeof(y)
        for check in [() -> is_janet_reducible(R(1), [y]),
                      () -> is_janet_reducible_univariate(R(1), [y]),
                      () -> normal_form(R(0), [y]),
                      () -> multiplicative_variables([x, y]),
                      ]
            @test_throws ArgumentError check()
        end
        @test is_janet_reducible(x, [x]) !== false
        @test iszero(normal_form(x, [x]).nf)
    end
    R, x = difference_polynomial_ring(QQ, :x, 1)
    S, y = difference_polynomial_ring(QQ, :other, 1)
    data = __action_janet_data([y])
    @test_throws ArgumentError __is_janet_reducible(R(0), data)
    @test_throws ArgumentError Oscar.__algebraic_conversion_data_from_polys(R, [y])
    @test_throws ArgumentError Oscar.__algebraic_conversion_data_from_polys(R, [x, y])
    @test Oscar.__algebraic_conversion_data_from_polys(R, [x]) !== nothing
end
