using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Differential minimal Janet basis" begin
    R, u =
        differential_polynomial_ring(
            QQ,
            :u,
            1,
        )

    du =
        apply_action(
            u,
            1,
        )

    @test is_quasi_simple([u])
    @test is_simple([u])

    # {u, u_x} is Janet complete and passive, but it is not a minimal
    # Janet leader set: u_x is not required in the Janet completion of {u}.
    @test !is_quasi_simple([u, du])
    @test !is_simple([u, du])
end

@testset "Differential certifications remain separate" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    system = DifferentialSystem([u^2])
    @test is_quasi_simple(system)
    @test system.simple === nothing
    @test !is_simple(system)
    @test system.simple === false && system.quasi_simple === true
    @test !is_simple_as_algebraic_system(system)
    D, v = differential_polynomial_ring(QQ, :v, 2)
    nonpassive = DifferentialSystem([v[1,0]-v, v[0,1]-v^2])
    @test is_quasi_simple_as_algebraic_system(nonpassive.equations, nonpassive.inequations)
    @test !is_quasi_simple(nonpassive)
    @test nonpassive.quasi_simple === false
    @test !is_simple(nonpassive)
    regular = DifferentialSystem([u[1]-u], [u])
    @test is_simple(regular)
    @test regular.quasi_simple === true
    @test is_quasi_simple(regular)
    @test normal_form(u[2], regular) == normal_form(u[2], regular.equations)
    @test is_janet_reducible(u[2], regular) == is_janet_reducible(u[2], regular.equations)
    @test is_janet_reducible_univariate(u[2], regular) == is_janet_reducible_univariate(u[2], regular.equations)
    @test is_quasi_simple(DifferentialSystem(R, [], [u]))
end

@testset "Finite conversions and ordinary window systems" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    input = DifferentialSystem([u[1]-u], [u])
    @test is_quasi_simple(input)
    algebraic = __conversion_algebraic_system(input)
    @test algebraic isa AlgebraicSystem
    @test algebraic.quasi_simple === nothing
    @test length(algebraic.equations) == 1 && length(algebraic.inequations) == 1
    @test is_quasi_simple(algebraic)
    empty_system = __conversion_algebraic_system(DifferentialSystem(R))
    @test ngens(empty_system.ring) == 0 && empty_system.quasi_simple === nothing
    @test isempty(empty_system.equations) && isempty(empty_system.inequations)
#    R, (x, z) = polynomial_ring(QQ, [:x, :z])
#    input = AlgebraicSystem([(z-x)*(z+x)], [x, z])
#    branches = ordinary_split(input; left_variables=[x], right_variables=[z], prefixes=[[x], [x,z]])
#    @test length(branches) == 2
#    @test all(b -> b isa OrdinaryWindowBranch && b.certificate.holds, branches)
#    @test all(b -> b.system.quasi_simple === nothing && b.window_system.quasi_simple === nothing, branches)
#    @test length(ordinary_prime_charts(input)) == 2
#    @test length(ordinary_visible_primes(input)) == 2
#    @test !isone(ordinary_saturated_window_ideal(input))
#    first_branch = first(branches)
#    guards = first_branch.guards
#    zero!(guards[1])
#    @test all(!iszero, first_branch.guards)
#    is_quasi_simple(first_branch.system)
#    @test copy(first_branch).system.quasi_simple === nothing
end

@testset "Literature: heat equation (Robertz 2018, Example 2.43)" begin
    # Derivation order: (t, x); degree-reverse lexicographic ranking.
    R, u = differential_polynomial_ring(QQ, :u, 2; index_ordering_name=:degrevlex)

    ut  = u[1, 0]
    uxx = u[0, 2]
    h = ut - uxx

    @test leader(h) == uxx
    @test is_janet_complete([h])
    @test is_passive([h])

    S = DifferentialSystem(R, [h], elem_type(R)[])
    @test is_quasi_simple(S)
    @test is_simple(S)
    @test is_simple_as_algebraic_system(S)
end

@testset "Literature: minimal Janet basis (Robertz 2018, Example 2.44)" begin
    # Derivation order: (x, y, z).
    R, u = differential_polynomial_ring(QQ, :u, 3)

    E = [
        u[1, 1, 0],  # u_xy
        u[2, 1, 0],  # u_xxy
        u[3, 0, 1],  # u_xxxz
        u[3, 1, 0],  # u_xxxy
    ]

    expected_MV = [
        BitVector([false, true,  true ]),
        BitVector([false, true,  true ]),
        BitVector([true,  false, true ]),
        BitVector([true,  true,  true ]),
    ]

    @test multiplicative_variables(E) == expected_MV
    @test is_janet_complete(E)
    @test is_passive(E)

    # Explicit non-multiplicative prolongations from the Janet table.
    @test is_zero(normal_form(apply_action(E[1], 1), E).nf)
    @test is_zero(normal_form(apply_action(E[2], 1), E).nf)
    @test is_zero(normal_form(apply_action(E[3], 2), E).nf)

    S = DifferentialSystem(R, E, elem_type(R)[])
    @test is_quasi_simple(S)
    @test is_simple(S)
end

@testset "Literature: involutivity/passivity (BGLR 2012, Examples 3.1 and 3.4)" begin
    # Use derivation order (t, x), so :degrevlex gives u_t > u_x,
    # matching the ranking displayed in the paper.
    R, u = differential_polynomial_ring(
        QQ, :u, 2;
        index_ordering_name=:degrevlex,
    )

    ut  = u[1, 0]
    ux  = u[0, 1]
    uxx = u[0, 2]

    p1 = ut + u*ux
    p2 = uxx
    E = [p1, p2]

    @test leader(p1) == ut
    @test leader(p2) == uxx
    @test multiplicative_variables(E) == [
        BitVector([true, true]),
        BitVector([false, true]),
    ]
    @test is_janet_complete(E)

    # The non-reductive t-prolongation of u_xx reduces to zero.
    @test is_zero(normal_form(apply_action(p2, 1), E).nf)
    @test is_passive(E)

    S = DifferentialSystem(R, E, elem_type(R)[])
    @test is_quasi_simple(S)
    @test is_simple(S)
end
