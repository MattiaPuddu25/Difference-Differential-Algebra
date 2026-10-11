using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end



@testset "Branch representations and output selection" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached=false)
    Q, (u, v) = polynomial_ring(QQ, [:x, :y]; cached=false)
    E, I = [x^2-1], [y]
    system = AlgebraicSystem(E, I)
    for split in (quasi_simple_splitting, thomas_decomposition)
        branches = split(E, I)
        @test branches isa Vector{<:AlgebraicSystem}
        @test all(s -> s.quasi_simple === true, branches)
        @test branches == split(system)
        @test split(E, I; vector_output=true) == branch_conversion(branches)
        @test split(system; vector_output=true) == branch_conversion(branches)
        @test split(E, []; vector_output=true) == branch_conversion(split(E))
        @test_throws ArgumentError split([x], [u])
        @test_throws ArgumentError split([], [])
    end
    @test system.quasi_simple === nothing
    branches = quasi_simple_splitting(system)
    pairs = branch_conversion(branches)
    mixed = [branches[1], pairs[2]]
    @test branch_conversion(mixed) == pairs
    @test branch_conversion(pairs) == pairs
    @test isempty(branch_conversion(AlgebraicSystem[]))
    @test isempty(branch_conversion(typeof(system)[]))
    @test_throws ArgumentError branch_conversion([AlgebraicSystem(R), AlgebraicSystem(Q)])
    @test_throws ArgumentError branch_conversion([([x], [u])])
    @test_throws ArgumentError branch_conversion([([1], [])])
    detached = branch_conversion(mixed)
    zero!(detached[1][1][1]); empty!(detached[2][2])
    @test branch_conversion(mixed) == pairs
    @test !algebraic_branch_intersection(branches[1], pairs[2])
    @test !algebraic_branch_intersection(pairs[1], branches[2])
    # Algebraic pairs use the general entry point without a tuple-specific method.
    @test !algebraic_branch_intersection(pairs[1], pairs[2])
    @test algebraic_branch_intersection(pairs[1], pairs[1])
    @test algebraic_branch_intersection(([], []), ([], []))
    @test_throws ArgumentError algebraic_branch_intersection(AlgebraicSystem(Q), pairs[1])
    for input in (system, (E, I)), representation in (branches, pairs, mixed)
        @test splitting_covers_system(input, representation)
    end
    @test splitting_covers_system(E, I, mixed)
    @test splitting_covers_system(E, quasi_simple_splitting(E))
    missing = splitting_covers_system(E, I, mixed[1:1]; return_remainder=true)
    @test !missing.covers && !isempty(missing.missing) && isempty(missing.extra)
    @test all(s -> s.quasi_simple === nothing, missing.missing)
    raw = splitting_covers_system(system, pairs[1:1]; return_remainder=true, vector_output=true)
    @test raw.missing == branch_conversion(missing.missing)
    extra = splitting_covers_system([x], [([x], []), AlgebraicSystem([y])]; return_remainder=true)
    @test !extra.covers && isempty(extra.missing) && !isempty(extra.extra)
    @test all(s -> s.quasi_simple === nothing, extra.extra)
    @test_throws ArgumentError splitting_covers_system(AlgebraicSystem(Q), branches)
    @test_throws ArgumentError splitting_covers_system(E, I, [AlgebraicSystem(Q)])
    universal = AlgebraicSystem(R)
    @test only(quasi_simple_splitting(universal; vector_output=true)) == (elem_type(R)[], elem_type(R)[])
    @test isempty(quasi_simple_splitting([R(1)]; vector_output=true))
    @test splitting_covers_system([], [], [universal])
    @test splitting_covers_system(universal, [([], [])])
    remainder = splitting_covers_system(universal, []; return_remainder=true)
    @test only(remainder.missing).ring === R
    @test only(remainder.missing).quasi_simple === nothing
    @test !splitting_covers_system([], [], [])
    @test_throws ArgumentError splitting_covers_system([], [], []; return_remainder=true)
    @test !splitting_covers_system([], [], []; return_remainder=true, vector_output=true).covers
    for constructor in (difference_polynomial_ring, differential_polynomial_ring)
        A, z = constructor(QQ, :z, 1)
        @test splitting_covers_system([z], [([z], [])])
        remainder = splitting_covers_system([z], []; return_remainder=true)
        @test only(remainder.missing) isa AlgebraicSystem
        @test only(remainder.missing).quasi_simple === nothing
    end
end

@testset "Joint reduction preserves the lower context" begin
    R, (x, y, z) = polynomial_ring(QQ, [:x, :y, :z]; cached=false)
    equations = [y^2-z^3]
    polynomials = [x*y^2+y^3+1, y^3+y+1]
    original = deepcopy(polynomials)
    batch = __algebraic_normal_form_mod_equations(polynomials, equations)
    @test batch == [__algebraic_normal_form_mod_equations(p, equations) for p in polynomials]
    @test polynomials == original
    @test __algebraic_normal_form_mod_equations(elem_type(R)[], equations) == elem_type(R)[]
    Q, (u, v, w) = polynomial_ring(QQ, [:x, :y, :z]; cached=false)
    @test_throws ArgumentError __algebraic_normal_form_mod_equations(polynomials, [v-w])
    system = AlgebraicSystem([x^2+y^3, y^2-z^3], [x+y^3])
    children = __algebraic_step(system; factorize=false)
    @test splitting_covers_system(system, children)
end

@testset "Finite algebraic intersections of action-system objects" begin
    K, _ = rational_function_field(QQ, "t")
    R, u = differential_polynomial_ring(K, :u, [action_derivation(K)])
    A = DifferentialSystem([u])
    B = DifferentialSystem([u[1]-1])
    A_pair = ([u], elem_type(R)[])
    B_pair = ([u[1]-1], elem_type(R)[])

    # Compatible finite jets do not prove differential compatibility.
    # Objects and untyped pairs must reach the same conversion as typed pairs.
    for left in (A, A_pair, (Any[u], Any[])),
        right in (B, B_pair, (Any[u[1]-1], Any[]))
        @test algebraic_branch_intersection(left, right)
        @test algebraic_branch_intersection(right, left)
    end
    nonzero = DifferentialSystem(R, elem_type(R)[], [u])
    for left in (A, A_pair, (Any[u], Any[])),
        right in (nonzero, (elem_type(R)[], [u]), (Any[], Any[u]))
        @test !algebraic_branch_intersection(left, right)
        @test !algebraic_branch_intersection(right, left)
    end
    @test !algebraic_branch_intersection(A, DifferentialSystem([u-1]))
    @test algebraic_branch_intersection(DifferentialSystem(R), ([], []))
    @test algebraic_branch_intersection(DifferentialSystem(R), A)
    @test A.quasi_simple === nothing && B.quasi_simple === nothing

    pairs = branch_conversion([A, B_pair])
    @test pairs == [([u], elem_type(R)[]), B_pair]
    zero!(pairs[1][1][1])
    @test A.equations == [u]

    S, v = differential_polynomial_ring(K, :v, [action_derivation(K)])
    @test_throws ArgumentError algebraic_branch_intersection(A, DifferentialSystem([v]))
    @test_throws ArgumentError algebraic_branch_intersection(DifferentialSystem(R), DifferentialSystem(S))
    @test_throws ArgumentError algebraic_branch_intersection(A, ([v], elem_type(S)[]))
end
