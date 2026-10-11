using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Algebraic systems own their data" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached=false)
    Q, (u, v) = polynomial_ring(QQ, [:x, :y]; cached=false)
    E, I = [x-y], [y]
    system = AlgebraicSystem(E, I)
    @test system.ring === R
    @test system.quasi_simple === nothing
    @test system.simple === nothing
    @test system.equations == E
    @test system.inequations == I
    @test_throws ArgumentError AlgebraicSystem(R, [u], I)
    @test_throws ArgumentError AlgebraicSystem([x], [u])
    @test_throws ArgumentError AlgebraicSystem(elem_type(R)[])
    @test_throws ArgumentError AlgebraicSystem(R, [1], [])
    @test_throws ArgumentError AlgebraicSystem(QQ)
    @test AlgebraicSystem(R).ring === R
    @test isempty(AlgebraicSystem(R).equations)
    @test isempty(AlgebraicSystem(R).inequations)
    @test is_quasi_simple(system)
    @test system.quasi_simple === true
    key = __algebraic_key(system)
    hash_before = hash(system)
    zero!(E[1])
    empty!(I)
    working = system.equations
    zero!(working[1])
    push!(working, R(1))
    @test system.equations == [x-y]
    @test system.inequations == [y]
    @test system.quasi_simple === true
    @test hash(system) == hash_before
    @test __algebraic_key(system) == key
    @test_throws Exception setproperty!(system, :quasi_simple, false)
    for copied in (AlgebraicSystem(system), deepcopy(system))
        @test copied.quasi_simple === nothing
        @test copied.ring === R
        @test copied == system
        @test parent(only(copied.equations)) === R
    end
    changed = AlgebraicSystem(R, [system.equations; x+y], system.inequations)
    @test changed.quasi_simple === nothing
    @test !is_quasi_simple(changed)
    @test changed.quasi_simple === false
    @test system.quasi_simple === true
    @test __algebraic_cleanup(system).quasi_simple === nothing
end

@testset "Owned algebraic keys share data without changing normalization" begin
    for K in (ZZ, QQ, GF(7))
        R, (x, y) = polynomial_ring(K, [:x, :y])
        for system in (AlgebraicSystem([x+y, x+y], [y, y]),
                       AlgebraicSystem(Val(:normalized), R, [x+y], [y]))
            key = __algebraic_key(system)
            @test length(key.equations) == length(key.inequations) == 1
            @test only(key.equations) === first(getfield(system, :equations))
            @test only(key.inequations) === first(getfield(system, :inequations))
            for copied in (AlgebraicSystem(system), deepcopy(system))
                copied_key = __algebraic_key(copied)
                @test copied == system && hash(copied) == hash(system)
                @test only(copied_key.equations) === first(getfield(copied, :equations))
                @test only(copied_key.equations) !== only(key.equations)
                @test only(copied_key.inequations) !== only(key.inequations)
            end
        end
        # An independently built key must still detach its caller-owned input.
        E, I = [deepcopy(x)], [deepcopy(y)]
        key = __polynomial_system_key(R, E, I)
        zero!(E[1]); zero!(I[1])
        @test key.equations == (x,)
        @test key.inequations == (y,)
    end
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    system = AlgebraicSystem([-2*x-2*y, x+y], [3*y])
    copied = AlgebraicSystem(system)
    @test copied.equations == [-2*x-2*y, x+y]
    @test copied.inequations == [3*y]
    @test copied == AlgebraicSystem([x+y], [y])
    @test __algebraic_key(copied).equations == (x+y,)
    @test only(__algebraic_key(copied).equations) !== only(__algebraic_key(system).equations)
    @test AlgebraicSystem(AlgebraicSystem(R)) == AlgebraicSystem(R)
end

@testset "Set membership" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached=false)
    Q, (u, v) = polynomial_ring(QQ, [:x, :y]; cached=false)
    a = AlgebraicSystem([x, y], [x+y])
    b = AlgebraicSystem([3*y, -2*x, x], [-4*x-4*y])
    @test isequal(a, b)
    @test hash(a) == hash(b)
    @test length(Set([a, b])) == 1
    @test a != AlgebraicSystem([x+y], [x, y])
    @test a != AlgebraicSystem([u, v], [u+v])
    @test AlgebraicSystem(R) != AlgebraicSystem(Q)
    @test AlgebraicSystem([x]) != AlgebraicSystem([x^2])
    set = Set([a])
    is_quasi_simple(a)
    @test a in set && b in set
    @test !is_quasi_simple(a)
    @test b.quasi_simple === nothing
    E, I = [deepcopy(x)], [deepcopy(y)]
    key = __algebraic_key(E, I)
    keys = Set([key])
    zero!(E[1]); empty!(I)
    @test __algebraic_key([x], [y]) in keys
    @test __algebraic_key(E, I) != key
    @test length(__algebraic_deduplicate_branch_list([([x,y], [x+y]), ([y,x,x], [x+y])])) == 1
end

@testset "Splitting creates uncertified children and certifies results" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    input = AlgebraicSystem([x^2-1])
    @test is_quasi_simple(input)
    @test __algebraic_target_issue(input; check_discriminants=false, factorize=true) !== nothing
    @test input.quasi_simple === true
    children = __algebraic_step(input; check_discriminants=false, factorize=true)
    @test length(children) == 2
    @test all(child -> child.quasi_simple === nothing, children)
    @test input.equations == [x^2-1]
    result = quasi_simple_splitting(input)
    @test all(child -> child.quasi_simple === true, result)
    @test splitting_covers_system(input, result)
    @test splitting_covers_system(input, AlgebraicSystem[result...])
    @test !splitting_covers_system(input, AlgebraicSystem[])
    @test !algebraic_branch_intersection(result[1], result[2])
    @test result == quasi_simple_splitting([x^2-1])
    discriminant_input = AlgebraicSystem([x^2-y])
    @test is_quasi_simple(discriminant_input)
    @test __algebraic_target_issue(input; check_discriminants=false, factorize=true,) !== nothing
    @test discriminant_input.quasi_simple === true
    discriminant_result = thomas_decomposition(discriminant_input)
    @test all(child -> child.quasi_simple === true, discriminant_result)
    @test splitting_covers_system(discriminant_input, discriminant_result)
    @test discriminant_result == thomas_decomposition([x^2-y])
    universal = quasi_simple_splitting(AlgebraicSystem(R))
    @test length(universal) == 1 && only(universal).quasi_simple === true
    @test only(universal).ring === R
    @test isempty(quasi_simple_splitting(AlgebraicSystem(R, [R(1)])))
    @test isempty(quasi_simple_splitting(AlgebraicSystem(R, [], [R(0)])))
    missing = splitting_covers_system(input, result[1:1]; return_remainder=true)
    @test !missing.covers
    @test !isempty(missing.missing)
    @test all(s -> s.quasi_simple === nothing, missing.missing)
    @test_throws ArgumentError algebraic_branch_intersection(input, AlgebraicSystem(first(polynomial_ring(QQ, [:other]))))
    io = IOBuffer(); print_splitting_result(result; io)
    @test occursin("Branch 1", String(take!(io)))
end

@testset "Native ActionPolyRing coefficient maps" begin
    K, t = rational_function_field(QQ, "t")
    sigma = action_shift(map_from_func(K, K, c -> c(t + 1)))
    delta = action_derivation(map_from_func(K, K, derivative))
    R, x = difference_polynomial_ring(K, :x, [sigma])
    D, u = differential_polynomial_ring(K, :u, [delta])
    @test apply_action(t*x, 1) == (t+1)*x[1]
    @test apply_action(t*u, 1) == t*u[1]+u
    @test only(action_maps(R))(t) == t+1
    @test only(action_maps(D))(t) == 1

    # Rings with the same coefficient field retain their own native maps.
    S, y = difference_polynomial_ring(K, :y, 1)
    B, v = differential_polynomial_ring(K, :v, 1)
    @test action_map(S, 1)(t) == t
    @test action_map(B, 1)(t) == 0
    @test apply_action(t*y, 1) == t*y[1]
    @test apply_action(t*v, 1) == t*v[1]
    @test apply_action(t*x, 1) == (t+1)*x[1]
    @test apply_action(t*u, 1) == t*u[1]+u

    # Loading the project must not replace the native arithmetic methods.
    for p in (x, u), index_type in (Int, Vector{Int})
        @test which(apply_action, (typeof(p), index_type)).module === Oscar
    end

    S, y = difference_polynomial_ring(K, :y, [sigma, action_shift(K)])
    B, v = differential_polynomial_ring(K, :v, [delta, action_derivation(K)])
    @test apply_action(t*y, [1, 1]) == (t+1)*y[1, 1]
    @test apply_action(t*v, [1, 1]) == v[0, 1]+t*v[1, 1]
end

@testset "Several native coefficient maps" begin
    A, (s, t) = polynomial_ring(QQ, [:s, :t])
    K = fraction_field(A)
    s, t = K(s), K(t)
    f = hom(A, K, [s+1, t])
    g = hom(A, K, [s, t+1])
    sigma1(c) = f(numerator(c)) // f(denominator(c))
    tau(c) = g(numerator(c)) // g(denominator(c))
    shifts = [action_shift(map_from_func(K, K, sigma1)),
              action_shift(map_from_func(K, K, tau))]
    derivations = [action_derivation(map_from_func(K, K, c -> derivative(c, i)))
                   for i in 1:2]
    R, x = difference_polynomial_ring(K, :x, shifts)
    D, u = differential_polynomial_ring(K, :u,
        derivations)
    action = apply_action
    p = s/t*x + s^2/(t+1)
    @test action(p, [1, 1]) == (s+1)/(t+1)*x[1,1] + (s+1)^2/(t+2)
    @test action(s*t*u, [1, 1]) == s*t*u[1,1]+s*u[1,0]+t*u[0,1]+u
    @test action(p, [1, 1]) == action(action(p, 1), 2)
    @test action(s*t*u, [1, 1]) == action(action(s*t*u, 2), 1)
    for q in (p, s*t*u)
        @test action(q, [0, 0]) == q
        @test action(zero(parent(q)), [1, 1]) == 0
        @test_throws ArgumentError action(q, 0)
        @test_throws ArgumentError action(q, 3)
        @test_throws ArgumentError action(q, [1])
        @test_throws ArgumentError action(q, [-1, 0])
    end

    @test_throws ArgumentError difference_polynomial_ring(K, :x, shifts[1:0])
    @test_throws ArgumentError differential_polynomial_ring(K, :u, derivations[1:0])

    other_A, _ = polynomial_ring(QQ, [:s, :t]; cached=false)
    other_K = fraction_field(other_A)
    @test_throws ArgumentError difference_polynomial_ring(other_K, :x, shifts)
    @test_throws ArgumentError differential_polynomial_ring(other_K, :u, derivations)

    for (make_ring, maps) in ((difference_polynomial_ring, shifts),
                             (differential_polynomial_ring, derivations))
        S, variables = make_ring(K, [:a, :b], maps)
        @test length(variables) == 2
        @test action_map(S, 1)(s) == maps[1](s)
        @test action_map(S, 2)(t) == maps[2](t)
    end

    for (make_ring, coefficient_action) in (
        (difference_polynomial_ring, action_shift(QQ)),
        (differential_polynomial_ring, action_derivation(QQ)))
        S, variables = make_ring(QQ, 2, [coefficient_action])
        @test length(variables) == 2
        @test all(v -> parent(v) === S, variables)
        @test action(variables[1], [1]) == variables[1][1]
    end
end

@testset "Action systems own their data and certificates" begin
    (constructor, ring_constructor) = (DifferentialSystem, differential_polynomial_ring)
        R, (u, v) = ring_constructor(QQ, [:u, :v], 1)
        S, (a, b) = ring_constructor(QQ, [:u, :v], 1)
        E = [deepcopy(u)]
        I = [deepcopy(v)]
        system = constructor(R, E, I)
        data = getfield(system, :_data)
        @test only(data.key.equations) === only(data.equations)
        @test only(data.key.inequations) === only(data.inequations)
        @test only(data.equations) !== only(E)
        @test only(data.inequations) !== only(I)
        @test system.ring === R
        @test system.quasi_simple === nothing
        @test __require_same_parent(system, E, I) === R
        @test_throws ArgumentError __require_same_parent(constructor(R), constructor(S))
        @test_throws ArgumentError constructor(R, [u], [b])
        @test_throws ArgumentError constructor([u, a])
        @test_throws ArgumentError constructor(R, [1])
        @test_throws ArgumentError constructor(elem_type(R)[])
        @test_throws MethodError constructor(R, E, I, true)
        @test_throws Exception setproperty!(system, :quasi_simple, true)
        @test is_quasi_simple(system)
        @test system.quasi_simple === true
        # Mutating either the input or a returned polynomial cannot stale a certificate.
        zero!(E[1]); empty!(I)
        working = system.equations
        zero!(working[1])
        @test system.equations == [u]
        @test system.inequations == [v]
        @test system.quasi_simple === true
        for new in (constructor(system), copy(system), deepcopy(system))
            @test new == system
            @test new.ring === R
            @test all(name -> getproperty(new, name) === nothing, __system_certificate_names(new))
            new_data = getfield(new, :_data)
            @test only(new_data.key.equations) === only(new_data.equations)
            @test only(new_data.key.equations) !== only(data.key.equations)
            @test only(new_data.key.inequations) !== only(data.key.inequations)
        end
        duplicated = constructor([u, v, u], [v, v])
        @test length(duplicated.equations) == 3
        @test length(duplicated.inequations) == 2
        @test length(__action_system_key(duplicated).equations) == 2
        @test length(__action_system_key(duplicated).inequations) == 1
        @test constructor(R) != constructor(S)
        @test constructor([u, v, u]) == constructor([v, u])
        @test hash(constructor([u, v])) == hash(constructor([v, u, v]))
        @test constructor([u], [v]) != constructor([v], [u])
        @test constructor([u]) != constructor([2*u])
        key = hash(system)
        seen = Set([system])
        apply_action(u, [20]) # Parent gains new jet variables
        @test hash(system) == key
        @test constructor([u], [v]) in seen
        @test is_quasi_simple(constructor(R))
        @test is_janet_complete(constructor(R))
        @test is_passive(constructor(R))
        @test normal_form(u, constructor(R)) == (nf=u, multiplier=one(R))
        @test_throws ArgumentError normal_form(a, constructor(R))
        @test_throws ArgumentError is_janet_reducible(a, constructor(R))
        @test_throws ArgumentError is_janet_reducible_univariate(a, constructor(R))
        @test occursin("quasi_simple = true", sprint(show, system))
        @test occursin("Branch 1", sprint(io -> print_splitting_result([system]; io)))
    end

@testset "System vector conversion returns detached constraints" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    D, u = differential_polynomial_ring(QQ, :u, 1)
    for system in (AlgebraicSystem([x-y], [y]), DifferentialSystem([u[1]-u], [u]))
        expected = system.equations, system.inequations
        key_hash = hash(system)
        for (E, I) in (system_to_vectors(system), only(system_to_vectors([system])))
            @test (E, I) == expected
            zero!(first(E))
            empty!(I)
            @test (system.equations, system.inequations) == expected
            @test hash(system) == key_hash
        end
    end
end

@testset "Factored printing keeps each equation as one constraint" begin
    R, (x, y) = polynomial_ring(QQ, [:x, :y])
    system = AlgebraicSystem([x^2-y^2])
    output = sprint(io -> print_splitting_result([system]; io))
    @test occursin("(x + y)*(x - y)", output) || occursin("(x - y)*(x + y)", output)
    @test !occursin("Branch 2", output)
    @test system.equations == [x^2-y^2]
end
