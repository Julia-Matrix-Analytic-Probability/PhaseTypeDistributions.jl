@testset "MAPHDist" begin
    # Small 2-phase, 2-absorbing example
    α = [0.6, 0.4]
    T = [-3.0 1.0; 0.5 -2.0]
    # Row sums: [-2.0, -1.5]; choose D so T·1 + D·1 = 0
    D = [1.5 0.5; 1.0 0.5]
    d = MAPHDist(α, T, D)

    @testset "basic accessors" begin
        @test nphases(d) == 2
        @test nabsorbing(d) == 2
        @test initial_prob(d) == α
        @test subgenerator(d) == T
        @test exit_rate_matrix(d) == D
        @test exit_rates(d) ≈ [3.0, 2.0]
    end

    @testset "validation" begin
        @test_throws ArgumentError MAPHDist([0.5, 0.3], T, D)           # α doesn't sum to 1
        @test_throws ArgumentError MAPHDist([-0.1, 1.1], T, D)          # negative α
        @test_throws DimensionMismatch MAPHDist([1.0], T, D)            # α size mismatch
        @test_throws ArgumentError MAPHDist(α, T, -D)                   # negative D
        # T·1 + D·1 must equal 0
        D_bad = copy(D); D_bad[1, 1] += 1.0
        @test_throws ArgumentError MAPHDist(α, T, D_bad)
    end

    @testset "marginal absorption probabilities" begin
        π = marginal_absorption(d)
        @test length(π) == 2
        @test all(π .>= 0)
        @test sum(π) ≈ 1.0 atol=1e-10
        # π = -α T⁻¹ D
        @test π ≈ -α' * (T \ D) |> vec atol=1e-10
    end

    @testset "marginal τ is PH(α, T)" begin
        ph = PHDist(d)
        @test initial_prob(ph) == α
        @test subgenerator(ph) == T
        # E[τ] from PH should match sum_k E[τ · 𝟙{κ=k}]
        total_mean = sum(kth_joint_moment(d, k, 1) for k in 1:nabsorbing(d))
        @test mean(ph) ≈ total_mean atol=1e-10
    end

    @testset "absorption_probs matrix R" begin
        R = absorption_probs(d)
        @test size(R) == (2, 2)
        @test all(R .>= -1e-12)
        # Rows sum to 1 (under the irreducibility assumption)
        @test all(isapprox.(sum(R; dims=2), 1.0; atol=1e-10))
    end

    @testset "pdf/cdf/ccdf joint sub-density" begin
        # Integrating f(u, k) over u gives π_k
        π = marginal_absorption(d)
        for k in 1:2
            # Trapezoidal with fine grid
            grid = range(0.0, 20.0; length=10_000)
            vals = [pdf(d, u, k) for u in grid]
            approx_mass = sum(vals) * (grid[2] - grid[1])
            @test approx_mass ≈ π[k] atol=5e-3
            # cdf + ccdf = π_k
            for u in [0.1, 0.5, 1.0, 3.0]
                @test cdf(d, u, k) + ccdf(d, u, k) ≈ π[k] atol=1e-10
            end
        end
        @test pdf(d, -1.0, 1) == 0.0
        @test cdf(d, -1.0, 1) == 0.0
        @test ccdf(d, 0.0, 1) ≈ π[1] atol=1e-10
    end

    @testset "kth_joint_moment" begin
        # First moment E[τ · 𝟙{κ=k}] = α T⁻² D_k
        for k in 1:2
            expected = α' * (T \ (T \ D[:, k]))
            @test kth_joint_moment(d, k, 1) ≈ expected atol=1e-10
        end
        @test_throws ArgumentError kth_joint_moment(d, 1, 0)
        @test_throws ArgumentError kth_joint_moment(d, 3, 1)
    end

    @testset "conditional time τ | κ=k is a valid PH" begin
        for k in 1:2
            ph_k = conditional_time(d, k)
            @test sum(initial_prob(ph_k)) ≈ 1.0 atol=1e-10
            # E[τ | κ=k] = E[τ · 𝟙{κ=k}] / P(κ=k)
            π_k = marginal_absorption(d)[k]
            expected = kth_joint_moment(d, k, 1) / π_k
            @test mean(ph_k) ≈ expected atol=1e-8
        end
    end

    @testset "random sampling" begin
        Random.seed!(123)
        N = 20_000
        samples = rand(d, N)
        @test length(samples) == N
        @test all(s -> s[1] >= 0 && 1 <= s[2] <= 2, samples)

        # Empirical marginal absorption matches
        π = marginal_absorption(d)
        emp_π = [count(s -> s[2] == k, samples) / N for k in 1:2]
        @test isapprox(emp_π, π; atol=0.02)

        # Empirical conditional mean matches E[τ | κ=k]
        for k in 1:2
            τs = [s[1] for s in samples if s[2] == k]
            expected = kth_joint_moment(d, k, 1) / π[k]
            @test isapprox(Statistics.mean(τs), expected; rtol=0.05)
        end
    end

    @testset "PH → MAPH embedding (n=1)" begin
        ph = PHDist([1.0, 0.0], [-2.0 2.0; 0.0 -2.0])
        maph = MAPHDist(ph)
        @test nabsorbing(maph) == 1
        @test marginal_absorption(maph) ≈ [1.0] atol=1e-10
        ph_back = PHDist(maph)
        @test initial_prob(ph_back) ≈ initial_prob(ph)
        @test subgenerator(ph_back) ≈ subgenerator(ph)
        # Conditional τ|κ=1 equals the marginal since there's only one absorbing state
        ph_cond = conditional_time(maph, 1)
        @test mean(ph_cond) ≈ mean(ph) atol=1e-8
    end

    @testset "Block-diagonal construction from per-category PHs" begin
        he = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])
        er = ErlangPHDist(3, 2.0)
        π = [0.3, 0.7]
        maph = MAPHDist(AbstractPHDist[he, er], π)
        @test nphases(maph) == nphases(he) + nphases(er)
        @test nabsorbing(maph) == 2
        @test marginal_absorption(maph) ≈ π atol=1e-10

        # Conditional τ|κ=k should match phs[k] distributionally
        for (k, ph_true) in enumerate([he, er])
            ph_cond = conditional_time(maph, k)
            @test mean(ph_cond) ≈ mean(ph_true) atol=1e-8
            @test var(ph_cond) ≈ var(ph_true) atol=1e-8
        end
    end

    @testset "Moment-matched construction" begin
        π_target = [0.25, 0.45, 0.30]
        μ_target = [1.0, 3.0, 5.0]
        σ²_target = [0.5, 12.0, 4.0]   # c² = [0.5, 4/3, 0.16]  → mix of hypo/hyper
        maph = MAPHDist(π_target, μ_target, σ²_target)

        @test marginal_absorption(maph) ≈ π_target atol=1e-8
        for k in 1:3
            ph_k = conditional_time(maph, k)
            @test mean(ph_k) ≈ μ_target[k] rtol=1e-4
            @test var(ph_k) ≈ σ²_target[k] rtol=5e-2
        end
    end

    @testset "(α, q, R, U) parameterization roundtrip" begin
        # Construct via (α, T, D), extract R, define a compatible U, rebuild, compare
        R = absorption_probs(d)
        q = -diag(T)
        # Derive U: u_ij = T_ij/q_i · R_{jk}/R_{ik} with k=1
        m = 2
        U = zeros(m, m)
        for i in 1:m, j in 1:m
            U[i, j] = i == j ? 0.0 : (T[i, j] / q[i]) * R[j, 1] / R[i, 1]
        end
        d_rebuilt = MAPHDist(α, q, R, U)
        @test isapprox(d_rebuilt, d; atol=1e-8)
    end

    @testset "params / show / isapprox" begin
        α2, T2, D2 = params(d)
        @test α2 == α && T2 == T && D2 == D
        s = sprint(show, d)
        @test occursin("MAPHDist", s)
        @test occursin("m=2", s)
        @test occursin("n=2", s)

        d_copy = MAPHDist(α, T, D)
        @test isapprox(d, d_copy)
    end
end
