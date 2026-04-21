@testset "HyperExponentialDist" begin
    he = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])

    @testset "basic properties" begin
        @test nphases(he) == 2
        @test initial_prob(he) == [0.4, 0.6]
        @test subgenerator(he) ≈ [-2.0 0.0; 0.0 -5.0]
        @test exit_rates(he) == [2.0, 5.0]
    end

    @testset "efficient mean/var" begin
        μ = 0.4/2.0 + 0.6/5.0  # = 0.32
        @test mean(he) ≈ μ atol=1e-10
        ph = PHDist(he)
        @test mean(he) ≈ mean(ph) atol=1e-10
        @test var(he) ≈ var(ph) atol=1e-10
        @test std(he) ≈ sqrt(var(he)) atol=1e-10
    end

    @testset "pdf/cdf/ccdf match general PHDist" begin
        ph = PHDist(he)
        for x in [0.0, 0.1, 0.5, 1.0, 3.0]
            @test pdf(he, x) ≈ pdf(ph, x) atol=1e-10
            @test cdf(he, x) ≈ cdf(ph, x) atol=1e-10
            @test ccdf(he, x) ≈ ccdf(ph, x) atol=1e-10
            @test cdf(he, x) + ccdf(he, x) ≈ 1.0 atol=1e-10
        end
    end

    @testset "logpdf" begin
        for x in [0.1, 0.5, 1.0]
            @test logpdf(he, x) ≈ log(pdf(he, x)) atol=1e-10
        end
        @test logpdf(he, -1.0) == -Inf
    end

    @testset "SCV ≥ 1" begin
        @test scv(he) >= 1.0 - 1e-10
    end

    @testset "skewness/kurtosis match general PHDist" begin
        ph = PHDist(he)
        @test skewness(he) ≈ skewness(ph) atol=1e-8
        @test kurtosis(he) ≈ kurtosis(ph) atol=1e-8
    end

    @testset "quantile/median" begin
        ph = PHDist(he)
        for p in [0.1, 0.5, 0.9]
            q = quantile(he, p)
            @test cdf(he, q) ≈ p atol=1e-8
            @test q ≈ quantile(ph, p) atol=1e-6
        end
        @test median(he) ≈ quantile(he, 0.5) atol=1e-10
    end

    @testset "random sampling" begin
        samples = [rand(he) for _ in 1:10_000]
        @test isapprox(Statistics.mean(samples), mean(he); atol=0.05)

        xs = rand(he, 1000)
        @test length(xs) == 1000
        @test all(x -> x >= 0, xs)
    end

    @testset "mean/scv constructor" begin
        he2 = HyperExponentialDist(2.0, 3.0)
        @test isapprox(mean(he2), 2.0; atol=1e-6)
        @test isapprox(scv(he2), 3.0; atol=1e-6)
    end

    @testset "moments and mgf" begin
        ph = PHDist(he)
        @test kth_moment(he, 1) ≈ kth_moment(ph, 1) atol=1e-10
        @test kth_moment(he, 3) ≈ kth_moment(ph, 3) atol=1e-10
        @test mgf(he, 0.5) ≈ mgf(ph, 0.5) atol=1e-10
    end

    @testset "params" begin
        probs, rates = params(he)
        @test probs == [0.4, 0.6]
        @test rates == [2.0, 5.0]
    end

    @testset "show" begin
        s = sprint(show, he)
        @test occursin("HyperExponentialDist", s)
    end

    @testset "constructor validation" begin
        @test_throws DimensionMismatch HyperExponentialDist([0.5, 0.5], [1.0])
        @test_throws ArgumentError HyperExponentialDist([0.5, 0.3], [1.0, 2.0])   # doesn't sum to 1
        @test_throws ArgumentError HyperExponentialDist([-0.1, 1.1], [1.0, 2.0])  # negative prob
        @test_throws ArgumentError HyperExponentialDist([0.5, 0.5], [1.0, 0.0])   # non-positive rate
        @test_throws ArgumentError HyperExponentialDist(1.0, 0.5)                  # SCV ≤ 1
    end
end

@testset "HypoExponentialDist" begin
    ho = HypoExponentialDist([3.0, 5.0])

    @testset "basic properties" begin
        @test nphases(ho) == 2
        @test initial_prob(ho) == [1.0, 0.0]
        T = subgenerator(ho)
        @test T ≈ [-3.0 3.0; 0.0 -5.0]
        @test exit_rates(ho) ≈ [0.0, 5.0]
    end

    @testset "efficient mean/var" begin
        @test mean(ho) ≈ 1/3 + 1/5 atol=1e-10
        @test var(ho) ≈ 1/9 + 1/25 atol=1e-10
        ph = PHDist(ho)
        @test mean(ho) ≈ mean(ph) atol=1e-10
        @test var(ho) ≈ var(ph) atol=1e-10
    end

    @testset "pdf/cdf/ccdf match general PHDist" begin
        ph = PHDist(ho)
        for x in [0.01, 0.1, 0.5, 1.0, 2.0]
            @test pdf(ho, x) ≈ pdf(ph, x) atol=1e-8
            @test cdf(ho, x) ≈ cdf(ph, x) atol=1e-8
            @test ccdf(ho, x) ≈ ccdf(ph, x) atol=1e-8
            @test cdf(ho, x) + ccdf(ho, x) ≈ 1.0 atol=1e-10
        end
    end

    @testset "single-phase HypoExp = Exponential" begin
        ho1 = HypoExponentialDist([2.0])
        exp_d = Exponential(0.5)
        for x in [0.1, 1.0, 3.0]
            @test pdf(ho1, x) ≈ pdf(exp_d, x) atol=1e-10
            @test ccdf(ho1, x) ≈ ccdf(exp_d, x) atol=1e-10
        end
    end

    @testset "SCV ≤ 1" begin
        @test scv(ho) <= 1.0 + 1e-10
    end

    @testset "skewness/kurtosis" begin
        ph = PHDist(ho)
        @test skewness(ho) ≈ skewness(ph) atol=1e-6
        @test kurtosis(ho) ≈ kurtosis(ph) atol=1e-6
    end

    @testset "quantile/median" begin
        for p in [0.1, 0.5, 0.9]
            q = quantile(ho, p)
            @test cdf(ho, q) ≈ p atol=1e-8
        end
        @test median(ho) ≈ quantile(ho, 0.5) atol=1e-10
    end

    @testset "random sampling" begin
        samples = [rand(ho) for _ in 1:10_000]
        @test isapprox(Statistics.mean(samples), mean(ho); atol=0.05)

        xs = rand(ho, 500)
        @test length(xs) == 500
    end

    @testset "mean/scv constructor" begin
        ho2 = HypoExponentialDist(2.0, 0.5)
        @test isapprox(mean(ho2), 2.0; atol=1e-6)
        @test isapprox(scv(ho2), 0.5; atol=0.1)
    end

    @testset "mgf" begin
        ph = PHDist(ho)
        @test mgf(ho, 0.5) ≈ mgf(ph, 0.5) atol=1e-10
    end

    @testset "params" begin
        (rates,) = params(ho)
        @test rates == [3.0, 5.0]
    end

    @testset "show" begin
        s = sprint(show, ho)
        @test occursin("HypoExponentialDist", s)
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError HypoExponentialDist(Float64[])
        @test_throws ArgumentError HypoExponentialDist([1.0, -2.0])
        @test_throws ArgumentError HypoExponentialDist(2.0, 1.5)  # SCV must be < 1
    end
end

@testset "ErlangPHDist" begin
    er = ErlangPHDist(3, 2.0)

    @testset "basic properties" begin
        @test nphases(er) == 3
        @test initial_prob(er) == [1.0, 0.0, 0.0]
        T = subgenerator(er)
        @test T ≈ [-2.0 2.0 0.0; 0.0 -2.0 2.0; 0.0 0.0 -2.0]
        @test exit_rates(er) ≈ [0.0, 0.0, 2.0]
    end

    @testset "matches Distributions.Erlang" begin
        d_erlang = Erlang(3, 0.5)  # shape=3, scale=1/rate=0.5
        @test mean(er) ≈ Distributions.mean(d_erlang) atol=1e-10
        @test var(er) ≈ Distributions.var(d_erlang) atol=1e-10
        for x in [0.1, 0.5, 1.0, 2.0, 5.0]
            @test pdf(er, x) ≈ pdf(d_erlang, x) atol=1e-10
            @test cdf(er, x) ≈ cdf(d_erlang, x) atol=1e-10
            @test ccdf(er, x) ≈ ccdf(d_erlang, x) atol=1e-10
            @test logpdf(er, x) ≈ logpdf(d_erlang, x) atol=1e-10
        end
        # Erlang has skewness 2/sqrt(k), excess kurtosis 6/k
        @test skewness(er) ≈ 2 / sqrt(3) atol=1e-8
        @test kurtosis(er) ≈ 6 / 3 atol=1e-8
    end

    @testset "matches general PHDist" begin
        ph = PHDist(er)
        @test mean(er) ≈ mean(ph) atol=1e-10
        @test var(er) ≈ var(ph) atol=1e-10
        for x in [0.1, 0.5, 1.0, 2.0]
            @test pdf(er, x) ≈ pdf(ph, x) atol=1e-10
            @test cdf(er, x) ≈ cdf(ph, x) atol=1e-10
            @test ccdf(er, x) ≈ ccdf(ph, x) atol=1e-10
        end
    end

    @testset "pdf numerical stability at large shape" begin
        # The old direct formula overflows here; logpdf/pdf via logpdf stays finite.
        er_big = ErlangPHDist(50, 1.0)
        @test isfinite(logpdf(er_big, 50.0))
        @test isfinite(pdf(er_big, 50.0))
        @test pdf(er_big, 50.0) > 0
    end

    @testset "random sampling" begin
        samples = [rand(er) for _ in 1:10_000]
        @test isapprox(Statistics.mean(samples), 1.5; atol=0.05)
    end

    @testset "quantile/median" begin
        d_erlang = Erlang(3, 0.5)
        for p in [0.1, 0.5, 0.9]
            @test quantile(er, p) ≈ quantile(d_erlang, p) atol=1e-6
        end
    end

    @testset "moments and mgf" begin
        @test kth_moment(er, 1) ≈ 1.5 atol=1e-10
        @test mgf(er, 0.5) ≈ (2.0/1.5)^3 atol=1e-10
    end

    @testset "params" begin
        shape, rate = params(er)
        @test shape == 3
        @test rate == 2.0
    end

    @testset "show" begin
        s = sprint(show, er)
        @test occursin("ErlangPHDist", s)
        @test occursin("shape", s)
        @test occursin("rate", s)
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError ErlangPHDist(0, 1.0)
        @test_throws ArgumentError ErlangPHDist(-1, 1.0)
        @test_throws ArgumentError ErlangPHDist(3, 0.0)
        @test_throws ArgumentError ErlangPHDist(3, -1.0)
    end
end

@testset "CoxianDist" begin
    # 3-phase Coxian: rates [3, 4, 5], exit probs [0.2, 0.3] (last phase always absorbs)
    cox = CoxianDist([3.0, 4.0, 5.0], [0.2, 0.3])

    @testset "basic properties" begin
        @test nphases(cox) == 3
        @test initial_prob(cox) == [1.0, 0.0, 0.0]
        t0 = exit_rates(cox)
        @test t0[1] ≈ 3.0 * 0.2
        @test t0[2] ≈ 4.0 * 0.3
        @test t0[3] ≈ 5.0
    end

    @testset "matches general PHDist" begin
        ph = PHDist(cox)
        @test mean(cox) ≈ mean(ph) atol=1e-10
        @test var(cox) ≈ var(ph) atol=1e-10
        for x in [0.01, 0.1, 0.5, 1.0, 2.0]
            @test pdf(cox, x) ≈ pdf(ph, x) atol=1e-8
            @test cdf(cox, x) ≈ cdf(ph, x) atol=1e-8
            @test ccdf(cox, x) ≈ ccdf(ph, x) atol=1e-8
        end
        @test skewness(cox) ≈ skewness(ph) atol=1e-6
        @test kurtosis(cox) ≈ kurtosis(ph) atol=1e-6
    end

    @testset "Coxian with exit_probs=0 is hypoexponential" begin
        cox_hypo = CoxianDist([3.0, 5.0], [0.0])
        ho = HypoExponentialDist([3.0, 5.0])
        @test mean(cox_hypo) ≈ mean(ho) atol=1e-10
        @test var(cox_hypo) ≈ var(ho) atol=1e-10
        for x in [0.1, 0.5, 1.0]
            @test cdf(cox_hypo, x) ≈ cdf(ho, x) atol=1e-8
            @test ccdf(cox_hypo, x) ≈ ccdf(ho, x) atol=1e-8
        end
    end

    @testset "single-phase Coxian = Exponential" begin
        cox1 = CoxianDist(2.0)
        @test mean(cox1) ≈ 0.5 atol=1e-10
        @test var(cox1) ≈ 0.25 atol=1e-10
    end

    @testset "quantile/median" begin
        for p in [0.1, 0.5, 0.9]
            q = quantile(cox, p)
            @test cdf(cox, q) ≈ p atol=1e-8
        end
    end

    @testset "random sampling" begin
        samples = [rand(cox) for _ in 1:10_000]
        @test isapprox(Statistics.mean(samples), mean(cox); atol=0.05)
    end

    @testset "params" begin
        rates, exit_probs = params(cox)
        @test rates == [3.0, 4.0, 5.0]
        @test exit_probs == [0.2, 0.3]
    end

    @testset "show" begin
        s = sprint(show, cox)
        @test occursin("CoxianDist", s)
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError CoxianDist(Float64[], Float64[])
        @test_throws ArgumentError CoxianDist([1.0, -2.0], [0.5])
        @test_throws DimensionMismatch CoxianDist([1.0, 2.0, 3.0], [0.5])   # wrong length
        @test_throws ArgumentError CoxianDist([1.0, 2.0], [1.5])             # exit prob > 1
        @test_throws ArgumentError CoxianDist([1.0, 2.0], [-0.1])            # exit prob < 0
    end
end

# Sampling-quality test: empirical CDF at several points should match the true CDF.
@testset "Sampling quality — empirical vs true CDF" begin
    Random.seed!(42)
    N = 50_000
    tol = 5 / sqrt(N)  # ~DKW bound, loose

    for d in [
        HyperExponentialDist([0.4, 0.6], [2.0, 5.0]),
        HypoExponentialDist([3.0, 5.0]),
        ErlangPHDist(3, 2.0),
        CoxianDist([3.0, 4.0, 5.0], [0.2, 0.3]),
        PHDist([0.6, 0.4], [-3.0 1.0; 0.0 -2.0]),
    ]
        samples = rand(d, N)
        for x in [quantile(d, 0.1), quantile(d, 0.5), quantile(d, 0.9)]
            emp = count(≤(x), samples) / N
            @test abs(emp - cdf(d, x)) < tol
        end
    end
end
