# --- Comparison helpers for non-identifiable PH distributions ---

"""
    moment_vector(d::AbstractPHDist, order::Int) -> Vector{Float64}

Compute a vector of raw moments [E[X], E[X²], ..., E[X^order]].
"""
function moment_vector(d::AbstractPHDist, order::Int)
    order >= 1 || throw(ArgumentError("order must be >= 1"))
    return [kth_moment(d, k) for k in 1:order]
end

"""
    moments_isapprox(d1::AbstractPHDist, d2::AbstractPHDist;
                     order=4, atol=1e-6, rtol=1e-6) -> Bool

Compare two PH distributions by their moments up to the given order.
Returns true if all moments (and derived quantities: mean, var, scv) are approximately equal.

Note: Two distributions can match in moments but still differ in shape (e.g., in the tails).
Moment matching is necessary but not sufficient for distributional equality.
"""
function moments_isapprox(d1::AbstractPHDist, d2::AbstractPHDist;
                          order::Int=4, atol::Real=1e-6, rtol::Real=1e-6)
    m1 = moment_vector(d1, order)
    m2 = moment_vector(d2, order)
    return all(isapprox.(m1, m2; atol=atol, rtol=rtol))
end

"""
    distribution_isapprox(d1::AbstractPHDist, d2::AbstractPHDist;
                          n_points=100, atol=1e-6) -> Bool

Compare two PH distributions by their CDF values on an adaptive grid.
This is a stronger test than moment comparison — it checks the full distributional shape.

The grid spans from 0 to a point well into the tail, chosen based on the mean and
standard deviation of both distributions.
"""
function distribution_isapprox(d1::AbstractPHDist, d2::AbstractPHDist;
                                n_points::Int=100, atol::Real=1e-6)
    # Build adaptive grid based on both distributions
    μ1, μ2 = mean(d1), mean(d2)
    σ1, σ2 = sqrt(var(d1)), sqrt(var(d2))
    x_max = max(μ1 + 5σ1, μ2 + 5σ2)
    x_min = 0.0
    xs = range(x_min, x_max; length=n_points)

    for x in xs
        if !isapprox(cdf(d1, x), cdf(d2, x); atol=atol)
            return false
        end
    end
    return true
end
