model_size(ph::PHDist) = length(ph.α)
model_size(maph::MAPHDist) = (m = size(maph.T, 1), n = size(maph.D, 2)) #transient is m and abosrbing is n


absorption_probs(maph::MAPHDist) = maph.α' *maph.R

#Conditional moment moment conditional on absorbing
kth_moment(maph::MAPHDist, k::Real) = (-1.0)^(k+1) * factorial(k) .* maph.α' * inv(maph.T)^(k+1) * maph.D ./ absorption_probs(maph)
mean(maph::MAPHDist) = kth_moment(maph, 1)
var(maph::MAPHDist) = kth_moment(maph, 2) - mean(maph).^2
scv(maph::MAPHDist) = var(maph) ./ (mean(maph).^2) 


function sub_distribution(maph::MAPHDist, k::Real, xs::Vector{Float64})
    return reduce(vcat, map(x ->  maph.α * (exp(x * maph.T) - I) * inv(maph.T) * maph.D[:,k], xs))
end

function sub_pdf(maph::MAPHDist, k::Real, xs::Vector{Float64})
    return  reduce(vcat, map(x ->  maph.α * exp(maph.T*x) * maph.D[:, k], xs))
end

function mgf(maph::MAPHDist, k::Real, zs::Vector{Real})
    
    return reduce(vcat, map(z -> -maph.α * inv(z*I + maph.T) * maph.D[:, k], zs))
end

kth_moment(ph::PHDist, k::Real) = (-1)^k * factorial(k) * ph.α * inv(ph.T)^k * ones(model_size(ph))
mean(ph::PHDist) = kth_moment(ph, 1)
var(ph::PHDist) = kth_moment(ph, 2) - mean(ph).^2
scv(ph::PHDist) = var(ph) ./ (mean(ph).^2)
