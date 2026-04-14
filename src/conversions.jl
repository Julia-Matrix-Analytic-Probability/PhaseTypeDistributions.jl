# --- Conversions to general PHDist ---

"""Convert any AbstractPHDist subtype to a general PHDist(α, T) representation."""
PHDist(d::AbstractPHDist) = PHDist(initial_prob(d), subgenerator(d))
PHDist(d::PHDist) = d  # no-op for PHDist itself

"""Convert a Distributions.Exponential to a 1-phase PHDist."""
PHDist(d::Exponential) = PHDist([1.0], [-1.0/Distributions.mean(d);;])

"""Convert a Distributions.Erlang to a k-phase PHDist."""
function PHDist(d::Erlang)
    k = Int(Distributions.shape(d))
    λ = 1.0 / Distributions.scale(d)
    return PHDist(ErlangPHDist(k, λ))
end
