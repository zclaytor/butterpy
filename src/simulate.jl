using Random
using DataFrames
using Distributions

export generate_simdata

function generate_simdata(Nlc::Integer)
    inclination = asin.(sqrt.(rand(Nlc)))
    ar = 10 .^ rand(Uniform(-2, 1), Nlc)
    clen = 10 .^ rand(Uniform(0, 1.6), Nlc)
    cover = 10 .^ rand(Uniform(-1, 0.5), Nlc)
    θ_low = rand(Uniform(0, 40), Nlc)
    θ_high = rand.(Uniform.(θ_low, 80))
    period = 10 .^ rand(Uniform(-1, 2), Nlc)
    τ_evol = 10 .^ rand(Nlc)
    butterfly = Bool.(rand(Bernoulli(0.8), Nlc))
    Δω = 10 .^ rand(Uniform(-1, 0), Nlc)

    ω = 24.5 ./ period
    Δω .*= ω

    return DataFrame(
        ar = ar,
        clen = clen,
        cover = cover,
        inclination = inclination,
        θ_low = θ_low,
        θ_high = θ_high,
        period = period,
        ω = ω,
        Δω = Δω,
        τ_decay = τ_evol,
        butterfly = butterfly
    )
end