using RecipesBase
using Statistics

@recipe function f(spots::Vector{Spot})
    seriestype --> :scatter
    xlabel --> "Time (days)"
    ylabel --> "Latitude (deg)"
    size --> 10 .* [s.Φmax for s in spots] ./ median([s.Φmax for s in spots])
    alpha --> 0.5
    color --> "#996699"
    markerstrokealpha --> 0

    [s.nday for s in spots], rad2deg.([s.lat for s in spots])
end

@recipe function f(spots::Spots)
    spots = spots.spots

    seriestype --> :scatter
    xlabel --> "Time (days)"
    ylabel --> "Latitude (deg)"
    size --> 10 .* [s.Φmax for s in spots] ./ median([s.Φmax for s in spots])
    alpha --> 0.5
    color --> "#996699"
    markerstrokealpha --> 0

    [s.nday for s in spots], rad2deg.([s.lat for s in spots])
end