using RecipesBase
using Statistics

@recipe function f(spots::AbstractVector{Spot})
    nday = [s.nday for s in spots]
    lat = [s.lat for s in spots]
    Φmax = [s.Φmax for s in spots]

    seriestype --> :scatter
    xlabel --> "Time (days)"
    ylabel --> "Latitude (deg)"
    markersize --> 3Φmax ./ median(Φmax)(spots.Φmax)
    alpha --> 0.5
    color --> "#996699"
    markerstrokealpha --> 0

    nday, rad2deg.(lat)
end

@recipe function f(r::SpotDynamics)
    spots = r.spots
    nday = [s.nday for s in spots]
    lat = [s.lat for s in spots]
    Φmax = [s.Φmax for s in spots]

    seriestype --> :scatter
    xlabel --> "Time (days)"
    ylabel --> "Latitude (deg)"
    markersize --> 3Φmax ./ median(Φmax)
    alpha --> 0.5
    color --> "#996699"
    markerstrokealpha --> 0

    nday, rad2deg.(lat)
end