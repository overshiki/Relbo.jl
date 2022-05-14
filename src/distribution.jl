using Distributions

function ConstrainedBeta(a, b)
    a, b = sqrt(a^2), sqrt(b^2)
    beta = Beta(a, b)
    return beta
end