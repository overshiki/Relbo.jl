using Relbo 
using Relbo: train


# create some data with 6 observed heads and 4 observed tails
data = []
for _ in 1:8
    push!(data, 1.0)
end
for _ in 1:2
    push!(data, 0.0)
end

# function square(x)
#     return (x+1e-3)^2
# end

elbo = @ELBO ga, gb begin 
    (i, j, k, l)::Index
    data::Observe(q)
    (a, b)::Param 
    # (ga, gb)::Param(exp)

    a | i 
    b | i
    data | i
    ga | i 
    gb | i 

    # prior is a Beta distribution
    z ~ ConstrainedBeta(a, b)
    # likelihood is a Bernoulli
    q ~ Bernoulli(z)
    # binding likelihood q and data
    obsq = q(data)

    # guide is also a Beta distribution, this makes sence since Beta and Bernoulli are conjugate, generating a Beta posterior
    # here we use ≈ symbol to indicate it is guide for posterior of z
    guide ~ ConstrainedBeta(ga, gb) ≈ z

    return Expectation(guide, obsq)

end

# our prior for Beta is a=b=10
a = 10.0 
b = 10.0

# initially, we guess ga=gb=15
ga = 15.0
gb = 5.0

# during variational inference, for each data point, we draw 1000 samples
sampling_size = 1000


# currently, we do sampling using repeat experiment idea
data = cat([data for _ in 1:sampling_size]..., dims=1)
g = GD(elbo, :data, true; ga=ga, gb=gb, a=a, b=b)

# we train for 500 steps
n_epochs = 5
train(g::GD, data, n_epochs; verbose=true, is_distributed=true)