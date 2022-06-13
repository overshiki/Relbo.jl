using Relbo 
using Relbo: train, eval_elbo




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


# create some data with 6 observed heads and 4 observed tails
data = []
for _ in 1:6
    push!(data, 1.0)
end
for _ in 1:4
    push!(data, 0.0)
end


# our prior for Beta is a=b=10
a = 10.0 
b = 10.0

# initially, we guess ga=gb=15
ga = 15.0
gb = 15.0

# during variational inference, for each data point, we draw 1000 samples
# sampling_size = 1000
sampling_size = 10000


# currently, we do sampling using repeat experiment idea
data = cat([data for _ in 1:sampling_size]..., dims=1)
g = GD(elbo, :data, true, 5e-4; ga=ga, gb=gb, a=a, b=b)

# we train for 500 steps
n_epochs = 100
for i in 1:50
    train(g, data, n_epochs, verbose=false)
    elbo_value = eval_elbo(g, data)
    @show i, i*n_epochs, elbo_value
end

alpha_q = sqrt(g.param_dict[:ga].data^2)
beta_q = sqrt(g.param_dict[:gb].data^2)

# here we use some facts about the Beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * sqrt(factor)

report_str = "Based on the data and our prior belief, the fairness of the coin is $inferred_mean +- $inferred_std"
println(report_str)