# using Relbo
# using Relbo: train
# using Relbo: Index, Atom, AtomOperation, Dist, Param, FunctorOperation, Integral, ExprTerm, ParamOperation, sampling_fun_generator, global_expr_dict
# using Relbo: arguments, get_symbol, integral2dist_sampling_rewrite_rule, Elbo_container
using Distributed
# @everywhere using Distributions
# @everywhere function myBeta(a, b)
#     # a, b = log(1+exp(a)), log(1+exp(b))
#     a, b = sqrt(a^2), sqrt(b^2)
#     # @show a, b
#     beta = Beta(a, b)
#     # @show beta
#     return beta
# end

@everywhere include("../src/distribution.jl")
@everywhere include("../src/term.jl")
@everywhere include("../src/pattern_match.jl")
@everywhere include("../src/rewrite.jl")
@everywhere include("../src/dsl.jl")
@everywhere include("../src/term2latex.jl")
@everywhere include("../src/codegen.jl")
@everywhere include("../src/to_tree.jl")
@everywhere include("../src/gradient_descent.jl")

i, j, k, l = map(Index, [:i, :j, :k, :l])

data = Atom(AtomOperation(:q, :observe, FunctorOperation(:data)), Param(:data, i))

# latent variable z
z = Atom(AtomOperation(:z, :dist_sampling, Dist(:ConstrainedBeta)), [Param(:a, i), Param(:b, i)])
q = Atom(AtomOperation(:q, :observe, Dist(:Bernoulli)), z)
obsq = ExprTerm(FunctorOperation(:observe), [q, data])

guide = Atom(AtomOperation(:z, :distribution, Dist(:ConstrainedBeta)), [Param(:ga, i), Param(:gb, i)])

# guide_s = ExprTerm(FunctorOperation(:sampling), guide)

# elbo = ExprTerm(ParamOperation(:integral, :z), [guide, obsq])
elbo = Integral(:z, guide, obsq)



""" generate real elbo """
guide, terms = arguments(elbo)[1], arguments(elbo)[2:end]
if length(terms)==1
    term = terms[1]
else 
    term = ExprTerm(FunctorOperation(:*), terms)
end

term = ExprTerm(FunctorOperation(:log), term)
tguide = ExprTerm(FunctorOperation(:log), guide)
term = ExprTerm(FunctorOperation(:-), [term, tguide])

elbo = Integral(get_symbol(guide), guide, term)
""" generate real elbo end """


grad_elbo = ExprTerm(ParamOperation(:grad, [:ga, :gb]), elbo)



# grad_elbo = integral2dist_sampling_rewrite_rule(grad_elbo)
# elbo = integral2dist_sampling_rewrite_rule(elbo)
# println()
elbo_c = Elbo_container(elbo, grad_elbo, [:ga, :gb])


# our prior for Beta is a=b=10
a = 10.0 
b = 10.0

# initially, we guess ga=gb=15
ga = 15.0
gb = 5.0

# during variational inference, for each data point, we draw 1000 samples
# sampling_size = 1000
sampling_size = 5000



# create some data with 6 observed heads and 4 observed tails
data = []
for _ in 1:8
    push!(data, 1.0)
end
for _ in 1:2
    push!(data, 0.0)
end

# currently, we do sampling using repeat experiment idea
data = cat([data for _ in 1:sampling_size]..., dims=1)
g = GD(elbo_c, :data, true, 5e-3; ga=ga, gb=gb, a=a, b=b)

# we train for 500 steps
n_epochs = 100
for i in 1:50
    train(g, data, n_epochs, verbose=false)
    elbo_value = eval(g, data)
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