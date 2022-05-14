using Relbo 
using Relbo: train


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

    z ~ ConstrainedBeta(a, b)
    q ~ InverseGaussian(z)
    obsq = q(data)
    guide ~ ConstrainedBeta(ga, gb) ≈ z

    
    # return 𝔼_guide(obsq)
    return Expectation(guide, obsq)

end

# # @show grad_elbo
# sf_grad_elbo_eval = grad_elbo |> sf_grad_rule |> integral2sampler_rule #|> cgen

# # grad_func = sampling_fun_generator(code, var, true; ga=12, gb=4)

# data = rand(100) 
# g = GD(sf_grad_elbo_eval, :data; ga=12.0, gb=4.0)
# train(g::GD, data, 10)

data = ones(100)
g = GD(elbo, :data, true; ga=12.0, gb=4.0, a=10.0, b=3.0)
train(g::GD, data, 10)