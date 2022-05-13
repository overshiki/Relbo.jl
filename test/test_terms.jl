using Relbo 
using Relbo: Index, Atom, AtomOperation, Dist, Param, FunctorOperation, Integral, ExprTerm, ParamOperation, sampling_fun_generator, global_expr_dict
using AbstractTrees

i, j, k, l = map(Index, [:i, :j, :k, :l])

# latent variable z
z = Atom(AtomOperation(:z, :dist_sampling, Dist(:Beta)), [Param(:a, i), Param(:b, i)])


data = Atom(AtomOperation(:q, :observe, FunctorOperation(:data)), Param(:data, i))
q = Atom(AtomOperation(:q, :observe, Dist(:InverseGaussian)), z)
obsq = ExprTerm(FunctorOperation(:observe), [q, data])

guide = Atom(AtomOperation(:z, :distribution, Dist(:Beta)), [Param(:ga, i), Param(:gb, i)])
elbo = Integral(:z, guide, obsq)
grad_elbo = ExprTerm(ParamOperation(:grad, [:ga, :gb]), elbo)


sf_grad_elbo = sf_grad_rule(grad_elbo)
print_tree(sf_grad_elbo |> to_tree; maxdepth=20)


sf_grad_elbo_eval = integral2sampler_rule(sf_grad_elbo)
print_tree(sf_grad_elbo_eval |> to_tree; maxdepth=20)

code = cgen(sf_grad_elbo_eval)
println(code)
display(global_expr_dict)

var = :data

# grad_func = sampling_fun_generator(code, var, true; ga=12, gb=4)
grad_func = sampling_fun_generator(code, [var, :ga, :gb, :a, :b], true)

ga = 12
gb = 4
a = 10 
b = 11

data = rand(100) 
@time grad_func.(data, Ref(ga), Ref(gb), Ref(a), Ref(b))

grad = sum(grad_func.(data, Ref(ga), Ref(gb), Ref(a), Ref(b)))
@show size(grad)

println()

