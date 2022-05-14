### Relbo.jl: Rewriting ELBO -variational inference using symbolic rewriting techniques

`Relbo.jl` uses expression tree to represent computational graph of variational inference, doing this, symbolic rewriting technique based on `Metatheory.jl` could be used to manipulate the computation procedure at compile time. Rao-Blackwellization rewriting for variance reduction could be easily implemented. It also takes advantages of codegen(meta programming) and automatic differentiation techniques(`Zygote.jl`) to handle the numerical calculation, in an efficient way.

#### Motivations

This project is motivated by projects in JuliaSymbolic comminities(`Metatheory.jl`, `Symbolics.jl`) for symbolic rewriting ideas; `pyro` for the variational inference as PPL idea. Related projects include `Turing.jl` and `Soss.jl`. The difference between `Relbo.jl` and `Turing.jl`|`Soss.jl` is that `Turing.jl`|`Soss.jl` focuses on `HMC` based posterial sampling while `Relbo.jl` focuses on variational inference(currently it is a very small trial project).  

#### Quick show case
The easest way of using `Relbo.jl` is through the dsl and the provided helper functions:  
```julia
using Relbo 
using Relbo: train


elbo = @ELBO ga, gb begin 
    (i, j, k, l)::Index
    data::Observe(q)
    (a, b)::Param 

    a | i 
    b | i
    data | i
    ga | i 
    gb | i 

    z ~ Beta(a, b)
    q ~ InverseGaussian(z)
    obsq = q(data)
    guide ~ Beta(ga, gb) ≈ z

    return Expectation(guide, obsq)
end

data = ones(100)
g = GD(elbo, :data; ga=12.0, gb=4.0, a=10.0, b=3.0)
train(g::GD, data, 10)

```
#### What does the expression tree of `Relbo.jl` looks like?
The expression tree of `sf_grad_elbo_eval` could be ploted using AbstractTrees:
```julia
using Relbo 
using AbstractTrees
print_tree(sf_grad_elbo_eval |> to_tree; maxdepth=20)
```
The result is as below:
```julia
"ExprTerm_*"
└─ Dict{Any, Any}
   ├─ ""
   │  └─ "ExprTerm_*"
   │     └─ Dict{Any, Any}
   │        ├─ ""
   │        │  └─ "ExprTerm_observe"
   │        │     └─ Dict{Any, Any}
   │        │        ├─ ""
   │        │        │  └─ "Atom_q_InverseGaussian_observe"
   │        │        │     └─ ""
   │        │        │        └─ "Atom_z_Beta_sampling"
   │        │        │           └─ Dict{Any, Any}
   │        │        │              ├─ ""
   │        │        │              │  └─ "Param_ga"
   │        │        │              │     └─ ""
   │        │        │              │        └─ "nothing"
   │        │        │              └─ ""
   │        │        │                 └─ "Param_gb"
   │        │        │                    └─ ""
   │        │        │                       └─ "nothing"
   │        │        └─ ""
   │        │           └─ "Atom_q_data_observe"
   │        │              └─ ""
   │        │                 └─ "Param_data"
   │        │                    └─ ""
   │        │                       └─ "nothing"
   │        └─ ""
   │           └─ "ExprTerm_observe"
   │              └─ Dict{Any, Any}
   │                 ├─ ""
   │                 │  └─ "Atom_z_data_observe"
   │                 │     └─ ""
   │                 │        └─ "Param_z"
   │                 │           └─ ""
   │                 │              └─ "Atom_z_Beta_sampling"
   │                 │                 └─ Dict{Any, Any}
   │                 │                    ├─ ""
   │                 │                    │  └─ "Param_gb"
   │                 │                    │     └─ ""
   │                 │                    │        └─ "nothing"
   │                 │                    └─ ""
   │                 │                       └─ "Param_ga"
   │                 │                          └─ ""
   │                 │                             └─ "nothing"
   │                 └─ ""
   │                    └─ "Atom_z_Beta_observe"
   │                       └─ Dict{Any, Any}
   │                          ├─ ""
   │                          │  └─ "Param_a"
   │                          │     └─ ""
   │                          │        └─ "nothing"
   │                          └─ ""
   │                             └─ "Param_b"
   │                                └─ ""
   │                                   └─ "nothing"
   └─ ""
      └─ "ExprTerm_grad"
         └─ ""
            └─ "ExprTerm_log"
               └─ ""
                  └─ "ExprTerm_observe"
                     └─ Dict{Any, Any}
                        ├─ ""
                        │  └─ "Atom_z_Beta_observe"
                        │     └─ Dict{Any, Any}
                        │        ├─ ""
                        │        │  └─ "Param_gb"
                        │        │     └─ ""
                        │        │        └─ "nothing"
                        │        └─ ""
                        │           └─ "Param_ga"
                        │              └─ ""
                        │                 └─ "nothing"
                        └─ ""
                           └─ "Atom_z_data_observe"
                              └─ ""
                                 └─ "Param_z"
                                    └─ ""
                                       └─ "Atom_z_Beta_sampling"
                                          └─ Dict{Any, Any}
                                             ├─ ""
                                             │  └─ "Param_ga"
                                             │     └─ ""
                                             │        └─ "nothing"
                                             └─ ""
                                                └─ "Param_gb"
                                                   └─ ""
                                                      └─ "nothing"

```

#### How does `Relbo.jl` handle the rewriting
It just use the term rewriting functions provided by `Metatheory.jl`. Below shows how to define a `score function estimator rewriter` to transform a gradient over intergal into a score function version, which has smaller variance. 
```julia
function sf_estimator(x::ExprTerm)
    grad_op = x.op
    @assert is_single_arg(x)
    elbo = get_single_arg(x)

    guide = elbo.args[1]

    nelbo, nguide = copy(elbo), copy(guide)

    log_guide = ExprTerm(FunctorOperation(:log), nguide)
    grad_log_guide = ExprTerm(grad_op, log_guide)
    push!(nelbo.args, grad_log_guide)
    return nelbo
end

function sf_estimator_2expr(x::ExprTerm)
    elbo = sf_estimator(x)
    return :($elbo)
end

sf_grad_rule = @rule x x::ExprTerm => sf_estimator_2expr(x) where is_sf_grad(x)
sf_grad_rule = Postwalk(PassThrough(sf_grad_rule))
```
Fow more informations of how to manipulate the expression tree, see `src/rewrite.jl`

#### Code generation
The expression tree of `Relbo.jl` could be easily transformed into runnable code using `cgen` functions provided in `src/codegen.jl`, the code for gradients are generated using `Zygote.jl`. The resulting code is easily broadcastable along batch dimension, which allows it to be scaled up easily.
```julia
code = cgen(sf_grad_elbo_eval)
grad_func = sampling_fun_generator(code, [:data, :ga, :gb, :a, :b], true)

ga = 12
gb = 4
a = 10 
b = 11

data = rand(100) 
@time grad_func.(data, Ref(ga), Ref(gb), Ref(a), Ref(b))

grad = sum(grad_func.(data, Ref(ga), Ref(gb), Ref(a), Ref(b)))
@show size(grad)
```

The generated `grad_func` is as below:
```julia
begin
    (data, ga, gb, a, b)->begin
            begin
                z = Beta(ga, gb)
                z_observe = rand(z)
                var"z##328" = Beta(a, b)
                q = InverseGaussian(z_observe)
            end
            return (pdf(q, data) * pdf(var"z##328", z_observe)) * collect(gradient(((ga, gb)->begin
                                    begin
                                        z = Beta(ga, gb)
                                    end
                                    log(pdf(z, z_observe))
                                end), (ga, gb)...))
        end
end

```

For more information, see `test/test_terms.jl`