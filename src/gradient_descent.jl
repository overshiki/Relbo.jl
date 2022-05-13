

mutable struct Input_params 
    symbol::Symbol
    data
    is_require_grad::Bool
end

mutable struct GD
    eval_func::Function
    grad_func::Function
    param_dict::Dict
end

# function GD(grad_elbo_sampler::ExprTerm, input_var; initi_params...)
function GD(elbo::Elbo_container, input_var; initi_params...)


    param_dict = Dict()
    init_param_dict = Dict(initi_params)
    for k in keys(init_param_dict)
        is_require_grad = k in elbo.grad_vars
        param_dict[k] = Input_params(k, init_param_dict[k], is_require_grad)
    end


    params = [input_var, ]
    for k in keys(initi_params)
        push!(params, k)
    end

    sf_grad_elbo_eval = elbo.grad_elbo |> sf_grad_rule |> integral2sampler_rule
    code = cgen(sf_grad_elbo_eval)
    grad_func = sampling_fun_generator(code, params, false)

    eval_code = elbo.elbo |> sf_grad_rule |> integral2sampler_rule |> cgen
    eval_func = sampling_fun_generator(eval_code, params, false)

    return GD(eval_func, grad_func, param_dict)
end


function train(g::GD, input, num_epochs; lr=1e-2, verbose=false)
    println("initiate elbo: ", eval(g, input))
    for _ in 1:num_epochs
        params = [Ref(x.data) for x in values(g.param_dict)]
        grad = sum(g.grad_func.(input, params...))
        # for (i,k) in enumerate(keys(g.param_dict))
        i = 0
        for k in keys(g.param_dict)
            if g.param_dict[k].is_require_grad
                i += 1
                g.param_dict[k].data -= grad[i] * lr
            end

            # if verbose
            #     @show g.param_dict[k], grad[i]
            # end
        end
    end
    # @show g.param_dict
    println("final elbo: ", eval(g, input))

end

function eval(g::GD, input)
    params = [Ref(x.data) for x in values(g.param_dict)]
    evidence = sum(g.eval_func.(input, params...))
    return evidence
end


# function GD(grad_elbo_sampler, input_var, input; initi_params...)
#     param_dict = Dict(initi_params)
#     code = cgen(grad_elbo_sampler)
#     grad_func = sampling_fun_generator(code, input_var, false; param_dict...)

#     for _ in 1:100
#         params = [Ref(x) for x in values(param_dict)]
#         grad = sum(grad_func.(input, params...))
#         for (i,k) in enumerate(keys(param_dict))
#             param_dict[k] -= grad[i]
#         end
#     end
#     @show param_dict
# end