
mutable struct GD
    grad_func::Function
    param_dict::Dict
end

function GD(grad_elbo_sampler::ExprTerm, input_var; initi_params...)
    param_dict = Dict(initi_params)
    code = cgen(grad_elbo_sampler)
    grad_func = sampling_fun_generator(code, input_var, false; param_dict...)
    return GD(grad_func, param_dict)
end


function train(g::GD, input, num_epochs; lr=1e-2, verbose=false)
    for _ in 1:num_epochs
        params = [Ref(x) for x in values(g.param_dict)]
        grad = sum(g.grad_func.(input, params...))
        for (i,k) in enumerate(keys(g.param_dict))
            if verbose
                @show g.param_dict[k], grad[i]
            end
            g.param_dict[k] -= grad[i] * lr
        end
    end
    @show g.param_dict

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