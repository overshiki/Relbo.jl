using Flux
# using LoopVectorization
# using ThreadsX
using Distributed

mutable struct Input_params 
    symbol::Symbol
    data
    is_require_grad::Bool
end

mutable struct GD
    eval_func::Function
    grad_func::Function
    param_dict::Dict
    params_var::Vector{Symbol}
    opt # optimizer
end

# function GD(grad_elbo_sampler::ExprTerm, input_var; initi_params...)
function GD(elbo::Elbo_container, input_var, verbose=false, lr=1e-2; initi_params...)
    opt = ADAM(lr, (0.9, 0.999))

    param_dict = Dict()
    init_param_dict = Dict(initi_params)
    for k in keys(init_param_dict)
        is_require_grad = k in elbo.grad_vars
        param_dict[k] = Input_params(k, init_param_dict[k], is_require_grad)
    end


    params = [input_var, ]
    params_var = []
    for k in keys(initi_params)
        push!(params, k)
        push!(params_var, k)
    end

    sf_grad_elbo_eval = elbo.grad_elbo |> sf_grad_rule |> integral2sampler_rule
    code = cgen(sf_grad_elbo_eval)
    grad_func = sampling_fun_generator(code, params, verbose)

    eval_code = elbo.elbo |> sf_grad_rule |> integral2sampler_rule |> cgen
    eval_func = sampling_fun_generator(eval_code, params, verbose)

    return GD(eval_func, grad_func, param_dict, params_var, opt)
end

function get_params(g::GD)
    params = []
    for sym in g.params_var
        if sym in keys(g.param_dict)
            x = g.param_dict[sym]
            # push!(params, Ref(x.data))
            push!(params, x.data)                
        end
    end
    return params
end

function get_grad_params(g::GD)
    params = Float32[]
    for sym in g.params_var
        if sym in keys(g.param_dict)
            x = g.param_dict[sym]
            if x.is_require_grad
                push!(params, x.data)                
            end
        end
    end
    return params
end


function broadcast_sum(func, input, params)
    params = [Ref(x) for x in params]
    return sum(func.(input, params...))
end


function distributed_sum(func, input, params)
    c = @distributed (+) for i = 1:length(input)
        func(input[i], params...)
    end
    return c
end


function train(g::GD, input, num_epochs; verbose=false, is_distributed=false)
    if verbose
        println("initiate elbo: ", eval_elbo(g, input))
        display(g.param_dict)
    end
    # opt = ADAM(lr, (0.9, 0.8))
    for epoch in 1:num_epochs
        params = get_params(g)

        if is_distributed
            grad = distributed_sum(g.grad_func, input, params)
        else 
            grad = broadcast_sum(g.grad_func, input, params)
        end

        grad_params = get_grad_params(g)

        # grad.*-1 because we want gradient ascent
        Flux.Optimise.update!(g.opt, grad_params, grad.*-1)
        
        # @show grad_params

        i = 0
        for k in keys(g.param_dict)
            if g.param_dict[k].is_require_grad
                i += 1
                g.param_dict[k].data = grad_params[i]

            end

            # if verbose
            #     @show g.param_dict[k], grad[i]
            # end
        end
    end
    if verbose
        println("final elbo: ", eval_elbo(g, input))
        display(g.param_dict)
    end

end

function eval_elbo(g::GD, input; is_distributed=false)
    # params = [Ref(x.data) for x in values(g.param_dict)]
    params = get_params(g)

    # elbo = sum(g.eval_func.(input, params...))
    if is_distributed
        elbo = distributed_sum(g.eval_func, input, params)
    else
        elbo = broadcast_sum(g.eval_func, input, params)
    end

    return elbo
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