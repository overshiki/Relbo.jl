using TermInterface
TermInterface.istree(::ExprTerm) = true 
TermInterface.exprhead(::ExprTerm) = nothing
TermInterface.operation(e::ExprTerm) = e.op 
TermInterface.arguments(e::ExprTerm) = e.args

TermInterface.istree(::Atom) = true 
TermInterface.exprhead(::Atom) = nothing
TermInterface.operation(e::Atom) = e.op 
TermInterface.arguments(e::Atom) = e.params

TermInterface.istree(::Param) = true 
TermInterface.exprhead(::Param) = nothing
TermInterface.operation(e::Param) = e.head 
TermInterface.arguments(e::Param) = e.data

TermInterface.istree(::EmptyTerm) = true 
TermInterface.exprhead(::EmptyTerm) = nothing
TermInterface.operation(e::EmptyTerm) = e.head 
TermInterface.arguments(e::EmptyTerm) = e.data

TermInterface.similarterm(::ExprTerm, head::Operation, args::Vector{<:Term}; exprhead=nothing) = ExprTerm(head, args)

TermInterface.similarterm(::Atom, head::AtomOperation, args::Vector{<:Union{Param, Atom, Term}}; exprhead=nothing) = Atom(head, args)

TermInterface.similarterm(::Param, head::ParamHead, args::Vector; exprhead=nothing) = Param(head, args)

TermInterface.similarterm(::EmptyTerm, head, args; exprhead=nothing) = EmptyTerm()



# function similarterm(x, head, args, symtype = nothing; metadata = nothing, exprhead = nothing)
#     !istree(x) ? head : head(args...)
# end

using Metatheory: Prewalk, Postwalk, PassThrough, Fixpoint


# get_op(x::ExprTerm) = x.op
# get_args(x::ExprTerm) = x.args
get_args_length(x::ExprTerm) = length(x.args)
is_single_arg(x::ExprTerm) = get_args_length(x)==1

function get_single_arg(x::ExprTerm)
    @assert is_single_arg(x)
    return arguments(x)[1]
end 

function is_grad_operation(o::T) where {T<:Operation}
    return o isa ParamOperation && o.symbol==:grad
end
is_grad_operation(e::ExprTerm) = is_grad_operation(e.op)

function is_integral_operation(o::T) where {T<:Operation}
    return o isa ParamOperation && o.symbol==:integral
end
is_integral_operation(e::ExprTerm) = is_integral_operation(e.op)

function is_sf_grad(x::ExprTerm)
    if is_grad_operation(x)
        if is_single_arg(x)
            obj = get_single_arg(x)
            if obj isa ExprTerm
                if is_integral_operation(obj)
                    return true 
                end 
            end 
        end
    end 

    return false
end


function sf_estimator(x::ExprTerm)
    grad_op = operation(x)
    @assert is_single_arg(x)
    elbo = get_single_arg(x)

    guide = arguments(elbo)[1]

    nelbo, nguide = copy(elbo), copy(guide)
    # nelbo = similarterm(elbo, operation(elbo), arguments(elbo))
    # nguide = similarterm(guide, operation(guide), arguments(guide))

    log_guide = ExprTerm(FunctorOperation(:log), nguide)
    grad_log_guide = ExprTerm(grad_op, log_guide)
    push!(nelbo.args, grad_log_guide)
    return nelbo
end

function sf_estimator_2expr(x::ExprTerm)
    elbo = sf_estimator(x)
    return :($elbo)
end


function integral2sampler_2expr(x::ExprTerm)
    guide, terms = arguments(x)[1], arguments(x)[2:end]
    symbol = get_symbol(guide)

    if length(terms)==1
        term = terms[1]
    else 
        term = ExprTerm(FunctorOperation(:*), terms)
    end


    function sampling_from_guide(guide::Atom)
        g = copy(guide)
        symbol = get_symbol(g)
        # g.type = :sampling
        g = change_type(g, :sampling)

        param = Param(ParamHead(symbol, nothing), [g])
        data = Atom(AtomOperation(symbol, :observe, FunctorOperation(:data)), param)
        return data
    end

    """ rewrite distribution"""
    function is_rewrite_dist(a::Atom)
        # println("is_rewrite")
        return get_symbol(a) == get_symbol(guide) && get_type(a)==:distribution
    end

    function rewrite_dist_obs(q::Atom)
        data = sampling_from_guide(guide)
        # q.type = :observe
        q = change_type(q, :observe)
        obsq = ExprTerm(FunctorOperation(:observe), [q, data])
        return obsq
    end

    function rewrite_dist_obs_2expr(q::Atom)
        obsq = rewrite_dist_obs(q)
        return :($obsq)
    end 

    r1 = @rule q q::Atom => rewrite_dist_obs_2expr(q) where is_rewrite_dist(q)
    r1 = Postwalk(PassThrough(r1))


    """ rewrite sampling"""
    function is_rewrite_sampling(a::Atom)
        # println("is_rewrite")
        return get_symbol(a) == get_symbol(guide) && get_type(a)==:dist_sampling
    end

    function rewrite_sampling_obs(q::Atom)
        data = sampling_from_guide(guide)
        # q.type = :observe
        q = change_type(q, :observe)
        obsq = ExprTerm(FunctorOperation(:passing_and_observe), [q, data])
        return obsq
    end

    function rewrite_sampling_obs_2expr(q::Atom)
        obsq = rewrite_sampling_obs(q)
        return :($obsq)
    end 

    r2 = @rule q q::Atom => rewrite_sampling_obs_2expr(q) where is_rewrite_sampling(q)
    r2 = Postwalk(PassThrough(r2))

    #TODO: try RefTerm and Term_dict idea

    function is_has_passing_and_observe(q::Term)
        for arg in arguments(q)
            if arg isa ExprTerm && operation(arg).symbol == :passing_and_observe
                return true 
            end 

            if operation(arg).symbol != :observe
                if is_has_passing_and_observe(arg)
                    return true 
                end
            end
        end

        return false
    end

    function is_observe_and_has_passing_and_observe(q::ExprTerm)
        if operation(q).symbol==:observe
            return is_has_passing_and_observe(q)
        end
        return false
    end


    function rewrite_and_return_passing_and_observe(q::Term)
        for (i, arg) in enumerate(arguments(q))
            if arg isa ExprTerm && operation(arg).symbol == :passing_and_observe
                # modify arg recursive, in equivalent to inplace operation
                @assert length(arguments(arg))==2
                sampling_arg = nothing 
                for a in arguments(arg)
                    if a isa Atom && a.op.distribution==FunctorOperation(:data)
                        @assert length(arguments(a))==1
                        param = arguments(a)[1]
                        @assert param isa Param 
                        @assert length(arguments(param))==1
                        sampling_arg = arguments(param)[1]
                        @assert sampling_arg isa Atom 
                    end 
                end 
                @assert sampling_arg isa Atom

                # nqargs = arguments(q)
                # nqargs = Vector{<:Union{Param, Atom, Term}}(copy(nqargs))
                # nqargs[i] = sampling_arg
                nqargs = Term[]
                for (ii, _nqarg) in enumerate(arguments(q))
                    if i==ii 
                        push!(nqargs, sampling_arg)
                    else 
                        push!(nqargs, _nqarg)
                    end
                end

                nq = similarterm(q, operation(q), nqargs)



                # arg_return = copy(arg)

                # op_symbol = operation(arg).symbol 
                # operation(arg_return).symbol = :observe
                op = FunctorOperation(:observe)
                arg_return = ExprTerm(op, arguments(arg))
                return arg_return, nq
            end 

            if operation(arg).symbol != :observe
                term, sampling_arg = rewrite_and_return_passing_and_observe(arg)
                if term!==nothing 
                    # modify arg recursive, in equivalent to inplace operation
                    # nqargs = arguments(q)
                    # nqargs[i] = sampling_arg

                    nqargs = Term[]
                    for (ii, _nqarg) in enumerate(arguments(q))
                        if i==ii 
                            push!(nqargs, sampling_arg)
                        else 
                            push!(nqargs, _nqarg)
                        end
                    end

                    nq = similarterm(q, operation(q), nqargs)

                    return term, nq
                end
            end
        end

        return nothing, nothing
    end



    function rewrite_passing_and_observe_2expr(q::ExprTerm)
        term, nq = rewrite_and_return_passing_and_observe(q)
        @assert term!==nothing
        term_return = ExprTerm(FunctorOperation(:*), [nq, term])
        return :($term_return)
    end

    r3 = @rule q q::ExprTerm => rewrite_passing_and_observe_2expr(q) where is_observe_and_has_passing_and_observe(q)
    r3 = Postwalk(PassThrough(r3))

    term = term |> r1 |> r2 |> r3
    return :($term)


end

integral2sampler_rule = @rule x x::ExprTerm => integral2sampler_2expr(x) where is_integral_operation(x)
integral2sampler_rule = Postwalk(PassThrough(integral2sampler_rule))














function is_grad_prod(x::ExprTerm)
    if is_grad_operation(x)
        @assert length(x.args)==1
        expr = x.args[1] 
        operation = expr.op 
        return operation.symbol == :* 
    end
    return false
end

function grad_passthrough(x::ExprTerm, ::Val{:*})
    grad_op = x.op 
    @assert is_single_arg(x)
    expr = get_single_arg(x)

    prod_terms = expr.args

    sum_terms = Term[]
    for i in 1:length(prod_terms)
        g_prod_terms = Term[]
        for j in 1:length(prod_terms)
            if i==j
                t = ExprTerm(grad_op, prod_terms[j])
            else 
                t = prod_terms[j]
            end 
            push!(g_prod_terms, t)
        end
        g_prod = ExprTerm(FunctorOperation(:*), g_prod_terms)
    end
    g_sum = ExprTerm(FunctorOperation(:+), sum_terms)
    return g_sum 
end

function grad_passthrough_2expr(x::ExprTerm, ::Val{:*})
    g_sum = grad_passthrough(x, Val(:*))
    return :($g_sum)
end


sf_grad_rule = @rule x x::ExprTerm => sf_estimator_2expr(x) where is_sf_grad(x)
passthrough_grad_rule = @rule x x::ExprTerm => grad_passthrough_2expr(x, Val(:*)) where is_grad_prod(x)

sf_grad_rule = Postwalk(PassThrough(sf_grad_rule))
passthrough_grad_rule = Postwalk(PassThrough(passthrough_grad_rule))





function integral2dist_sampling_rewrite_2expr(x::ExprTerm)
    guide, terms = arguments(x)[1], arguments(x)[2:end]
    @assert guide isa Atom
    guide_symbol = get_symbol(guide)

    if length(terms)==1
        term = terms[1]
    else 
        term = ExprTerm(FunctorOperation(:*), terms)
    end


    function is_dist_sampling_rewriting(x::Atom)#, guide_symbol::Symbol)

        if get_dist_op(x) isa Dist #&& operation(x).type==:observe
            for arg in arguments(x)
                # x is a Atom with Dist type, and x has a param that is a Atom with Dist type
                if arg isa Atom
                    if get_symbol(arg)==guide_symbol
                        if get_dist_op(arg) isa Dist && operation(arg).type==:distribution
                            return true 
                        end 
                    end 
                end 
            end 
        end 
        return false
    end


    function dist_sampling_rewrite_2expr(x::Atom)#, guide_symbol::Symbol)
        # if get_dist_op(x) isa Dist #&& operation(x).type==:observe
        narg_dict = Dict()
        for (index, arg) in enumerate(arguments(x))
            # x is a Atom with Dist type, and x has a param that is a Atom with Dist type
            if arg isa Atom
                if get_symbol(arg)==guide_symbol
                    if get_dist_op(arg) isa Dist && operation(arg).type==:distribution
                        op = operation(arg)
                        nop = AtomOperation(op.symbol, :dist_sampling, op.distribution)
                        narg = similarterm(arg, nop, arguments(arg))
                        narg_dict[index] = narg
                    end 
                end 
            end 
        end 


        nargs = Term[]
        for (index, arg) in enumerate(arguments(x))
            if index in keys(narg_dict)
                push!(nargs, narg_dict[index])
            else 
                push!(nargs, arg)
            end
        end
        nx = similarterm(x, operation(x), nargs)
        return :($nx)
    end

    # r = @rule q q::Atom => dist_sampling_rewrite_2expr(q, guide_symbol) where is_dist_sampling_rewriting(q, guide_symbol)
    r = @rule q q::Atom => dist_sampling_rewrite_2expr(q) where is_dist_sampling_rewriting(q)
    r = Postwalk(PassThrough(r))

    term = r(term)

    re_integral = Integral(guide_symbol, guide, term)
    return :($re_integral)

end

integral2dist_sampling_rewrite_rule = @rule x x::ExprTerm => integral2dist_sampling_rewrite_2expr(x) where is_integral_operation(x)
integral2dist_sampling_rewrite_rule = Postwalk(PassThrough(integral2dist_sampling_rewrite_rule))