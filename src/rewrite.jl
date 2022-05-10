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

TermInterface.similarterm(::ExprTerm, head::Operation, args::Vector{<:Term}; exprhead=nothing) = ExprTerm(head, args)

TermInterface.similarterm(::Atom, head::AtomOperation, args::Vector{<:Union{Param, Atom, Term}}; exprhead=nothing) = Atom(head, args)

TermInterface.similarterm(::Param, head::ParamHead, args::Vector; exprhead=nothing) = Param(head, args)



# function similarterm(x, head, args, symtype = nothing; metadata = nothing, exprhead = nothing)
#     !istree(x) ? head : head(args...)
# end

using Metatheory: Prewalk, Postwalk, PassThrough, Fixpoint


get_op(x::ExprTerm) = x.op
get_args(x::ExprTerm) = x.args
get_args_length(x::ExprTerm) = length(x.args)

is_single_arg(x::ExprTerm) = get_args_length(x)==1

function get_single_arg(x::ExprTerm)
    @assert is_single_arg(x)
    return get_args(x)[1]
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
    grad_op = x.op
    @assert is_single_arg(x)
    elbo = get_single_arg(x)

    guide = elbo.args[1]

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
    # println(x)
    # println("==============>")
    guide, terms = x.args[1], x.args[2:end]
    # symbol = guide.symbol
    symbol = get_symbol(guide)

    # @show guide isa Atom

    function sampling_from_guide(guide::Atom)
        g = copy(guide)
        symbol = get_symbol(g)
        # g.type = :sampling
        g = change_type(g, :sampling)

        param = Param(ParamHead(symbol, nothing), [g])
        data = Atom(AtomOperation(symbol, :observe, FunctorOperation(:data)), param)
        return data
    end

    function is_rewrite(a::Atom)
        # println("is_rewrite")
        return get_symbol(a) == get_symbol(guide) && get_type(a)==:distribution
    end

    function rewrite_obs(q::Atom)
        data = sampling_from_guide(guide)
        # q.type = :observe
        q = change_type(q, :observe)
        obsq = ExprTerm(FunctorOperation(:observe), [q, data])
        return obsq
    end

    function rewrite_obs_2expr(q::Atom)
        # println("here")
        obsq = rewrite_obs(q)
        # println(obsq)
        # println("++++++++++++")
        return :($obsq)
    end 

    r = @rule q q::Atom => rewrite_obs_2expr(q) where is_rewrite(q)
    r = Postwalk(PassThrough(r))

    if length(terms)==1
        term = terms[1]
    else 
        term = ExprTerm(FunctorOperation(:*), terms)
    end

    term = r(term)
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



