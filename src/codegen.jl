using Zygote
using Distributions

global_expr_dict = Dict() 
global_mapping = Dict()
global_hash_dict = Dict()


function cgen(x::Union{Real, Vector})
    return x
end

function cgen(x::Param)
    @assert length(x.data)==1
    data = x.data[1]
    # if data!==nothing
    if !(data isa EmptyTerm)
        d = cgen(data)
        # if d isa Atom 
        #     # return :(rand($(x.data)))
        #     return cgen(d)
        # else
        #     return d
        # end
        return d
    else 
        return get_symbol(x)
    end
    # global_expr_dict[id] = d
    # return d, id
end

function check_type(x::Atom)
    op = get_dist_op(x)
    if op isa FunctorOperation
        if op.symbol == :data
            return Val(:data)
        end 
    end 

    if op isa Dist && get_type(x)==:observe
        return Val(:observe_dist)
    end 

    if op isa FunctorOperation && get_type(x)==:observe
        return Val(:observe_data)
    end

    if op isa Dist && (get_type(x)==:sampling || get_type(x)==:distribution)
        return Val(:sampling)
    end 
    @show op, get_type(x)
    error()
end

function cgen(x::Atom)
    _type = check_type(x)
    return cgen(x, _type)
end


function cgen(x::Atom, ::Val{:data})
    param = x.params[1]
    return cgen(param)
end

function atom2hash(x::Atom, symbol::Symbol)
    hash_symbol = nothing
    for ak in keys(global_hash_dict)
        if approx_eqaul(x, global_hash_dict[ak])
            hash_symbol = ak 
        end 
    end 
    
    is_has_symbol = symbol in keys(global_hash_dict)
    
    # @show is_has_symbol, symbol, hash_symbol
    # if is_has_symbol
    #     ex = global_hash_dict[symbol]
    #     println(x)
    #     println(ex)
    #     println(approx_eqaul(x.op, ex.op))
    #     println(is_same_set(x.params, ex.params))
    #     print(Set(x.params)==Set(ex.params))
    # end

    # for _ in 1:5 
    #     println()
    # end

    if is_has_symbol
        if hash_symbol==symbol
            "do nothing"
        else
            "hash to a different symbol"
            symbol = Symbol(symbol, gensym())
            global_hash_dict[symbol] = x
        end 
    else 
        if hash_symbol!==nothing 
            symbol = hash_symbol
        else
            "register to global_hash_dict"
            global_hash_dict[symbol] = x
        end
    end
    return symbol
end



"""
for example:
    obsq = ExprTerm(FunctorOperation(:observe), [q, data])
    observe_dist stands for q
        cgen(x::Atom, ::Val{:observe_dist}) should return the distribution generating expr
    observe_data stands for data
        cgen(x::Atom, ::Val{:observe_data}) should return cgen(param), which stands for expr of data sampling or data of Union{Real, Vector} type
"""
function cgen(x::Atom, ::Val{:observe_dist})
    # id = gensym()

    op = get_dist_op(x)
    params = x.params 
    params = map(x->cgen(x), params)
    symbol = op.symbol

    expr = :($(symbol)())
    mapping_vec = []
    for param in params
        push!(expr.args, param)
        if param isa Symbol
            push!(mapping_vec, param)
        end
    end 
    # func = expr |> eval 
    # return func
    # global_expr_dict[id] = expr

    symbol = get_symbol(x)
    symbol = atom2hash(x, symbol)

    expr = :($symbol = $expr)

    global_expr_dict[symbol] = expr
    global_mapping[symbol] = mapping_vec

    # return expr
    return symbol
end


function cgen(x::Atom, ::Val{:observe_data})
    params = x.params
    @assert length(params) == 1
    param = params[1]
    return cgen(param)
end

function cgen(x::Atom, ::Val{:sampling})
    symbol = cgen(x, Val(:observe_dist))

    obs_symbol = Symbol(symbol, "_observe")

    # obs_symbol = atom2hash(x, obs_symbol)
    # if obs_symbol==symbol
        # obs_symbol = Symbol(obs_symbol, gensym())
    # end

    expr = :($obs_symbol = rand($symbol))

    global_expr_dict[obs_symbol] = expr
    global_mapping[obs_symbol] = symbol

    # hash the Atom obj 




    # symbol = x.symbol
    # expr = :($symbol = $expr)

    # global_expr_dict[symbol] = expr
    # func = expr |> eval 

    # if d isa Distribution 
    # return :(rand($symbol))
    return obs_symbol
end



function check_type(x::ExprTerm)
    op = x.op 
    return Val(op.symbol)
end

function cgen(x::ExprTerm)
    _type = check_type(x)
    return cgen(x, _type)
end

"""
obsq = ExprTerm(FunctorOperation(:observe), [q, data]), return the probablity density expr
"""
function cgen(x::ExprTerm, ::Val{:observe})
    @assert length(x.args)==2
    q, data = x.args

    q_symbol = cgen(q)
    expr_data = cgen(data)

    expr = :(pdf($q_symbol, $expr_data))
    return expr
end

"""
grad_elbo = ExprTerm(ParamOperation(:grad, [:ga, :gb]), elbo)
"""
function cgen(x::ExprTerm, ::Val{:grad})
    op = x.op
    terms = x.args 

    @assert length(terms) == 1
    term = terms[1]
    func_expr = cgen(term)

    block = :(begin end)

    # @show term
    # k = collect(keys(global_expr_dict))
    all_symbols = get_data_symbols(term)
    for k in keys(global_expr_dict)
        if k in all_symbols
            push!(block.args, global_expr_dict[k])
        end
    end

    # expr1 = global_expr_dict[k[1]]

    # grad_expr = :(y->gradient($variables_symbols -> begin
    #                     $expr1
    #                     $func_expr
    #                 end, y)
    #                 )
    # variables_symbols = [gensym() for _ in 1:length(op.to_reduce)])#
    variables_symbols = Tuple(op.to_reduce)
    variables_symbols_placeholder = :(())
    # for _ in 1:length(op.to_reduce)
    for var in variables_symbols
        push!(variables_symbols_placeholder.args, var)
    end

    grad_expr = :(collect(gradient($variables_symbols_placeholder -> begin
                        $block
                        $func_expr
                    end, $variables_symbols_placeholder...))
                    )

    return grad_expr
end

function cgen(x::ExprTerm, ::Val{:*})
    terms = x.args

    expr = :((*)())
    for term in terms 
        push!(expr.args, cgen(term))
    end
    return expr
end

function cgen(x::ExprTerm, ::Val{:log})
    terms = x.args 
    @assert length(terms) == 1
    term = terms[1]
    func_expr = cgen(term)
    expr = :(log($func_expr))
    return expr
end


# function sampling_fun_generator(code, var, verbose=true; kwargs...)
function sampling_fun_generator(code, params_symbol_vec, verbose=true)
    @show verbose
    function change_order(orders, parants, current)
        if parants in keys(orders)
            parants_order = min(orders[parants], orders[current])
            current_order = max(orders[parants], orders[current])
            orders[parants] = parants_order
            orders[current] = current_order
        end 
        return orders
    end
    if verbose
        @show global_mapping
    end
    orders = Dict(keys(global_mapping).=>collect(1:length(global_mapping)))
    for current in keys(global_mapping)
        parants = global_mapping[current]
        if parants isa Symbol
            orders = change_order(orders, parants, current)
        elseif parants isa Vector
            for p in parants
                if p isa Symbol
                    orders = change_order(orders, p, current)
                end 
            end 
        else 
            error()
        end
    end
    orders = Dict(values(orders).=>keys(orders))

    if verbose
        @show orders
    end


    # block = :(begin end)
    # for k in keys(kwargs)
    #     v = kwargs[k]
    #     expr = :($k = $v)
    #     push!(block.args, expr)
    # end

    # block |> eval

    block = :(begin end)
    for i in 1:length(orders)
        key = orders[i]
        expr = global_expr_dict[key]
        # expr |> eval 
        push!(block.args, expr)
    end 

    Base.remove_linenums!(block)
    if verbose
        @show block
    end

    inputs = :(())
    for param in params_symbol_vec
        push!(inputs.args, param)
    end

    # inputs = :(($var, ))
    # # parameters = collect(keys(kwargs))
    # for k in keys(kwargs)
    #     push!(inputs.args, k)
    # end
    

    func_expr = quote 
        $inputs -> begin
            $block 
            return $code 
        end 
    end

    Base.remove_linenums!(func_expr)
    if verbose
        println(func_expr)
    end

    func = func_expr |> eval
    
    # block |> eval
    # func = :($var -> $code) |> eval 
    return func
end

