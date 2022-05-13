

function to_string(x::Atom)
    symbol = get_symbol(x)
    dist_symbol = get_dist_op(x).symbol 
    str = "Atom_" * string(symbol) * "_" * string(dist_symbol) * "_" * string(get_type(x))
    return str
end

function to_string(x::ExprTerm)
    op_symbol = x.op.symbol 
    str = "ExprTerm_" * string(op_symbol)
    return str
end

function to_string(x::Param)
    return "Param_" * string(get_symbol(x))
end

function to_string(x::EmptyTerm)
    return "nothing"
end

# function to_string(x::Nothing)
#     return "nothing"
# end

# function to_tree(x::Nothing)
#     return "nothing"
# end

function to_tree(x::EmptyTerm)
    return "nothing"
end

function to_tree(x::Union{ExprTerm, Atom, Param})
    # d = Dict()
    args = arguments(x)
    if length(args)==1
        arg = args[1]
        d_arg = "" => to_tree(arg)
    else
        d_arg = Dict()
        for arg in args
            d_arg[gensym()] = "" => to_tree(arg)
        end 
    end

    # d[gensym()] = to_string(x) => d_arg
    # return d
    return to_string(x) => d_arg
end