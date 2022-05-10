using Metatheory

abstract type Term end 
abstract type Operation end 

struct Index <: Term 
    index::Symbol 
    index_range::Symbol
end
Index(index::Symbol) = Index(index, gensym())

struct ParamHead <: Operation
    symbol::Symbol
    indices::Union{Vector{Index}, Nothing}
end


struct Param <: Term 
    # symbol::Symbol 
    # indices::Union{Vector{Index}, Nothing}
    head::ParamHead
    data::Vector
end

function get_data_symbols(x::Param)
    symbols = Symbol[]
    push!(symbols, get_symbol(x))
    for d in x.data
        if d isa Term 
            append!(symbols, get_data_symbols(d))
        end
    end
    return symbols
end

function get_symbol(x::Param)
    return x.head.symbol 
end 

function get_indices(x::Param)
    return x.head.indices 
end 

function Param(symbol::Symbol, indices::Vector{Index})
    return Param(ParamHead(symbol, indices), [nothing])
end

function Param(symbol::Symbol, index::Index)
    return Param(ParamHead(symbol, [index]), [nothing])
end


struct FunctorOperation <: Operation 
    symbol::Symbol
end

struct Dist <: Operation
    symbol::Symbol
end

struct ParamOperation <: Operation
    symbol::Symbol
    to_reduce::Vector{Symbol}
end

function ParamOperation(symbol::Symbol, to_reduce::Symbol)
    return ParamOperation(symbol, [to_reduce])
end


struct AtomOperation 
    symbol::Symbol
    type::Symbol
    distribution::Operation
end



struct Atom <: Term 
    # symbol::Symbol 
    # type::Symbol
    op::AtomOperation
    params::Vector{<:Union{Param, Atom, Term}}
    # indices::Vector{Index}
end

function get_data_symbols(x::Atom)
    symbols = Symbol[]
    push!(symbols, get_symbol(x))
    for param in x.params
        append!(symbols, get_data_symbols(param))
    end 
    return symbols
end

function get_symbol(x::Atom)
    return x.op.symbol 
end 

function get_type(x::Atom)
    return x.op.type
end

function get_dist_op(x::Atom)
    return x.op.distribution
end

function change_type(x::Atom, _type::Symbol)
    op = AtomOperation(get_symbol(x), _type, get_dist_op(x))
    return Atom(op, x.params)
end


function Base.copy(x::Atom)
    return Atom(x.op, x.params)
end

# (a::AtomOperation)(params...) = (a::AtomOperation)(params)

# function (a::AtomOperation)(params::Tuple{Vararg{<:Union{Param, Atom, Term}}})
#     params = [p for p in params]
#     return Atom(a, params)
# end


# function get_indices(params::Vector{<:Union{Param, Atom}})
#     indices = Index[]
#     for p in params
#         for index in p.indices
#             if !(index in indices)
#                 push!(indices, index)
#             end 
#         end 
#     end 
#     return indices
# end



# function Atom(op::AtomOperation, params::Vector{<:Union{Param, Atom}})
#     indices = Index[]
#     for p in params
#         for index in p.indices
#             if !(index in indices)
#                 push!(indices, index)
#             end 
#         end 
#     end 
#     return Atom(symbol, type, distribution, params, indices)
# end

function Atom(op::AtomOperation, param::Union{Param, Atom})
    # indices = param.indices
    return Atom(op, [param])
end



struct ExprTerm <: Term 
    op::Operation 
    args::Vector{<:Term}
end

function get_data_symbols(x::ExprTerm)
    symbols = Symbol[]
    for d in x.args
        if d isa Term 
            append!(symbols, get_data_symbols(d))
        end
    end
    return symbols
end

# struct AssignTerm <: Term 
#     assign::Atom 
#     val::Term
# end

function Base.copy(x::ExprTerm)
    return ExprTerm(x.op, x.args)
end

function ExprTerm(op::Operation, arg::Term)
    return ExprTerm(op, [arg])
end


function Integral(symbol::Symbol, guide::Term, terms::Vector{<:Term})
    if length(terms)>1
        prod = ExprTerm(FunctorOperation(:*), terms)
    elseif length(terms)==1
        prod = terms[1]
    else 
        error()
    end
    return ExprTerm(ParamOperation(:integral, symbol), [guide, prod])
end

function Integral(symbol::Symbol, guide::Term, terms::Term)
    return Integral(symbol, guide, [terms])
end
