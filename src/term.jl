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
Base.:(==)(a::ParamHead, b::ParamHead) = a.symbol==b.symbol && Set(a.indices)==Set(b.indices)


struct EmptyOperation <: Operation
    symbol::Symbol
end

function EmptyOperation()
    return EmptyOperation(:None)
end

struct EmptyTerm <: Term 
    head::EmptyOperation
    data::Vector
end
Base.:(==)(a::EmptyTerm, b::EmptyTerm) = a.head.symbol==b.head.symbol

function EmptyTerm()
    return EmptyTerm(EmptyOperation(), [])
end

function is_same_set(a::Vector, b::Vector)
    check = length(a)==length(b)
    count = 0
    self_count = 0
    for ai in a 
        for bi in b 
            if typeof(ai)==typeof(bi)
                # if operation(ai).symbol==operation(bi).symbol
                #     @show ai, bi, ai==bi, Base.:(==)(ai, bi)
                # end
                # if ai.head==bi.head
                #     @show is_same_set(ai.data, bi.data)
                # end
                if ai==bi 
                    count += 1 
                end 
            end 
        end 
        for aj in a 
            if typeof(ai)==typeof(aj)
                if ai==aj 
                    self_count += 1 
                end 
            end 
        end
    end 
    check = (count - (self_count - length(a)))==length(a)
    # println(count, " ", self_count, " ", length(a))
    return check
end


struct Param <: Term 
    # symbol::Symbol 
    # indices::Union{Vector{Index}, Nothing}
    head::ParamHead
    data::Vector
end

#TODO: support for data check
Base.:(==)(a::Param, b::Param) = a.head.symbol==b.head.symbol && is_same_set(a.data, b.data)#Set(a.data)==Set(b.data)

function get_data_symbols(x::Param)
    symbols = Symbol[]
    push!(symbols, get_symbol(x))
    for d in x.data
        if d isa Term && !(d isa EmptyTerm) 
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
    return Param(ParamHead(symbol, indices), [EmptyTerm()])
end

function Param(symbol::Symbol, index::Index)
    return Param(ParamHead(symbol, [index]), [EmptyTerm()])
end


struct FunctorOperation <: Operation 
    symbol::Symbol
end
Base.:(==)(a::FunctorOperation, b::FunctorOperation) = a.symbol==b.symbol

struct Dist <: Operation
    symbol::Symbol
end
Base.:(==)(a::Dist, b::Dist) = a.symbol==b.symbol

struct ParamOperation <: Operation
    symbol::Symbol
    to_reduce::Vector{Symbol}
end
Base.:(==)(a::ParamOperation, b::ParamOperation) = a.symbol==b.symbol && Set(a.to_reduce)==Set(b.to_reduce)

function ParamOperation(symbol::Symbol, to_reduce::Symbol)
    return ParamOperation(symbol, [to_reduce])
end


struct AtomOperation 
    symbol::Symbol
    type::Symbol
    distribution::Operation
end
function Base.:(==)(a::AtomOperation, b::AtomOperation) 
    check = a.symbol==b.symbol && a.type==b.type && typeof(a.distribution)==typeof(b.distribution)
    if check 
        check = check && a.distribution==b.distribution
    end 
    return check
end
function approx_eqaul(a::AtomOperation, b::AtomOperation)
    check = a.symbol==b.symbol && typeof(a.distribution)==typeof(b.distribution)
    if check 
        check = check && a.distribution==b.distribution
    end 
    return check
end


struct Atom <: Term 
    # symbol::Symbol 
    # type::Symbol
    op::AtomOperation
    # params::Vector{<:Union{Param, Atom, Term}}
    params::Vector{<:Term}
    # indices::Vector{Index}
end
Base.:(==)(a::Atom, b::Atom) = a.op==b.op && is_same_set(a.params, b.params)

approx_eqaul(a::Atom, b::Atom) = approx_eqaul(a.op, b.op) && is_same_set(a.params, b.params)

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

# Base.:(==)(a::ExprTerm, b::ExprTerm) = a.op==b.op && a.args==b.args

# Base.:(==)(a::ExprTerm, b::Atom) = false
# Base.:(==)(a::Atom, b::ExprTerm) = false

# function change_symbol(x::ExprTerm, _type::Symbol)
#     op = AtomOperation(get_symbol(x), _type, get_dist_op(x))
#     return Atom(op, x.params)
# end

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
