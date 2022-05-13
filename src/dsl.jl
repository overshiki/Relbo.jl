
using MacroTools: rmlines

mutable struct TypeInfo
    type::Val
    infos::Vector{Union{Expr, Index}}
    order::Int
end


function update_type_dict(index::Int, type_dict::Dict, symbol::Symbol, type_symbol::Union{Nothing, Val}, type_expr::Union{Symbol, Expr, Nothing, Index})
    if !(symbol in keys(type_dict))
        if type_symbol isa Val 
            type_dict[symbol] = TypeInfo(type_symbol, [], index)
        else 
            type_dict[symbol] = TypeInfo(Val(:default), [], -1)
        end
    else 
        if type_symbol isa Val
            @assert type_dict[symbol].type isa Val{:default}
            type_dict[symbol].type = type_symbol
            type_dict[symbol].order = index
        end 
    end

    if type_expr!==nothing
        push!(type_dict[symbol].infos, type_expr)
    end
    
    return type_dict
end

function process_type_assign(s::Symbol, ::Val{:Index})
    expr = :($s = Index($s))
    return expr
end

function process_type_assign(index::Int, type_dict::Dict, expr::Union{Expr, Symbol}, type_expr, ::Val{:data})
    if expr isa Expr    
        @assert expr.head==:tuple
        for sym in expr.args
            type_dict = update_type_dict(index, type_dict, sym, Val(:data), type_expr)
        end 
    elseif expr isa Symbol
        type_dict = update_type_dict(index, type_dict, expr, Val(:data), type_expr)
    else 
        error()
    end
    return type_dict
end


function process_type_assign(index::Int, type_dict::Dict, expr::Union{Expr, Symbol}, t::Val)
    expr = expr.args[1]

    if expr isa Expr    
        @assert expr.head==:tuple
        for sym in expr.args
            type_dict = update_type_dict(index, type_dict, sym, t, nothing)
        end 
    elseif expr isa Symbol
        type_dict = update_type_dict(index, type_dict, expr, t, nothing)
    else 
        error()
    end

    return type_dict
end


function generate_elbo_term(type_dict::Dict)
    function select_by_order(type_dict, order)
        ks = []
        for k in keys(type_dict)
            if type_dict[k].order==order 
                push!(ks, k)
            end
        end
        return ks
    end

    function process_type_info(s::Symbol, T::TypeInfo, ::Val{:Param})
        indices = Index[] 
        for ind in T.infos 
            @assert ind isa Index 
            push!(indices, ind)
        end

        p = ParamHead(s, indices)
        return Param(p, [EmptyTerm()])
    end

    function process_type_info(s::Symbol, T::TypeInfo, ::Val{:input_param})
        return process_type_info(s, T, Val(:Param))
    end

    function process_type_info(s::Symbol, T::TypeInfo, ::Val{:data})
        indices = Index[] 
        obs = nothing
        for ind in T.infos 
            if ind isa Index 
                push!(indices, ind)
            elseif ind isa Expr 
                obs = ind 
            end
        end
        @assert length(indices)+1 == length(T.infos)
        @assert obs isa Expr
        # dump(obs)
        @assert obs.head == :call && obs.args[1] == :Observe
        @assert length(obs.args)==2
        obs_symbol = obs.args[2]
        # @show s
        return Atom(AtomOperation(obs_symbol, :observe, FunctorOperation(:data)), Param(s, indices))
    end


    function process_type_info(s::Symbol, T::TypeInfo, ::Val{:Observe})
        @assert length(T.infos)==1 
        expr = T.infos[1]
        @assert expr isa Expr 
        # dump(expr)
        @assert expr.head == :call
        term_vec = Term[]
        for sym in expr.args 
            # @show sym
            @assert sym in keys(term_dict)
            push!(term_vec, term_dict[sym])
        end 
        # @show term_vec

        return ExprTerm(FunctorOperation(:observe), term_vec)
        # error()
    end

    function is_observe(s::Symbol)
        for k in keys(term_dict)
            t = term_dict[k]
            if t isa Atom 
                if get_symbol(t)==s 
                    if t.op.distribution == FunctorOperation(:data)
                        return true 
                    end 
                end 
            end 
        end 
        return false
    end

    function generate_atom(expr::Expr, sym::Symbol)
        func_symbol = expr.args[1]
        param_symbols = expr.args[2:end]

        term_vec = Term[]
        for param_sym in param_symbols
            # @show param_sym
            @assert param_sym in keys(term_dict)
            # @show term_dict[param_sym]
            push!(term_vec, term_dict[param_sym])
        end 
        
        # @show sym, func_symbol, is_observe(sym)
        if is_observe(sym)
            return Atom(AtomOperation(sym, :observe, Dist(func_symbol)), term_vec)
        else 
            return Atom(AtomOperation(sym, :distribution, Dist(func_symbol)), term_vec) 
        end
    end

    function process_type_info(s::Symbol, T::TypeInfo, ::Val{:Dist})
        @assert length(T.infos)==1 
        expr = T.infos[1]
        @assert expr isa Expr 
        @assert expr.args[1] == :~
        @assert length(expr.args)==3
        sym = expr.args[2]
        expr = expr.args[3]

        # dump(expr)

        func_symbol = expr.args[1]
        if func_symbol!==:≈
            return generate_atom(expr, sym)
        else 
            @assert length(expr.args)==3
            sym = expr.args[3]
            expr = expr.args[2]
            return generate_atom(expr, sym)
        end
    end

    function process_type_info(s::Symbol, T::TypeInfo, ::Val{:Expectation})
        @assert length(T.infos)==1 
        expr = T.infos[1]
        @assert expr isa Expr 
        @assert expr.head == :call
        @assert expr.args[1] == :Expectation
        param_symbols = expr.args[2:end]
        term_vec = Term[]
        for param_sym in param_symbols
            @assert param_sym in keys(term_dict)
            push!(term_vec, term_dict[param_sym])
        end 
        @assert length(term_vec)==2
        guide, obsq = term_vec 
        guide_symbol = get_symbol(guide)
        return Integral(guide_symbol, guide, obsq)

    end




    # println(type_dict)
    orders = unique(sort([type_dict[k].order for k in keys(type_dict)]))

    block = :(begin end)
    term_dict = Dict()
    for i in orders
        # group keys(type_dict) into subsets of equal orders
        for k in select_by_order(type_dict, i)
            T = type_dict[k]
            # @show k, T
            if T.type isa Val{:input_param} || T.type isa Val{:Param} || T.type isa Val{:data} || T.type isa Val{:Dist} || T.type isa Val{:Observe} || T.type isa Val{:Expectation}
                p = process_type_info(k, T, T.type)
                term_dict[k] = p 
            end
            
        end
    end
    return term_dict
end

struct Elbo_container 
    elbo::ExprTerm
    grad_elbo::ExprTerm
    grad_vars::Vector{Symbol}
end

macro ELBO(input_def::Expr, expr::Expr)
    type_dict = Dict()
    # block = :(begin end)
    expr = expr|> rmlines
    exprs = expr.args

    Index_dict = Dict()

    for (index, e) in enumerate(exprs)
        # println(e)
        if e.head == Symbol("::") #&& (e.args[2]==:Index || e.args[2]==:Param)
            @assert length(e.args)==2
            if e.args[2] isa Symbol
                type_dict = process_type_assign(index, type_dict, e, Val(e.args[2]))
            else 
                _expr = e.args[2]
                @assert _expr isa Expr
                @assert _expr.head == :call 
                @assert _expr.args[1] == :Observe
                type_dict = process_type_assign(index, type_dict, e.args[1], _expr, Val(:data))
                # dump(_expr)
            end
            # push!(block.args, ex)
        end

        if e.head == Symbol("=")
            # @show e 
            # dump(e)
            sym = e.args[1]
            e = e.args[2]
            type_dict = update_type_dict(index, type_dict, sym, Val(:Observe), e)
        end

        if e.head == :call
            if e.args[1] == :~
                sym = e.args[2]
                type_dict = update_type_dict(index, type_dict, sym, Val(:Dist), e)
            elseif e.args[1] == Symbol("|")
                symbol = e.args[2]
                index_symbol = e.args[3]

                @assert index_symbol in keys(type_dict)
                @assert type_dict[index_symbol].type isa Val{:Index}

                if !(index_symbol in keys(Index_dict))
                    index_obj = Index(index_symbol)
                    Index_dict[index_symbol] = index_obj
                else 
                    index_obj = Index_dict[index_symbol]
                end
                # index_obj = Index(index_symbol)
                type_dict = update_type_dict(index, type_dict, symbol, nothing, index_obj)

            end
            # break
        end

        if e.head == :return 
            e = e.args
            @assert length(e)==1 
            e = e[1]
            @assert e.head == :call
            @assert e.args[1]==:Expectation
            type_dict = update_type_dict(index, type_dict, :elbo, Val(:Expectation), e)
        end

    end

    # println(block)
    grad_vars = Symbol[]

    @assert input_def.head == :tuple
    for sym in input_def.args 
        push!(grad_vars, sym)
        @assert sym in keys(type_dict)
        type_dict[sym].type = Val(:input_param)
    end

    term_dict = generate_elbo_term(type_dict)

    elbo = term_dict[:elbo]

    input_symbols = input_def.args
    # @show input_symbols
    grad_elbo = ExprTerm(ParamOperation(:grad, input_symbols), elbo)

    grad_elbo = integral2dist_sampling_rewrite_rule(grad_elbo)
    elbo = integral2dist_sampling_rewrite_rule(elbo)
    # println()
    elbo_c = Elbo_container(elbo, grad_elbo, grad_vars)

    # return :($grad_elbo)
    return :($elbo_c)

end