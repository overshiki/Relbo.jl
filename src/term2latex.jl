using Formatting


function to_latex(x::Param)
    # indices = x.indices
    # indices_str = ""
    # for (i,index) in enumerate(indices)
    #     indices_str *= string(index.index)
    #     if i<length(indices)
    #         indices_str *= ","
    #     end 
    # end 
    # latex_str = format("{1}_{{2}}", string(x.symbol), indices_str)
    # return latex_str
    return string(x.symbol)
end

function to_latex(x::Atom)
    symbol = x.symbol 
    op = x.distribution
    if op isa Sampling
        params = x.params
        param_str = ""
        for (i,param) in enumerate(params)
            param_str *= to_latex(param)
            if i<length(params)
                param_str *= ","
            end 
        end
            
        # latex_str = format("{1}_{{2}}({3})", string(op.symbol), string(symbol), param_str)

        ops = string(op.symbol)
        s = string(symbol)
        latex_str = "$(ops)_{$s}($param_str)"
        # @show ops, latex_str
        return latex_str

        # latex_str = "$op.symbol_"
    elseif op.symbol == :data
        latex_str = ""
        return latex_str
    else 
        error() 
    end 
    
end

# function to_latex(x::Atom)
#     return to_latex(x, Val(:Sampling))
# end

function to_latex(x::Symbol)
    return string(x)
end

function connect_terms(terms, op::String)
    latex_str = "("
    for (i, term) in enumerate(terms) 
        latex_str *= to_latex(term)
        if i<length(terms)
            latex_str *= string(op)
        end 
    end 
    latex_str *= ")"
    return latex_str
end


function to_latex(x::ExprTerm)
    op = get_op(x)
    terms = get_args(x)

    if op.symbol == :*
        # latex_str = ""
        # for (i, term) in enumerate(terms) 
        #     latex_str *= to_latex(term)
        #     if i<length(terms)
        #         latex_str *= "*"
        #     end 
        # end 
        latex_str = connect_terms(terms, "*")
    elseif op.symbol == :integral
        # sub_str = ""
        # for (i, sym) in enumerate(op.to_reduce)
        #     sub_str *= string(sym)
        #     if i<length(op.to_reduce)
        #         sub_str *= ","
        #     end 
        # end

        sub_str = connect_terms(op.to_reduce, ",")

        # latex_str = format("\\int_{{1}}", sub_str)
        latex_str = "\\int_{$sub_str}"
        guide = terms[1]
        latex_str *= to_latex(guide)
        latex_str *= format("d{1}", string(guide.symbol))

        for term in terms[2:end] 
            latex_str *= to_latex(term)
        end 

    elseif op.symbol == :grad
        # sub_str = ""
        # for (i, sym) in enumerate(op.to_reduce)
        #     sub_str *= string(sym)
        #     if i<length(op.to_reduce)
        #         sub_str *= ","
        #     end 
        # end
        sub_str = connect_terms(op.to_reduce, ",")

        # latex_str = format("\\partial_{1}", sub_str)
        latex_str = "\\partial_{$sub_str}"
        for (i, term) in enumerate(terms) 
            latex_str *= to_latex(term)
        end 
    elseif op.symbol == :observe
        q = terms[1]
        # latex_str = format("\\hat{{1}}", to_latex(q))
        sub_str = to_latex(q)
        s = string(q.symbol)
        latex_str = "Obs_{$s}\\left($(sub_str)\\right)"
    # elseif op isa Sampling
    #     @show op.symbol
    elseif op isa FunctorOperation
        # q = terms[1]
        sub_str = connect_terms(terms, ",")
        latex_str = format("{1}({2})", string(op.symbol), sub_str)
    else
        @show op
        error()
    end

    return latex_str

end