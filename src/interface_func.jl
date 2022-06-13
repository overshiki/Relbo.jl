

function def_data(var::Symbol, param::Param)
    return Atom(AtomOperation(var, :observe, FunctorOperation(:data)), param)
end

function def_dist_sampling(var::Symbol, dist::Symbol, params::Vector{Param})
    return Atom(AtomOperation(var, :dist_sampling, Dist(dist)), params)
end

function def_dist(var::Symbol, dist::Symbol, params::Vector{Param})
    return Atom(AtomOperation(var, :distribution, Dist(dist)), params)
end

function def_dist_observe(var::Symbol, dist::Symbol, obs::Atom)
    return Atom(AtomOperation(var, :observe, Dist(dist)), obs)
end

function observe_bind(var_pair::Pair{Atom, Atom})
    return ExprTerm(FunctorOperation(:observe), collect(var_pair))
end

function gen_elbo(model)
    guide, terms = arguments(model)[1], arguments(model)[2:end]
    if length(terms)==1
        term = terms[1]
    else 
        term = ExprTerm(FunctorOperation(:*), terms)
    end
    
    term = ExprTerm(FunctorOperation(:log), term)
    tguide = ExprTerm(FunctorOperation(:log), guide)
    term = ExprTerm(FunctorOperation(:-), [term, tguide])
    
    elbo = Integral(get_symbol(guide), guide, term)
    return elbo
end