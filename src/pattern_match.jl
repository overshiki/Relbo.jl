


function is_integral_operation(o::T) where {T<:Operation}
    return o isa ParamOperation && o.symbol==:integral
end
is_integral_operation(e::ExprTerm) = is_integral_operation(e.op)
is_integral_operation(e::Atom) = false 
is_integral_operation(e::Param) = false