"ExprTerm_integral"
└─ Dict{Any, Any}
   ├─ ""
   │  └─ "Atom_z_Beta_distribution"
   │     └─ Dict{Any, Any}
   │        ├─ ""
   │        │  └─ "Param_ga"
   │        │     └─ ""
   │        │        └─ "nothing"
   │        └─ ""
   │           └─ "Param_gb"
   │              └─ ""
   │                 └─ "nothing"
   ├─ ""
   │  └─ "ExprTerm_observe"
   │     └─ Dict{Any, Any}
   │        ├─ ""
   │        │  └─ "Atom_q_InverseGaussian_observe"
   │        │     └─ ""
   │        │        └─ "Atom_z_Beta_sampling"
   │        │           └─ Dict{Any, Any}
   │        │              ├─ ""
   │        │              │  └─ "Param_a"
   │        │              │     └─ ""
   │        │              │        └─ "nothing"
   │        │              └─ ""
   │        │                 └─ "Param_b"
   │        │                    └─ ""
   │        │                       └─ "nothing"
   │        └─ ""
   │           └─ "Atom_q_data_observe"
   │              └─ ""
   │                 └─ "Param_data"
   │                    └─ ""
   │                       └─ "nothing"
   └─ ""
      └─ "ExprTerm_grad"
         └─ ""
            └─ "ExprTerm_log"
               └─ ""
                  └─ "Atom_z_Beta_distribution"
                     └─ Dict{Any, Any}
                        ├─ ""
                        │  └─ "Param_ga"
                        │     └─ ""
                        │        └─ "nothing"
                        └─ ""
                           └─ "Param_gb"
                              └─ ""
                                 └─ "nothing"
"ExprTerm_*"
└─ Dict{Any, Any}
   ├─ ""
   │  └─ "ExprTerm_observe"
   │     └─ Dict{Any, Any}
   │        ├─ ""
   │        │  └─ "Atom_q_InverseGaussian_observe"
   │        │     └─ ""
   │        │        └─ "Atom_z_Beta_sampling"
   │        │           └─ Dict{Any, Any}
   │        │              ├─ ""
   │        │              │  └─ "Param_a"
   │        │              │     └─ ""
   │        │              │        └─ "nothing"
   │        │              └─ ""
   │        │                 └─ "Param_b"
   │        │                    └─ ""
   │        │                       └─ "nothing"
   │        └─ ""
   │           └─ "Atom_q_data_observe"
   │              └─ ""
   │                 └─ "Param_data"
   │                    └─ ""
   │                       └─ "nothing"
   └─ ""
      └─ "ExprTerm_grad"
         └─ ""
            └─ "ExprTerm_log"
               └─ ""
                  └─ "ExprTerm_observe"
                     └─ Dict{Any, Any}
                        ├─ ""
                        │  └─ "Atom_z_data_observe"
                        │     └─ ""
                        │        └─ "Param_z"
                        │           └─ ""
                        │              └─ "Atom_z_Beta_sampling"
                        │                 └─ Dict{Any, Any}
                        │                    ├─ ""
                        │                    │  └─ "Param_gb"
                        │                    │     └─ ""
                        │                    │        └─ "nothing"
                        │                    └─ ""
                        │                       └─ "Param_ga"
                        │                          └─ ""
                        │                             └─ "nothing"
                        └─ ""
                           └─ "Atom_z_Beta_observe"
                              └─ Dict{Any, Any}
                                 ├─ ""
                                 │  └─ "Param_ga"
                                 │     └─ ""
                                 │        └─ "nothing"
                                 └─ ""
                                    └─ "Param_gb"
                                       └─ ""
                                          └─ "nothing"
pdf(q, data) * collect(gradient(((ga, gb)->begin
                    #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:192 =#
                    #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:193 =#
                    begin
                        #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:167 =#
                        z = Beta(ga, gb)
                    end
                    #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:194 =#
                    log(pdf(z, z_observe))
                end), (ga, gb)...))
Dict{Any, Any} with 3 entries:
  :z_observe => :(z_observe = rand(z))
  :z         => :(z = Beta(ga, gb))
  :q         => :(q = InverseGaussian(z_observe))
verbose = true
global_mapping = Dict{Any, Any}(:z_observe => :z, :z => Any[:ga, :gb], :q => Any[:z_observe])
orders = Dict(2 => :z_observe, 3 => :q, 1 => :z)
block = quote
    #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:266 =#
    z = Beta(ga, gb)
    z_observe = rand(z)
    q = InverseGaussian(z_observe)
end
begin
    #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:284 =#
    (data, ga, gb)->begin
            #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:284 =#
            #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:285 =#
            begin
                #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:266 =#
                z = Beta(ga, gb)
                z_observe = rand(z)
                q = InverseGaussian(z_observe)
            end
            #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:286 =#
            return pdf(q, data) * collect(gradient(((ga, gb)->begin
                                    #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:192 =#
                                    #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:193 =#
                                    begin
                                        #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:167 =#
                                        z = Beta(ga, gb)
                                    end
                                    #= /home/le/CODE/gitee/Relbo.jl/src/codegen.jl:194 =#
                                    log(pdf(z, z_observe))
                                end), (ga, gb)...))
        end
end
 29.035210 seconds (51.49 M allocations: 2.738 GiB, 3.78% gc time, 99.98% compilation time)
size(grad) = (2,)
