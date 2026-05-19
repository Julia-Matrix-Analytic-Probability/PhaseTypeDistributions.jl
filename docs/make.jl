using Documenter
using PhaseTypeDistributions

DocMeta.setdocmeta!(
    PhaseTypeDistributions,
    :DocTestSetup,
    :(using PhaseTypeDistributions, Distributions, Random);
    recursive = true,
)

makedocs(;
    modules = [PhaseTypeDistributions],
    authors = "Zhihao Qiao, Yoni Nazarathy, and contributors",
    sitename = "PhaseTypeDistributions.jl",
    format = Documenter.HTML(;
        canonical = "https://julia-matrix-analytic-probability.github.io/PhaseTypeDistributions.jl",
        edit_link = "main",
        assets = String[],
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "PH distributions" => "ph.md",
        "MAPH distributions" => "maph.md",
        "API reference" => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(;
    repo = "github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl",
    devbranch = "main",
    push_preview = true,
)
