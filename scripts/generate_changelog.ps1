param(
  [string]$OutputPath = "CHANGELOG.md"
)

# Requires git-cliff installed locally:
#   - Windows: `winget install git-cliff`
#   - Or: `cargo install git-cliff`

git cliff -c cliff.toml -o $OutputPath
