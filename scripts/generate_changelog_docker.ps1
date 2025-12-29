param(
  [string]$OutputPath = "CHANGELOG.md"
)

# Generate CHANGELOG.md using git-cliff from Docker.
# Requires: Docker Desktop (or docker engine) installed.
#
# This avoids installing git-cliff binary locally and is consistent across machines.

$repoRoot = (Resolve-Path ".").Path

docker run --rm `
  -v "${repoRoot}:/repo" `
  -w /repo `
  orhunp/git-cliff:latest `
  -c cliff.toml `
  -o $OutputPath
