#!/bin/bash
# Pull the chemistry-comparison track's Results from HCC (Swan) down to this
# machine. Run it locally, not on the login node.
#
# Generated from ../pull_results.sh the same way the chem_*.sbatch files in this
# directory were generated from their ../ counterparts: same rsync rules, same
# leaf selection, same failure reporting. The only real delta is MODELS.
#
# Why this exists separately rather than being a flag on ../pull_results.sh.
# That script mirrors all five sweep models, and its header documents the cost:
# generation_eval/<model>/functional_group/data/summary.json carries a
# "free_generation" block merged in locally by
# fc_group/Analysis/generation/score_generations.py, the file exists on both
# sides, and there is no --delete -- so every full pull silently reverts that
# block and it has to be re-derived. Fetching two new models should not put the
# already-analysed three at that risk. Keeping this as its own file also leaves
# the verified puller untouched, which is the same reasoning that gave this
# directory its own sbatch track instead of widening the shared sweep.
#
# That summary.json caveat does not apply to the two models here: nothing local
# has merged anything into their summaries, so there is nothing to lose. If they
# are later scored by score_generations.py --models, this comment stops being
# true and the note in ../pull_results.sh starts applying to them too.
#
# Usage: bash fc_group/scripts/chem/chem_pull_results.sh [--dry-run]

set -u

# The ~/.ssh/config alias, which already sets ControlMaster auto + ControlPersist
# 8h -- the first connection authenticates, the rest reuse the socket, so Duo is
# not prompted once per directory.
REMOTE=swan
REMOTE_ROOT=/work/ock/hoon/llm-perception/fc_group/Results
LOCAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)/fc_group/Results"

# Which leaf directories to take under each method. This track only ran the
# declarative 'functional_group' entity type (see chem_models.sh), so that is the
# only prompt-type leaf that exists remotely.
#
# tsne_plots is the exception, and for the same reason as in ../pull_results.sh:
# its third component is not a prompt type but the colour-by feature applied to
# one shared set of t-SNE coordinates. The silhouette score is computed on the
# 2-D coordinates, so it is only interpretable as a comparison between two
# colourings of the SAME tsne_data -- fc_group/tsne_functional_groups.py:159
# names functional_group vs template_index specifically. Taking one without the
# other leaves a number with nothing to compare against. The other twelve
# colourings (mw, pka, tpsa, hba/hbd, ...) are left on Swan.
prompt_dirs_for() {
  case "$1" in
    tsne_plots) echo "functional_group"; echo "template_index" ;;
    *)          echo "functional_group" ;;
  esac
}

METHODS=(
  ambiguity_metric
  anisotropy_diagnostic
  functional_group_analogy
  functional_group_analogy_retrieval
  functional_group_analogy_retrieval_carbon
  functional_group_analogy_retrieval_halide
  functional_group_probe
  generation_eval
  tsne_plots
)

# Directory spellings, not registry names: chem_models.sh says
# "weidawang/Chem-R-8B", the Results tree writes "weidawang-Chem-R-8B".
MODELS=(
  weidawang-Chem-R-8B
  OpenDFM-ChemDFM-v1.5-8B
)

DRY=""
if [ "${1:-}" = "--dry-run" ]; then
  DRY="-n"
  echo "DRY RUN -- nothing will be written."
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

echo "from: $REMOTE:$REMOTE_ROOT"
echo "to:   $LOCAL_ROOT"
echo

missing=()
for method in "${METHODS[@]}"; do
  for model in "${MODELS[@]}"; do
    while IFS= read -r leaf; do
      src="$REMOTE_ROOT/$method/$model/$leaf/"
      dst="$LOCAL_ROOT/$method/$model/$leaf/"
      printf '  %-42s %-26s %-16s ... ' "$method" "$model" "$leaf"
      [ -n "$DRY" ] || mkdir -p "$dst"
      # No -z (the payload is PNGs, already compressed) and no -v (thousands of
      # plot filenames drown the one line that matters). --partial so an
      # interrupted transfer resumes. Trailing slashes on both sides: copy the
      # contents in, so a re-run tops up rather than nesting another $leaf/.
      if rsync -a $DRY --partial "$REMOTE:$src" "$dst" 2>/tmp/chem_pull_results_err.$$; then
        echo "ok"
      else
        echo "FAILED"
        sed 's/^/      /' /tmp/chem_pull_results_err.$$
        missing+=("$method/$model/$leaf")
      fi
      rm -f /tmp/chem_pull_results_err.$$
    done < <(prompt_dirs_for "$method")
  done
done

echo
if [ ${#missing[@]} -gt 0 ]; then
  # Not fatal: a partially-complete HCC track should still yield what does exist.
  echo "${#missing[@]} directory(ies) did not transfer:"
  printf '  %s\n' "${missing[@]}"
  echo
fi

if [ -z "$DRY" ]; then
  echo "local sizes:"
  for model in "${MODELS[@]}"; do
    du -sh "$LOCAL_ROOT"/*/"$model" 2>/dev/null
  done
fi
