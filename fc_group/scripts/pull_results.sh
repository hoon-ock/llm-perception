#!/bin/bash
# Pull the analysis Results from HCC (Swan) down to this machine. Run it locally,
# not on the login node.
#
# The remote tree is Results/<method>/<model>/<leaf>, and only a slice of it is
# under analysis at a time. Copying the whole thing means ~10 GB dominated by
# leaves nobody is looking at, so the selection lives here as explicit lists
# rather than as a half-remembered rsync filter typed by hand.
#
# "Leaf" rather than "prompt type" because the third component means different
# things per method: for most it is the prompt type, but under tsne_plots it is
# the colour-by feature. prompt_dirs_for() below is where that difference lives.
#
# One rsync per source directory, with an explicit destination, instead of a
# single --files-from run: macOS ships openrsync (/usr/bin/rsync, "rsync 2.6.9
# compatible"), which takes --files-from but does not advertise the --relative
# that real rsync 3.x implies under it to rebuild the tree. Thirty small
# invocations over one multiplexed SSH connection cost nothing and put the
# destination layout beyond doubt.
#
# One file here is not a pure mirror. generation_eval/<model>/functional_group/data/
# summary.json carries a "free_generation" block merged in locally by
# fc_group/Analysis/generation/score_generations.py -- the behavioural readout that
# generation_eval.py writes generations.csv for but never scores. summary.json exists on
# both sides and there is no --delete here, so the Swan copy overwrites it and that block
# goes away. Nothing warns you: Results* is gitignored, so there is no diff to notice.
#
# Restore it after a pull with:
#
#   python fc_group/Analysis/generation/score_generations.py
#
# It is CPU-only and takes seconds. Excluding the file instead would make the sync rules
# conditional on one leaf, which is worse than re-deriving something this cheap.
#
# Usage: bash fc_group/scripts/pull_results.sh [--dry-run]

set -u

# The ~/.ssh/config alias, which already sets ControlMaster auto + ControlPersist
# 8h -- the first connection authenticates, the other 26 reuse the socket, so Duo
# is not prompted once per directory.
REMOTE=swan
REMOTE_ROOT=/work/ock/hoon/llm-perception/fc_group/Results
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)/fc_group/Results"

# Which leaf directories to take under each method. Sibling prompt types include
# names with spaces ("functional_group question"), so this names exact
# directories and never has to parse or glob them.
#
# tsne_plots is the exception. Its third component is not a prompt type at all
# but the colour-by feature applied to one shared set of t-SNE coordinates, and
# template_index is the colouring by which prompt template produced each row
# (built in tsne_functional_groups.py from the row layout). The comment at
# tsne_functional_groups.py:159 is the reason both are needed: the silhouette
# score is computed on the 2-D coordinates, so it is only interpretable as a
# relative comparison between two colourings of the SAME tsne_data -- it names
# functional_group vs template_index specifically. Taking one without the other
# leaves a number with nothing to compare against.
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

# Directory spellings, not registry names: model_registry.py says
# "meta-llama/Llama-3.1-8B", the Results tree writes "meta-llama-Llama-3.1-8B".
MODELS=(
  meta-llama-Llama-3.1-8B
  deepseek-ai-DeepSeek-R1-Distill-Llama-8B
  phenixace-Chem-R-Faithful
  # The chemistry-comparison track (fc_group/scripts/chem/). Listed here even
  # though it is not part of the five-model sweep: this list is what decides
  # whether results come down off Swan at all, and a model missing from it fails
  # silently -- the jobs run, the results are written, and nothing is pulled.
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
      printf '  %-42s %-42s %-16s ... ' "$method" "$model" "$leaf"
      [ -n "$DRY" ] || mkdir -p "$dst"
      # No -z (the payload is PNGs, already compressed) and no -v (thousands of
      # plot filenames drown the one line that matters). --partial so an
      # interrupted transfer resumes. Trailing slashes on both sides: copy the
      # contents in, so a re-run tops up rather than nesting another $leaf/.
      if rsync -a $DRY --partial "$REMOTE:$src" "$dst" 2>/tmp/pull_results_err.$$; then
        echo "ok"
      else
        echo "FAILED"
        sed 's/^/      /' /tmp/pull_results_err.$$
        missing+=("$method/$model/$leaf")
      fi
      rm -f /tmp/pull_results_err.$$
    done < <(prompt_dirs_for "$method")
  done
done

echo
if [ ${#missing[@]} -gt 0 ]; then
  # Not fatal: a partially-complete HCC sweep should still yield what does exist.
  echo "${#missing[@]} directory(ies) did not transfer:"
  printf '  %s\n' "${missing[@]}"
  echo
fi

if [ -z "$DRY" ]; then
  echo "local sizes:"
  du -sh "$LOCAL_ROOT"/* 2>/dev/null
  echo
  du -sh "$LOCAL_ROOT"
fi
