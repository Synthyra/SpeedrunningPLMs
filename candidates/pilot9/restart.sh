#!/usr/bin/env bash
# Restart 6 of 2026-10-30: the pilot of every round-7 switch alone on confirm1_01 at the 180 s rung (validity and compile
# cost; the proposers explore them at 600 s), then the 10-minute loop until its deadline. The 1-hour loop is restarted
# from the same tree by a second shipment: `bash autosearch.sh --search search_1h --after ... --until ... --count 4`.
# Runs on the box from the shipped tree: `ws remote run ... -- bash candidates/pilot9/restart.sh`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

bash run_gh200.sh python search.py --state /state/search/search.json propose --proposer manual --batch pilot9 --rungs 180 --genomes candidates/pilot9/product_keys.json candidates/pilot9/kv_share_2.json candidates/pilot9/grouping.json candidates/pilot9/terminus.json candidates/pilot9/muon_late_2.json --hypotheses "confirm1_01 with a product-key memory read at block 18: the stream reads 65,536 learned value rows (50M parameters outside the blocks, values from zero) through a product-key lookup, an extra block input (ideas/2026-10-30_capacity_at_fixed_shape.md)" "confirm1_01 with every second block borrowing the previous block's keys and values (cross-layer KV sharing, 21M fewer parameters, fewer matmuls a step; ideas/2026-10-30_attention_and_output_head.md)" "confirm1_01 with an auxiliary head over six physico-chemical residue classes, its weight falling linearly to zero by half the clock (ideas/2026-10-30_protein_input_priors.md)" "confirm1_01 with learned N- and C-terminus vectors added at each protein's first and last residue (ideas/2026-10-30_protein_input_priors.md)" "confirm1_01 with Muon's learning rate alone rising from 1x at 80 percent of the clock to 2x at the end (ideas/2026-10-30_short_run_schedules.md)"
exec bash autosearch.sh --until 2026-10-31T02:00:00Z --count 4
