# Evaluation Pipeline Quickstart

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r deepseek_latent_attention/requirements.txt
   ```

2. **Run benchmarks**
   - Core QA/MC:
     ```bash
     make eval.core cfg=deepseek_latent_attention/cfgs/v1_mla_sparse.yaml \
         tasks="mmlu,gsm8k,hellaswag,arc_easy,arc_challenge,boolq"
     ```
   - Language modelling perplexity:
     ```bash
     make eval.lm cfg=... corpora="wikitext,pg19"
     ```
   - Long-context:
     ```bash
     make eval.long cfg=... bench="longbench,scrolls"
     ```
   - Long Range Arena:
     ```bash
     make eval.lra cfg=... tasks="listops,text,retrieval"
     ```
   - Throughput & memory:
     ```bash
     make bench.lat cfg=... seq="2048,8192,32768" bs="1,4"
     make bench.mem cfg=... seq=8192 bs=4
     ```
   - Needle-in-Haystack:
     ```bash
     make eval.needle cfg=... depth="1000000"
     ```
   - Diagnostics dump:
     ```bash
     make dump.stats cfg=... run_id=$(date +%Y%m%d)
     ```

3. **Results layout**
   - Metrics JSON: `results/{model_tag}/{suite}/metrics.json`
   - Performance CSV: `results/{model_tag}/perf/latency_mem.csv`
   - Diagnostic NPZ: `results/{model_tag}/stats/{run_id}_{layer}.npz`

4. **Dashboard seed**
   - Open `notebooks/00_results_dashboard.ipynb` and run all cells to visualise quality,
     long-context, efficiency, and meta-token diagnostics.

> Tip: set `output=<dir>` when invoking `make` to collect results in a custom root.
