PYTHON ?= python

check_var = @if [ -z "$($(1))" ]; then \
		echo "Missing required variable: $(1)=..."; \
		exit 1; \
	fi

.PHONY: eval.core eval.lm eval.long eval.lra bench.lat bench.mem eval.needle dump.stats

eval.core:
	$(call check_var,cfg)
	$(call check_var,tasks)
	$(PYTHON) scripts/run_lmeval.py --cfg $(cfg) --tasks $(tasks) --suite core $(if $(output),--output-dir $(output),)

eval.lm:
	$(call check_var,cfg)
	$(call check_var,corpora)
	$(PYTHON) scripts/run_lmeval.py --cfg $(cfg) --tasks $(corpora) --suite lm $(if $(output),--output-dir $(output),)

eval.long:
	$(call check_var,cfg)
	$(call check_var,bench)
	@if echo "$(bench)" | grep -q "longbench"; then \
		$(PYTHON) scripts/run_longbench.py --cfg $(cfg) $(if $(long_tasks),--bench $(long_tasks),) $(if $(output),--output-dir $(output),); \
	fi
	@if echo "$(bench)" | grep -q "scrolls"; then \
		$(PYTHON) scripts/run_scrolls.py --cfg $(cfg) $(if $(scroll_tasks),--tasks $(scroll_tasks),) $(if $(output),--output-dir $(output),); \
	fi

eval.lra:
	$(call check_var,cfg)
	$(call check_var,tasks)
	$(PYTHON) scripts/run_lra.py --cfg $(cfg) --tasks $(tasks) $(if $(output),--output-dir $(output),)

bench.lat:
	$(call check_var,cfg)
	$(call check_var,seq)
	$(call check_var,bs)
	$(PYTHON) scripts/bench_throughput.py --cfg $(cfg) --seq $(seq) --bs $(bs) $(if $(output),--output-dir $(output),)

bench.mem:
	$(call check_var,cfg)
	$(call check_var,seq)
	$(call check_var,bs)
	$(PYTHON) scripts/bench_throughput.py --cfg $(cfg) --seq $(seq) --bs $(bs) --iters 1 $(if $(output),--output-dir $(output),)

eval.needle:
	$(call check_var,cfg)
	$(call check_var,depth)
	$(PYTHON) scripts/eval_needle.py --cfg $(cfg) --depth $(depth) $(if $(output),--output-dir $(output),)

dump.stats:
	$(call check_var,cfg)
	$(PYTHON) scripts/dump_stats.py --cfg $(cfg) $(if $(output),--output-dir $(output),)
