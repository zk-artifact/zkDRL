PYTHON ?= python3
export PYTHONPATH := $(CURDIR)
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: test check build register validate numerics audit analysis plots clean

test:
	$(PYTHON) -B -m unittest discover -s tests/numerics -v
	$(PYTHON) -B -m unittest discover -s tests/ledger -v

check: test

build:
	./build.sh

register:
	$(PYTHON) -B prepare.py register

validate:
	$(PYTHON) -B prepare.py validate

numerics:
	$(PYTHON) -B run_numerics.py run

audit:
	$(PYTHON) -B run_numerics.py audit

analysis:
	$(PYTHON) -B analyze_numerics.py

plots:
	$(PYTHON) -B plot_results.py numeric

clean:
	rm -rf runs .build bin dqn/elf dqn/elf-h32 dqn/elf-h64 dqn/elf-c128 dqn/elf-c256 a2c/elf
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
