.PHONY: setup test demo lint format

setup:
	python -m pip install -e ".[dev,full]"

test:
	pytest

demo:
	python scripts/run_demo.py --pack packs/toy_counting --out out/demo_toy_counting

lint:
	ruff .
	black --check .

format:
	black .
