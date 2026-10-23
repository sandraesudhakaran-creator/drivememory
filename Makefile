.PHONY: install demo test eval api docker lint

install:
	pip install -e ".[all]"

demo:
	drivememory demo --driver anna --weeks 4

test:
	pytest -q

lint:
	ruff check src tests eval

eval:
	python eval/run_eval.py --weeks 8

api:
	uvicorn drivememory.api.app:app --reload

docker:
	docker compose up --build
