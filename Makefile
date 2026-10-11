.PHONY: install generate preprocess train experiments test run clean

install:
	pip install -r requirements.txt

generate:
	python scripts/generate_dataset.py

validate:
	python scripts/validate_dataset.py

preprocess:
	python scripts/preprocess_data.py

train:
	python scripts/train_models.py

experiments:
	python scripts/run_experiments.py

init-db:
	python scripts/initialize_database.py

assets:
	python scripts/generate_report_assets.py

setup: generate validate preprocess train experiments init-db assets

test:
	pytest tests/ -v

run:
	streamlit run app.py

clean:
	rm -rf __pycache__ .pytest_cache artifacts/models/*.pkl data/grievance_route.db
