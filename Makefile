.PHONY: up run test gate audit down clean

up:            ## build, run the pipeline, then serve the API, dashboard, MLflow and Dagster UI
	docker compose up --build -d
	docker compose logs -f pipeline

run:           ## run only the pipeline
	docker compose run --rm pipeline

test:          ## unit and integration tests inside the image
	docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI=sqlite:////tmp/t.db pipeline pytest -q

gate:          ## release gate on a small fresh dataset, as CI runs it
	docker compose run --rm --no-deps pipeline python -m referral_mlops.gate --packets 40

audit:         ## PHI audit over logs and MLflow
	docker compose run --rm pipeline python -m referral_mlops.pipeline audit

down:
	docker compose down

clean:
	docker compose down -v && rm -rf data results mlflow_data
