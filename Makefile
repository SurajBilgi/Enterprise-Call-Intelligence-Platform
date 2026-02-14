# Makefile for Enterprise Call Intelligence Platform

.PHONY: help install setup run-pipeline run-api run-ui test clean docker-build docker-up docker-down

help:
	@echo "Enterprise Call Intelligence Platform - Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install dependencies"
	@echo "  make setup         - Setup environment and directories"
	@echo ""
	@echo "Development:"
	@echo "  make run-pipeline  - Run full processing pipeline"
	@echo "  make run-api       - Start FastAPI server"
	@echo "  make run-ui        - Start Streamlit UI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start services with docker-compose"
	@echo "  make docker-down   - Stop services"
	@echo ""
	@echo "Utilities:"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean generated files"

install:
	pip install -r requirements.txt

setup:
	mkdir -p storage/raw_transcripts
	mkdir -p storage/structured
	mkdir -p storage/vectors
	mkdir -p storage/cache
	mkdir -p storage/search
	mkdir -p logs
	cp .env.example .env
	@echo "✓ Setup complete. Please edit .env with your API keys."

run-pipeline:
	PYTHONPATH=. python pipelines/orchestrator.py --num-calls 1000

run-pipeline-fast:
	PYTHONPATH=. python pipelines/orchestrator.py --num-calls 100 --skip-enrichment

run-api:
	PYTHONPATH=. python -m api.main

run-ui:
	PYTHONPATH=. streamlit run ui/streamlit_app.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

test:
	pytest tests/ -v

clean:
	rm -f storage/raw_transcripts/*.json
	rm -f storage/structured/*.db
	rm -rf storage/vectors
	rm -rf storage/cache
	rm -rf storage/search
	rm -f logs/*.log
	mkdir -p storage/vectors storage/cache storage/search
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .
