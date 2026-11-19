.PHONY: install run test lint docker

install:
	uv sync

run:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

docker:
	docker build -t transcribe . && docker run --rm -p 127.0.0.1:8000:8000 --env-file .env transcribe
