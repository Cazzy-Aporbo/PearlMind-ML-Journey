FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[api]" && useradd --create-home learner
COPY configs ./configs
USER learner
WORKDIR /home/learner
ENV PYTHONUNBUFFERED=1
CMD ["pearlmind", "--help"]
