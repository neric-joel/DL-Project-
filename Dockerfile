# CPU image. By default it runs the tests and rebuilds every table and figure from the committed
# per-unit results (results/units/), which takes a few minutes and needs no dataset download.
# The full pipeline (download -> infer -> ...) also runs inside it; see docker-compose.yaml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 MPLBACKEND=Agg

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY results ./results
RUN pip install --no-deps -e .

CMD ["sh", "-c", "pytest -q && asrshift evaluate && asrshift figures && asrshift report"]
