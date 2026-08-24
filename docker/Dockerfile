FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /workspace
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY configs/ configs/

CMD ["python", "scripts/train.py"]