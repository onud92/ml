# NVIDIA Image
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# verhindern, dass Python .pyc Dateien schreibt (hält Container sauber)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# zusätzliche pakete installieren
RUN pip install --no-cache-dir matplotlib pandas scipy