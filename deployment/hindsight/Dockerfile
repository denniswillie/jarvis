FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY deployment/hindsight/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt \
    && useradd --system --uid 10001 --create-home hindsight
COPY backend/app/__init__.py /srv/app/__init__.py
COPY backend/app/knowledge/__init__.py /srv/app/knowledge/__init__.py
COPY backend/app/knowledge/service_policy.py backend/app/knowledge/service_entrypoint.py /srv/app/knowledge/
USER 10001
ENTRYPOINT ["python", "-m", "app.knowledge.service_entrypoint"]
CMD ["--host", "0.0.0.0", "--port", "8888"]
