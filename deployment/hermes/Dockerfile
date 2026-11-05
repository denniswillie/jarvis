FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
# Keep the pinned generic runtime cached across Jarvis application edits.
COPY deployment/hermes/requirements.txt /tmp/hermes-requirements.txt
RUN python -m venv /opt/hermes && /opt/hermes/bin/pip install --no-cache-dir -r /tmp/hermes-requirements.txt
COPY backend/pyproject.toml /app/pyproject.toml
COPY backend/app /app/app
RUN python -m venv /opt/jarvis && /opt/jarvis/bin/pip install --no-cache-dir '/app[sandbox]' \
    && /opt/jarvis/bin/python -c "import app.tasks.worker; import app.sandbox.images"
RUN groupadd --gid 10001 jarvis && useradd --uid 10001 --gid 10001 --create-home jarvis \
    && mkdir -p /var/lib/jarvis/hermes && chown -R 10001:10001 /var/lib/jarvis \
    && chmod 700 /var/lib/jarvis/hermes
ENV PATH="/opt/jarvis/bin:/usr/bin:/bin" SECOND_BRAIN_HERMES_PYTHON=/opt/hermes/bin/python \
    SECOND_BRAIN_HERMES_WORK_ROOT=/var/lib/jarvis/hermes
USER 10001:10001
CMD ["python","-m","app.tasks.worker","run"]
