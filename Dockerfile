# Multi-stage build: Python deps and Node MCP servers are built in their own stages;
# the runtime image is python:3.13-slim with a non-root user.

# ---- Python dependencies -----------------------------------------------------
# A virtualenv at /opt/venv (not `pip install --user`): /root is mode 700 in this
# image, so packages under /root/.local would be unreadable by the non-root user.
FROM python:3.13-slim AS python-deps
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Node MCP servers (filesystem, Tavily) -------------------------------------
# Installed from package-lock.json so the image runs the exact tested versions.
FROM node:22-slim AS node-deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# ---- Runtime ---------------------------------------------------------------------
FROM python:3.13-slim
WORKDIR /app

# Node runtime for the stdio MCP servers launched from node_modules/.bin
COPY --from=node-deps /usr/local/bin/node /usr/local/bin/node
COPY --from=node-deps /app/node_modules ./node_modules
COPY --from=python-deps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY . .

# Code stays root-owned (read-only for the app); only the session-data dir is writable.
RUN useradd --create-home appuser \
    && mkdir -p data \
    && chown appuser:appuser data
USER appuser

EXPOSE 8080
CMD ["python", "main.py"]
