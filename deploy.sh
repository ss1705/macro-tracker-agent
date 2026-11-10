#!/usr/bin/env bash
# Build the image with Cloud Build and deploy it to Cloud Run.
#
# Settings and API keys come from .env (never committed); nothing secret is written
# in this file. Every gcloud command passes --project explicitly instead of relying on
# `gcloud config set project`, so the machine's global gcloud config is left alone.
set -euo pipefail
cd "$(dirname "$0")"

set -a
source .env
set +a

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID in .env}"
for var in OPENAI_API_KEY MODEL_NAME NIM_API_KEY NIM_BASE_URL FALLBACK_MODEL_NAME \
           TAVILY_API_KEY USDA_API_KEY; do
  if [ -z "${!var:-}" ]; then
    echo "Missing $var in .env" >&2
    exit 1
  fi
done

SERVICE=csci599-a1
REGION=us-west1
IMAGE="gcr.io/${GCP_PROJECT_ID}/${SERVICE}"  # served by Artifact Registry

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  --project "$GCP_PROJECT_ID"

gcloud builds submit --tag "$IMAGE" --project "$GCP_PROJECT_ID"

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$GCP_PROJECT_ID" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 1 \
  --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY},MODEL_NAME=${MODEL_NAME},NIM_API_KEY=${NIM_API_KEY},NIM_BASE_URL=${NIM_BASE_URL},FALLBACK_MODEL_NAME=${FALLBACK_MODEL_NAME},TAVILY_API_KEY=${TAVILY_API_KEY},USDA_API_KEY=${USDA_API_KEY},MCP_SERVERS_CONFIG=/app/mcp_config.json"
