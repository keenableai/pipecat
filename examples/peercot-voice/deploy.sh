#!/usr/bin/env bash
# Deploy Voice PeerCoT to Cloud Run.
#
# First run only:
#   1) gcloud auth login && gcloud config set project YOUR_PROJECT
#   2) gcloud services enable run.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
#   3) Create secrets (skip if they already exist in the project):
#        printf '%s' "$XAI_API_KEY"      | gcloud secrets create XAI_API_KEY --data-file=-
#        printf '%s' "$KEENABLE_API_KEY"  | gcloud secrets create KEENABLE_API_KEY --data-file=-
#   4) Grant Cloud Run access:
#        PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
#        for s in XAI_API_KEY KEENABLE_API_KEY; do
#          gcloud secrets add-iam-policy-binding "$s" \
#            --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
#            --role=roles/secretmanager.secretAccessor
#        done

set -euo pipefail

SERVICE="${SERVICE:-peercot-voice}"
REGION="${REGION:-us-west1}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Cloud Run --source needs Dockerfile at source root
cp "$(dirname "$0")/Dockerfile" "$REPO_ROOT/Dockerfile"
cd "$REPO_ROOT"

gcloud run deploy "$SERVICE" \
  --source=. \
  --region="$REGION" \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=3 \
  --concurrency=10 \
  --cpu=2 --memory=1Gi \
  --timeout=3600 \
  --session-affinity \
  --set-secrets="XAI_API_KEY=XAI_API_KEY:latest,KEENABLE_API_KEY=KEENABLE_API_KEY:latest" \
  --set-env-vars="PEERCOT_MODE=debate,PEERCOT_TURNS=3"

# Clean up
rm -f "$REPO_ROOT/Dockerfile"

echo ""
echo "Deployed! Update topic with:"
echo "  gcloud run services update $SERVICE --region=$REGION --set-env-vars='PEERCOT_TOPIC=Will Bitcoin hit 150k'"
