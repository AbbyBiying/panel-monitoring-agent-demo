## Requirements
Cloud Run Functions requires `requirements.txt`.  Locally, we can use `uv` to install, but we need to make sure our function's requirements are also in `requirements.txt`.

## Deployment

### Prerequisites
1. Create a secret in Secret Manager for your LangSmith API key:
```bash
# Create the secret (one-time setup)
echo -n "your-langsmith-api-key" | gcloud secrets create LANGSMITH_API_KEY_CLOUD_FUNCTIONS --data-file=-

# Grant the Cloud Run service account access to the secret
gcloud secrets add-iam-policy-binding LANGSMITH_API_KEY_CLOUD_FUNCTIONS \
  --member="serviceAccount:$PROJECT_ID@appspot.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Deploy the Function

**Quick Deploy (uses `.env` file):**
```bash
# From project root
just deploy-pubsub
```

**Manual Deploy:**
```bash
# From gcp/functions/pubsub_to_langsmith/
gcloud functions deploy pubsub-to-langsmith \
  --region=us-central1 \
  --runtime=python313 \
  --source=. \
  --entry-point=pubsub_to_langsmith \
  --trigger-topic=user-event-signups \
  --set-env-vars LANGSMITH_PROJECT=your-project-name,LOG_LEVEL=INFO \
  --set-secrets LANGSMITH_API_KEY_CLOUD_FUNCTIONS=LANGSMITH_API_KEY_CLOUD_FUNCTIONS:latest
```

## Local Development

For local testing and deployment, create a `.env` file in the function directory:

```bash
# gcp/functions/pubsub_to_langsmith/.env
LANGSMITH_API_KEY_CLOUD_FUNCTIONS=lsv2_xxx_your_key_here
LANGSMITH_PROJECT=your-project-name
LOG_LEVEL=DEBUG
```

**Important:** Add `.env` to your `.gitignore` to avoid committing secrets!
