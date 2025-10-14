## Requirements
Cloud Run Functions requires `requirements.txt`.  Locally, we can use `uv` to install, but we need to make sure our function's requirements are also in `requirements.txt`.

```
gcloud run functions deploy pubsub-to-langsmith \
  --region=us-central1 \
  --runtime=python312 \
  --source=. \
  --entry-point=pubsub_to_langsmith \
  --trigger-topic=panel-events
  --set-secrets=LANGSMITH_API_KEY=projects/$PROJECT_ID/secrets/LANGSMITH_API_KEY:latest
```
