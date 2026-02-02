# Events report

Small local reporting page that monitors the Firestore **events** collection by `type`. Enter an event type (e.g. signup), load events, and view them in a table (panelist_id, type, status, decision, confidence), sorted by `updated_at` descending (most recent first).

## Run from repo root

1. Set environment variables (same as the rest of the repo):
   - **GCP_PROJECT** or **GOOGLE_CLOUD_PROJECT** – Firestore project ID
   - This app **always** uses **Application Default Credentials** (your local `gcloud auth application-default login`). It does not use `GOOGLE_APPLICATION_CREDENTIALS`.
   - **FIRESTORE_DATABASE_ID** (optional) – defaults to `panel-monitoring-agent-dev`
   - **ENVIRONMENT** (optional) – set to `local` to load credentials from file; otherwise ADC is used

2. From the **repo root**:
   ```bash
   uv run uvicorn main:app --reload --app-dir gcp/events-report
   ```

3. Open [http://localhost:8000](http://localhost:8000), enter an event type, and click **Load events**.

## Firestore composite index

The query filters by `type` and orders by `updated_at` descending. Firestore requires a **composite index** for this.

- **Index**: collection `events`, fields:
  - `type` (Ascending)
  - `updated_at` (Descending)

### Option A: Firebase Console

1. Open [Firebase Console](https://console.firebase.google.com) → your project → Firestore → Indexes.
2. Add a composite index on collection **events** with `type` (Ascending) and `updated_at` (Descending).

### Option B: Firebase CLI (`firestore.indexes.json`)

This repo includes `gcp/events-report/firestore.indexes.json`. From a directory that contains (or links to) this file in a Firebase project:

```bash
firebase deploy --only firestore:indexes
```

### Option C: Google Cloud Console

1. Open [Cloud Console](https://console.cloud.google.com) → Firestore → Indexes.
2. Create a composite index on **events** with `type` (Ascending) and `updated_at` (Descending).

If the index is missing, the first load will fail with an error that includes a link to create the index in the console.
