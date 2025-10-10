from langsmith import Client

example = {
  "inputs": {
    "raw_event": {
      "case_id": "youtube_vs_organic_mismatch",
      "email": "818rahim@gmail.com",
      "age": 21,
      "gender": "male",
      "city": "Austin",
      "state": "TX",
      "claimed_source": "youtube",
      "recorded_source": "organic",
      "text": "My email is 818rahim@gmail.com. I'm a 21 year-old man from Austin, TX. When asked where I signed up, I said 'from youtube'. My recorded source was an Organic signup, not coming from an advertising campaign."
    },
    "case_id": "youtube_vs_organic_mismatch",
    "min_confidence": 0.6
  },
  "outputs": {
    "signals": {
      "suspicious_signup": False,
      "normal_signup": True,
      "confidence": 0.75,
      "reason": "Self-reported YouTube source with 'organic' tracking is plausible (direct/SEO); details are consistent and non-automated."
    },
    "classification": "normal",
    "action": "allow",
    "log_entry": "log-normal-youtube-001",
    "explanation_report": "Signup appears human with consistent demographics; YouTube vs. organic attribution is a common tracking discrepancy."
  },
  "metadata": {
    "source": "seed_example",
    "reason_contains": "tracking discrepancy plausible"
  }
}

client = Client()
ds = client.create_dataset(
  dataset_name="Panel Monitoring Cases",
  description="Panel monitoring cases with reference outputs aligned to GraphState.",
)
client.create_examples(dataset_id=ds.id, examples=[example])
