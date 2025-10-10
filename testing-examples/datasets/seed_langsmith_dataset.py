from langsmith import Client

examples = [
    {
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
                "text": (
                    "My email is 818rahim@gmail.com. I'm a 21 year-old man from Austin, TX. "
                    "When asked where I signed up, I said 'from youtube'. "
                    "My recorded source was an Organic signup, not coming from an advertising campaign."
                ),
            },
            "case_id": "youtube_vs_organic_mismatch",
            "min_confidence": 0.6,
        },
        "outputs": {
            "signals": {
                "suspicious_signup": False,
                "normal_signup": True,
                "confidence": 0.75,
                "reason": "Self-reported YouTube source with 'organic' tracking is plausible; details are consistent and non-automated.",
            },
            "classification": "normal",
            "action": "allow",
            "log_entry": "log-normal-youtube-001",
            "explanation_report": "Signup appears human with consistent demographics; YouTube vs. organic attribution is a common tracking discrepancy.",
        },
        "metadata": {
            "source": "seed_example",
            "reason_contains": "tracking discrepancy plausible",
        },
    },
    {
        "inputs": {
            "raw_event": {
                "case_id": "geo_ip_mismatch_ct_vs_ca",
                "name": "Pierre",
                "email": "starttony916@gmail.com",
                "age": 23,
                "gender": "male",
                "address": {
                    "street": "425 Grant Street",
                    "city": "Bridgeport",
                    "state": "CT",
                    "postal_code": "06610",
                },
                "stated_city": "Bridgeport",
                "stated_state": "CT",
                "ip_geo_city": "Los Angeles",
                "ip_geo_state": "CA",
                "text": (
                    "My name is Pierre and my email address is starttony916@gmail.com. "
                    "I'm a 23 year-old male from Bridgeport, Ct. My address is 425 Grant Street, "
                    "Bridgeport, Connecticut, 06610. But I signed up with an IP address identified "
                    "as located in Los Angeles, CA."
                ),
            },
            "case_id": "geo_ip_mismatch_ct_vs_ca",
            "min_confidence": 0.7,
        },
        "outputs": {
            "signals": {
                "suspicious_signup": True,
                "normal_signup": False,
                "confidence": 0.85,
                "reason": "Geolocation mismatch: stated address in CT but signup IP resolves to Los Angeles, CA.",
            },
            "classification": "suspicious",
            "action": "challenge",
            "log_entry": "log-suspicious-geoip-001",
            "explanation_report": "IP geolocation indicates Los Angeles, CA while user claims residence in Bridgeport, CT; treat as high-risk until verified.",
        },
        "metadata": {
            "source": "seed_example",
            "reason_contains": "geolocation mismatch",
        },
    },
    {
        # Added from your runtime log (final_action = remove_account)
        "inputs": {
            "raw_event": {
                "case_id": "geo_ip_mismatch_ct_vs_ca_runtime",
                "text": (
                    "My name is Pierre and my email address is starttony916@gmail.com. "
                    "I'm a 23 year-old male from Bridgeport, Ct. My address is 425 Grant Street, "
                    "Bridgeport, Connecticut, 06610. But I signed up with an IP address identified "
                    "as located in Los Angeles, CA."
                ),
            },
            "case_id": "geo_ip_mismatch_ct_vs_ca_runtime",
            "min_confidence": 0.9,
        },
        "outputs": {
            "signals": {
                "suspicious_signup": True,
                "normal_signup": False,
                "confidence": 0.9,
                "reason": "Stated address is in Bridgeport, CT, but the signup IP address is located in Los Angeles, CA.",
            },
            "classification": "suspicious",
            "action": "remove_account",
            "log_entry": "2025-10-10T03:20:28.539852+00:00",
            "explanation_report": "Stated address is in Bridgeport, CT, but the signup IP address is located in Los Angeles, CA.",
        },
        "metadata": {
            "source": "runtime_log",
            "reason_contains": "geolocation mismatch",
        },
    },
]

client = Client()
dataset = client.create_dataset(
    dataset_name="Panel Monitoring Cases",
    description="Panel monitoring cases with reference outputs aligned to GraphState.",
)

client.create_examples(dataset_id=dataset.id, examples=examples)
