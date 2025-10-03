import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(
    project="panel-monitoring-agent", location="us-central1", vertexai=True
)

response = client.models.generate_content(
    model="gemini-2.5-pro", contents="Tell me a story in 300 words."
)
print(response.text)

print(response.model_dump_json(exclude_none=True, indent=4))
