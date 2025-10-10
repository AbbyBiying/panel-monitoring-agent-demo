# testing-examples/datasets/tag_dataset_version.py
from langsmith import Client
from dotenv import load_dotenv


load_dotenv()
client = Client()

print([d.name for d in client.list_datasets(limit=100)])
ds = client.read_dataset(dataset_name="Panel Monitoring Cases")

# Get the latest version, then tag it "v1" (or "prod")
v = client.read_dataset_version(dataset_name=ds.name, tag="latest")
client.update_dataset_tag(dataset_name=ds.name, as_of=v.as_of, tag="v1")

print(f"[OK] Tagged dataset '{ds.name}' at {v.as_of} as 'v1'")
