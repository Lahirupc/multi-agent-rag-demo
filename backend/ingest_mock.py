import json
from typing import List

# Mock data mapping to the requirements
MOCK_DATA = [
    {
        "id": "inc-001",
        "text": "Payment gateway outage during Q3 due to excessive token timeout thresholds.",
        "attribution": "Incident Report #2904",
        "metadata": {
            "department": "payments",
            "document_type": "incident",
            "access_level": "internal",
            "created_date": "2025-01-01"
        }
    },
    {
        "id": "run-001",
        "text": "To restart the core payment routing engine, clear the Redis cache and bounce the pod.",
        "attribution": "Payment Routing Runbook",
        "metadata": {
            "department": "payments",
            "document_type": "runbook",
            "access_level": "internal",
            "created_date": "2023-11-12"
        }
    },
    {
        "id": "arch-001",
        "text": "The payments architecture relies on RabbitMQ queues passing messages to the Settlement Engine.",
        "attribution": "System Architecture V2.1",
        "metadata": {
            "department": "payments",
            "document_type": "architecture",
            "access_level": "restricted",
            "created_date": "2024-05-30"
        }
    },
    {
        "id": "spec-001",
        "text": "Product specification: Commercial Tier accounts get up to 10k transfers/day with 0.1% processing fee.",
        "attribution": "Product Spec - Commercial Tier",
        "metadata": {
            "department": "product",
            "document_type": "specification",
            "access_level": "public",
            "created_date": "2025-02-14"
        }
    }
]

def generate_mock_json():
    with open("mock_pinecone_data.json", "w") as f:
        json.dump(MOCK_DATA, f, indent=4)
        
    print(f"Generated {len(MOCK_DATA)} mock records in mock_pinecone_data.json")

if __name__ == "__main__":
    generate_mock_json()
    # In a full run, we would initialize Pinecone, embed the texts (OpenAI/Gemini),
    # compute BM25/SPLADE, and upsert vectors matching the structure.
