"""
Simple script to test the batch endpoint with proper UTF-8 handling.
"""
import requests
import json

# Load input
with open("test_batch_input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Make request
response = requests.post(
    "http://localhost:8001/batch",
    json=data,
    headers={"Content-Type": "application/json; charset=utf-8"}
)

# Save output with proper UTF-8
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(response.json(), f, ensure_ascii=False, indent=2)

print("✅ Response saved to output.json")
print("\n--- Preview ---")
print(json.dumps(response.json(), ensure_ascii=False, indent=2)[:2000])
