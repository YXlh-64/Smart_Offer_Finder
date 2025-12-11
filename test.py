import requests

API_KEY = "4EVnEyJP7s8zagpmGPQrhhd41D7Imi2EArfoZHatqnSUk9YC0gmEdlMVTBVHirjLLpVXiwqxFUd_65B8ZuBdog"  # <-- Put your key here privately (don’t share it)
API_URL = "https://api.deepseek.com"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
}

response = requests.post(API_URL, headers=headers, json=data)

if response.status_code == 200:
    print("Response:")
    print(response.json()["choices"][0]["message"]["content"])
else:
    print("Error:", response.status_code, response.text)
