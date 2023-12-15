import urllib.request
import requests
import json
from datetime import datetime


data = requests.get("https://www.reddit.com/r/Cooking/new/.json?limit=10")
#with urllib.request.urlopen() as page:
print(type(data.text))
data = json.loads(data.text)

for i, post in enumerate(data["data"]["children"], start=1):
    # post["data"]["selftext"]
    print(i, datetime.fromtimestamp(post["data"]["created"]), post["data"]["title"])
