import requests

import pandas as pd

print("Fetching OpenPhish URLs...")
response = requests.get("https://openphish.com/feed.txt")
openphish_urls = response.text.strip().split("\n")

print(f"OpenPhish URLs fetched: {len(openphish_urls)}")
print(openphish_urls[:5])