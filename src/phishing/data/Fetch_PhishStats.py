import requests
import time

all_phishing = []
page = 1
TARGET = 10000

print("Fetching PhishStats URLs (target: 10,000)...")
print("=" * 40)

while page <= 100:  # 100 pages × 100 = 10,000 URLs
    try:
        url = f"https://api.phishstats.info/api/phishing?_sort=-date&_p={page}&_size=100"
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            data = response.json()

            if not data:
                print(f"No more data on page {page} — stopping early.")
                break

            for entry in data:
                all_phishing.append(entry.get("url", ""))
            print(f"Page {page}/100 — Records: {len(all_phishing)}/{TARGET} ✅")
            page += 1
            time.sleep(7)

        elif response.status_code == 429:
            print(f"Rate limited — waiting 30s before retrying page {page}...")
            time.sleep(30)  # retry same page, don't skip

        else:
            print(f"Failed: {response.status_code} — skipping page {page}")
            page += 1

    except Exception as e:
        print(f"Error: {e} — skipping page {page}")
        page += 1
        time.sleep(10)

print("=" * 40)
print(f"Total URLs fetched : {len(all_phishing)}")
print(f"Target             : {TARGET}")
print(f"Pages scraped      : {page - 1}/100")