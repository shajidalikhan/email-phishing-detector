import requests

websites = {
    "PhishTank"  : "https://www.phishtank.com",
    "OpenPhish"  : "https://openphish.com/feed.txt",
    "PhishStats" : "https://api.phishstats.info/api/phishing?_sort=-date&_p=1&_size=1",
    "URLhaus"    : "https://urlhaus-api.abuse.ch/v1/urls/recent/limit/10/",
    "AlienVault" : "https://otx.alienvault.com/api/v1/indicators/domain/phishing.com/general"
}

print("Checking all phishing websites...")
print("=" * 50)

for name, url in websites.items():
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        if response.status_code == 200:
            print(f"✅ {name} — Status: 200 — Working!")
        elif response.status_code == 403:
            if "cloudflare" in response.text.lower():
                print(f"❌ {name} — Status: 403 — Cloudflare Protected!")
            else:
                print(f"❌ {name} — Status: 403 — Forbidden!")
        elif response.status_code == 429:
            print(f"⚠️ {name} — Status: 429 — Rate Limited!")
        elif response.status_code == 401:
            print(f"❌ {name} — Status: 401 — Unauthorized!")
        else:
            print(f"⚠️ {name} — Status: {response.status_code}")
    except Exception as e:
        print(f"❌ {name} — Error: {e}")

print("=" * 50)
#  valid are --> OpenPhish ,PhishStats and Alienvault
