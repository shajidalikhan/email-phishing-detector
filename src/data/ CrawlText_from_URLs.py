from bs4 import BeautifulSoup
import time

# Combine OpenPhish + PhishStats URLs
all_urls = openphish_urls + all_phishing
print(f"Total URLs to crawl: {len(all_urls)}")
print("=" * 40)

results = []
failed = 0

for i, url in enumerate(all_urls):
    try:
        res = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)[:500]

        if text.strip():
            results.append({
                "Email": text,
                "Email Type": "Phishing Email"
            })
        else:
            failed += 1

    except Exception:
        failed += 1

    if (i + 1) % 100 == 0:
        print(f"Progress: {i+1}/{len(all_urls)} — Success: {len(results)} — Failed: {failed}")

    time.sleep(1)

phishing_text_df = pd.DataFrame(results)
phishing_text_df.to_csv("phishing_crawled_text.csv", index=False)

print("=" * 40)
print(f"Success  : {len(results)}")
print(f"Failed   : {failed}")
print(f"Saved to : phishing_crawled_text.csv")
print(phishing_text_df.head())