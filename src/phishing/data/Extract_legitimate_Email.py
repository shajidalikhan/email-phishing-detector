import tarfile
import email as email_lib
import requests
import pandas as pd

# Download
print("Downloading Enron Dataset (~400MB)...")
enron_url = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
response = requests.get(enron_url, stream=True)

with open("enron_mail.tar.gz", "wb") as f:
    total = 0
    for chunk in response.iter_content(chunk_size=1024*1024):
        f.write(chunk)
        total += len(chunk)
        print(f"Downloaded: {total / (1024*1024):.1f} MB", end="\r")

print("\nDownload complete!")

# Extract
print("Extracting 500 legitimate emails...")
all_legit = []
limit = 500

with tarfile.open("enron_mail.tar.gz", "r:gz") as tar:
    for member in tar.getmembers():
        if len(all_legit) >= limit:
            break
        if not member.isfile():
            continue
        try:
            f = tar.extractfile(member)
            if f is None:
                continue
            raw = f.read().decode("utf-8", errors="ignore")
            msg = email_lib.message_from_string(raw)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="ignore")
            if body.strip():
                all_legit.append({
                    "Email": body.strip()[:500],
                    "Email Type": "Safe Email"
                })
                if len(all_legit) % 100 == 0:
                    print(f"Parsed: {len(all_legit)}/{limit}")
        except Exception:
            continue

legit_df = pd.DataFrame(all_legit)
legit_df.to_csv("legitimate_emails.csv", index=False)
print(f"Saved {len(legit_df)} legitimate emails!")
print(legit_df.head())