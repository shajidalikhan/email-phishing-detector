import pandas as pd

# Load all files
phishing_df = pd.read_csv("phishing_crawled_text.csv")
legit_df = pd.read_csv("legitimate_emails.csv")

print(f"Phishing records : {len(phishing_df)}")
print(f"Legitimate records: {len(legit_df)}")

# Merge
final_df = pd.concat([phishing_df, legit_df], ignore_index=True)
print(f"Combined total   : {len(final_df)}")
print(final_df["Email Type"].value_counts())