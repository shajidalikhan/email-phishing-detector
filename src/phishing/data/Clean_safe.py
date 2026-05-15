from sklearn.utils import shuffle

# ── LOAD FRESH ────────────────────────────────────────────────────────────────
phishing_df = pd.read_csv("phishing_crawled_text.csv")
legit_df = pd.read_csv("legitimate_emails.csv")

# ── MERGE ─────────────────────────────────────────────────────────────────────
final_df = pd.concat([phishing_df, legit_df], ignore_index=True)

# ── LABEL ENCODING FIRST ──────────────────────────────────────────────────────
final_df["label"] = final_df["Email Type"].map({
    "Phishing Email" : 1,
    "Safe Email"     : 0
})

# ── RENAME COLUMNS ────────────────────────────────────────────────────────────
final_df = final_df.rename(columns={"Email": "body"})

# ── KEEP ONLY NEEDED COLUMNS — body first, label second ──────────────────────
final_df = final_df[["body", "label"]]

# ── DROP NULLS ────────────────────────────────────────────────────────────────
final_df.dropna(subset=["body", "label"], inplace=True)

# ── DROP EMPTY ────────────────────────────────────────────────────────────────
final_df = final_df[final_df["body"].str.strip() != ""]

# ── DROP DUPLICATES ───────────────────────────────────────────────────────────
final_df.drop_duplicates(subset=["body"], inplace=True)

# ── STRIP WHITESPACE ──────────────────────────────────────────────────────────
final_df["body"] = final_df["body"].str.strip()

# ── FIX LABELS ────────────────────────────────────────────────────────────────
final_df = final_df[final_df["label"].isin([0, 1])]

# ── SHUFFLE ───────────────────────────────────────────────────────────────────
final_df = shuffle(final_df, random_state=42)
final_df.reset_index(drop=True, inplace=True)

# ── SAVE ──────────────────────────────────────────────────────────────────────
final_df.to_csv("phishing_dataset_final.csv", index=False)

print("=" * 50)
print("      FINAL DATASET SAVED ✅")
print("=" * 50)
print(f"Total records  : {len(final_df)}")
print(f"Phishing  (1)  : {int(final_df['label'].sum())}")
print(f"Legitimate (0) : {(final_df['label'] == 0).sum()}")
print(f"Columns        : {final_df.columns.tolist()}")
print("=" * 50)
print(final_df.sample(5))