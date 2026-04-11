import pandas as pd
from feature_extractor import extract_features

# Load datasets
df1 = pd.read_csv("data/phishing1.csv")
df2 = pd.read_csv("data/phishing2.csv")

df = pd.concat([df1, df2], ignore_index=True)

# Detect URL column
url_col = None
for col in df.columns:
    if "url" in col.lower():
        url_col = col
        break

# Detect label column
label_col = None
for col in df.columns:
    if col.lower() in ["label", "class", "target", "phishing"]:
        label_col = col
        break

if url_col is None or label_col is None:
    raise ValueError(f"Columns found: {list(df.columns)}")

print("Using URL column:", url_col)
print("Using label column:", label_col)

# Extract features
features_list = []

for i, row in df.iterrows():
    try:
        features = extract_features(str(row[url_col]))
        features["label"] = row[label_col]
        features_list.append(features)
    except:
        continue

processed_df = pd.DataFrame(features_list)

processed_df.to_csv("data/processed_urls.csv", index=False)

print("Saved: data/processed_urls.csv")
print("Shape:", processed_df.shape)