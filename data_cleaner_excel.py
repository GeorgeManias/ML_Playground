import pandas as pd
import numpy as np

# 1. Read Excel
df = pd.read_excel("eur_usd_data.xls", engine="xlrd")


# 3. Parse dates
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 4. Sort by date ascending
df = df.sort_values("Date").reset_index(drop=True)

# 5. If prices are imported like 11544 instead of 1.1544,
#    uncomment the next block
# for col in ["Price", "Open", "High", "Low"]:
#     df[col] = df[col] / 10000

# 6. Create features
df["return_1"] = df["Price"].pct_change()
df["range_hl"] = df["High"] - df["Low"]
df["open_close_diff"] = df["Price"] - df["Open"]

# 7. Create target
df["target"] = (df["Price"].shift(-1) > df["Price"]).astype(int)

# 8. Remove missing rows
df = df.dropna().reset_index(drop=True)

# 9. Build X and y
X = df[["return_1", "range_hl", "open_close_diff"]].values
y = df["target"].values

print("X shape:", X.shape)
print("y shape:", y.shape)
print(df.head())