import os
import pandas as pd

# CONFIG
folder_path = "full_dataset_folder" 
output_file = "BoT_IoT_5    .csv"
chunk_size = 100_000
target_rows = 3_600_000
rows_written = 0

# Remove existing output if needed
if os.path.exists(output_file):
    os.remove(output_file)

print(f"🔍 Scanning folder: {folder_path}")

for filename in os.listdir(folder_path):
    if not filename.endswith(".csv"):
        continue

    filepath = os.path.join(folder_path, filename)
    print(f"📁 Reading: {filename}")

    for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        if 'category' not in chunk.columns:
            print(f"⚠️ Skipping: 'category' column missing in {filename}")
            continue

        chunk['target'] = (chunk['category'] == 'DDoS').astype(int)
        sample = chunk.sample(frac=0.05, random_state=42)

        if rows_written == 0:
            sample.to_csv(output_file, index=False, mode='w')
        else:
            sample.to_csv(output_file, index=False, mode='a', header=False)

        rows_written += len(sample)
        print(f"  ➕ Written {len(sample):,} rows. Total so far: {rows_written:,}")

        if rows_written >= target_rows:
            print(f"\n✅ DONE: Reached {rows_written:,} rows.")
            break

    if rows_written >= target_rows:
        break




#         DoS               1650263
#         Normal                479
#         Reconnaissance      91100
#         Theft                  83
#         DDoS              1861597
# dtype: int64