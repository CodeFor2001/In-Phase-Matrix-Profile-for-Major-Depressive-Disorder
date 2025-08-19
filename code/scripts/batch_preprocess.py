import pandas as pd, subprocess, shlex, os

idx = "data/interim/modma_eeg_index.csv"
df = pd.read_csv(idx)

for _, row in df.iterrows():
  fp = row["filepath"]
  sid = str(row["subject_id"])

  if not os.path.exists(fp):
      print("Missing:", fp)
      continue

  out = f"data/interim/sub-{sid}_epochs.npz"
  if os.path.exists(out):
      print("Skip (exists):", sid)
      continue

  cmd = f'python code/preprocess/preprocess_one.py "{fp}" {sid}'
  print("RUN:", cmd)
  subprocess.run(shlex.split(cmd), check=False)
