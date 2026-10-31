"""Semantic distance per system on the LoRA pilot windows. usage: semdist_lora.py TEXTS_CSV"""
import re
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

UNIN = re.compile(r"<\s*(?:UNIN|INAUDIBLE_SPEECH)\s*/?\s*>", re.I)
TAGS = re.compile(r"</?\s*[A-Za-z_]+\s*/?\s*>")
clean = lambda s: re.sub(r"\s+", " ", TAGS.sub(" ", UNIN.sub(" ", str(s)))).strip()

t = pd.read_csv(sys.argv[1], keep_default_na=False)
t["consultation"] = t.unit_id.str.extract(r"pm_win_(day\d+_consultation\d+)_")[0]
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
ref = m.encode(t.reference.map(clean).tolist(), batch_size=32, normalize_embeddings=True)
sd = {}
for s in ("small_gpu", "base", "lora", "turbo"):
    e = m.encode(t[s].map(clean).tolist(), batch_size=32, normalize_embeddings=True)
    sd[s] = 1 - (ref * e).sum(1)
    print(f"{s:<10} semdist mean {sd[s].mean():.4f}  windows > 0.1: {(sd[s] > 0.1).mean():.1%}")
d = pd.DataFrame({"c": t.consultation, "x": sd["lora"] - sd["base"]})
g = d.groupby("c").x.agg(["sum", "count"])
rng = np.random.default_rng(598)
v = [g.loc[s, "sum"].sum() / g.loc[s, "count"].sum() for s in (rng.choice(g.index, len(g)) for _ in range(1000))]
print(f"lora - base: {d.x.mean():+.4f}  95% CI [{np.quantile(v, .025):+.4f}, {np.quantile(v, .975):+.4f}]")
