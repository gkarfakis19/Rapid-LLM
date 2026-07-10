
import csv, ast, math, sys
GBS=64
src, dst = sys.argv[1], sys.argv[2]
rows=list(csv.DictReader(open(src), delimiter="\t"))
kept=[]
for row in rows:
    p=ast.literal_eval(row["parallelism"])
    dp,pp=p["train"]["dp"],p["pp"]
    mini=math.ceil(GBS/dp); micro=math.ceil(mini/p["mb"]) if pp>1 else mini
    eff=dp*(p["mb"]*micro if pp>1 else mini)
    if eff==GBS: kept.append(row)
w=csv.DictWriter(open(dst,"w",newline=""), fieldnames=rows[0].keys(), delimiter="\t")
w.writeheader(); [w.writerow(r) for r in kept]
print(f"{src}: {len(rows)} -> {len(kept)} honest rows")
