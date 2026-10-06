import pandas as pd
import requests
from pathlib import Path

BASE = "https://raw.githubusercontent.com/alissonlinneker/DataUrnas-BR/main/data/parquet"
FILES = ["secoes.parquet", "votos_t1_a.parquet", "votos_t1_b.parquet"]
OUT = Path("resultado")
OUT.mkdir(exist_ok=True)
NORDESTE = ["AL","BA","CE","MA","PB","PE","PI","RN","SE"]

for name in FILES:
    p = Path(name)
    if not p.exists():
        with requests.get(f"{BASE}/{name}", stream=True, timeout=240) as r:
            r.raise_for_status()
            with p.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk: fh.write(chunk)

sec = pd.read_parquet("secoes.parquet")
sec["uf"] = sec["uf"].astype(str).str.upper()
sec["turno_num"] = pd.to_numeric(sec["turno"], errors="coerce")
sec["modelo_urna"] = pd.to_numeric(sec["modelo_urna"], errors="coerce")
sec["erros_num"] = pd.to_numeric(sec["erros_log"], errors="coerce").fillna(0)

ne = sec[(sec["turno_num"]==1) & (sec["uf"].isin(NORDESTE)) & (sec["modelo_urna"].isin([2013,2015,2020]))].copy()
ne["grupo_modelo"] = ne["modelo_urna"].map({2013:"2013/2015",2015:"2013/2015",2020:"2020"})

ids = set(ne["id"].astype(str))
va = pd.read_parquet("votos_t1_a.parquet")
vb = pd.read_parquet("votos_t1_b.parquet")
v = pd.concat([va,vb], ignore_index=True)
v["secao_id"] = v["secao_id"].astype(str)
v = v[v["secao_id"].isin(ids)].copy()
cargo = v["cargo"].astype(str).str.upper()
pres = v[cargo.str.contains("PRESIDENT", na=False)].copy()
pres["codigo_candidato"] = pd.to_numeric(pres["codigo_candidato"], errors="coerce")
pres["quantidade"] = pd.to_numeric(pres["quantidade"], errors="coerce").fillna(0)
nom = pres[pres["tipo_voto"].astype(str).str.lower().eq("nominal")].copy()

pv=(nom[nom["codigo_candidato"].isin([13,22])]
    .pivot_table(index="secao_id",columns="codigo_candidato",values="quantidade",aggfunc="sum",fill_value=0)
    .rename(columns={13:"lula_13",22:"bolsonaro_22"})
    .reset_index())
for c in ["lula_13","bolsonaro_22"]:
    if c not in pv.columns: pv[c]=0

res = ne.merge(pv,left_on="id",right_on="secao_id",how="left")
res["lula_13"]=res["lula_13"].fillna(0).astype(int)
res["bolsonaro_22"]=res["bolsonaro_22"].fillna(0).astype(int)
res["total_13_22"]=res["lula_13"]+res["bolsonaro_22"]
res["pct_lula_secao"]=100*res["lula_13"]/res["total_13_22"].replace(0,float("nan"))

def summarize(df, by):
    g=(df.groupby(by,dropna=False)
       .agg(secoes=("id","count"),
            com_erro=("erros_num",lambda s:(s>0).sum()),
            soma_erros=("erros_num","sum"),
            votos_lula=("lula_13","sum"),
            votos_bolsonaro=("bolsonaro_22","sum"))
       .reset_index())
    g["total_13_22"]=g["votos_lula"]+g["votos_bolsonaro"]
    den=g["total_13_22"].astype(float).replace(0,float("nan"))
    g["pct_lula"]=(100*g["votos_lula"]/den).round(2)
    g["pct_bolsonaro"]=(100*g["votos_bolsonaro"]/den).round(2)
    g["pct_secoes_com_erro"]=(100*g["com_erro"]/g["secoes"]).round(2)
    return g

summarize(res,["grupo_modelo"]).to_csv(OUT/"comparacao_modelos_nordeste.csv",index=False,encoding="utf-8-sig")
summarize(res,["uf","grupo_modelo"]).to_csv(OUT/"comparacao_modelos_por_uf.csv",index=False,encoding="utf-8-sig")
summarize(res,["modelo_urna"]).to_csv(OUT/"comparacao_modelos_individuais.csv",index=False,encoding="utf-8-sig")

# Comparação dentro do mesmo município: média ponderada da diferença do % Lula entre 2020 e 2013/2015.
mun=(res.groupby(["uf","municipio","grupo_modelo"],dropna=False)
     .agg(votos_lula=("lula_13","sum"),votos_bolsonaro=("bolsonaro_22","sum"),secoes=("id","count"))
     .reset_index())
mun["total"]=mun["votos_lula"]+mun["votos_bolsonaro"]
mun["pct_lula"]=100*mun["votos_lula"]/mun["total"].replace(0,float("nan"))
p=mun.pivot_table(index=["uf","municipio"],columns="grupo_modelo",values=["pct_lula","total","secoes"],aggfunc="first")
p.columns=["_".join(map(str,c)) for c in p.columns]
p=p.reset_index()
needed=["pct_lula_2013/2015","pct_lula_2020","total_2013/2015","total_2020"]
for c in needed:
    if c not in p.columns: p[c]=pd.NA
p=p.dropna(subset=["pct_lula_2013/2015","pct_lula_2020"]).copy()
p["dif_pct_lula_2020_menos_antigas"]=p["pct_lula_2020"]-p["pct_lula_2013/2015"]
p["peso"]=p[["total_2013/2015","total_2020"]].min(axis=1)
p.to_csv(OUT/"comparacao_mesmo_municipio.csv",index=False,encoding="utf-8-sig")

weighted=(p["dif_pct_lula_2020_menos_antigas"]*p["peso"]).sum()/p["peso"].sum()
simple=p["dif_pct_lula_2020_menos_antigas"].mean()
median=p["dif_pct_lula_2020_menos_antigas"].median()
share_pos=(p["dif_pct_lula_2020_menos_antigas"]>0).mean()*100

with (OUT/"RESUMO_COMPARACAO.txt").open("w",encoding="utf-8") as fh:
    fh.write("Nordeste 2022 - 1º turno - comparação modelos 2013/2015 vs 2020\n")
    fh.write(summarize(res,["grupo_modelo"]).to_string(index=False))
    fh.write("\n\nComparação dentro do mesmo município (Lula % em 2020 menos Lula % em 2013/2015):\n")
    fh.write(f"Municípios comparáveis: {len(p)}\n")
    fh.write(f"Diferença média simples: {simple:.3f} p.p.\n")
    fh.write(f"Diferença mediana: {median:.3f} p.p.\n")
    fh.write(f"Diferença média ponderada: {weighted:.3f} p.p.\n")
    fh.write(f"Percentual de municípios em que 2020 teve % Lula maior: {share_pos:.2f}%\n")

print(summarize(res,["grupo_modelo"]).to_string(index=False))
print("municipios",len(p),"media",simple,"mediana",median,"ponderada",weighted,"sharepos",share_pos)
