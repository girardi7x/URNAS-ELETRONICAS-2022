import pandas as pd
import requests
from pathlib import Path

BASE = "https://raw.githubusercontent.com/alissonlinneker/DataUrnas-BR/main/data/parquet"
FILES = ["secoes.parquet", "issues.parquet", "votos_t1_b.parquet"]
OUT = Path("resultado")
OUT.mkdir(exist_ok=True)

for name in FILES:
    p = Path(name)
    if not p.exists():
        print(f"Baixando {name}...")
        with requests.get(f"{BASE}/{name}", stream=True, timeout=180) as r:
            r.raise_for_status()
            with p.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)

sec = pd.read_parquet("secoes.parquet")
sec["modelo_urna"] = sec["modelo_urna"].astype(str)
sec["uf"] = sec["uf"].astype(str).str.upper()
sec["turno_num"] = pd.to_numeric(sec["turno"], errors="coerce")
sec["erros_num"] = pd.to_numeric(sec["erros_log"], errors="coerce").fillna(0)
sec["modelo_norm"] = sec["modelo_urna"].str.upper().str.replace(" ", "", regex=False)

diag_rs = sec[(sec["turno_num"]==1) & (sec["uf"]=="RS")]
(diag_rs.groupby("modelo_urna")
 .agg(secoes=("id","count"), com_erro=("erros_num",lambda s:(s>0).sum()), soma_erros=("erros_num","sum"))
 .reset_index()
 .sort_values("secoes",ascending=False)
 .to_csv(OUT/"diagnostico_modelos_rs.csv",index=False,encoding="utf-8-sig"))

target = diag_rs[
    diag_rs["modelo_norm"].str.contains("2013|2015", regex=True, na=False) &
    (diag_rs["erros_num"] > 0)
].copy()

print("Modelos RS:")
print(diag_rs["modelo_urna"].value_counts().head(30).to_string())
print("\nSeções alvo com erro:", len(target))

ids = set(target["id"].astype(str))

issues = pd.read_parquet("issues.parquet")
issues["secao_id"] = issues["secao_id"].astype(str)
iss = issues[issues["secao_id"].isin(ids)].copy()
iss.to_csv(OUT/"issues_ue2013_ue2015_rs.csv", index=False, encoding="utf-8-sig")

v = pd.read_parquet("votos_t1_b.parquet")
v["secao_id"] = v["secao_id"].astype(str)
v = v[v["secao_id"].isin(ids)].copy()
cargo = v["cargo"].astype(str).str.upper()
pres = v[cargo.str.contains("PRESIDENT", na=False)].copy()
pres["codigo_candidato"] = pd.to_numeric(pres["codigo_candidato"], errors="coerce")
pres["quantidade"] = pd.to_numeric(pres["quantidade"], errors="coerce").fillna(0)
nom = pres[pres["tipo_voto"].astype(str).str.lower().eq("nominal")].copy()
pv = (nom[nom["codigo_candidato"].isin([13,22])]
      .pivot_table(index="secao_id", columns="codigo_candidato", values="quantidade", aggfunc="sum", fill_value=0)
      .rename(columns={13:"lula_13",22:"bolsonaro_22"})
      .reset_index())
for c in ["lula_13","bolsonaro_22"]:
    if c not in pv.columns: pv[c]=0

res = target.merge(pv,left_on="id",right_on="secao_id",how="left")
res["lula_13"] = res["lula_13"].fillna(0).astype(int)
res["bolsonaro_22"] = res["bolsonaro_22"].fillna(0).astype(int)
res["dois_candidatos"] = res["lula_13"]+res["bolsonaro_22"]
res["pct_lula_entre_13_22"]=(100*res["lula_13"]/res["dois_candidatos"].replace(0,pd.NA)).round(2)
res["pct_bolsonaro_entre_13_22"]=(100*res["bolsonaro_22"]/res["dois_candidatos"].replace(0,pd.NA)).round(2)
res["vencedor_13_22"] = res.apply(lambda r:"Lula" if r["lula_13"]>r["bolsonaro_22"] else ("Bolsonaro" if r["bolsonaro_22"]>r["lula_13"] else "Empate"),axis=1)

cols=[c for c in ["id","uf","municipio","zona","secao","modelo_urna","tipo_urna","versao_sw","eleitores_aptos","comparecimento","reboots","erros_log","alertas_mesario","substituicoes","has_issues","n_issues","lula_13","bolsonaro_22","pct_lula_entre_13_22","pct_bolsonaro_entre_13_22","vencedor_13_22"] if c in res.columns]
res[cols].sort_values(["modelo_urna","municipio","zona","secao"]).to_csv(OUT/"secoes_com_erros_e_votos_rs.csv",index=False,encoding="utf-8-sig")

summary=(res.groupby("modelo_urna",dropna=False)
         .agg(secoes=("id","count"),soma_erros=("erros_num","sum"),votos_lula=("lula_13","sum"),votos_bolsonaro=("bolsonaro_22","sum"))
         .reset_index())
summary["total_13_22"]=summary["votos_lula"]+summary["votos_bolsonaro"]
summary["pct_lula"]=(100*summary["votos_lula"]/summary["total_13_22"].replace(0,pd.NA)).round(2)
summary["pct_bolsonaro"]=(100*summary["votos_bolsonaro"]/summary["total_13_22"].replace(0,pd.NA)).round(2)
summary.to_csv(OUT/"resumo_por_modelo.csv",index=False,encoding="utf-8-sig")

winner=res["vencedor_13_22"].value_counts(dropna=False)
with (OUT/"RESUMO.txt").open("w",encoding="utf-8") as f:
    f.write("Análise 2022 - RS - 1º turno - modelos contendo 2013/2015 com erros_log > 0\n")
    f.write(f"Seções encontradas: {len(res)}\n\n")
    f.write("Vencedor entre Lula e Bolsonaro por seção:\n")
    f.write(winner.to_string())
    f.write("\n\nResumo por modelo:\n")
    f.write(summary.to_string(index=False))
    f.write("\n\nObservação: erros_log vem do .logjez da urna e NÃO é o mesmo que erro de transmissão/RecArquivos.\n")
print(summary.to_string(index=False))
print(winner.to_string())
