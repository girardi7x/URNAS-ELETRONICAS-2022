import pandas as pd
import requests
from pathlib import Path

BASE = "https://raw.githubusercontent.com/alissonlinneker/DataUrnas-BR/main/data/parquet"
FILES = ["secoes.parquet", "issues.parquet", "votos_t1_a.parquet", "votos_t1_b.parquet"]
OUT = Path("resultado")
OUT.mkdir(exist_ok=True)

for name in FILES:
    p = Path(name)
    if not p.exists():
        print(f"Baixando {name}...")
        with requests.get(f"{BASE}/{name}", stream=True, timeout=240) as r:
            r.raise_for_status()
            with p.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        fh.write(chunk)

NORDESTE = ["AL","BA","CE","MA","PB","PE","PI","RN","SE"]

sec = pd.read_parquet("secoes.parquet")
sec["modelo_urna"] = sec["modelo_urna"].astype(str)
sec["uf"] = sec["uf"].astype(str).str.upper()
sec["turno_num"] = pd.to_numeric(sec["turno"], errors="coerce")
sec["erros_num"] = pd.to_numeric(sec["erros_log"], errors="coerce").fillna(0)
sec["modelo_norm"] = sec["modelo_urna"].str.upper().str.replace(" ", "", regex=False)

ne = sec[(sec["turno_num"]==1) & (sec["uf"].isin(NORDESTE))].copy()

diag=(ne.groupby(["uf","modelo_urna"])
      .agg(secoes=("id","count"),
           com_erro=("erros_num",lambda s:(s>0).sum()),
           soma_erros=("erros_num","sum"))
      .reset_index())
diag["pct_com_erro"]=(100*diag["com_erro"]/diag["secoes"]).round(2)
diag.to_csv(OUT/"diagnostico_modelos_nordeste.csv",index=False,encoding="utf-8-sig")

target = ne[
    ne["modelo_norm"].str.contains("2013|2015", regex=True, na=False) &
    (ne["erros_num"] > 0)
].copy()
ids=set(target["id"].astype(str))
print("Seções alvo Nordeste UE2013/UE2015 com erro:",len(target))

issues=pd.read_parquet("issues.parquet")
issues["secao_id"]=issues["secao_id"].astype(str)
iss=issues[issues["secao_id"].isin(ids)].copy()
iss.to_csv(OUT/"issues_ue2013_ue2015_nordeste.csv",index=False,encoding="utf-8-sig")

va=pd.read_parquet("votos_t1_a.parquet")
vb=pd.read_parquet("votos_t1_b.parquet")
v=pd.concat([va,vb],ignore_index=True)
v["secao_id"]=v["secao_id"].astype(str)
v=v[v["secao_id"].isin(ids)].copy()
cargo=v["cargo"].astype(str).str.upper()
pres=v[cargo.str.contains("PRESIDENT",na=False)].copy()
pres["codigo_candidato"]=pd.to_numeric(pres["codigo_candidato"],errors="coerce")
pres["quantidade"]=pd.to_numeric(pres["quantidade"],errors="coerce").fillna(0)
nom=pres[pres["tipo_voto"].astype(str).str.lower().eq("nominal")].copy()

pv=(nom[nom["codigo_candidato"].isin([13,22])]
    .pivot_table(index="secao_id",columns="codigo_candidato",values="quantidade",aggfunc="sum",fill_value=0)
    .rename(columns={13:"lula_13",22:"bolsonaro_22"})
    .reset_index())
for c in ["lula_13","bolsonaro_22"]:
    if c not in pv.columns: pv[c]=0

res=target.merge(pv,left_on="id",right_on="secao_id",how="left")
res["lula_13"]=res["lula_13"].fillna(0).astype(int)
res["bolsonaro_22"]=res["bolsonaro_22"].fillna(0).astype(int)
res["dois_candidatos"]=res["lula_13"]+res["bolsonaro_22"]
den=res["dois_candidatos"].astype(float).replace(0,float("nan"))\nres["pct_lula_entre_13_22"]=(100*res["lula_13"]/den).round(2)
res["pct_bolsonaro_entre_13_22"]=(100*res["bolsonaro_22"]/den).round(2)
res["vencedor_13_22"]=res.apply(lambda r:"Lula" if r["lula_13"]>r["bolsonaro_22"] else ("Bolsonaro" if r["bolsonaro_22"]>r["lula_13"] else "Empate"),axis=1)

cols=[c for c in ["id","uf","municipio","zona","secao","modelo_urna","tipo_urna","versao_sw","eleitores_aptos","comparecimento","reboots","erros_log","alertas_mesario","substituicoes","has_issues","n_issues","lula_13","bolsonaro_22","pct_lula_entre_13_22","pct_bolsonaro_entre_13_22","vencedor_13_22"] if c in res.columns]
res[cols].sort_values(["uf","modelo_urna","municipio","zona","secao"]).to_csv(OUT/"secoes_com_erros_e_votos_nordeste.csv",index=False,encoding="utf-8-sig")

summary_model=(res.groupby("modelo_urna",dropna=False)
               .agg(secoes=("id","count"),soma_erros=("erros_num","sum"),votos_lula=("lula_13","sum"),votos_bolsonaro=("bolsonaro_22","sum"))
               .reset_index())
summary_model["total_13_22"]=summary_model["votos_lula"]+summary_model["votos_bolsonaro"]
denm=summary_model["total_13_22"].astype(float).replace(0,float("nan"))\nsummary_model["pct_lula"]=(100*summary_model["votos_lula"]/denm).round(2)
summary_model["pct_bolsonaro"]=(100*summary_model["votos_bolsonaro"]/denm).round(2)
summary_model.to_csv(OUT/"resumo_por_modelo_nordeste.csv",index=False,encoding="utf-8-sig")

summary_uf=(res.groupby("uf",dropna=False)
            .agg(secoes=("id","count"),soma_erros=("erros_num","sum"),votos_lula=("lula_13","sum"),votos_bolsonaro=("bolsonaro_22","sum"))
            .reset_index())
summary_uf["total_13_22"]=summary_uf["votos_lula"]+summary_uf["votos_bolsonaro"]
denu=summary_uf["total_13_22"].astype(float).replace(0,float("nan"))\nsummary_uf["pct_lula"]=(100*summary_uf["votos_lula"]/denu).round(2)
summary_uf["pct_bolsonaro"]=(100*summary_uf["votos_bolsonaro"]/denu).round(2)
summary_uf.to_csv(OUT/"resumo_por_uf_nordeste.csv",index=False,encoding="utf-8-sig")

summary_uf_model=(res.groupby(["uf","modelo_urna"],dropna=False)
                  .agg(secoes=("id","count"),soma_erros=("erros_num","sum"),votos_lula=("lula_13","sum"),votos_bolsonaro=("bolsonaro_22","sum"))
                  .reset_index())
summary_uf_model["total_13_22"]=summary_uf_model["votos_lula"]+summary_uf_model["votos_bolsonaro"]
denum=summary_uf_model["total_13_22"].astype(float).replace(0,float("nan"))\nsummary_uf_model["pct_lula"]=(100*summary_uf_model["votos_lula"]/denum).round(2)
summary_uf_model["pct_bolsonaro"]=(100*summary_uf_model["votos_bolsonaro"]/denum).round(2)
summary_uf_model.to_csv(OUT/"resumo_por_uf_e_modelo_nordeste.csv",index=False,encoding="utf-8-sig")

winner=res["vencedor_13_22"].value_counts(dropna=False)
with (OUT/"RESUMO_NORDESTE.txt").open("w",encoding="utf-8") as fh:
    fh.write("Análise 2022 - Nordeste - 1º turno - UE2013/UE2015 com erros_log > 0\n")
    fh.write(f"Seções encontradas: {len(res)}\n\n")
    fh.write("Vencedor entre Lula e Bolsonaro por seção:\n")
    fh.write(winner.to_string())
    fh.write("\n\nResumo por modelo:\n")
    fh.write(summary_model.to_string(index=False))
    fh.write("\n\nResumo por UF:\n")
    fh.write(summary_uf.to_string(index=False))
    fh.write("\n\nObservação: erros_log vem do .logjez da urna e NÃO é o mesmo que erro de transmissão/RecArquivos.\n")

print(summary_model.to_string(index=False))
print(summary_uf.to_string(index=False))
print(winner.to_string())
