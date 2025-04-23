# streamlit_app.py

import streamlit as st
import pandas as pd
import polars as pl
import spectrochempy as scp
import zipfile, tempfile, io, os, re
from natsort import natsorted

st.set_page_config(page_title="OMNIC FTIR Data Processing", layout="wide")


# ─── Helpers ───────────────────────────────────────────────────────────────────

def file_has_header(path):
    try:
        first = open(path).readline().lower()
        return any(k in first for k in ["wavenumbers", "cm^-1", "absorbance", "a.u."])
    except:
        return False

def extract_time_value(fn):
    m = re.search(r"t\s*=\s*([\d\.]+)", fn)
    return float(m.group(1)) if m else None

def calculate_cv_voltage(t, p):
    E0, E1, E2, sr, Teq = p["E_begin"], p["E_vertex1"], p["E_vertex2"], p["scan_rate"], p["T_eq"]
    if t < Teq: return E0
    if E0 == E2:
        t1 = abs(E1-E0)/sr; T = 2*t1; tc = (t-Teq)%T
        if tc<t1: return E0 + (E1-E0)*(tc/t1)
        else:     return E1 + (E0-E1)*((tc-t1)/t1)
    else:
        t1 = abs(E1-E0)/sr; t2 = abs(E2-E1)/sr; t3 = abs(E1-E0)/sr
        T = t1+t2+t3; tc = (t-Teq)%T
        if tc<t1:      return E0 + (E1-E0)*(tc/t1)
        elif tc<t1+t2: return E1 + (E2-E1)*((tc-t1)/t2)
        else:          return E2 + (E0-E2)*((tc-t1-t2)/t3)


# ─── SPA → CSV ─────────────────────────────────────────────────────────────────

def convert_spa(uploaded):
    zio = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp,"out")
        os.makedirs(out)
        errs=[]
        for f in uploaded:
            fp = os.path.join(tmp,f.name)
            open(fp,"wb").write(f.getbuffer())
            try:
                ds = scp.read_omnic(fp)
                if ds is None: raise ValueError("bad SPA")
                base,_ = os.path.splitext(f.name)
                path = os.path.join(out, base+".csv")
                (ds if ds.ndim==1 else ds[0]).write_csv(path)
            except Exception as e:
                errs.append(f"{f.name}: {e}")
        with zipfile.ZipFile(zio,"w") as z:
            for fn in os.listdir(out):
                z.write(os.path.join(out,fn), fn)
    zio.seek(0)
    return zio, errs


# ─── SRS → CSV ─────────────────────────────────────────────────────────────────

def convert_srs(uploaded):
    zio = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, uploaded.name)
        open(fp,"wb").write(uploaded.getbuffer())
        ds = scp.read_omnic(fp)
        if ds is None: raise ValueError("bad SRS")
        out = os.path.join(tmp,"out"); os.makedirs(out)
        if ds.ndim==2:
            for i,sub in enumerate(ds):
                sub.write_csv(os.path.join(out,f"{os.path.splitext(uploaded.name)[0]}_{i+1}.csv"))
        else:
            ds.write_csv(os.path.join(out,os.path.splitext(uploaded.name)[0]+".csv"))
        with zipfile.ZipFile(zio,"w") as z:
            for fn in os.listdir(out):
                z.write(os.path.join(out,fn), fn)
    zio.seek(0)
    return zio


# ─── Combine Series CSV ────────────────────────────────────────────────────────

def combine_series(files):
    with tempfile.TemporaryDirectory() as tmp:
        paths=[]
        for f in files:
            p=os.path.join(tmp,f.name)
            open(p,"wb").write(f.getbuffer())
            paths.append(p)
        paths = natsorted(paths, key=os.path.basename)
        combined=None
        for p in paths:
            skip = 1 if file_has_header(p) else 0
            df = pl.read_csv(p, has_header=False, skip_rows=skip,
                             new_columns=["Wavenumber",os.path.basename(p)])
            df = df.with_columns((pl.col("Wavenumber")//0.1*0.1).alias("Wavenumber"))
            if combined is None:
                combined=df
            else:
                tmpdf=df.rename({"Wavenumber":"Wavenumber_tmp"})
                combined=combined.join(tmpdf, left_on="Wavenumber",
                                      right_on="Wavenumber_tmp", how="full").drop("Wavenumber_tmp")
        return combined.to_pandas()


# ─── Combine Time-Resolved CSV ─────────────────────────────────────────────────

def combine_time(files):
    combined=None
    for f in files:
        if "static" in f.name.lower(): continue
        tv = extract_time_value(f.name)
        if tv is None: continue
        df = pd.read_csv(io.BytesIO(f.getbuffer()), header=None,
                         skiprows=1 if file_has_header(f.name) else 0)
        if df.shape[1]<2: continue
        df.columns=["Wavenumber",f"{tv:.2f}"]
        df["Wavenumber"] = df["Wavenumber"].floordiv(0.1).mul(0.1)
        combined = df if combined is None else pd.merge(combined, df, on="Wavenumber", how="outer")
    if combined is None: return None
    cols = ["Wavenumber"]+sorted([c for c in combined if c!="Wavenumber"], key=lambda x: float(x))
    return combined[cols]


# ─── Rename CV Headers ─────────────────────────────────────────────────────────

def rename_cv(f, T_eq, E0, E1, E2, sr, scans):
    df = pd.read_csv(io.BytesIO(f.getbuffer()))
    n = df.shape[1]-1
    params={"E_begin":E0,"E_vertex1":E1,"E_vertex2":E2,"scan_rate":sr,"T_eq":T_eq}
    if E0==E2:
        t1=abs(E1-E0)/sr; cycle=2*t1
    else:
        t1=abs(E1-E0)/sr; t2=abs(E2-E1)/sr; t3=t1; cycle=t1+t2+t3
    total=T_eq+scans*cycle; dt=total/n
    cols=["Wavenumber"]
    for i in range(n):
        t=i*dt; v=E0 if t<T_eq else calculate_cv_voltage(t,params)
        cols.append(f"{v:.4f} V")
    df.columns=cols
    return df


# ─── Rename LV Headers ─────────────────────────────────────────────────────────

def rename_lv(f, T_eq, E0, Eend, sr):
    df = pd.read_csv(io.BytesIO(f.getbuffer()))
    n = df.shape[1]-1
    ramp=abs(Eend-E0)/sr; total=T_eq+ramp; dt=total/n
    cols=["Wavenumber"]
    for i in range(n):
        t=i*dt
        if t<T_eq: v=E0
        else:
            tr=t-T_eq
            v=(min(E0+sr*tr,Eend) if Eend>E0 else max(E0-sr*tr,Eend))
        cols.append(f"{v:.4f} V")
    df.columns=cols
    return df


# ─── Rename Time-Based Headers ─────────────────────────────────────────────────

def rename_time(f, total):
    df = pd.read_csv(io.BytesIO(f.getbuffer()))
    n=df.shape[1]-1; dt=total/n
    cols=["Wavenumber"]+[f"{i*dt:.2f}s" for i in range(n)]
    df.columns=cols
    return df


# ─── Background Reprocess ──────────────────────────────────────────────────────

def bg_reproc(f, bgcol):
    df = pd.read_csv(io.BytesIO(f.getbuffer()))
    out=pd.DataFrame({"Wavenumber":df["Wavenumber"]})
    for c in df.columns[1:]:
        out[c] = 0 if c==bgcol else df[c]-df[bgcol]
    return out


# ─── Reduce Spectral Columns ────────────────────────────────────────────────────

def reduce_spec(f, n):
    df = pd.read_csv(io.BytesIO(f.getbuffer()))
    tot=df.shape[1]; keep=[0,1]
    cycle=n+1
    for i in range(2,tot-1):
        if (i-2)%cycle==n: keep.append(i)
    keep.append(tot-1)
    return df.iloc[:,keep]


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

st.title("OMNIC FTIR Data Processing")

mode = st.sidebar.selectbox("Operation:", [
    "Convert SPA→CSV","Convert SRS→CSV",
    "Combine Series CSV","Combine Time‐Resolved CSV",
    "Rename CV headers","Rename LV headers",
    "Rename Time headers","Background reprocess",
    "Reduce spectral columns"
])

if mode=="Convert SPA→CSV":
    files = st.file_uploader("*.spa files", type="spa", accept_multiple_files=True)
    if files and st.button("Run"):
        z, errs = convert_spa(files)
        if errs: st.error("\n".join(errs))
        st.download_button("Download ZIP", z, "spa_to_csv.zip", mime="application/zip")

elif mode=="Convert SRS→CSV":
    f = st.file_uploader("*.srs file", type="srs")
    if f and st.button("Run"):
        try:
            z = convert_srs(f)
            st.download_button("Download ZIP", z, "srs_to_csv.zip", mime="application/zip")
        except Exception as e:
            st.error(e)

elif mode=="Combine Series CSV":
    files = st.file_uploader("CSV files", type="csv", accept_multiple_files=True)
    if files and st.button("Run"):
        df = combine_series(files)
        st.dataframe(df)
        st.download_button("DL CSV", df.to_csv(index=False).encode(), "combined.csv")

elif mode=="Combine Time‐Resolved CSV":
    files = st.file_uploader("CSV files", type="csv", accept_multiple_files=True)
    if files and st.button("Run"):
        df = combine_time(files)
        if df is None: st.error("No data")
        else:
            st.dataframe(df)
            st.download_button("DL CSV", df.to_csv(index=False).encode(), "time_combined.csv")

elif mode=="Rename CV headers":
    f = st.file_uploader("CSV", type="csv")
    if f:
        Teq = st.number_input("T_eq",value=0.0)
        Eb  = st.number_input("E_begin",value=0.0)
        Ev1 = st.number_input("E_vertex1",value=0.0)
        Ev2 = st.number_input("E_vertex2",value=0.0)
        sr  = st.number_input("scan_rate",value=0.0)
        scans=st.number_input("num_scans",min_value=1,value=1,step=1)
        if st.button("Run"):
            out = rename_cv(f,Teq,Eb,Ev1,Ev2,sr,scans)
            st.dataframe(out)
            st.download_button("DL CSV", out.to_csv(index=False).encode(), "cv_renamed.csv")

elif mode=="Rename LV headers":
    f = st.file_uploader("CSV", type="csv")
    if f:
        Teq = st.number_input("T_eq",value=0.0, key="Teq2")
        Eb  = st.number_input("E_begin",value=0.0, key="Eb2")
        Ee  = st.number_input("E_end",value=0.0, key="Ee2")
        sr  = st.number_input("scan_rate",value=0.0, key="sr2")
        if st.button("Run"):
            out = rename_lv(f,Teq,Eb,Ee,sr)
            st.dataframe(out)
            st.download_button("DL CSV", out.to_csv(index=False).encode(), "lv_renamed.csv")

elif mode=="Rename Time headers":
    f = st.file_uploader("CSV", type="csv")
    if f:
        tot = st.number_input("Total time (s)",value=0.0)
        if st.button("Run"):
            out = rename_time(f,tot)
            st.dataframe(out)
            st.download_button("DL CSV", out.to_csv(index=False).encode(), "time_renamed.csv")

elif mode=="Background reprocess":
    f = st.file_uploader("CSV", type="csv")
    if f:
        df = pd.read_csv(io.BytesIO(f.getbuffer()))
        col = st.selectbox("BG column", [c for c in df.columns if c!="Wavenumber"])
        if st.button("Run"):
            out = bg_reproc(f, col)
            st.dataframe(out)
            st.download_button("DL CSV", out.to_csv(index=False).encode(), "bg_reproc.csv")

elif mode=="Reduce spectral columns":
    f = st.file_uploader("CSV", type="csv")
    if f:
        n = st.number_input("skip n", min_value=0, value=0, step=1)
        if st.button("Run"):
            out = reduce_spec(f,n)
            st.dataframe(out)
            st.download_button("DL CSV", out.to_csv(index=False).encode(), "reduced.csv")
