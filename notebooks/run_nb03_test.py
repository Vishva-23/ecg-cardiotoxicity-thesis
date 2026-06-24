"""
Dry-run of NB03 logic to verify the notebook executes correctly
and produces both outputs (figure + CSV).
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, welch
import pywt
from PyEMD import EMD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR    = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

FS=1000; BANDPASS_LOW_HZ=0.5; BANDPASS_HIGH_HZ=150.0; BANDPASS_ORDER=2
REFRACTORY_MS=55; PRE_R_MS=100; POST_R_MS=150; HR_ACCEPT_LOW=300; HR_ACCEPT_HIGH=700
WAVELET="db6"; WAVELET_LEVELS=6; WAVELET_MODE="soft"; EMD_NOISE_IMFS=2
ANIMAL_ID=107; WINDOW_START=1_395_000; WINDOW_END=1_425_000

def parse_header_and_layout(path):
    info={}; n_header=0; first_data_row=None
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            s=line.lstrip()
            if not s or s[0].isdigit() or s[0] in "+-.":
                first_data_row=line.rstrip("\n"); break
            n_header+=1
            if "=" in line:
                key,_,rest=line.partition("=")
                info[key.strip()]=rest.strip("\n").strip("\t").split("\t")
    titles=[t.strip() for t in info.get("ChannelTitle",[]) if t.strip()]
    n_channels=max(len(titles),1); n_cols=len(first_data_row.split("\t")) if first_data_row else (n_channels+1)
    lead_cols=max(1,n_cols-n_channels); ch3_idx=next((i for i,t in enumerate(titles) if t.lower()=="channel 3"),0)
    return n_header, lead_cols+ch3_idx

def load_ecg_file(path, n_header, ecg_col):
    voltages=[]
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for _ in range(n_header): f.readline()
        for line in f:
            parts=line.rstrip("\n").split("\t")
            if len(parts)<=ecg_col: continue
            try: v=float(parts[ecg_col])
            except ValueError: v=np.nan
            voltages.append(v)
    voltage=np.asarray(voltages,dtype=float)
    if np.isnan(voltage).any():
        idx=np.arange(len(voltage)); good=~np.isnan(voltage)
        if good.any(): voltage=np.interp(idx,idx[good],voltage[good])
    return voltage

def denoise_butterworth(x):
    nyq=0.5*FS; b,a=butter(BANDPASS_ORDER,[BANDPASS_LOW_HZ/nyq,BANDPASS_HIGH_HZ/nyq],btype="band")
    y=filtfilt(b,a,x)
    for f0 in (50,100,150):
        bn,an=iirnotch(f0/(FS/2),Q=30); y=filtfilt(bn,an,y)
    return y

def denoise_wavelet(x):
    coeffs=pywt.wavedec(x,WAVELET,level=WAVELET_LEVELS)
    sigma=np.median(np.abs(coeffs[-1]))/0.6745; thr=sigma*np.sqrt(2*np.log(len(x)))
    coeffs_thr=[coeffs[0]]+[pywt.threshold(c,thr,mode=WAVELET_MODE) for c in coeffs[1:]]
    return pywt.waverec(coeffs_thr,WAVELET)[:len(x)]

def denoise_emd(x):
    emd=EMD(); emd.MAX_ITERATION=2000; imfs=emd.emd(x)
    if imfs.shape[0]<=EMD_NOISE_IMFS: return x.copy()
    return imfs[EMD_NOISE_IMFS:].sum(axis=0)

def _peaks_with_mad(sig):
    if sig.size==0: return np.array([],dtype=int),np.nan
    med=float(np.median(sig)); mad=float(np.median(np.abs(sig-med)))
    if mad<=0: return np.array([],dtype=int),np.nan
    height=med+4*mad; prominence=max(0.3*mad,0.005); distance=int(REFRACTORY_MS*FS/1000)
    peaks,_=find_peaks(sig,height=height,distance=distance,prominence=prominence)
    return peaks,height

def detect_r_peaks(sig):
    def hr(p): return 60000./np.mean(np.diff(p)*1000./FS) if len(p)>=2 else np.nan
    pp,_=_peaks_with_mad(sig); phr=hr(pp)
    if not np.isnan(phr) and HR_ACCEPT_LOW<=phr<=HR_ACCEPT_HIGH: return pp,False,phr
    ip,_=_peaks_with_mad(-sig); ihr=hr(ip)
    if not np.isnan(ihr) and HR_ACCEPT_LOW<=ihr<=HR_ACCEPT_HIGH: return ip,True,ihr
    def dist(h): return np.inf if np.isnan(h) else max(0,HR_ACCEPT_LOW-h,h-HR_ACCEPT_HIGH)
    if dist(ihr)<dist(phr): return ip,True,ihr
    return pp,False,phr

def extract_beats(sig,peaks):
    pre=int(PRE_R_MS*FS/1000); post=int(POST_R_MS*FS/1000)
    t_ms=(np.arange(pre+post)-pre)*(1000./FS)
    beats=[sig[r-pre:r+post] for r in peaks if r-pre>=0 and r+post<=len(sig)]
    if not beats: return np.empty((0,pre+post)),t_ms
    return np.vstack(beats),t_ms

def estimate_snr(sig):
    f,pxx=welch(sig,fs=FS,nperseg=min(len(sig),4096))
    cardiac=np.trapezoid(pxx[(f>=4)&(f<=45)],f[(f>=4)&(f<=45)])
    total=np.trapezoid(pxx,f)
    return float(cardiac/total) if total>0 else np.nan

def wave_amp(tmpl,tms,lo,hi,polarity=+1):
    mask=(tms>=lo)&(tms<=hi)
    if not mask.any(): return np.nan
    seg=tmpl[mask]; return float(seg.max() if polarity>=0 else seg.min())

def measure_qt(tmpl,tms):
    baseline=float(np.median(tmpl[tms<-70])) if (tms<-70).any() else 0.
    r_amp=float(tmpl[np.argmax(tmpl)]); tol=0.05*r_amp
    for i in np.where(tms>=40)[0]:
        if abs(tmpl[i]-baseline)<=tol: return float(tms[i])
    return np.nan

# Load
path=next(DATA_DIR.glob(f"*_{ANIMAL_ID}.txt"),None)
n_header,ecg_col=parse_header_and_layout(path)
full_voltage=load_ecg_file(path,n_header,ecg_col)
raw=full_voltage[WINDOW_START:WINDOW_END].copy()
t_sec=np.arange(len(raw))/FS
print(f"Raw loaded: {len(raw)} samples")

# Denoise
print("Butterworth...",end="",flush=True); sig_butter=denoise_butterworth(raw); print(" done")
print("Wavelet...",end="",flush=True); sig_wavelet=denoise_wavelet(raw); print(" done")
print("EMD (slow)...",end="",flush=True); sig_emd=denoise_emd(raw); print(" done")

# Peaks
signals={"Raw":raw,"Butterworth+Notch":sig_butter,"Wavelet (db6)":sig_wavelet,"EMD":sig_emd}
results={}
print("\nPeak detection:")
for name,sig in signals.items():
    peaks,inv,hr=detect_r_peaks(sig)
    sig_o=-sig if inv else sig
    beats,tms=extract_beats(sig_o,peaks)
    tmpl=beats.mean(axis=0) if len(beats)>0 else None
    rr=np.diff(peaks)*1000./FS if len(peaks)>1 else np.array([])
    snr=estimate_snr(sig)
    results[name]=dict(sig=sig_o,peaks=peaks,inv=inv,hr=hr,beats=beats,t_ms=tms,template=tmpl,
                       rr_mean=float(np.mean(rr)) if len(rr) else np.nan,
                       rr_std=float(np.std(rr,ddof=1)) if len(rr)>1 else np.nan,
                       n_beats=len(peaks),snr=snr)
    print(f"  {name:<22} HR={hr:.1f} bpm  beats={len(peaks)}  SNR={snr:.4f}")

# Figure
METHOD_NAMES=["Raw","Butterworth+Notch","Wavelet (db6)","EMD"]
METHOD_COLORS=["#555555","#2563eb","#16a34a","#9333ea"]
PEAK_COLOR="crimson"

def clim(vals,pad=0.05):
    lo,hi=np.nanmin(vals),np.nanmax(vals); rng=hi-lo; return lo-pad*rng,hi+pad*rng

fig=plt.figure(figsize=(22,16))
fig.suptitle(f"ECG Denoising Comparison - Animal {ANIMAL_ID}  (window {WINDOW_START//FS}-{WINDOW_END//FS} s)\nButterworth+Notch = NB02 pipeline baseline",fontsize=14,fontweight="bold",y=0.98)
gs=gridspec.GridSpec(3,4,figure=fig,hspace=0.52,wspace=0.32,top=0.93,bottom=0.06,left=0.06,right=0.98)

all_sigs=np.concatenate([results[n]["sig"] for n in METHOD_NAMES]); ylim_sig=clim(all_sigs)
all_tmpl=np.concatenate([results[n]["template"] for n in METHOD_NAMES if results[n]["template"] is not None]); ylim_tmpl=clim(all_tmpl)

f_raw_ref=None; pxx_raw_ref=None
for col,name in enumerate(METHOD_NAMES):
    r=results[name]; sig=r["sig"]; clr=METHOD_COLORS[col]
    ax0=fig.add_subplot(gs[0,col])
    ax0.plot(t_sec,sig,lw=0.55,color=clr,alpha=0.85)
    if len(r["peaks"]): ax0.scatter(t_sec[r["peaks"]],sig[r["peaks"]],s=12,color=PEAK_COLOR,zorder=5)
    ax0.set_ylim(ylim_sig)
    ax0.set_title(f"{name}\nHR={r['hr']:.1f} bpm  n={r['n_beats']}  SNR={r['snr']:.3f}",fontsize=9.5,loc="left")
    ax0.set_xlabel("Time (s)",fontsize=8)
    if col==0: ax0.set_ylabel("Voltage (mV)",fontsize=8)
    ax0.tick_params(labelsize=7); ax0.grid(alpha=0.22); ax0.set_xlim(0,t_sec[-1])

    ax1=fig.add_subplot(gs[1,col]); tmpl=r["template"]; tms=r["t_ms"]
    if tmpl is not None:
        ax1.plot(tms,tmpl,lw=1.8,color=clr)
        ax1.axvline(0,color=PEAK_COLOR,lw=0.8,linestyle=":",alpha=0.7)
        ax1.axhline(0,color="k",lw=0.35,linestyle="--",alpha=0.35)
        for wname,wlo,whi,wcol in [("P",-55,-15,"#4e79a7"),("J",12,35,"#0891b2"),("T",40,90,"#059669")]:
            ax1.axvspan(wlo,whi,alpha=0.10,color=wcol)
            wmask=(tms>=wlo)&(tms<=whi)
            if wmask.any():
                wt=tms[wmask][np.argmax(tmpl[wmask])]; wa=tmpl[wmask].max()
                ax1.scatter([wt],[wa],s=30,color=wcol,zorder=5)
                ax1.text(wt,wa+0.03*(ylim_tmpl[1]-ylim_tmpl[0]),wname,fontsize=7.5,color=wcol,ha="center",fontweight="bold")
        ax1.set_ylim(ylim_tmpl)
    ax1.set_xlim(-PRE_R_MS,POST_R_MS); ax1.set_xlabel("ms from R-peak",fontsize=8)
    if col==0: ax1.set_ylabel("Voltage (mV)",fontsize=8)
    ax1.set_title("Averaged beat template",fontsize=8.5,loc="left")
    ax1.tick_params(labelsize=7); ax1.grid(alpha=0.22)

    ax2=fig.add_subplot(gs[2,col])
    if col==0: f_raw_ref,pxx_raw_ref=welch(raw,fs=FS,nperseg=min(len(raw),4096))
    f_s,pxx_s=welch(sig,fs=FS,nperseg=min(len(sig),4096))
    if col>0 and pxx_raw_ref is not None: ax2.semilogy(f_raw_ref,pxx_raw_ref,lw=0.8,color="#999",alpha=0.6,label="raw (ref)",linestyle="--")
    ax2.semilogy(f_s,pxx_s,lw=1.2,color=clr,label=name)
    for fline in (50,100,150): ax2.axvline(fline,color="red",lw=0.6,linestyle=":",alpha=0.5)
    ax2.set_xlim(0,200); ax2.set_xlabel("Frequency (Hz)",fontsize=8)
    if col==0: ax2.set_ylabel("PSD (mV2/Hz)",fontsize=8)
    ax2.set_title("Power spectral density",fontsize=8.5,loc="left")
    ax2.tick_params(labelsize=7); ax2.grid(alpha=0.22)
    if col>0: ax2.legend(fontsize=6.5,loc="upper right")

out_fig=FIGURES_DIR/"denoising_comparison.png"
fig.savefig(out_fig,dpi=150,bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {out_fig}")

# Metrics
rows=[]
for name in METHOD_NAMES:
    r=results[name]; tmpl=r["template"]; tms=r["t_ms"]
    r_amp=wave_amp(tmpl,tms,-5,5,+1) if tmpl is not None else np.nan
    j_amp=wave_amp(tmpl,tms,12,35,+1) if tmpl is not None else np.nan
    t_amp=wave_amp(tmpl,tms,40,90,+1) if tmpl is not None else np.nan
    p_amp=wave_amp(tmpl,tms,-55,-15,+1) if tmpl is not None else np.nan
    s_amp=wave_amp(tmpl,tms,1,12,-1) if tmpl is not None else np.nan
    qt=measure_qt(tmpl,tms) if tmpl is not None else np.nan
    qtc=(qt/np.sqrt(r["rr_mean"]/100.)) if (not np.isnan(qt) and not np.isnan(r["rr_mean"]) and r["rr_mean"]>0) else np.nan
    rows.append(dict(method=name,HR_bpm=round(r["hr"],1),n_beats=r["n_beats"],
                     rr_mean_ms=round(r["rr_mean"],1),rr_std_ms=round(r["rr_std"],1),SNR=round(r["snr"],4),
                     R_amp_mV=round(r_amp,4) if not np.isnan(r_amp) else np.nan,
                     P_amp_mV=round(p_amp,4) if not np.isnan(p_amp) else np.nan,
                     S_amp_mV=round(s_amp,4) if not np.isnan(s_amp) else np.nan,
                     J_amp_mV=round(j_amp,4) if not np.isnan(j_amp) else np.nan,
                     T_amp_mV=round(t_amp,4) if not np.isnan(t_amp) else np.nan,
                     QT_ms=round(qt,1) if not np.isnan(qt) else np.nan,
                     QTc_Mitchell=round(qtc,1) if not np.isnan(qtc) else np.nan))
metrics_df=pd.DataFrame(rows).set_index("method")
csv_path=OUTPUTS_DIR/"denoising_comparison_results.csv"
metrics_df.reset_index().to_csv(csv_path,index=False)
print(f"CSV saved: {csv_path}")
print()
print(metrics_df.T.to_string())
print()
print("--- Change vs Butterworth+Notch baseline ---")
ref=metrics_df.loc["Butterworth+Notch"]
numeric_cols=["HR_bpm","SNR","R_amp_mV","J_amp_mV","T_amp_mV","QT_ms","QTc_Mitchell"]
for name in ["Wavelet (db6)","EMD"]:
    row=metrics_df.loc[name]
    print(f"  {name}:")
    for col in numeric_cols:
        rv,bv=row[col],ref[col]
        if not (np.isnan(rv) or np.isnan(bv)):
            delta=rv-bv; pct=100*delta/bv if bv!=0 else np.nan
            print(f"    {col:<18} delta={delta:+.4f}  ({pct:+.1f}%)")
