import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# パラメータ
# =========================
BASE_DIR = "/content/outputs/U25_FULL_ensemble"
TAG = "N10"
WIN_PRE = 15
WIN_POST = 30

# =========================
# 1. τ_delay読み込み
# =========================
tau_df = pd.read_csv(os.path.join(BASE_DIR, f"{TAG}_tau_delay.csv"))
print("τ_delay summary:")
print(tau_df.describe())

# =========================
# 2. ヒストグラム（全runs）
# =========================
taus_all = tau_df["tau_delay"].dropna().to_numpy()

plt.figure(figsize=(6.2, 4.2))
plt.hist(taus_all, bins=min(15, max(5, len(taus_all)//2)), edgecolor='black')
plt.xlabel("τ_delay (steps)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title(f"τ_delay Distribution (N={len(tau_df)})", fontsize=14)
plt.axvline(np.mean(taus_all), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(taus_all):.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, f"{TAG}_tau_delay_hist.pdf"), dpi=300)
plt.show()

# =========================
# 3. Event-aligned（t_global基準）
# =========================
segs_alive = []
segs_L_local = []

for idx, row in tau_df.iterrows():
    run_id = int(row["run"])
    t_global = int(row["t_global"])

    # 該当runのtimeseries.csvを読み込み
    run_dir = os.path.join(BASE_DIR.replace("_ensemble", f"_{TAG}_run{run_id:02d}"))
    ts_path = os.path.join(run_dir, "timeseries.csv")

    if not os.path.exists(ts_path):
        print(f"Warning: {ts_path} not found, skipping run {run_id}")
        continue

    ts = pd.read_csv(ts_path)

    # 範囲チェック
    L_idx = t_global - WIN_PRE
    R_idx = t_global + WIN_POST + 1

    if L_idx >= 0 and R_idx <= len(ts):
        segs_alive.append(ts["alive"].iloc[L_idx:R_idx].to_numpy())
        # Λ_localの代理：E_intの平均を使う（ない場合はaliveで代用）
        if "L_local_mean" in ts.columns:
            segs_L_local.append(ts["L_local_mean"].iloc[L_idx:R_idx].to_numpy())
        else:
            # 代理指標：alive減少率でΛ_localを近似
            segs_L_local.append(ts["alive"].iloc[L_idx:R_idx].to_numpy())

# =========================
# 4. Aliveプロット
# =========================
if len(segs_alive) > 0:
    seg_arr = np.array(segs_alive)
    mean_seg = np.mean(seg_arr, axis=0)
    sem_seg = np.std(seg_arr, axis=0) / np.sqrt(seg_arr.shape[0])
    ci = 1.96 * sem_seg
    x = np.arange(-WIN_PRE, WIN_POST + 1)

    plt.figure(figsize=(9.5, 4.0))
    plt.plot(x, mean_seg, linewidth=2.5, label='Mean Alive')
    plt.fill_between(x, mean_seg - ci, mean_seg + ci, alpha=0.25, label='95% CI')
    plt.axvline(0, linestyle='--', color='red', linewidth=2, label='t_global (Social Death)')
    plt.xlabel("Time from t_global (steps)", fontsize=12)
    plt.ylabel("Alive Fraction", fontsize=12)
    plt.title(f"Event-aligned: Alive around Social Death (N={len(segs_alive)})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{TAG}_tau_event_aligned_alive.pdf"), dpi=300)
    plt.show()

# =========================
# 5. Λ_localプロット
# =========================
if len(segs_L_local) > 0:
    seg_arr = np.array(segs_L_local)
    mean_seg = np.mean(seg_arr, axis=0)
    sem_seg = np.std(seg_arr, axis=0) / np.sqrt(seg_arr.shape[0])
    ci = 1.96 * sem_seg
    x = np.arange(-WIN_PRE, WIN_POST + 1)

    plt.figure(figsize=(9.5, 4.0))
    plt.plot(x, mean_seg, linewidth=2.5, label='Mean Λ_local (or proxy)')
    plt.fill_between(x, mean_seg - ci, mean_seg + ci, alpha=0.25, label='95% CI')
    plt.axvline(0, linestyle='--', color='red', linewidth=2, label='t_global (Social Death)')
    plt.xlabel("Time from t_global (steps)", fontsize=12)
    plt.ylabel("Λ_local [mean normalized]", fontsize=12)
    plt.title(f"Event-aligned: Individual Energy Ratio around Social Death (N={len(segs_L_local)})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{TAG}_tau_event_aligned_Llocal.pdf"), dpi=300)
    plt.show()

print("\n✅ All plots saved in:", BASE_DIR)
