"""
实现logic:
批量处理 CWRU_HP0 下 12 个轴承文件：
  xlsx -> 高通滤波 -> 0.5s 窗口 50% 重叠切分
       -> 每段信号做特征模态分解 FMD(L=30, n=6, mode_num=3, max_iter=20) 4个参数设置
       -> 3 个模态做连续小波变换 CWT + 双线性缩放 224×224
       -> 保存到 <BASE>/<stem>/seg_XXXX.npy

支持断点续跑：已存在的 seg_XXXX.npy 自动跳过。
"""
import os
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import zoom
from scipy.signal import firwin, hilbert, correlate, butter, filtfilt
from numpy.linalg import inv


# ============特征模态分解FMD===========
def TT(y, fs):
    M = fs
    na = correlate(y, y, mode='full', method='auto')[-M - 1:]
    na = na[len(na) // 2:]
    zeroposi = None
    for lag in range(1, len(na)):
        if na[lag - 1] > 0 and na[lag] < 0:
            zeroposi = lag
            break
    if zeroposi is None:
        zeroposi = 0
    na = na[zeroposi:]
    return zeroposi + int(np.argmax(na))


def CK(x, T, M=2):
    x = x.flatten()
    N = len(x)
    xs = np.zeros((M + 1, N))
    xs[0] = x.copy()
    for m in range(1, M + 1):
        xs[m, T:] = xs[m - 1, :-T]
    return np.sum(np.prod(xs, axis=0) ** 2) / (np.sum(x ** 2) ** (M + 1))


def max_IJ(X):
    tempI = np.argmax(X, axis=0)
    M = np.max(X, axis=0)
    J = int(np.argmax(M))
    I = int(tempI[J])
    return I, J, X[I, J]


def xxc_mckd(fs, x, f_init, term_iter, T=None, M=3):
    x = x.flatten()
    L = len(f_init)
    N = len(x)

    if T is None:
        env = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
        T = TT(env, fs)
    T = int(round(T))
    if T <= 0 or T >= N:
        T = max(1, min(N - 1, fs // 100))

    XmT = np.zeros((L, N, M + 1))
    for m in range(M + 1):
        for l in range(L):
            if l == 0:
                XmT[l, m * T:, m] = x[:N - m * T]
            else:
                XmT[l, 1:, m] = XmT[l - 1, :-1, m]
    Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

    f = f_init.copy()
    y_final = np.zeros((N, term_iter))
    f_final = np.zeros((L, term_iter))
    T_final = T

    for n in range(term_iter):
        y = (f.T @ XmT[:, :, 0]).T
        yt = np.zeros((N, M + 1))
        yt[:, 0] = y.flatten()
        for m in range(1, M + 1):
            yt[m * T:, m] = yt[:N - m * T, m - 1]

        alpha = np.zeros((N, M + 1))
        for m in range(M + 1):
            cols = [k for k in range(M + 1) if k != m]
            alpha[:, m] = np.prod(yt[:, cols], axis=1) ** 2 * yt[:, m]
        beta = np.prod(yt, axis=1)

        Xalpha = np.zeros(L)
        for m in range(M + 1):
            Xalpha += XmT[:, :, m] @ alpha[:, m]

        f = (np.sum(y ** 2) / (2 * np.sum(beta ** 2))) * Xinv @ Xalpha
        f /= np.sqrt(np.sum(f ** 2))

        env = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
        T = int(round(TT(env, fs)))
        if T <= 0 or T >= N:
            T = max(1, min(N - 1, fs // 100))
        T_final = T

        XmT = np.zeros((L, N, M + 1))
        for m in range(M + 1):
            for l in range(L):
                if l == 0:
                    XmT[l, m * T:, m] = x[:N - m * T]
                else:
                    XmT[l, 1:, m] = XmT[l - 1, :-1, m]
        Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

        y_final[:, n] = np.convolve(x, f, mode='same')
        f_final[:, n] = f

    return y_final, f_final, T_final


def FMD(fs, x, filter_size, cut_num, mode_num, max_iter_num):
    freq_bound = np.arange(0, 1, 1 / cut_num)
    temp_filters = np.zeros((filter_size, cut_num))
    for n in range(cut_num):
        temp_filters[:, n] = firwin(
            filter_size,
            [freq_bound[n] + 1e-15, freq_bound[n] + 1 / cut_num - 1e-15],
            window='hann', pass_zero=False)

    temp_sig = np.tile(x, (cut_num, 1)).T
    iter_count = 0

    while True:
        iter_num = (max_iter_num - (cut_num - mode_num) * 2
                    if iter_count == 0 else 2)

        current = []
        for n in range(cut_num):
            y_iter, f_iter, T_iter = xxc_mckd(
                fs, temp_sig[:, n], temp_filters[:, n], iter_num, None, 1)
            current.append({'y': y_iter[:, -1], 'f': f_iter[:, -1], 'T': T_iter})

        temp_sig = np.array([m['y'] for m in current]).T
        temp_filters = np.array([m['f'] for m in current]).T

        if cut_num == mode_num:
            break

        corr = np.abs(np.corrcoef(temp_sig.T))
        corr = np.triu(corr, 1)
        I, J, _ = max_IJ(corr)
        ki = CK(temp_sig[:, I], current[I]['T'])
        kj = CK(temp_sig[:, J], current[J]['T'])
        out = J if ki > kj else I

        temp_sig = np.delete(temp_sig, out, axis=1)
        temp_filters = np.delete(temp_filters, out, axis=1)
        cut_num -= 1
        iter_count += 1

    return temp_sig


# =========== CWT + 双线性缩放 ============
FREQ_MIN = 10
N_FREQS = 5990
WAVELET = "morl"
TARGET = 224


def cwt_tf(signal, fs):
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    signal = signal - np.mean(signal)

    freqs = np.linspace(FREQ_MIN, fs / 2, N_FREQS)
    cf = pywt.central_frequency(WAVELET)
    scales = cf * fs / freqs
    coeffs, cwt_freqs = pywt.cwt(signal, scales, WAVELET,
                                 sampling_period=1.0 / fs)
    order = np.argsort(cwt_freqs)
    return coeffs[order, :], cwt_freqs[order]


def segment_to_npy(seg, fs, L, n, mode_num, max_iter):
    modes = FMD(fs, seg, L, n, mode_num, max_iter)  # (6000, 3)
    images = []
    for i in range(modes.shape[1]):
        coeffs, _ = cwt_tf(modes[:, i], fs)
        mag_log = np.log10(np.abs(coeffs) + 1e-12)
        zf = (TARGET / mag_log.shape[0], TARGET / mag_log.shape[1])
        images.append(zoom(mag_log, zf, order=1))
    return np.stack(images, axis=0).astype(np.float32)


# ======================== 单文件处理 =========================
# 按照时间为1.0s分割，step=50%win,win表示窗口，6k则为0.5s，12k则是1s。
def process_file(src, base_out, fs=12000, win=12000, step=6000,
                 L=30, n=6, mode_num=3, max_iter=20, overwrite=False):
    stem = Path(src).stem
    out_dir = Path(base_out) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    df = pd.read_excel(src, sheet_name=0, header=0, engine="openpyxl")
    col = next((c for c in df.columns if "DE_time" in str(c)), df.columns[0])
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)

    # 高通滤波
    b, a = butter(4, 10.0 / (0.5 * fs), btype="high", analog=False)
    x = filtfilt(b, a, x)

    N = len(x)
    n_segs = (N - win) // step + 1
    print(f"  列: {col} | 长度: {N} | 时长: {N/fs:.2f}s | 片段: {n_segs}")

    ok = skipped = fail = 0
    t0 = time.time()
    for i in range(n_segs):
        save_path = out_dir / f"seg_{i:04d}.npy"
        if save_path.exists() and not overwrite:
            skipped += 1
            continue

        seg = x[i * step: i * step + win]
        try:
            arr = segment_to_npy(seg, fs, L, n, mode_num, max_iter)
            np.save(save_path, arr)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"    [失败] seg_{i:04d}: {e}")

        if (i + 1) % 5 == 0 or i == n_segs - 1:
            el = time.time() - t0
            print(f"    [{i+1:3d}/{n_segs}] ok={ok} skip={skipped} fail={fail} "
                  f"| 已用时 {el:.1f}s")

    return dict(stem=stem, col=col, N=N, n_segs=n_segs,
                ok=ok, skipped=skipped, fail=fail,
                out=str(out_dir), elapsed=time.time() - t0)


# 主流程
# 要处理的文件
DATA_ROOT = Path(__file__).resolve().parents[1] / "Data"
FILES = [str(DATA_ROOT / "HP_0" / name) for name in ['B007_0.xlsx', 'B014_0.xlsx', 'B021_0.xlsx', 'B028_0.xlsx', 'IR007_0.xlsx', 'IR014_0.xlsx', 'IR021_0.xlsx', 'IR028_0.xlsx', 'Normal_0.xlsx', 'OR007@6_0.xlsx', 'OR014@6_0.xlsx', 'OR021@6_0.xlsx']]

# 输出路径修改为自己的路径
BASE_OUT = str(DATA_ROOT / "HP_0_processed")


if __name__ == "__main__":
    t_all = time.time()
    summary = []

    for idx, src in enumerate(FILES, 1):
        print(f"\n{'=' * 72}")
        print(f"[{idx}/{len(FILES)}] {src}")
        if not Path(src).is_file():
            print("  [跳过] 文件不存在")
            summary.append(dict(stem=Path(src).stem, error="missing"))
            continue
        try:
            s = process_file(src, BASE_OUT)
            summary.append(s)
        except Exception:
            print("  [错误] 该文件整体失败：")
            traceback.print_exc()
            summary.append(dict(stem=Path(src).stem, error="exception"))

    # ---------- 汇总 ----------
    total_ok = sum(s.get("ok", 0) for s in summary)
    total_seg = sum(s.get("n_segs", 0) for s in summary)
    total_fail = sum(s.get("fail", 0) for s in summary)
    print(f"\n{'=' * 72}")
    print(f"全部完成，总用时 {(time.time()-t_all)/60:.1f} min")
    print(f"{'文件':<14}{'段数':>6}{'成功':>6}{'跳过':>6}{'失败':>6}{'用时(s)':>10}")
    for s in summary:
        if s.get("error") == "missing":
            print(f"{s['stem']:<14}{'文件不存在':>10}")
        elif s.get("error") == "exception":
            print(f"{s['stem']:<14}{'处理异常':>10}")
        else:
            print(f"{s['stem']:<14}{s['n_segs']:>6}{s['ok']:>6}"
                  f"{s['skipped']:>6}{s['fail']:>6}{s['elapsed']:>10.1f}")
    print(f"\n总片段: {total_seg} | 本次成功: {total_ok} | 失败: {total_fail}")
    print(f"输出根目录: {BASE_OUT}")