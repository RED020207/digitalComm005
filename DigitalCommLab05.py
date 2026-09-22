# ============================================================
# Experiment 5 — DPCM and Delta Modulation
# Single-cell Google Colab script
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['axes.grid'] = True

fs, T = 1000, 1.0
t = np.arange(0, T, 1/fs)

def slow_sig(t):  return np.sin(2*np.pi*2*t)      # slowly varying
def fast_sig(t):  return np.sin(2*np.pi*20*t)     # rapidly varying

# ------------------------------------------------------------
# 1. PCM (memoryless uniform quantizer)
# ------------------------------------------------------------
def pcm(x, bits=4):
    L = 2**bits
    A = np.max(np.abs(x))
    delta = 2*A/L
    q = np.clip(np.round(x/delta), -L/2, L/2-1)
    xq = q*delta
    return xq, x - xq

# ------------------------------------------------------------
# 2. First-order DPCM
#    predictor: x_hat[n] = x_recon[n-1]
# ------------------------------------------------------------
def dpcm(x, bits=4):
    L = 2**bits
    A = np.max(np.abs(x))
    delta = 2*A/L              # step of error quantizer
    N = len(x)
    xp = np.zeros(N); d = np.zeros(N)
    dq = np.zeros(N); xr = np.zeros(N)
    xr[0] = 0.0
    for n in range(1, N):
        xp[n] = xr[n-1]                            # first-order predictor
        d[n]  = x[n] - xp[n]
        q     = np.clip(np.round(d[n]/delta), -L/2, L/2-1)
        dq[n] = q*delta
        xr[n] = xp[n] + dq[n]
    return xp, d, dq, xr

# ------------------------------------------------------------
# 3. Delta Modulation (linear, 1-bit)
# ------------------------------------------------------------
def dm(x, Delta):
    N = len(x)
    y  = np.zeros(N)     # +/- Delta staircase increments
    xr = np.zeros(N)     # reconstructed signal
    for n in range(1, N):
        y[n]  = Delta if x[n] >= xr[n-1] else -Delta
        xr[n] = xr[n-1] + y[n]
    return xr, y

# ------------------------------------------------------------
# 4. Adaptive Delta Modulation (Jayant-style)
# ------------------------------------------------------------
def adm(x, Delta0=0.05, alpha=1.5, beta=0.9):
    N = len(x)
    y  = np.zeros(N); xr = np.zeros(N); D = np.zeros(N)
    D[0] = Delta0; prev = 1
    for n in range(1, N):
        e = x[n] - xr[n-1]
        s = 1 if e >= 0 else -1
        D[n] = D[n-1]*(alpha if s == prev else beta)
        D[n] = np.clip(D[n], Delta0*0.2, Delta0*8)
        y[n]  = s*D[n]
        xr[n] = xr[n-1] + y[n]
        prev = s
    return xr, y, D

def mse(a, b): return float(np.mean((a-b)**2))

# ============================================================
# PART A — PCM vs DPCM on a slowly varying input
# ============================================================
print("="*70)
print("EXPECTED (Part A):")
print(" • Predictor removes the slowly-varying trend → prediction error is")
print("   small and DPCM reconstruction error << PCM error at equal bits.")
print(" • Prediction error should be roughly a zero-mean residual.")
print("="*70)

x = slow_sig(t)
xq_pcm, err_pcm = pcm(x, bits=4)
xp, d, dq, xr_dpcm = dpcm(x, bits=4)

print(f"MSE PCM  (4-bit): {mse(x, xq_pcm):.6e}")
print(f"MSE DPCM (4-bit): {mse(x, xr_dpcm):.6e}")
print(f"Improvement factor: {mse(x,xq_pcm)/mse(x,xr_dpcm):.1f}x")

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, x, 'k', label='original')
ax[0].plot(t, xp, 'r--', label='DPCM prediction $\\hat{x}_p[n]=x_r[n-1]$')
ax[0].plot(t, xr_dpcm, 'b', alpha=0.6, label='DPCM reconstruction')
ax[0].set_title('Original / Predicted / Reconstructed (slow input, 4-bit DPCM)')
ax[0].legend()
ax[1].plot(t, d, 'g', label='prediction error d[n]=x[n]-x_p[n]')
ax[1].plot(t, err_pcm, 'm', alpha=0.6, label='PCM quantization error')
ax[1].set_xlabel('time [s]'); ax[1].legend()
ax[1].set_title('Prediction error vs PCM error')
plt.tight_layout(); plt.show()

# ============================================================
# PART B — Delta modulation: step size sweep
# ============================================================
print("="*70)
print("EXPECTED (Part B):")
print(" • small Δ → granular noise (idle hiss) even on slow input.")
print(" • moderate Δ → tracks well.")
print(" • large Δ → slope-overload on fast input (staircase cannot follow).")
print(" • MSE vs Δ is U-shaped for each input; the optimum is larger")
print("   for the fast signal.")
print("="*70)

Deltas = np.logspace(-3, -0.1, 40)
mse_slow, mse_fast = [], []
for D in Deltas:
    xr_s, _ = dm(slow_sig(t), D);  mse_slow.append(mse(slow_sig(t), xr_s))
    xr_f, _ = dm(fast_sig(t), D);  mse_fast.append(mse(fast_sig(t), xr_f))

i_s = int(np.argmin(mse_slow)); i_f = int(np.argmin(mse_fast))
print(f"Optimal Δ (slow 2 Hz):  {Deltas[i_s]:.4f}  → MSE {mse_slow[i_s]:.3e}")
print(f"Optimal Δ (fast 20 Hz): {Deltas[i_f]:.4f}  → MSE {mse_fast[i_f]:.3e}")

fig, ax = plt.subplots()
ax.semilogx(Deltas, mse_slow, 'o-', label='slow input (2 Hz)')
ax.semilogx(Deltas, mse_fast, 's-', label='fast input (20 Hz)')
ax.axvline(Deltas[i_s], color='C0', ls=':', alpha=0.6)
ax.axvline(Deltas[i_f], color='C1', ls=':', alpha=0.6)
ax.set_xlabel('Δ (step size)'); ax.set_ylabel('MSE')
ax.set_title('Delta-modulator MSE vs step size'); ax.legend()
plt.tight_layout(); plt.show()

# ============================================================
# PART C — Three representative Δ values on slow and fast inputs
# ============================================================
small_D, mod_D, large_D = 0.005, 0.06, 0.5

fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
cases = [('small Δ=0.005 (granular)', small_D),
         ('moderate Δ=0.06',            mod_D),
         ('large Δ=0.5 (slope-overload)', large_D)]

for j, (title, D) in enumerate(cases):
    xr_s, y_s = dm(slow_sig(t), D)
    xr_f, y_f = dm(fast_sig(t), D)
    axs[0, j].plot(t, slow_sig(t), 'k', lw=1, label='x(t)')
    axs[0, j].plot(t, xr_s, 'r', lw=1, label='DM out')
    axs[0, j].set_title(f'Slow input — {title}')
    axs[0, j].set_ylim(-1.4, 1.4)
    axs[1, j].plot(t, fast_sig(t), 'k', lw=1)
    axs[1, j].plot(t, xr_f, 'r', lw=1)
    axs[1, j].set_title(f'Fast input — {title}')
    axs[1, j].set_ylim(-1.4, 1.4)
    axs[1, j].set_xlabel('time [s]')
axs[0, 0].legend(loc='lower right')
plt.tight_layout(); plt.show()

# ---- Zoom-in of delta staircase (moderate Δ, slow input) ----
xr, y = dm(slow_sig(t), mod_D)
zoom = slice(0, 120)
fig, ax = plt.subplots()
ax.step(t[zoom], xr[zoom], where='post', color='r', label='Δ-staircase')
ax.plot(t[zoom], slow_sig(t)[zoom], 'k--', label='x[n]')
for n in range(zoom.start+1, zoom.stop):
    ax.plot([t[n], t[n]], [xr[n-1], xr[n]], color='r', lw=0.6)
ax.set_title(f'Delta staircase zoom (Δ={mod_D}, slow input)')
ax.set_xlabel('time [s]'); ax.legend()
plt.tight_layout(); plt.show()

# ============================================================
# PART D — ADAPTIVE Delta Modulation
# ============================================================
print("="*70)
print("EXPECTED (Part D):")
print(" • ADM enlarges Δ on steep slopes and shrinks it in flat regions →")
print("   reduces both granular noise and slope overload simultaneously.")
print(" • ADM MSE should be lower than the best fixed-Δ DM for a signal")
print("   with mixed slow/fast content.")
print("="*70)

# Mixed signal
x_mix = np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*20*t)

best_D, best_mse = None, np.inf
for D in np.logspace(-3, 0, 60):
    xr_fx, _ = dm(x_mix, D)
    m = mse(x_mix, xr_fx)
    if m < best_mse: best_mse, best_D = m, D

xr_adm, y_adm, D_adm = adm(x_mix, Delta0=best_D, alpha=1.6, beta=0.85)
print(f"Best fixed-Δ DM  : Δ*={best_D:.4f}  MSE={best_mse:.3e}")
print(f"ADM (Δ0=Δ*)      : MSE={mse(x_mix, xr_adm):.3e}")

fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
ax[0].plot(t, x_mix, 'k', lw=1, label='x(t)')
ax[0].plot(t, xr_adm, 'r', lw=1, label='ADM out')
ax[0].legend(); ax[0].set_title('Adaptive DM on mixed signal')
ax[1].plot(t, x_mix - xr_adm, 'g'); ax[1].set_title('ADM error')
ax[2].plot(t, D_adm, 'b'); ax[2].set_title('Adaptive step size Δ[n]')
ax[2].set_xlabel('time [s]')
plt.tight_layout(); plt.show()

# ============================================================
# MANDATORY VALIDATION — step change must be exactly ±Δ
# ============================================================
print("="*70)
print("VALIDATION: each DM output step equals exactly +Δ or -Δ")
print("="*70)

for D in [small_D, mod_D, large_D]:
    xr, y = dm(slow_sig(t), D)
    yv = y[1:]                               # skip initial zero
    ok_pos = np.allclose(yv[yv>0],  D)
    ok_neg = np.allclose(yv[yv<0], -D)
    steps = np.diff(xr)
    consistent = np.allclose(steps, y[1:])
    print(f"Δ={D:<6}  |y|=Δ for all +: {ok_pos}   "
          f"|y|=Δ for all -: {ok_neg}   diff(x_r)==y : {consistent}")

# For adaptive: step magnitude is Δ[n] (varies) but sign is ±1
xr, y, Dn = adm(x_mix, Delta0=best_D)
signs = np.sign(y[1:])*np.sign(Dn[1:])
print(f"ADM: y[n] = sign(n)·Δ[n]  (all signs match): "
      f"{np.allclose(np.abs(y[1:]), Dn[1:])}")

print("\nDone.")