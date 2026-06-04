# V32 — RESET: Code lại đúng idea cốt lõi DINN/NEC/NPC

**Mục tiêu:** xoá mọi nhánh đi xa (h_total scheme, PHYS_SCALE_FACTOR=2500², continuity loss, infectious_case_loss, log_mse mode tuỳ chọn, gradient balancer, adaptive w_phys, v.v.) và code lại sạch theo đúng 3 idea đã thống nhất.

**Lý do reset:** từ V23 → V31 code đã tích lũy nhiều thử nghiệm sai (mỗi version vá một lỗi mới sinh). V3.1 stripped luôn physics residual khiến `phys=0.000e+00` toàn bộ training → cal overfit, proj fail. Làm sạch từ đầu rẻ hơn vá tiếp.

---

## 1. ĐỊNH NGHĨA CỐ ĐỊNH (KHÔNG ĐƯỢC DIỄN GIẢI LẠI)

### 1.1 Mô hình ODE — 7 compartment mosquito–human (giống NPC)
States: `Ms, Me, Mi, Hs, He, Hi, Hr` (+ `Itot` cumulative aux, derivative = `theta_h * He`).

Tham số học được (8): `lambda, beta_m, mu_m, theta_m, mu_h, beta_h, theta_h, gamma_h`.
**Lưu ý:** `mu_h = 0.0004` cố định theo NPC paper, KHÔNG học (chỉ 7 params trainable, dù bounds vẫn lưu 8 dòng).

RHS (NPC x7m.m, KHÔNG có hệ số 2500 ở mức global):
```
dMs/dt  = lambda - beta_m * Ms * Hi - mu_m * Ms
dMe/dt  = beta_m * Ms * Hi - (mu_m + theta_m) * Me
dMi/dt  = theta_m * Me - mu_m * Mi
dHs/dt  = -beta_h * Hs * Mi
dHe/dt  =  beta_h * Hs * Mi - theta_h * He
dHi/dt  = theta_h * He - gamma_h * Hi - mu_h * Hi
dHr/dt  = gamma_h * Hi
dItot/dt= theta_h * He
```

### 1.2 NPC paradigm (data protocol)
- Mỗi region có **HAI outbreak độc lập** lưu trong `data_cali.csv` và `data_proj.csv`.
- Proj **KHÔNG phải forecasting nối tiếp** sau cal — là outbreak khác (calendar có thể trước hoặc sau cal).
- **cal = training set**, **proj = hold-out test set**.
- Cấm gradient từ proj. Cấm continuity loss giữa cuối-cal và đầu-proj. Cấm dùng IC cuối cal làm IC đầu proj.

### 1.3 DINN (Shaier 2022) — xác nhận từ repo
- Một MLP duy nhất `t → 7 states` (Hr, Hs, He, Hi, Me, Ms, Mi).
- Itot tính ngoài MLP: cumulative trapezoid của `theta_h * He` hoặc state thứ 8 cũng được, miễn nhất quán.
- Tham số ODE 7 cái là `nn.Parameter` (sigmoid-bounded trong DINN_PARAM_BOUNDS).
- **Loss = data MSE (trên các state quan sát) + physics residual MSE (autograd dstate/dt vs RHS)**, weight 1:1 trong repo gốc. Project ta dùng `w_phys = 3.0 fixed` (đã chốt).

### 1.4 NEC — dual network (user khẳng định)
- **Net A — Calibration**: `input baseline_features → 8 IC (log1p) + 7 params (linear, mu_h fixed)`. Output 15 scalars, sigmoid-bounded theo `Range7_{region}.mat`.
- **Net B — Emulation**: `MLP(t) → 7 states` giống DINN MLP, dùng để compute autograd residual và data fit. Itot từ trajectory này.
- Loss giống DINN: `data_MSE + w_phys * residual_MSE`, nhưng residual dùng params từ Net A (gradient chảy qua cả 2 net).
- **KHÔNG dùng RK45 solver bên trong NEC** — đó là NPC. NEC là PINN-style giống DINN, chỉ khác có thêm head calibration.

### 1.5 Quan sát + scaling
- Chỉ `Itot` (cumulative cases) là observable bắt buộc. `Hi` (incidence) có thể bật khi data có sẵn — V32 mặc định **chỉ fit Itot** để đơn giản.
- Data loss: MSE trên `log1p(Itot_pred)` vs `log1p(Itot_true)` trên grid cal.
- Residual loss: MSE thuần trên 7 ODE residuals trên cùng grid cal, **không chia PHYS_SCALE_FACTOR**, không weight per-state.

### 1.6 Range7 IC + param bounds (16 × 2, [lower, upper])
Đọc từ `Range7_{region}-2.mat` (sample bello-2 đã verify):
- Rows 0–7: IC bounds trong **log1p space**, thứ tự `[Itot, Hi, Me, Hr, Hs, He, Ms, Mi]`.
- Rows 8–15: param bounds **linear space**, thứ tự `[beta_h, beta_m, gamma_h, lambda, mu_h, mu_m, theta_h, theta_m]`.
- `mu_h` row giữ nguyên nhưng predicted value override = 0.0004.

---

## 2. NHỮNG GÌ XOÁ KHỎI CODE (KHÔNG VIẾT LẠI)

| Item | Lý do xoá |
|---|---|
| `h_total = Hs + He + Hi + Hr` scheme | Sai với NPC RHS, gây V27 fail |
| `PHYS_SCALE_FACTOR = 2500**2` chia residual | V28 over-correction, NPC không dùng global scale |
| `loss_continuity` (nối cuối-cal sang đầu-proj) | Vi phạm NPC paradigm (V29 root cause) |
| `loss_infectious_case` aux | Không có trong DINN/NPC, làm loud signal trong logs |
| `loss_itot_linear` denom = peak² | Sai V30, bỏ luôn linear variant |
| `loss_itot_proj` trong gradient | Data leak (user feedback) |
| `GradBalancer`, `adaptive w_phys` | Sai với "fixed w_phys=3.0" |
| `itot_loss_mode` switch (log_mse vs linear) | Cố định log MSE, bỏ branch |
| `_attach_shifted_proj_weeks` | Hệ quả của continuity bug |
| `w_phys_real = 0.3` default | Lệch với 3.0 đã chốt |
| Hardcode `loss_residual_* = tensor(0.0)` | Bug V3.1, khôi phục residual thật |
| Log `terms=4` khi gradient chỉ 1 term | Cosmetic bug, fix bằng cách khôi phục đúng số term |

---

## 3. CẤU TRÚC CODE V32 (baseline_experiments-7.py)

### 3.1 Module layout
```
baseline_experiments-7.py
├── constants:        STATE_COLS, DINN_PARAM_BOUNDS, MU_H_FIXED=0.0004
├── load_range7():    đọc Range7_{region}-2.mat → dict {ic_bounds, param_bounds}
├── load_npc_data():  đọc data_cali.csv / data_proj.csv → (t_cal, itot_cal, t_proj, itot_proj)
├── ODESystem:        RHS function (Ms..Hr + Itot), pure torch ops
├── DINNNet:          MLP(t) → 7 states; sigmoid params (7, mu_h injected)
├── NECNet:           CalibrationHead (15 scalars) + EmulationMLP (7 states)
├── compute_loss():   data_loss + w_phys * residual_loss (cal only)
├── train_one():      Adam + ReduceLROnPlateau, 10000 ep
├── evaluate():       trên proj dưới no_grad, log R²/MAE/MSE
└── run_all():        loop 6 regions × 2 models {dinn, nec}
```

### 3.2 Loss (CHỈ 2 TERM TRONG GRADIENT)
```python
# trên cal grid:
itot_pred_cal   = trajectory_cal[:, IDX_ITOT]
loss_data       = F.mse_loss(torch.log1p(itot_pred_cal.clamp(min=0)),
                             torch.log1p(itot_cal.clamp(min=0)))

# autograd residual trên cùng cal grid:
residuals       = ode_system.residual(t_cal, trajectory_cal, params)  # (T, 7)
loss_residual   = (residuals ** 2).mean()

loss = loss_data + W_PHYS * loss_residual   # W_PHYS = 3.0 fixed

# proj: chỉ no_grad, log metric, KHÔNG vào loss
with torch.no_grad():
    itot_pred_proj = ...
    metric_proj = R²(itot_pred_proj, itot_proj)
```

### 3.3 Sigmoid bounding (DINN params + NEC IC/params)
```python
param_raw  = nn.Parameter(torch.zeros(...))     # init 0
param_val  = lower + (upper - lower) * sigmoid(param_raw)
```
NEC IC head: output 8 raw scalars → sigmoid → expand vào bounds log1p → `expm1` về linear → đưa vào `Net B` qua initial-state constraint (xem 3.4).

### 3.4 Net B emulation constraint
Standard DINN trick để MLP(t=t0) ≈ IC: dùng hard constraint
```
state(t) = IC + (1 - exp(-(t - t0))) * MLP(t)
```
hoặc soft IC loss riêng. Chọn **hard constraint** (đơn giản hơn, không thêm loss term).

### 3.5 Config
```python
@dataclass
class V32Config:
    epochs: int = 10000
    lr: float = 5e-4
    weight_decay: float = 1e-5
    w_phys: float = 3.0          # FIXED, KHÔNG adaptive
    seed: int = 1234
    device: str = "cpu"
    scheduler_factor: float = 0.7
    scheduler_patience: int = 3000
    min_lr: float = 1e-7
```

---

## 4. ACCEPTANCE CRITERIA

Code v32 PASS khi tất cả thoả:

1. Grep file:
   - Không còn `h_total`, `PHYS_SCALE_FACTOR`, `loss_continuity`, `infectious_case`, `GradBalancer`, `adaptive`, `itot_loss_mode`, `w_phys_real`.
   - Không có `loss_residual_* = torch.tensor(0.0)` (residual phải là tensor có grad).
   - Có đúng 1 dòng `W_PHYS = 3.0` (hoặc `w_phys: float = 3.0`).
2. Loss in log mỗi 1000 epoch hiển thị `loss_data=..., loss_phys=...` với **loss_phys > 0** sau 500 ep đầu.
3. Trên Bello/SanJuan: `R²_cal > 0.85` và `R²_proj > 0.0` (proj không cần đẹp như cal, miễn không âm rõ rệt).
4. Proj evaluation hoàn toàn dưới `torch.no_grad()` (grep `loss_itot_proj` không xuất hiện trong tổng loss).
5. Tham số `mu_h` không phải `nn.Parameter` (constant 0.0004 inject vào RHS).

---

## 5. WORKFLOW CHO CLAUDE/CODEX

Paste-in block dưới đây vào Claude/Codex coding agent:

```
Task: viết lại từ đầu file baseline_experiments-7.py theo spec V32_RESET_CORE_IDEA_PROMPT.md.

Constraints (must):
- Đọc V32_RESET_CORE_IDEA_PROMPT.md trước khi code.
- KHÔNG copy logic từ baseline_experiments-6.py — chỉ tham khảo cấu trúc CSV output.
- Output CSV: baseline_results_v32.csv (region, model, R²_cal, R²_proj, MAE_cal, MAE_proj, loss_data_final, loss_phys_final, epochs_run, runtime_s).
- Output predictions: baseline_predictions_v32.csv (region, model, split, t, itot_true, itot_pred).
- Notebook companion: baseline_all_results-7.ipynb chạy được end-to-end trên CPU.
- Dataset: /tmp/npc_repo/data/Dengue/{region}/data_{cali,proj}.csv, range: Range7_{region}-2.mat.

Forbidden:
- adaptive balancer, continuity loss, h_total scheme, PHYS_SCALE_FACTOR, infectious_case_loss, log_mse switch.
- gradient từ projection data.
- mu_h trainable.

Đầu ra mong đợi sau 1 lần chạy đầy đủ: 6 region × 2 model = 12 row CSV, R²_cal trung bình > 0.7.
```

---

## 6. CHECKLIST TRƯỚC KHI CHẠY

- [ ] Đã đọc lại NPC paper section 2.3 (cal/proj definition) và DINN repo dengue.ipynb (loss structure).
- [ ] Đã copy Range7_{region}-2.mat về workspace cho cả 6 region (bello, cambodia, iquitos, philippines, sanjuan, vietnam) — hiện chỉ có bello-2, **5 region còn lại cần user cung cấp hoặc fallback về Range7_{region}-1.mat**.
- [ ] Đã xác nhận `/tmp/npc_repo/data/Dengue/{region}/data_{cali,proj}.csv` tồn tại cho 6 region.
- [ ] Backup `baseline_experiments-6.py` (giữ làm reference, không sửa).

---

## 7. NHỮNG GÌ KHÔNG LÀM TRONG V32

- KHÔNG thêm Hi observable (chờ V33 nếu V32 pass).
- KHÔNG thêm Rt computation, peak detection, time-to-peak metric.
- KHÔNG đổi optimizer (giữ Adam).
- KHÔNG thử LBFGS finetune cuối.
- KHÔNG implement ensemble seed.
- KHÔNG so sánh với V3.0 baseline_results.csv trong cùng file — để table riêng nếu cần.

Một mục tiêu duy nhất: **chạy được đúng idea DINN/NEC/NPC, score reasonable, code clean ≤ 800 dòng**.
