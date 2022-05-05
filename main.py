import pandas as pd
import numpy as np
import datetime
import util 
import matplotlib.pyplot as plt
import japanize_matplotlib

''' パラメータ設定 '''
mean_generation_time = 2.1  # オミクロン株の平均世代時間（日）
mean_hospitalized_days = 9  # 入院患者の平均入院日数
mean_severe_days = 12  # 重症患者の平均入院日数

# 転帰日数について指数分布に従うと仮定
gamma = 1/mean_generation_time
gamma_H = 1/mean_hospitalized_days
gamma_ICU = 1/mean_severe_days

epsilon_o1 = (0.33 + 0.62)/2  # ワクチン1回目の感染予防効果（高齢者以外）
epsilon_o2 = 0.5  # ワクチン2回目の感染予防効果（高齢者以外）
epsilon_o3 = 0.85  # ワクチン3回目の感染予防効果（高齢者以外）

epsilon_e1 = epsilon_o1  # ワクチン1回目の感染予防効果（高齢者）
epsilon_e2 = epsilon_o2  # ワクチン2回目の感染予防効果（高齢者）
epsilon_e3 = epsilon_o3  # ワクチン3回目の感染予防効果（高齢者）

vaccine_3rd_share_max = 70 / 100  # ワクチン3回目の接種率上限
graph_start_date = datetime.datetime(2021, 12, 15)
wave_start_date = datetime.datetime(2021, 12, 15)
wave_end_date = datetime.datetime(2022, 4, 13)
today = datetime.datetime.today()

T_sim = 7*8  # シミュレーション期間（日）
T_sim_weekly = 8  # シミュレーション期間（週）
T_beta_target = 7*2  # 基本再生産数が何日で BRN_target に到達するか
T_delta_target = 2  # 入院率・重症化率・致死率が何週間で delta_target に到達するか
W_V = 7  # ワクチンの接種スピードについて、直近何日分の平均を取るか

pref_name_JP = '兵庫県'
BRNs = [3.35, 3.4, 3.45]  # 楽観シナリオ, 基本シナリオ, 悲観シナリオ
colors = ['b', 'g', 'r']
num_scenarios = len(BRNs)


''' データの読み込み '''
df = pd.read_csv('https://docs.google.com/spreadsheets/d/1OOwRFo5sh_kaDQF79BdpAHhI_WXXcXpV5tj4NXYQBHk/export?format=csv&gid=1016364865')
df['date'] = pd.to_datetime(df['date'])
df_pref = df[df['都道府県'] == pref_name_JP].reset_index(drop=True)
df_pref_name = pd.read_csv('pref_name.csv')
pref_name_JP_list = df_pref_name['都道府県'].dropna().tolist()


''' 過去データの計算 '''
### Daily data ###
# ワクチン
V_o1 = df_pref['V_o1'].values
V_o2 = df_pref['V_o2'].values
V_o3 = df_pref['V_o3'].values
V_e1 = df_pref['V_e1'].values
V_e2 = df_pref['V_e2'].values
V_e3 = df_pref['V_e3'].values
V_u1 = df_pref['V_u1'].values
V_u2 = df_pref['V_u2'].values
V_u3 = df_pref['V_u3'].values
V_m1 = df_pref['V_m1'].values
V_m2 = df_pref['V_m2'].values

V_o1 = V_o1 + V_u1 + V_m1
V_o2 = V_o2 + V_u2 + V_m2
V_o3 = V_o3 + V_u3

V_2 = V_o2 + V_e2
V_3 = V_o3 + V_e3

V = \
epsilon_o1 * (V_o1-V_o2) + epsilon_o2 * (V_o2-V_o3) + epsilon_o3 * V_o3 +\
epsilon_e1 * (V_e1-V_e2) + epsilon_e2 * (V_e2-V_e3) + epsilon_e3 * V_e3

# Daily SIR model
P = df_pref.loc[0, 'population']  # 人口
cumN = df_pref['cumN'].values
S = P - V - cumN
df_pref['S'] = S
N = df_pref['N'].rolling(7).mean().fillna(0).values
T = N.shape[0]
I = util.calc_I(N, gamma)
beta = util.calc_beta(S, I, N, P)
BRN = beta / gamma
ERN = BRN * S / P
date_index = df_pref['date']
wave_start_idx = np.where(date_index == wave_start_date)[0][0]
wave_end_idx = np.where(date_index == wave_end_date)[0][0]
last_date = date_index.iloc[-1]

### Weekly data ###
df_pref['N'] = df_pref['N'].rolling(7).mean().fillna(0)
df_pref_weekly = df_pref.dropna().reset_index().copy()
N_weekly = df_pref_weekly['N'].values * 7
week_index = df_pref_weekly['date']

H_weekly = df_pref_weekly['H'].values  # 入院患者数
ICU_weekly = df_pref_weekly['ICU'].values  # 重症患者数
D_weekly = df_pref_weekly['D'].values  # 累計死亡者数

# 新規入院患者数
dH_weekly = H_weekly[1:] - H_weekly[:-1] + H_weekly[:-1] * gamma_H * 7
dH_weekly[dH_weekly < 0] = 0

# 新規重症患者数
dICU_weekly = ICU_weekly[1:] - ICU_weekly[:-1] + ICU_weekly[:-1] * gamma_ICU * 7
dICU_weekly[dICU_weekly < 0] = 0

# 新規死亡者数
dD_weekly = D_weekly[1:] - D_weekly[:-1] 

# 入院率の計算
is_N_weekly_zero = N_weekly[:-1] == 0
delta_H_weekly = np.zeros(N_weekly[:-1].shape[0])
delta_H_weekly[~is_N_weekly_zero] = \
dH_weekly[~is_N_weekly_zero] / N_weekly[:-1][~is_N_weekly_zero]
delta_H_weekly[delta_H_weekly > 1] = 1

# 重症化率の計算
is_H_weekly_zero = H_weekly[:-1] == 0
delta_ICU_weekly = np.zeros(H_weekly[:-1].shape[0])
delta_ICU_weekly[~is_H_weekly_zero] = \
dICU_weekly[~is_H_weekly_zero] / H_weekly[:-1][~is_H_weekly_zero]
delta_ICU_weekly[delta_ICU_weekly > 1] = 1

# 致死率の計算
delta_D_weekly = np.zeros(H_weekly[:-1].shape[0])
delta_D_weekly[~is_H_weekly_zero] = \
dD_weekly[~is_H_weekly_zero] / H_weekly[:-1][~is_H_weekly_zero]
delta_D_weekly[delta_D_weekly > 1] = 1


''' 見通し '''
### ワクチンの見通し ###
V_o1_pred = util.pred_linear(V_o1, W_V, T_sim)
V_o2_pred = util.pred_linear(V_o2, W_V, T_sim)
V_o3_pred = util.pred_linear(V_o3, W_V, T_sim)
V_e1_pred = util.pred_linear(V_e1, W_V, T_sim)
V_e2_pred = util.pred_linear(V_e2, W_V, T_sim)
V_e3_pred = util.pred_linear(V_e3, W_V, T_sim)
V_3_max = P*vaccine_3rd_share_max 
V_o3_max = V_3_max * V_o3_pred[0] / (V_o3_pred[0] + V_e3_pred[0])
V_e3_max = V_3_max * V_e3_pred[0] / (V_o3_pred[0] + V_e3_pred[0])
V_o3_pred[V_o3_pred > V_o3_max] = V_o3_max 
V_e3_pred[V_e3_pred > V_e3_max] = V_e3_max 
V_2_pred = V_o2_pred + V_e2_pred
V_3_pred = V_o3_pred+ V_e3_pred

V_pred = \
epsilon_o1 * (V_o1_pred-V_o2_pred) + epsilon_o2 * (V_o2_pred-V_o3_pred) + epsilon_o3 * V_o3_pred +\
epsilon_e1 * (V_e1_pred-V_e2_pred) + epsilon_e2 * (V_e2_pred-V_e3_pred) + epsilon_e3 * V_e3_pred

sim_end_date = last_date + datetime.timedelta(days=T_sim)
date_index_pred = pd.date_range(last_date, sim_end_date)

wave_start_idx_weekly = np.where(week_index == wave_start_date)[0][0]
wave_end_idx_weekly = np.where(week_index == wave_end_date)[0][0]

N_weekly_6th_wave = N_weekly[wave_start_idx_weekly-1:wave_end_idx_weekly]

# 入院率の見通し
delta_H_weekly_avg = np.average(delta_H_weekly[wave_start_idx_weekly:wave_end_idx_weekly+1], weights=N_weekly_6th_wave)
delta_H_weekly_pred = util.calc_beta_pred(delta_H_weekly, delta_H_weekly_avg, W=T_delta_target, T=T_sim_weekly)
delta_H_weekly_pred_half = util.calc_beta_pred(delta_H_weekly, delta_H_weekly_avg/2, W=T_delta_target, T=T_sim_weekly)

# 重症化率の見通し
H_weekly_6th_wave = H_weekly[wave_start_idx_weekly-1:wave_end_idx_weekly]
delta_ICU_weekly_avg = np.average(delta_ICU_weekly[wave_start_idx_weekly:wave_end_idx_weekly+1], weights=H_weekly_6th_wave)
delta_ICU_weekly_pred = util.calc_beta_pred(delta_ICU_weekly, delta_ICU_weekly_avg, W=T_delta_target, T=T_sim_weekly)
delta_ICU_weekly_pred_half = util.calc_beta_pred(delta_ICU_weekly, delta_ICU_weekly_avg/2, W=T_delta_target, T=T_sim_weekly)

# 致死率の見通し
delta_D_weekly_avg = np.average(delta_D_weekly[wave_start_idx_weekly:wave_end_idx_weekly+1], weights=H_weekly_6th_wave)
delta_D_weekly_pred = util.calc_beta_pred(delta_D_weekly, delta_D_weekly_avg, W=T_delta_target, T=T_sim_weekly)
delta_D_weekly_pred_half = util.calc_beta_pred(delta_D_weekly, delta_D_weekly_avg/2, W=T_delta_target, T=T_sim_weekly)

last_date_weekly = df_pref_weekly['date'].iloc[-1]
diff_days = (last_date - last_date_weekly).days
sim_end_date_weekly = last_date_weekly + datetime.timedelta(days=T_sim_weekly * 7)
week_index_pred = pd.date_range(last_date_weekly, sim_end_date_weekly, freq='7D')

# initialization
beta_pred = np.zeros((num_scenarios, T_sim+1))
S_pred = np.zeros((num_scenarios, T_sim+1))
I_pred = np.zeros((num_scenarios, T_sim+1))
N_pred = np.zeros((num_scenarios, T_sim+1))
BRN_pred = np.zeros((num_scenarios, T_sim+1))
ERN_pred = np.zeros((num_scenarios, T_sim+1))

N_weekly_pred = np.zeros((num_scenarios, T_sim_weekly))
H_weekly_pred = np.zeros((num_scenarios, T_sim_weekly+1))
H_weekly_pred_half = np.zeros((num_scenarios, T_sim_weekly+1))
ICU_weekly_pred = np.zeros((num_scenarios, T_sim_weekly+1))
ICU_weekly_pred_half = np.zeros((num_scenarios, T_sim_weekly+1))
dD_weekly_pred = np.zeros((num_scenarios, T_sim_weekly+1))
dD_weekly_pred_half = np.zeros((num_scenarios, T_sim_weekly+1))

for i, BRN_target in enumerate(BRNs):
    # 基本再生産数が直近の水準から T_beta_target（日）で BRN_target に線形のパスで到達するという仮定
    beta_target = BRN_target * gamma
    beta_pred[i] = util.calc_beta_pred(beta, beta_target, T_beta_target, T_sim)
    S_pred[i], I_pred[i], N_pred[i] = util.SIR(S, I, N, V_pred, P, beta_pred[i], gamma, T_sim)
    BRN_pred[i] = beta_pred[i] / gamma
    ERN_pred[i] = BRN_pred[i] * S_pred[i] / P

    # 新規陽性者数のシナリオ作成
    N_weekly_pred[i] = np.hstack([N[-diff_days:], N_pred[i][1:]])[6::7][:T_sim_weekly] * 7
    # 入院患者数の見通し計算
    H_weekly_pred[i] = util.pred_H_weekly(N_weekly_pred[i], H_weekly, gamma_H, delta_H_weekly_pred, T_sim_weekly)
    H_weekly_pred_half[i] = util.pred_H_weekly(N_weekly_pred[i], H_weekly, gamma_H, delta_H_weekly_pred/2, T_sim_weekly)
    # 重症患者数の見通し計算
    ICU_weekly_pred[i] = util.pred_ICU_weekly(ICU_weekly, H_weekly_pred[i], gamma_ICU, delta_ICU_weekly_pred, T_sim_weekly)
    ICU_weekly_pred_half[i] = util.pred_ICU_weekly(ICU_weekly, H_weekly_pred[i], gamma_ICU, delta_ICU_weekly_pred_half, T_sim_weekly)
    # 新規死亡者数の見通し計算
    dD_weekly_pred[i] = util.pred_dD_weekly(dD_weekly, H_weekly_pred[i], delta_D_weekly_pred, T_sim_weekly)
    dD_weekly_pred_half[i] = util.pred_dD_weekly(dD_weekly, H_weekly_pred[i], delta_D_weekly_pred_half, T_sim_weekly)


''' グラフの出力 '''
plt.figure(figsize=(16, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.6)

### 新規陽性者数 ###
plt.subplot(3, 4, 1)
plt.title(f'新規陽性者数（{pref_name_JP}）')
plt.plot(date_index, N, c='black')
for i, brn in enumerate(BRNs):
    plt.plot(date_index_pred, N_pred[i], label=f'BRN = {BRN_target}', c=colors[i])

y_max = N.max() * 4
plt.axvline(x=last_date, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim([graph_start_date, sim_end_date_weekly])
plt.ylim([0, y_max])

plt.subplot(3, 4, 5)
plt.title(f'ワクチン接種率（{pref_name_JP}）')
plt.plot(date_index, V_2/P*100, c='black', linewidth=0.5)
plt.plot(date_index, V_3/P*100, c='black')
plt.plot(date_index_pred, V_2_pred/P*100, label='2本目累計', c='black', linewidth=0.5)
plt.plot(date_index_pred, V_3_pred/P*100, label='3本目累計', c='black')
plt.legend()
plt.axvline(x=last_date, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim([graph_start_date, sim_end_date_weekly])
plt.ylim([0, 100])


### 入院患者数 ###
plt.subplot(3, 4, 2)
plt.title(f'入院患者数 （{pref_name_JP}）\nー入院率が第6波と同じー')
plt.plot(week_index, H_weekly, c='black')
for i, brn in enumerate(BRNs):
    plt.plot(week_index_pred, H_weekly_pred[i], label=f'BRN = {BRN_target}', c=colors[i])

plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim([graph_start_date, sim_end_date_weekly])
y_max = max(H_weekly[wave_start_idx_weekly:].max(), H_weekly_pred.max()) * 1.1
plt.ylim([0, y_max])

plt.subplot(3, 4, 6)
plt.title(f'入院患者数 （{pref_name_JP}）\nー入院率が第6波の半分ー')
plt.plot(week_index, H_weekly, c='black')
for i, brn in enumerate(BRNs):
    plt.plot(week_index_pred, H_weekly_pred_half[i], label=f'BRN = {BRN_target}', c=colors[i], linewidth=0.5)

plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim([graph_start_date, sim_end_date_weekly])
plt.ylim([0, y_max])


### 重症患者数 ###
plt.subplot(3, 4, 3)
plt.title(f'重症患者数【自治体基準】（{pref_name_JP}）\nー重症化率が第6波と同じー')
plt.plot(week_index, ICU_weekly, c='black')
for i, brn in enumerate(BRNs):
    plt.plot(week_index_pred, ICU_weekly_pred[i], label=f'BRN = {BRN_target}', c=colors[i])

plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim([graph_start_date, sim_end_date_weekly])
y_max = max(ICU_weekly[wave_start_idx_weekly:].max(), ICU_weekly_pred.max()) * 1.1
if y_max != 0:
    plt.ylim([0, y_max])
else:
    plt.ylim([0, 10])

plt.subplot(3, 4, 7)
plt.title(f'重症患者数【自治体基準】（{pref_name_JP}）\nー重症化率が第6波の半分ー')
plt.plot(week_index, ICU_weekly, c='black')
for i, brn in enumerate(BRNs):
    plt.plot(week_index_pred, ICU_weekly_pred_half[i], label=f'BRN = {BRN_target}', c=colors[i], linewidth=0.5)

plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim([graph_start_date, sim_end_date_weekly])
if y_max != 0:
    plt.ylim([0, y_max])
else:
    plt.ylim([0, 10])


### 新規死亡者数 ###
plt.subplot(3, 4, 4)
plt.title(f'新規死亡者数（{pref_name_JP}）\nー致死率が第6波と同じー')
plt.plot(week_index[1:], dD_weekly / 7, c='black')
for i, brn in enumerate(BRNs):
    plt.plot(week_index_pred, dD_weekly_pred[i] / 7, label=f'BRN = {BRN_target}', c=colors[i])

plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim([graph_start_date, sim_end_date_weekly])
y_max = max(dD_weekly[wave_start_idx_weekly-1:].max(), dD_weekly_pred.max()) / 7 * 1.1
plt.ylim([0, y_max])

plt.subplot(3, 4, 8)
plt.title(f'新規死亡者数（{pref_name_JP}）\nー致死率が第6波の半分ー')
plt.plot(week_index[1:], dD_weekly / 7, c='black')
for i, brn in enumerate(BRNs):
    plt.plot(week_index_pred, dD_weekly_pred_half[i] / 7, label=f'BRN = {BRN_target}', c=colors[i], linewidth=0.5)

plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim([graph_start_date, sim_end_date_weekly])
plt.ylim([0, y_max])


### 重要パラメータの推移 ###
plt.subplot(3, 4, 9)
plt.title(f'実効再生産数の推移（{pref_name_JP})')
plt.plot(date_index, ERN, c='black')
for i, brn in enumerate(BRNs):
    plt.plot(date_index_pred, ERN_pred[i], label=f'BRN = {BRN_target}', c=colors[i])
plt.axvline(x=last_date, c='black', linewidth=0.5)
plt.xlim([graph_start_date, sim_end_date_weekly])
y_max = ERN[wave_start_idx:].max()
plt.ylim([0, 2])
plt.xticks(rotation=45)

plt.subplot(3, 4, 10)
plt.title(f'入院率の推移（{pref_name_JP})')
plt.plot(week_index[1:], delta_H_weekly, c='black')
plt.plot(week_index_pred, delta_H_weekly_pred, color='blue')
plt.plot(week_index_pred, delta_H_weekly_pred_half, color='blue', linewidth=0.5)
plt.xlim([graph_start_date, sim_end_date_weekly])
plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
plt.ylim([0, 0.1])

plt.subplot(3, 4, 11)
plt.title(f'重症化率の推移（{pref_name_JP})')
plt.plot(week_index[1:], delta_ICU_weekly, c='black')
plt.plot(week_index_pred, delta_ICU_weekly_pred, color='blue')
plt.plot(week_index_pred, delta_ICU_weekly_pred_half, color='blue', linewidth=0.5)
plt.xlim([graph_start_date, sim_end_date_weekly])
plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
y_max = delta_ICU_weekly_pred.max() * 2
plt.ylim([0, y_max])

plt.subplot(3, 4, 12)
plt.title(f'致死率の推移（{pref_name_JP})')
plt.plot(week_index[1:], delta_D_weekly / 7, c='black')
plt.plot(week_index_pred, delta_D_weekly_pred / 7, color='blue')
plt.plot(week_index_pred, delta_D_weekly_pred_half / 7, color='blue', linewidth=0.5)
plt.xlim([graph_start_date, sim_end_date_weekly])
plt.axvline(x=last_date_weekly, c='black', linewidth=0.5)
plt.xticks(rotation=45)
y_max = delta_D_weekly_pred.max() / 7 * 2
plt.ylim([0, y_max])
plt.show()
