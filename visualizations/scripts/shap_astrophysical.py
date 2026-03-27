# SHAP analysis with astrophysical context for each feature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

DATA = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/script/kepler_clean.csv"
OUT  = "/Users/ayaanfarook/VU/Machine Learning/ProjMachineLearning/visualizations/images/shap_astrophysical.png"

# --- data ---
df = pd.read_csv(DATA)
df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration'].replace(0, np.nan)
df['prad_srad_ratio']      = df['koi_prad']  / df['koi_srad'].replace(0, np.nan)
df['snr_depth_ratio']      = df['koi_model_snr'] / df['koi_depth'].replace(0, np.nan)
for col in ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']:
    df[col] = df[col].fillna(df[col].median())

all_cols  = [c for c in df.columns if c != 'target']
unc_cols  = [c for c in all_cols if 'err1' in c or 'err2' in c]
eng_cols  = ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']
base_cols = [c for c in all_cols if c not in unc_cols and c not in eng_cols]
features  = base_cols + unc_cols + eng_cols

X = df[features]
y = df['target'].values

sc = StandardScaler()
X_sc = sc.fit_transform(X)
X_sc_df = pd.DataFrame(X_sc, columns=features)

print("Training XGBoost...")
model = XGBClassifier(n_estimators=150, eval_metric='logloss', random_state=42, verbosity=0)
model.fit(X_sc_df, y)

print("Computing SHAP values...")
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_sc_df)

# mean absolute SHAP for ranking
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
top_n = 18
top_idx   = np.argsort(mean_abs_shap)[::-1][:top_n]
top_feats = [features[i] for i in top_idx]
top_shap  = shap_vals[:, top_idx]
top_vals  = X_sc_df.values[:, top_idx]

# --- astrophysical annotations ---
# Format: feature -> (category, plain_name, astrophysical_meaning)
ASTRO = {
    # Transit geometry
    'koi_model_snr':    ('quality',   'Transit SNR',
                         "Signal-to-noise of transit model fit.\n"
                         "HIGH → clear, repeatable transit. Strong planet signal.\n"
                         "LOW → noisy or irregular dip. Likely false positive."),
    'koi_depth':        ('transit',   'Transit Depth',
                         "Fraction of stellar flux blocked during transit.\n"
                         "Encodes planet-to-star area ratio. Too deep or too shallow\n"
                         "can indicate eclipsing binary (false positive)."),
    'koi_prad':         ('transit',   'Planet Radius',
                         "Estimated planet radius (Earth radii).\n"
                         "Derived from transit depth × stellar radius.\n"
                         "Very large 'planets' are often eclipsing binaries."),
    'koi_duration':     ('transit',   'Transit Duration',
                         "How long the planet takes to cross the star (hours).\n"
                         "Depends on orbital speed and impact parameter.\n"
                         "Inconsistent with stellar density → false positive flag."),
    'koi_period':       ('transit',   'Orbital Period',
                         "Time between transits (days). Long-period planets\n"
                         "are harder to confirm. Short-period planets are\n"
                         "more likely to have follow-up confirmation data."),
    'koi_impact':       ('transit',   'Impact Parameter',
                         "How centrally the planet crosses the stellar disk (0-1).\n"
                         "0 = central transit (deep, V-shaped).\n"
                         "Near 1 = grazing transit (shallow, harder to confirm)."),
    'koi_time0bk':      ('transit',   'First Transit Time',
                         "Barycentric Julian Date of first observed transit.\n"
                         "Mainly an observational reference. Indirectly encodes\n"
                         "which part of Kepler mission the planet was detected in."),
    # Stellar properties
    'koi_steff':        ('stellar',   'Stellar Temperature',
                         "Effective temperature of the host star (Kelvin).\n"
                         "Hotter stars are physically larger — affects radius estimate.\n"
                         "Also affects planet habitability interpretation."),
    'koi_slogg':        ('stellar',   'Stellar Surface Gravity',
                         "Log10 of stellar surface gravity (cm/s²).\n"
                         "Distinguishes dwarfs (high log g) from giants (low log g).\n"
                         "Planet transits around giants are often false positives."),
    'koi_srad':         ('stellar',   'Stellar Radius',
                         "Radius of the host star (solar radii).\n"
                         "Directly converts transit depth to planet radius.\n"
                         "Uncertainty here propagates to planet radius uncertainty."),
    'koi_teq':          ('stellar',   'Equilibrium Temperature',
                         "Estimated planet equilibrium temperature (Kelvin).\n"
                         "Function of stellar luminosity and orbital distance.\n"
                         "Patterns differ between planets and false positives."),
    'koi_insol':        ('stellar',   'Insolation Flux',
                         "Stellar flux received at planet's orbit (Earth units).\n"
                         "High insolation → hot planet close to star.\n"
                         "Correlates with period and stellar temperature."),
    'koi_kepmag':       ('stellar',   'Kepler Magnitude',
                         "Brightness of host star in Kepler bandpass.\n"
                         "Fainter stars have noisier light curves → higher false positive rate.\n"
                         "Brighter stars are easier to follow up with ground telescopes."),
    'koi_tce_plnt_num': ('transit',   'Planet Number in System',
                         "Which planet number this candidate is in the system.\n"
                         "Multi-planet systems strongly suggest real planets\n"
                         "(rare for false positives to occur in pairs)."),
    # Engineered
    'depth_duration_ratio': ('engineered', 'Depth/Duration Ratio',
                             "Transit depth divided by transit duration.\n"
                             "A proxy for transit 'sharpness'. Real planets tend\n"
                             "to have characteristic sharpness for their geometry."),
    'prad_srad_ratio':      ('engineered', 'Planet/Star Radius Ratio',
                             "Planet radius divided by stellar radius.\n"
                             "Directly related to transit depth (√ratio ≈ depth).\n"
                             "Extreme values flag giant 'planets' = likely binary."),
    'snr_depth_ratio':      ('engineered', 'SNR/Depth Ratio',
                             "Signal-to-noise divided by transit depth.\n"
                             "Captures how much SNR you get per unit of depth.\n"
                             "Real planets with the same depth should score higher SNR."),
    # Uncertainty features (any err1/err2)
}

CATEGORY_COLORS = {
    'transit':     '#5c9be0',
    'stellar':     '#5ccc8c',
    'quality':     '#e09b5c',
    'engineered':  '#b07de0',
    'uncertainty': '#ff6b6b',
}

def get_cat(feat):
    for key in ASTRO:
        if feat == key:
            return ASTRO[key][0]
    if 'err1' in feat or 'err2' in feat:
        return 'uncertainty'
    return 'transit'

def get_plain_name(feat):
    if feat in ASTRO:
        return ASTRO[feat][1]
    return feat.replace('koi_', '').replace('_err1', ' ±upper').replace('_err2', ' ±lower')

top_cats   = [get_cat(f) for f in top_feats]
top_labels = [get_plain_name(f) for f in top_feats]
top_colors = [CATEGORY_COLORS[c] for c in top_cats]

# --- figure ---
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("SHAP Feature Importance with Astrophysical Interpretation\n"
             "Why does each feature predict exoplanet confirmation?",
             color="white", fontsize=14, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4,
                       top=0.93, bottom=0.06, left=0.04, right=0.97)

# --- panel 1: coloured beeswarm (manual) ---
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_facecolor("#0f0f1a")

# subsample for speed
idx_sample = np.random.choice(len(shap_vals), min(1500, len(shap_vals)), replace=False)

for feat_rank in range(top_n - 1, -1, -1):
    shap_col = top_shap[idx_sample, feat_rank]
    val_col  = top_vals[idx_sample, feat_rank]

    # jitter y positions
    y_jitter = feat_rank + np.random.uniform(-0.35, 0.35, len(shap_col))

    scatter = ax1.scatter(shap_col, y_jitter,
                          c=val_col, cmap='coolwarm', vmin=-2, vmax=2,
                          s=4, alpha=0.5, linewidths=0)

ax1.axvline(0, color='white', lw=1.0, alpha=0.5)
ax1.set_yticks(range(top_n))
ax1.set_yticklabels([f"{lbl}  [{cat}]" for lbl, cat in zip(top_labels, top_cats)],
                    color='white', fontsize=8)
# colour the y tick labels by category
for tick_label, cat in zip(ax1.get_yticklabels(), top_cats):
    tick_label.set_color(CATEGORY_COLORS[cat])

ax1.set_xlabel("SHAP Value\n(positive = pushes toward 'confirmed planet')", color='white', fontsize=9)
ax1.set_title("Feature Impact on Predictions\n(each dot = one Kepler object | colour = feature value)",
              color='white', fontsize=9)
ax1.tick_params(colors='white')
ax1.grid(axis='x', alpha=0.2, color='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')

# colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(-2, 2))
cb = plt.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02)
cb.ax.tick_params(colors='white')
cb.set_label("Feature value\n(scaled: blue=low, red=high)", color='white', fontsize=7)

# legend for categories
legend_patches = [mpatches.Patch(color=c, label=cat.title())
                  for cat, c in CATEGORY_COLORS.items()]
ax1.legend(handles=legend_patches, facecolor='#1a1a2e', labelcolor='white',
           fontsize=7, loc='lower right')

# --- panel 2: mean |SHAP| bar chart ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0f0f1a")
y2 = np.arange(top_n)
vals2 = mean_abs_shap[top_idx][::-1]
labels2 = top_labels[::-1]
cats2   = top_cats[::-1]
colors2 = [CATEGORY_COLORS[c] for c in cats2]

ax2.barh(y2, vals2, color=colors2, alpha=0.85)
ax2.set_yticks(y2)
ax2.set_yticklabels(labels2, color='white', fontsize=8)
for lbl, cat in zip(ax2.get_yticklabels(), cats2):
    lbl.set_color(CATEGORY_COLORS[cat])
ax2.set_xlabel("Mean |SHAP| (average impact on prediction)", color='white', fontsize=9)
ax2.set_title("Feature Importance Ranking\n(mean absolute SHAP value)", color='white', fontsize=9)
ax2.tick_params(colors='white')
ax2.grid(axis='x', alpha=0.2, color='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

# --- panel 3: astrophysical annotation for top 4 features ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#0f0f1a")
ax3.axis('off')

# SHAP dependence for top feature: shap value vs feature value
top_feat_idx = top_idx[0]
top_feat_name = features[top_feat_idx]
top_feat_plain = get_plain_name(top_feat_name)
top_feat_cat   = get_cat(top_feat_name)

ax3.set_facecolor("#0f0f1a")
ax3.set_visible(False)

ax3b = fig.add_subplot(gs[1, 1])
ax3b.set_facecolor("#0f0f1a")
feat_vals_all  = X_sc_df[top_feat_name].values
shap_vals_all  = shap_vals[:, top_feat_idx]
subsample      = np.random.choice(len(feat_vals_all), min(2000, len(feat_vals_all)), replace=False)
sc_plot = ax3b.scatter(feat_vals_all[subsample], shap_vals_all[subsample],
                       c=y[subsample], cmap='RdBu', alpha=0.4, s=6,
                       vmin=0, vmax=1)
ax3b.axhline(0, color='white', lw=0.8, alpha=0.4)
ax3b.set_xlabel(f"{top_feat_plain} (standardised)", color='white', fontsize=9)
ax3b.set_ylabel("SHAP value", color='white', fontsize=9)
ax3b.set_title(f"SHAP Dependence: {top_feat_plain}\n(red=confirmed planet, blue=false positive)",
               color='white', fontsize=9)
ax3b.tick_params(colors='white')
ax3b.grid(alpha=0.2, color='white')
for spine in ax3b.spines.values():
    spine.set_edgecolor('#444')
cb2 = plt.colorbar(sc_plot, ax=ax3b, fraction=0.04, pad=0.02)
cb2.ax.tick_params(colors='white')
cb2.set_label("True Label", color='white', fontsize=7)

# --- panel 4: astrophysical interpretation cards for top 4 ---
ax4 = fig.add_subplot(gs[:, 2])
ax4.set_facecolor("#0f0f1a")
ax4.axis('off')
ax4.set_title("Why These Features?\nAstrophysical Interpretation",
              color='white', fontsize=10, fontweight='bold', pad=12)

y_cursor = 0.97
for rank in range(min(6, top_n)):
    feat = top_feats[rank]
    cat  = top_cats[rank]
    plain = top_labels[rank]
    color = CATEGORY_COLORS[cat]

    if feat in ASTRO:
        meaning = ASTRO[feat][2]
    else:
        meaning = ("Measurement uncertainty feature.\n"
                   "Tighter errors → more consistent transit → more likely real planet.\n"
                   "Large errors → noisy or irregular signal → possible false positive.")

    ax4.text(0.02, y_cursor, f"#{rank+1} {plain}  [{cat}]",
             color=color, fontsize=9, fontweight='bold',
             transform=ax4.transAxes, va='top')
    ax4.text(0.04, y_cursor - 0.03, meaning,
             color='#cccccc', fontsize=7.8,
             transform=ax4.transAxes, va='top', linespacing=1.55)

    y_cursor -= 0.16
    if y_cursor < 0.02:
        break

for spine in ax4.spines.values():
    spine.set_edgecolor('#444')

plt.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"Saved {OUT}")
