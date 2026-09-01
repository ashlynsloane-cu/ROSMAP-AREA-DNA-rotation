import matplotlib
matplotlib.use('Agg')  # headless safe
import matplotlib.pyplot as plt
import numpy as np
import os

# Set standard styles
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['svg.fonttype'] = 'none'

# Define final physical dimensions (170 mm wide x 160 mm high)
# 170 mm = 6.69 inches, 160 mm = 6.30 inches
fig_width_inch = 6.69
fig_height_inch = 6.30

fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))

# Set up GridSpec: 
# Top subplot (continuous comparison, 8 variables)
# Bottom subplot (binary counts, 5 variables)
# Height ratios roughly 60% for top, 40% for bottom
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 1, height_ratios=[1.7, 1.0], hspace=0.35)

ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1])

# --- TOP PANEL: Continuous / Ordinal attributes (parallel comparison) ---
# Ordered by biological categories: Cognition, Neuropathology, Disease Timing, Systemic
variables_top = [
    "Body Mass Index (BMI)",
    "Age at Death",
    "Age at AD Onset",
    "Braak Tangles",
    "CERAD Plaques",
    "Consensus Cognition",
    "Global Cognition",
    "MMSE Score"
]

# Raw counts unchanged (reproduced from genome-wide screen)
counts_weighted = np.array([101,  391, 3721, 3500,  765, 2063, 1930, 3478]) # Dosage-like
counts_regular  = np.array([1423,  107,    0,   10,   53,   25,  451, 1661]) # Threshold-like
counts_shared   = np.array([1989, 2485, 1110, 3501, 7772, 6503, 9139, 7562]) # Shared/robust

# Calculate total significant genes per attribute
totals_top = counts_weighted + counts_regular + counts_shared

# Compute percentages
pct_weighted = (counts_weighted / totals_top) * 100
pct_regular  = (counts_regular / totals_top) * 100
pct_shared   = (counts_shared / totals_top) * 100

# Colors matching requested colorblind palette
color_dosage = "#1f77b4"     # Blue (Dosage-like)
color_threshold = "#ff7f0e"  # Orange (Threshold-like)
color_shared = "#2ca02c"     # Green (Shared)

y_indices = np.arange(len(variables_top))
bar_height = 0.65

# Plot stacked bars
ax_top.barh(y_indices, pct_weighted, color=color_dosage, height=bar_height, edgecolor='white', linewidth=0.3, label="Dosage-like response")
ax_top.barh(y_indices, pct_regular, left=pct_weighted, color=color_threshold, height=bar_height, edgecolor='white', linewidth=0.3, label="Threshold-like response")
ax_top.barh(y_indices, pct_shared, left=pct_weighted + pct_regular, color=color_shared, height=bar_height, edgecolor='white', linewidth=0.3, label="Shared response")

# Add internal percentages (only for segments >= 5%)
for i in range(len(variables_top)):
    left_accum = 0.0
    
    # 1. Weighted segment
    val_w = pct_weighted[i]
    if val_w >= 5.0:
        ax_top.text(left_accum + val_w/2, i, f"{int(round(val_w))}%", va='center', ha='center', color='white', fontsize=7.5, fontweight='bold')
    left_accum += val_w
    
    # 2. Regular segment
    val_r = pct_regular[i]
    if val_r >= 5.0:
        ax_top.text(left_accum + val_r/2, i, f"{int(round(val_r))}%", va='center', ha='center', color='white', fontsize=7.5, fontweight='bold')
    left_accum += val_r
    
    # 3. Shared segment
    val_s = pct_shared[i]
    if val_s >= 5.0:
        ax_top.text(left_accum + val_s/2, i, f"{int(round(val_s))}%", va='center', ha='center', color='white', fontsize=7.5, fontweight='bold')

# Annotate N values on the right
for i, (total, name) in enumerate(zip(totals_top, variables_top)):
    ax_top.text(101.5, i, f"N={total:,}", va='center', ha='left', fontsize=8, color="#2c3e50", fontweight='bold')

# Formatting top panel
ax_top.set_xlim(0, 100)
ax_top.set_yticks(y_indices)
ax_top.set_yticklabels(variables_top, fontsize=8.5, color="#2c3e50")
ax_top.set_xticklabels([]) # Hide x axis ticks on top panel
ax_top.xaxis.grid(True, linestyle="--", alpha=0.3, color="#bdc3c7")
ax_top.set_axisbelow(True)

# Add subtle separators to group by biological categories
for split_idx in [0.5, 2.5, 4.5]:
    ax_top.axhline(split_idx, color="#bdc3c7", linestyle="-", alpha=0.3, linewidth=0.7)

# Remove top and right spines
for spine in ["top", "right", "bottom"]:
    ax_top.spines[spine].set_visible(False)
ax_top.spines["left"].set_color("#7f8c8d")
ax_top.spines["left"].set_linewidth(0.5)


# --- BOTTOM PANEL: Binary attributes analyzed with a single binary model ---
variables_bot = [
    "APOE \u03b54 Carrier",
    "Diabetes History",
    "Hypertension History",
    "Biological Sex",
    "Stroke History"
]
counts_bot = [26, 43, 570, 2294, 2907] # Raw counts exactly unchanged

y_indices_bot = np.arange(len(variables_bot))

# Plot simple horizontal bars using neutral blue-gray
color_binary = "#7f8c8d"
ax_bot.barh(y_indices_bot, counts_bot, color=color_binary, height=bar_height, edgecolor='white', linewidth=0.3)

# Annotate absolute values on the right of the bars
for i, count in enumerate(counts_bot):
    ax_bot.text(count + 35, i, f"N={count:,}", va='center', ha='left', fontsize=8, color="#2c3e50", fontweight='bold')

# Set labels and formatting
ax_bot.set_yticks(y_indices_bot)
ax_bot.set_yticklabels(variables_bot, fontsize=8.5, color="#2c3e50")
ax_bot.set_xlabel("Number of significant genes", fontsize=8.5, fontweight='bold', labelpad=4)
# Since the bottom axis represents counts, we'll configure its own x-labels
ax_bot.set_xlim(0, 3200)
ax_bot.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
ax_bot.set_xticklabels(["0", "500", "1,000", "1,500", "2,000", "2,500", "3,000"], fontsize=8, color="#2c3e50")
ax_bot.xaxis.grid(True, linestyle="--", alpha=0.3, color="#bdc3c7")
ax_bot.set_axisbelow(True)

# Remove top and right spines
for spine in ["top", "right"]:
    ax_bot.spines[spine].set_visible(False)
ax_bot.spines["left"].set_color("#7f8c8d")
ax_bot.spines["left"].set_linewidth(0.5)
ax_bot.spines["bottom"].set_color("#7f8c8d")
ax_bot.spines["bottom"].set_linewidth(0.5)


# --- Master Titles & Legend ---
# 1. Main Title
fig.text(0.04, 0.95, "Clinical attributes show distinct dosage-like, threshold-like,\nand shared transcriptomic response architectures", 
         fontsize=10.0, fontweight='bold', color="#2c3e50", ha='left', va='top')

# 2. Take-home subtitle
fig.text(0.04, 0.89, "AD onset and Braak contain large dosage-like components, BMI is threshold-enriched, while CERAD and cognition are predominantly shared.",
         fontsize=7.8, fontweight='medium', color="#34495e", ha='left', va='top')

# 3. Methodological category definition
fig.text(0.04, 0.84, "Among BH-FDR < 0.05 genes, bars show whether associations are detected only when phenotype severity is retained continuously\n(dosage-like), only after discretization (threshold-like), or under both representations (shared response).",
         fontsize=7.0, color="#7f8c8d", style='italic', ha='left', va='top')

# 4. Binary Section Heading
fig.text(0.04, 0.35, "Binary attributes analyzed with a single binary model",
         fontsize=8.5, fontweight='bold', color="#2c3e50", ha='left', va='center')

# Legend for the stacked bar
handles, labels = ax_top.get_legend_handles_labels()
ax_top.legend(reversed(handles), labels[::-1], loc="lower center", bbox_to_anchor=(0.5, 1.04), 
              frameon=False, ncol=3, fontsize=7.8, handletextpad=0.5, columnspacing=1.0)

# Adjust margins precisely without bbox_inches='tight' to guarantee exact physical sizes
plt.subplots_adjust(left=0.28, right=0.88, top=0.74, bottom=0.10)

# Path resolution: support both remote container execution and local MacBook execution
save_dir = "."
if os.path.exists("/workspace/scratch") and os.access("/workspace/scratch", os.W_OK):
    save_dir = "/workspace/scratch"
elif os.path.exists("results"):
    save_dir = "results"
elif os.path.exists("exploratory/visualization"):
    save_dir = "exploratory/visualization"

print(f"Output files will be saved directly to: {save_dir}")

# Save high-res PNG (300 DPI) and vector PDF
fig.savefig(os.path.join(save_dir, "unbiased_transcriptomic_archetypes_v2.png"), dpi=300)
fig.savefig(os.path.join(save_dir, "unbiased_transcriptomic_archetypes_v2.pdf"))

# Generate web-optimized versions at exact pixel widths (1200px and 600px)
# At width = 6.69 inches:
# 1200 pixels / 6.69 in = 179.37 DPI
# 600 pixels / 6.69 in = 89.68 DPI
fig.savefig(os.path.join(save_dir, "unbiased_transcriptomic_archetypes_v2_1200.png"), dpi=179.37)
fig.savefig(os.path.join(save_dir, "unbiased_transcriptomic_archetypes_v2_600.png"), dpi=89.68)

plt.close()
print("Successfully generated all updated journal and web-optimized figures.")
