import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Filter data for Q1 of 2024 and 2025
q1_2024_df = df[(df['year'] == 2024) & (df['quarter'] == 1)]
q1_2025_df = df[(df['year'] == 2025) & (df['quarter'] == 1)]

# Calculate total sales for each coffee in Q1 2024 and Q1 2025
sales_2024_by_coffee = q1_2024_df.groupby('coffee_name')['price'].sum()
sales_2025_by_coffee = q1_2025_df.groupby('coffee_name')['price'].sum()

# Get common coffee names, sorted alphabetically
common_coffees = sorted(set(sales_2024_by_coffee.index) & set(sales_2025_by_coffee.index))

# Prepare data for plotting
sales_2024 = [sales_2024_by_coffee[c] for c in common_coffees]
sales_2025 = [sales_2025_by_coffee[c] for c in common_coffees]

# Calculate % change
pct_change = [(s25 - s24) / s24 * 100 if s24 > 0 else 0 for s24, s25 in zip(sales_2024, sales_2025)]

# Create figure with two subplots (main bar chart + % change)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#f9f9f9')

# Bar width and positions
bar_width = 0.38
x = np.arange(len(common_coffees))

# Colors
color_2024 = '#2196F3'
color_2025 = '#E53935'

# Plot bars
bars_2024 = ax1.bar(x - bar_width/2, sales_2024, bar_width, label='Q1 2024',
                    color=color_2024, edgecolor='white', linewidth=0.8, alpha=0.88)
bars_2025 = ax1.bar(x + bar_width/2, sales_2025, bar_width, label='Q1 2025',
                    color=color_2025, edgecolor='white', linewidth=0.8, alpha=0.88)

# Add value labels on bars
for bar in bars_2024:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 5, f'${h:.0f}',
             ha='center', va='bottom', fontsize=8, color='#333333', fontweight='bold')

for bar in bars_2025:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 5, f'${h:.0f}',
             ha='center', va='bottom', fontsize=8, color='#333333', fontweight='bold')

# Gridlines
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.set_axisbelow(True)
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc')
ax1.set_facecolor('#f9f9f9')

# Labels and title
ax1.set_ylabel('Total Sales ($)', fontsize=12, labelpad=10)
ax1.set_title('Q1 Coffee Sales Comparison: 2024 vs 2025', fontsize=15, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(common_coffees, rotation=30, ha='right', fontsize=10)
ax1.legend(fontsize=11, framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Extend y-axis a bit for labels
ymax = max(max(sales_2024), max(sales_2025))
ax1.set_ylim(0, ymax * 1.18)

# --- Bottom subplot: % change ---
bar_colors = ['#43A047' if p >= 0 else '#E53935' for p in pct_change]
bars_pct = ax2.bar(x, pct_change, bar_width * 1.5, color=bar_colors, edgecolor='white', linewidth=0.8, alpha=0.85)

for bar, pct in zip(bars_pct, pct_change):
    h = bar.get_height()
    va = 'bottom' if h >= 0 else 'top'
    offset = 2 if h >= 0 else -2
    ax2.text(bar.get_x() + bar.get_width()/2, h + offset, f'{pct:+.1f}%',
             ha='center', va=va, fontsize=8.5, fontweight='bold', color='#333333')

ax2.axhline(0, color='#555555', linewidth=0.9, linestyle='-')
ax2.set_axisbelow(True)
ax2.grid(axis='y', linestyle='--', alpha=0.4, color='#cccccc')
ax2.set_facecolor('#f9f9f9')
ax2.set_xticks(x)
ax2.set_xticklabels(common_coffees, rotation=30, ha='right', fontsize=10)
ax2.set_ylabel('YoY Change (%)', fontsize=10, labelpad=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

pct_abs_max = max(abs(p) for p in pct_change)
ax2.set_ylim(-pct_abs_max * 1.35, pct_abs_max * 1.35)

plt.tight_layout(h_pad=2.5)
plt.savefig('images/chart_v2.png', dpi=300, bbox_inches='tight')
plt.close()