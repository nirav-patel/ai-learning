import matplotlib.pyplot as plt

# Filter data for Q1 of 2024 and 2025
q1_2024_df = df[(df['year'] == 2024) & (df['quarter'] == 1)]
q1_2025_df = df[(df['year'] == 2025) & (df['quarter'] == 1)]

# Calculate total sales for each coffee in Q1 2024 and Q1 2025
sales_2024_by_coffee = q1_2024_df.groupby('coffee_name')['price'].sum().sort_values(ascending=False)
sales_2025_by_coffee = q1_2025_df.groupby('coffee_name')['price'].sum().sort_values(ascending=False)

# Get common coffee names for alignment on x-axis
common_coffees = sorted(set(sales_2024_by_coffee.index) & set(sales_2025_by_coffee.index))

# Prepare data for plotting
coffee_names = common_coffees
sales_2024 = [sales_2024_by_coffee[coffee] for coffee in coffee_names]
sales_2025 = [sales_2025_by_coffee[coffee] for coffee in coffee_names]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Bar width and positions
bar_width = 0.35
index = range(len(coffee_names))

# Plot bars for 2024 and 2025
bars_2024 = ax.bar(index, sales_2024, bar_width, label='2024', color='skyblue')
bars_2025 = ax.bar([i + bar_width for i in index], sales_2025, bar_width, label='2025', color='lightcoral')

# Configure plot details
ax.set_xlabel('Coffee Name', fontsize=12)
ax.set_ylabel('Total Sales ($)', fontsize=12)
ax.set_title('Q1 Coffee Sales Comparison: 2024 vs 2025', fontsize=14)
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(coffee_names, rotation=45, ha='right', fontsize=10)
ax.legend()

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Save the figure
plt.savefig('images/chart_v1.png', dpi=300, bbox_inches='tight')

# Close the plot
plt.close()