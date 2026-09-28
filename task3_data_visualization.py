# CodeAlpha Data Analytics Internship - Task 3
# Data Visualization: Retail Sales Performance Analysis (2024-2025)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Generate the synthetic retail dataset described in the report
n = 15648
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
categories = ["Electronics", "Clothing", "Home", "Beauty", "Books", "Sports"]
regions = ["North", "South", "East", "West"]

df = pd.DataFrame({
    "order_date": np.random.choice(dates, n),
    "category": np.random.choice(categories, n),
    "region": np.random.choice(regions, n),
    "quantity": np.random.randint(1, 8, n),
    "unit_price": np.round(np.random.uniform(10, 500, n), 2),
    "discount": np.round(np.random.choice([0, .05, .10, .15, .20],
                                          n, p=[.55, .10, .15, .12, .08]), 2)
})

# Make Electronics higher priced, as described in the report
df.loc[df["category"] == "Electronics", "unit_price"] *= 2.2
df["unit_price"] = df["unit_price"].clip(upper=1000)
df["revenue"] = np.round(
    df["quantity"] * df["unit_price"] * (1 - df["discount"]), 2
)

# Save dataset
df.to_csv("retail_sales_2024_2025.csv", index=False)

# 1. Monthly Revenue Trend
monthly = df.set_index("order_date").resample("ME")["revenue"].sum()
plt.figure(figsize=(10, 5))
monthly.plot(marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.savefig("01_monthly_revenue_trend.png", dpi=150)
plt.close()

# 2. Revenue by Product Category
category_revenue = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
category_revenue.plot(kind="bar")
plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("02_revenue_by_category.png", dpi=150)
plt.close()

# 3. Revenue Share by Region
region_revenue = df.groupby("region")["revenue"].sum()
plt.figure(figsize=(6, 6))
region_revenue.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Revenue Share by Region")
plt.ylabel("")
plt.tight_layout()
plt.savefig("03_revenue_share_region.png", dpi=150)
plt.close()

# 4. Revenue by Day of Week
df["day_of_week"] = df["order_date"].dt.day_name()
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
day_revenue = df.groupby("day_of_week")["revenue"].sum().reindex(day_order)
plt.figure(figsize=(9, 5))
day_revenue.plot(kind="bar")
plt.title("Revenue by Day of Week")
plt.xlabel("Day")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("04_revenue_by_day.png", dpi=150)
plt.close()

# 5. Unit Price Distribution by Category
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="category", y="unit_price")
plt.title("Unit Price Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Unit Price ($)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("05_unit_price_distribution.png", dpi=150)
plt.close()

# 6. Revenue Heatmap: Category vs Region
heatmap_data = pd.pivot_table(
    df, values="revenue", index="category", columns="region", aggfunc="sum"
)
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".0f")
plt.title("Revenue Heatmap: Category vs Region")
plt.tight_layout()
plt.savefig("06_revenue_heatmap.png", dpi=150)
plt.close()

print("Task 3 completed.")
print("Total orders:", len(df))
print("Total revenue: $", round(df["revenue"].sum(), 2))
print("Average order value: $", round(df["revenue"].mean(), 2))
print("Files and charts generated successfully.")
