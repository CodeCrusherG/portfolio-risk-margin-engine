import pandas as pd

df = pd.read_excel("risk_report.xlsx", sheet_name="Exceptions")
df["Date"] = pd.to_datetime(df["Date"])
print("Exceptions per year:")
print(df.groupby(df["Date"].dt.year).size(), "\n")
print("Months with the most exceptions:")
print(df.groupby(df["Date"].dt.to_period("M")).size()
        .sort_values(ascending=False).head(5))