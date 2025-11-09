from multiprocessing import Pool
import pandas as pd
import sqlite3


def mapper(row):
    return (row["Month"], row["Temperature_Celsius"])


def reducer(mapped_data):
    result = {}
    for month, temp in mapped_data:
        result.setdefault(month, []).append(temp)
    return {m: sum(v) / len(v) for m, v in result.items()}


def run_mapreduce(df):
    with Pool() as p:
        mapped = p.map(mapper, [row for _, row in df.iterrows()])
    reduced = reducer(mapped)

    print("\n Average Temperature per Month:")
    for m, t in reduced.items():
        print(f"{m}: {t:.2f}")
    return reduced


def top_fire_months(df, top_n=5):
    top = df.groupby("Month")["Burned_Area_hectares"].mean().sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} Months with Largest Fire Area:\n{top}\n")
    return top


def temperature_area_correlation(df):
    corr = df["Temperature_Celsius"].corr(df["Burned_Area_hectares"])
    print(f" Correlation between Temperature and Fire Area: {corr:.2f}")
    return corr


def query_avg_area_by_month(conn):
    query = """
        SELECT Month, AVG(Burned_Area_hectares) AS avg_area
        FROM forestfires
        GROUP BY Month
        ORDER BY avg_area DESC;
    """
    result = pd.read_sql_query(query, conn)
    print("\nAverage Burned Area by Month (from SQL):")
    print(result)
    return result


def run_pipeline():
    print("===  Forest Fire Analysis Pipeline Started ===\n")

    df = pd.read_csv("forestfires.csv")
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    conn = sqlite3.connect("forestfires.db")
    df.to_sql("forestfires", conn, if_exists="replace", index=False)
    print(" Data saved to SQLite database.\n")

    run_mapreduce(df)
    top_fire_months(df)
    temperature_area_correlation(df)
    query_avg_area_by_month(conn)

    print("\n===  Pipeline Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline()


################################################33 substitute
# from mrjob.job import MRJob
# import pandas as pd
# import sqlite3
# import os


# # ====== 1. MRJob Definition ======
# class MRAverageTemperature(MRJob):
#     """
#     This MRJob computes the average temperature per month.
#     It replaces the manual multiprocessing MapReduce.
#     """

#     def mapper(self, _, line):
#         # Skip the header
#         if line.startswith("X_Coordinate"):
#             return

#         parts = line.strip().split(",")
#         if len(parts) < 13:
#             return  # skip malformed rows

#         try:
#             month = parts[2]              # Month column
#             temperature = float(parts[8]) # Temperature_Celsius column
#             yield month, temperature
#         except ValueError:
#             pass  # skip rows where temperature is not numeric

#     def reducer(self, month, temps):
#         temps = list(temps)
#         if temps:
#             yield month, sum(temps) / len(temps)


# # ====== 2. Other Analysis Functions ======
# def top_fire_months(df, top_n=5):
#     top = (
#         df.groupby("Month")["Burned_Area_hectares"]
#         .mean()
#         .sort_values(ascending=False)
#         .head(top_n)
#     )
#     print(f"\nTop {top_n} Months with Largest Fire Area:\n{top}\n")
#     return top


# def temperature_area_correlation(df):
#     corr = df["Temperature_Celsius"].corr(df["Burned_Area_hectares"])
#     print(f"Correlation between Temperature and Fire Area: {corr:.2f}")
#     return corr


# def query_avg_area_by_month(conn):
#     query = """
#         SELECT Month, AVG(Burned_Area_hectares) AS avg_area
#         FROM forestfires
#         GROUP BY Month
#         ORDER BY avg_area DESC;
#     """
#     result = pd.read_sql_query(query, conn)
#     print("\nAverage Burned Area by Month (from SQL):")
#     print(result)
#     return result


# # ====== 3. Main Pipeline ======
# def run_pipeline():
#     print("=== Forest Fire Analysis Pipeline Started ===\n")

#     df = pd.read_csv("forestfires.csv")
#     print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

#     # Save to SQLite
#     conn = sqlite3.connect("forestfires.db")
#     df.to_sql("forestfires", conn, if_exists="replace", index=False)
#     print("Data saved to SQLite database.\n")

#     # ---- Run MRJob ----
#     print("Running MRJob to calculate average temperature per month...\n")
#     mr_input = "forestfires.csv"

#     # job = MRAverageTemperature(args=[mr_input, "--no-conf", "--quiet"])
#     job = MRAverageTemperature(args=[mr_input, "-r", "local", "--no-conf", "--quiet"])

#     with job.make_runner() as runner:
#         runner.run()
#         for line in runner.stream_output():
#             key, value = job.parse_output_line(line)
#             print(f"{key}: {value:.2f}")

#     # ---- Other analyses ----
#     top_fire_months(df)
#     temperature_area_correlation(df)
#     query_avg_area_by_month(conn)

#     print("\n=== Pipeline Completed Successfully ===")


# if __name__ == "__main__":
#     run_pipeline()
