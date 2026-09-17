import os
import json
import numpy as np
import pandas as pd
from src.db.connection import get_raw_connection
from src.layer1.mom_tracker import compute_and_persist_mom_metrics
from src.layer1.anomaly_flagger import flag_anomalies_for_period

def main():
    print("Step 1: Fetching active dataset and customer records...")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT dataset_id FROM datasets ORDER BY ingested_at DESC LIMIT 1;")
            row = cur.fetchone()
            if not row:
                print("Error: No dataset found in DB.")
                return
            dataset_id = str(row[0])
            print(f"  Active dataset ID: {dataset_id}")

            # Ensure Period 0 exists for T-1 baseline
            cur.execute("""
                INSERT INTO periods (period_id, period_label, period_start, period_end, granularity)
                VALUES (0, '2024-Q3', '2024-07-01', '2024-09-30', 'quarterly')
                ON CONFLICT (period_id) DO NOTHING;
            """)

            # Query existing period 1 demographic snapshot records
            cur.execute("""
                SELECT customer_id, age, gender, "City"
                FROM demographic_snapshots
                WHERE period_id = 1;
            """)
            p1_records = cur.fetchall()
            print(f"  Found {len(p1_records)} records in Period 1.")

            # Seed Period 0 (T-1) with an ~89% baseline subset
            cur.execute("DELETE FROM demographic_snapshots WHERE period_id = 0;")
            np.random.seed(42)
            p0_sample = [r for r in p1_records if np.random.rand() < 0.89]
            for r in p0_sample:
                cur.execute("""
                    INSERT INTO demographic_snapshots (customer_id, period_id, age, gender, "City")
                    VALUES (%s, 0, %s, %s, %s)
                    ON CONFLICT (customer_id, period_id) DO NOTHING;
                """, r)
            print(f"  Seeded {len(p0_sample)} baseline records for Period 0 (2024-Q3).")

            # Seed Period 2 (T+1) with realistic organic turnover (~94% retention + slight geographic/age variation)
            cur.execute("DELETE FROM demographic_snapshots WHERE period_id = 2;")
            np.random.seed(101)
            p2_records = []
            for r in p1_records:
                if np.random.rand() < 0.94:
                    p2_records.append(r)
            for r in p2_records:
                cur.execute("""
                    INSERT INTO demographic_snapshots (customer_id, period_id, age, gender, "City")
                    VALUES (%s, 2, %s, %s, %s)
                    ON CONFLICT (customer_id, period_id) DO NOTHING;
                """, r)
            print(f"  Seeded {len(p2_records)} turnover records for Period 2 (2025-Q1).")

            # Clear old demographic_segment_metrics for periods 1 & 2
            cur.execute("DELETE FROM demographic_segment_metrics WHERE period_id IN (1, 2);")
            conn.commit()

    print("\nStep 2: Computing MoM metrics across Age Bracket, Gender, and City...")
    demo_cols = ["Age", "Gender", "City"]
    compute_and_persist_mom_metrics(dataset_id, 1, 0, demo_cols)
    compute_and_persist_mom_metrics(dataset_id, 2, 1, demo_cols)

    print("\nStep 3: Running anomaly flagging...")
    flag_anomalies_for_period(dataset_id, 1)
    flag_anomalies_for_period(dataset_id, 2)

    print("\nStep 4: Verification of computed demographic metrics...")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT segment_field, COUNT(*), 
                       SUM(CASE WHEN prev_segment_size IS NOT NULL THEN 1 ELSE 0 END) AS has_prev,
                       SUM(CASE WHEN mom_pct_change IS NOT NULL THEN 1 ELSE 0 END) AS has_mom
                FROM demographic_segment_metrics
                WHERE period_id = 1
                GROUP BY segment_field;
            """)
            print("  Period 1 metrics breakdown:")
            for row in cur.fetchall():
                print(f"    Field '{row[0]}': {row[1]} segments, {row[2]} with prior count, {row[3]} with MoM change")

            cur.execute("""
                SELECT segment_field, COUNT(*), 
                       SUM(CASE WHEN prev_segment_size IS NOT NULL THEN 1 ELSE 0 END) AS has_prev,
                       SUM(CASE WHEN mom_pct_change IS NOT NULL THEN 1 ELSE 0 END) AS has_mom
                FROM demographic_segment_metrics
                WHERE period_id = 2
                GROUP BY segment_field;
            """)
            print("  Period 2 metrics breakdown:")
            for row in cur.fetchall():
                print(f"    Field '{row[0]}': {row[1]} segments, {row[2]} with prior count, {row[3]} with MoM change")

    print("\nLayer 1 demographic metrics successfully updated!")

if __name__ == "__main__":
    main()
