-- ==========================================================
-- Customer Segmentation Studio - Augmented Seeder Script
-- Note: T+1 layer is synthetically generated for testing/dev
-- ==========================================================

INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES
  (1, '2024-Q4', '2024-10-01', '2024-12-31', 'quarterly'),
  (2, '2025-Q1', '2025-01-01', '2025-03-31', 'quarterly')
ON CONFLICT (period_id) DO NOTHING;

INSERT INTO cluster_baselines (baseline_id, adopted_at, is_active, adopted_reason, period_id) VALUES
  (1, '2025-01-01 00:00:00+00', TRUE, 'initial', 1)
ON CONFLICT (baseline_id) DO NOTHING;

INSERT INTO clusters (cluster_id, baseline_id, cluster_label, medoid_customer_id, intra_cluster_spread) VALUES
  (1, 1, 'Cluster_0', 'c310d2a2-0700-77c1-b1f9-4a097408f646', 1.3533709466708435),
  (2, 1, 'Cluster_1', 'ea4f4430-8153-9408-325b-1116acb812ac', 1.3011206847342935),
  (3, 1, 'Cluster_2', 'aa9313da-7637-dd64-6182-02ce9281fb0d', 1.3680541765865843),
  (4, 1, 'Cluster_3', 'c9342209-06b3-17e8-81e4-fcc6a5151678', 1.7297286122118378),
  (5, 1, 'Cluster_4', '7f70881b-60c2-3031-9342-f3ebf241ba41', 1.3474390846559121)
ON CONFLICT (cluster_id) DO NOTHING;

