-- Create table (adjust catalog and warehouse first)
CREATE TABLE demo.ml_images (
  id BIGINT,
  ts TIMESTAMP,
  label INT,
  path STRING,
  bytes BINARY
) USING iceberg
PARTITIONED BY (days(ts))
TBLPROPERTIES (
  'write.target-file-size-bytes'='536870912',
  'format-version'='2'
);

-- Add sort order
ALTER TABLE demo.ml_images WRITE ORDERED BY (ts, label);

-- Compact data files to target size
CALL demo.system.rewrite_data_files(
  table => 'demo.ml_images',
  options => map('target-file-size-bytes','536870912')
);

-- Rewrite delete files (binpack)
CALL demo.system.rewrite_position_deletes(table => 'demo.ml_images');

-- List snapshots
SELECT snapshot_id, parent_id, committed_at, operation FROM demo.ml_images.snapshots;
