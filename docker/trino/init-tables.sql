CREATE SCHEMA IF NOT EXISTS delta.analytics
WITH (location = 's3://gold/');

USE delta.analytics;

CALL delta.system.register_table(
  schema_name => 'analytics',
  table_name  => 'daily_sales_summary',
  table_location => 's3://gold/daily_sales_summary'
);

CALL delta.system.register_table(
  schema_name => 'analytics',
  table_name  => 'product_performance',
  table_location => 's3://gold/product_performance'
);

CALL delta.system.register_table(
  schema_name => 'analytics',
  table_name  => 'customer_segments',
  table_location => 's3://gold/customer_segments'
);

SHOW TABLES;
