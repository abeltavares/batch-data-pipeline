"""Gold layer: Business aggregations as Delta tables with parallel processing."""
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import TableNotFoundError

from src.processors.base_processor import BaseProcessor
from src.utils.database import DuckDBClient
from src.utils.metrics import track_table_size


class GoldProcessor(BaseProcessor):
    """
    Gold layer processor - creates business aggregations.
    Uses ThreadPoolExecutor for parallel table processing.
    """
    
    def _validate_config(self) -> None:
        """Validate Gold-specific configuration."""
        self.bronze_bucket = self._get_bucket('bronze')
        self.silver_bucket = self._get_bucket('silver')
        self.gold_bucket = self._get_bucket('gold')
        self.storage_options = self.storage.get_storage_options()
        
        self.logger.info(
            f"Gold config: bronze={self.bronze_bucket}, "
            f"silver={self.silver_bucket}, gold={self.gold_bucket}"
        )
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Create daily business metrics as Delta tables.
        Processes all aggregations in parallel using ThreadPoolExecutor.
        
        Returns:
            {
                "daily_sales_summary": {"status": "success", "rows": 1},
                "product_performance": {"status": "success", "rows": 50},
                "customer_segments": {"status": "success", "rows": 150},
                "overall_status": "success"
            }
        """
        try:
            date = datetime.utcnow().strftime("%Y-%m-%d")
            
            # Load Silver data once (shared across all aggregations)
            sales_df = self._load_silver_data()
            
            if sales_df is None or len(sales_df) == 0:
                self.logger.warning("No Silver data available, skipping aggregation")
                return {
                    "status": "skipped",
                    "reason": "no_silver_data"
                }
            
            # Define aggregations to run in parallel
            aggregations: List[tuple[str, Callable]] = [
                ('daily_sales_summary', lambda: self._aggregate_daily_sales(sales_df, date)),
                ('product_performance', lambda: self._aggregate_product_performance(sales_df, date)),
                ('customer_segments', lambda: self._aggregate_customer_segments(sales_df, date)),
            ]
            
            # Execute aggregations in parallel
            results = self._execute_parallel_aggregations(aggregations)
            
            # Determine overall status
            overall_status = "success" if all(
                r.get("status") == "success" for r in results.values()
            ) else "partial_failure"
            
            results["overall_status"] = overall_status
            
            success_count = len([
                r for k, r in results.items()
                if k != "overall_status" and isinstance(r, dict) and r.get("status") == "success"
            ])
            self.logger.info(
                f"✓ Gold aggregation complete: {overall_status} "
                f"({success_count} of {len(aggregations)} succeeded)"
            )
        
            return results
            
        except Exception as e:
            self._handle_error(e, "gold_aggregation")
    
    def _load_silver_data(self, date: str = None) -> Optional[pd.DataFrame]:
        """Load only today's partition for incremental aggregation."""
        silver_path = f"s3://{self.silver_bucket}/sales"
        
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        
        try:
            dt = DeltaTable(silver_path, storage_options=self.storage_options)
            
            sales_df = dt.to_pandas(
                filters=[("partition_date", "=", date)]
            )
            
            self.logger.info(
                f"Loaded {len(sales_df)} records for partition_date={date}"
            )
            return sales_df
            
        except TableNotFoundError:
            self.logger.warning("Silver table does not exist yet")
            return None
            
        except Exception as e:
            self.logger.error(
                f"Failed to load Silver table: {type(e).__name__}: {e}"
            )
            raise
    
    def _execute_parallel_aggregations(
        self, 
        aggregations: List[tuple[str, Callable]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute multiple aggregations in parallel using ThreadPoolExecutor.
        
        Args:
            aggregations: List of (name, function) tuples
        
        Returns:
            Dict mapping aggregation name to result
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all aggregations
            future_to_agg = {
                executor.submit(agg_func): name 
                for name, agg_func in aggregations
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agg):
                agg_name = future_to_agg[future]
                
                try:
                    result = future.result()
                    results[agg_name] = result
                    self.logger.info(f"✓ Completed {agg_name}")
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed {agg_name}: {type(e).__name__}: {e}",
                        exc_info=True
                    )
                    results[agg_name] = {
                        "status": "failed",
                        "error": str(e)
                    }
        
        return results
    
    def _aggregate_daily_sales(
        self, 
        sales_df: pd.DataFrame, 
        date: str
    ) -> Dict[str, Any]:
        """Calculate and write daily sales summary."""
        try:
            with DuckDBClient() as conn:
                conn.register('sales', sales_df)
                
                df = conn.execute(f"""
                    SELECT
                        partition_date,
                        COUNT(DISTINCT transaction_id) as total_transactions,
                        COUNT(DISTINCT customer_id) as unique_customers,
                        SUM(total_amount) as total_revenue,
                        AVG(total_amount) as avg_transaction_value,
                        SUM(quantity) as total_items_sold,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_transactions,
                        COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_transactions,
                        ROUND(100.0 * COUNT(CASE WHEN status = 'completed' THEN 1 END) /
                              NULLIF(COUNT(*), 0), 2) as completion_rate
                    FROM sales
                    WHERE partition_date = '{date}'
                    GROUP BY partition_date
                """).df()
            
            if len(df) == 0:
                self.logger.warning("No data for daily_sales_summary")
                return {"status": "skipped", "rows": 0}
            
            df['created_at'] = datetime.utcnow()
            df['data_date'] = date
            
            self._upsert_to_delta(
                table_name='daily_sales_summary',
                df=df,
                merge_keys=['data_date'],
                partition_by=['partition_date']
            )
            
            return {"status": "success", "rows": len(df)}
            
        except Exception as e:
            self.logger.error(f"Failed daily_sales_summary: {e}")
            raise
    
    def _aggregate_product_performance(
        self, 
        sales_df: pd.DataFrame, 
        date: str
    ) -> Dict[str, Any]:
        """Calculate and write product performance."""
        try:
            products_path = f"s3://{self.bronze_bucket}/dimensions/products.json"
            
            with DuckDBClient() as conn:
                conn.register('sales', sales_df)
                
                df = conn.execute(f"""
                    SELECT
                        p.product_id,
                        p.product_name,
                        p.category,
                        COUNT(s.transaction_id) as transaction_count,
                        SUM(s.quantity) as units_sold,
                        SUM(s.total_amount) as revenue,
                        AVG(s.unit_price) as avg_selling_price,
                        MIN(s.unit_price) as min_price,
                        MAX(s.unit_price) as max_price,
                        '{date}' as data_date
                    FROM sales s
                    JOIN read_json_auto('{products_path}') p
                        ON s.product_id = p.product_id
                    WHERE s.partition_date = '{date}'
                        AND s.status = 'completed'
                    GROUP BY p.product_id, p.product_name, p.category
                    ORDER BY revenue DESC
                """).df()
            
            if len(df) == 0:
                self.logger.warning("No data for product_performance")
                return {"status": "skipped", "rows": 0}
            
            df['created_at'] = datetime.utcnow()
            
            self._upsert_to_delta(
                table_name='product_performance',
                df=df,
                merge_keys=['product_id', 'data_date'],
                partition_by=['data_date']
            )
            
            return {"status": "success", "rows": len(df)}
            
        except Exception as e:
            self.logger.error(f"Failed product_performance: {e}")
            raise
    
    def _aggregate_customer_segments(
        self, 
        sales_df: pd.DataFrame, 
        date: str
    ) -> Dict[str, Any]:
        """Calculate and write customer segmentation."""
        try:
            customers_path = f"s3://{self.bronze_bucket}/dimensions/customers.json"
            
            with DuckDBClient() as conn:
                conn.register('sales', sales_df)
                
                df = conn.execute(f"""
                    SELECT
                        c.customer_id,
                        c.customer_name,
                        c.customer_segment,
                        c.city,
                        c.country,
                        COUNT(s.transaction_id) as purchase_count,
                        SUM(s.total_amount) as total_spend,
                        AVG(s.total_amount) as avg_order_value,
                        MAX(s.transaction_timestamp) as last_purchase_date,
                        MIN(s.transaction_timestamp) as first_purchase_date,
                        '{date}' as data_date,
                        CASE
                            WHEN SUM(s.total_amount) > 1000 THEN 'High Value'
                            WHEN SUM(s.total_amount) > 500 THEN 'Medium Value'
                            ELSE 'Low Value'
                        END as value_tier
                    FROM sales s
                    JOIN read_json_auto('{customers_path}') c
                        ON s.customer_id = c.customer_id
                    WHERE s.partition_date = '{date}'
                        AND s.status = 'completed'
                    GROUP BY c.customer_id, c.customer_name, c.customer_segment, c.city, c.country
                    ORDER BY total_spend DESC
                """).df()
            
            if len(df) == 0:
                self.logger.warning("No data for customer_segments")
                return {"status": "skipped", "rows": 0}
            
            df['created_at'] = datetime.utcnow()
            
            self._upsert_to_delta(
                table_name='customer_segments',
                df=df,
                merge_keys=['customer_id', 'data_date'],
                partition_by=['data_date']
            )
            
            return {"status": "success", "rows": len(df)}
            
        except Exception as e:
            self.logger.error(f"Failed customer_segments: {e}")
            raise
    
    def _upsert_to_delta(
        self,
        table_name: str,
        df: pd.DataFrame,
        merge_keys: List[str],
        partition_by: Optional[List[str]] = None
    ) -> None:
        """
        Upsert DataFrame to Delta table using MERGE operation.
        Idempotent - safe to re-run.
        
        Args:
            table_name: Name of the Delta table
            df: DataFrame to upsert
            merge_keys: List of columns for merge condition
            partition_by: Columns to partition by
        """
        table_path = f"s3://{self.gold_bucket}/{table_name}"
        df = df.reset_index(drop=True)
        
        try:
            dt = DeltaTable(table_path, storage_options=self.storage_options)
            table_exists = True
        except (TableNotFoundError, FileNotFoundError):
            table_exists = False
        except Exception as e:
            self.logger.error(
                f"Error checking table existence: {type(e).__name__}: {e}"
            )
            raise
        
        if table_exists:
            # Build composite merge condition
            merge_conditions = [f"target.{key} = source.{key}" for key in merge_keys]
            merge_condition = " AND ".join(merge_conditions)
            
            self.logger.info(f"Merge condition: {merge_condition}")
            
            try:
                (
                    dt.merge(
                        source=df,
                        predicate=merge_condition,
                        source_alias="source",
                        target_alias="target"
                    )
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute()
                )
                
                self.logger.info(
                    f"✓ Merged {len(df)} rows into {table_name} (idempotent)"
                )
                
            except Exception as e:
                self.logger.error(
                    f"MERGE operation failed: {type(e).__name__}: {e}"
                )
                raise
        else:
            try:
                write_deltalake(
                    table_path,
                    df,
                    mode='overwrite',
                    partition_by=partition_by,
                    storage_options=self.storage_options,
                )
                self.logger.info(
                    f"✓ Created {table_name} Delta table ({len(df)} rows)"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to create table: {type(e).__name__}: {e}"
                )
                raise
        
        # Track total table size
        try:
            dt = DeltaTable(table_path, storage_options=self.storage_options)
            total_rows = len(dt.to_pandas())
            track_table_size(table_name=table_name, row_count=total_rows)
        except Exception as e:
            self.logger.warning(
                f"Could not track table size for {table_name}: {e}"
            )
