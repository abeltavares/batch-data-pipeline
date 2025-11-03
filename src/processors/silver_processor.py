"""Silver layer: Cleaned and deduplicated data in Delta Lake."""
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import TableNotFoundError

from src.processors.base_processor import BaseProcessor
from src.utils.database import DuckDBClient
from src.config import get_config


class SilverProcessor(BaseProcessor):
    """
    Silver layer processor - cleans, validates, and deduplicates data.
    Writes to Delta Lake for ACID transactions.
    """
    
    def _validate_config(self) -> None:
        """Validate Silver-specific configuration."""
        self.bronze_bucket = self._get_bucket('bronze')
        self.silver_bucket = self._get_bucket('silver')
        self.storage_options = self.storage.get_storage_options()
        self.num_products = get_config('data_generation', 'num_products', default=50)
        
        self.logger.info(
            f"Silver config: bronze={self.bronze_bucket}, "
            f"silver={self.silver_bucket}"
        )
    
    def process(self, bronze_key: str, **kwargs) -> Dict[str, Any]:
        """
        Process Bronze data to Silver Delta table.
        
        Args:
            bronze_key: S3 key of the Bronze data file
        
        Returns:
            {
                "records_read": int,
                "records_cleaned": int,
                "records_written": int,
                "duplicates_removed": int,
                "status": "success"
            }
        """
        try:
            date_partition = datetime.utcnow().strftime("%Y-%m-%d")
            
            # Read from Bronze using DuckDB
            bronze_path = f"s3://{self.bronze_bucket}/{bronze_key}"
            
            with DuckDBClient() as conn:
                df = conn.execute(f"""
                    SELECT
                        CAST(transaction_id AS VARCHAR) AS transaction_id,
                        CAST(customer_id AS BIGINT) AS customer_id,
                        product_id,
                        quantity,
                        unit_price,
                        total_amount,
                        CAST(transaction_timestamp AS TIMESTAMP) as transaction_timestamp,
                        payment_method,
                        store_id,
                        status,
                        '{date_partition}' as partition_date,
                        CURRENT_TIMESTAMP as processed_at
                    FROM read_json_auto('{bronze_path}')
                """).df()
            
            records_read = len(df)
            self.logger.info(f"✓ Read {records_read} records from Bronze")
            
            # Clean the data
            df_cleaned = self._clean_data(df)
            records_cleaned = len(df_cleaned)
            
            self.logger.info(
                f"✓ Cleaned data: {records_read} → {records_cleaned} records "
                f"({records_read - records_cleaned} removed)"
            )
            
            # Deduplicate against existing Delta table
            silver_path = f"s3://{self.silver_bucket}/sales"
            df_to_write, duplicates_removed = self._deduplicate(
                df_cleaned, 
                silver_path
            )
            
            # Write to Delta Lake
            records_written = 0
            if df_to_write is not None and len(df_to_write) > 0:
                self._write_to_delta(df_to_write, silver_path)
                records_written = len(df_to_write)
                
                # Track metrics
                self._track_metrics(
                    layer='silver',
                    table='sales',
                    count=records_written
                )
            else:
                self.logger.info("⊘ No new records to write")
            
            return {
                "records_read": records_read,
                "records_cleaned": records_cleaned,
                "records_written": records_written,
                "duplicates_removed": duplicates_removed,
                "partition_date": date_partition,
                "status": "success"
            }
            
        except Exception as e:
            self._handle_error(e, "silver_processing")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data according to business rules.
        
        Removes:
        - Null required fields
        - Future timestamps
        - Zero/negative quantities
        - Negative amounts
        - Mismatched totals
        - Invalid payment methods
        - Invalid statuses
        - Invalid product IDs
        """
        df = df.copy()
        initial_count = len(df)

        # Use nullable Int64 dtype for consistent schema
        df['customer_id'] = df['customer_id'].astype('Int64')
        df['product_id'] = df['product_id'].astype('Int64')
        df['quantity'] = df['quantity'].astype('Int64')
        df['store_id'] = df['store_id'].astype('Int64')

        # Remove nulls in required fields
        df = df.dropna(subset=[
            'transaction_id', 
            'customer_id', 
            'product_id',
            'transaction_timestamp'
        ])
        self.logger.debug(
            f"Removed {initial_count - len(df)} records with null required fields"
        )
        
        # Remove future timestamps
        now = pd.Timestamp.utcnow().tz_localize(None)
        before_clean = len(df)
        df['transaction_timestamp'] = pd.to_datetime(
            df['transaction_timestamp'], 
            utc=False
        )
        df = df[df['transaction_timestamp'] <= now]
        self.logger.debug(
            f"Removed {before_clean - len(df)} records with future timestamps"
        )
        
        # Remove zero or negative quantities
        before_clean = len(df)
        df = df[df['quantity'] > 0]
        self.logger.debug(
            f"Removed {before_clean - len(df)} records with non-positive quantities"
        )
        
        # Remove negative amounts
        before_clean = len(df)
        df = df[df['total_amount'] > 0]
        self.logger.debug(
            f"Removed {before_clean - len(df)} records with negative amounts"
        )
        
        # Remove mismatched totals (0.01 tolerance for rounding)
        before_clean = len(df)
        df['amount_mismatch'] = abs(
            df['total_amount'] - (df['quantity'] * df['unit_price'])
        )
        df = df[df['amount_mismatch'] <= 0.01]
        df = df.drop('amount_mismatch', axis=1)
        self.logger.debug(
            f"Removed {before_clean - len(df)} records with mismatched totals"
        )
        
        # Keep only valid payment methods
        valid_methods = ['credit_card', 'debit_card', 'cash', 'digital_wallet']
        before_clean = len(df)
        df = df[df['payment_method'].isin(valid_methods)]
        self.logger.debug(
            f"Removed {before_clean - len(df)} records with invalid payment methods"
        )
        
        # Keep only valid statuses
        valid_statuses = ['completed', 'pending', 'cancelled']
        before_clean = len(df)
        df = df[df['status'].isin(valid_statuses)]
        self.logger.debug(
            f"Removed {before_clean - len(df)} records with invalid statuses"
        )
        
        # Filter to valid product ID range
        before_clean = len(df)
        df = df[(df['product_id'] > 0) & (df['product_id'] <= self.num_products)]
        self.logger.debug(
            f"Removed {before_clean - len(df)} records with invalid product IDs"
        )
        
        self.logger.info(
            f"Data cleaning complete: {initial_count} → {len(df)} records "
            f"({initial_count - len(df)} removed)"
        )
        
        return df
    
    def _deduplicate(
        self, 
        df: pd.DataFrame, 
        silver_path: str
    ) -> tuple[Optional[pd.DataFrame], int]:
        """
        Deduplicate records against existing Delta table using partition pruning.

        Returns:
            (df_new, duplicates_removed)
        """
        try:
            dt = DeltaTable(silver_path, storage_options=self.storage_options)

            today = datetime.utcnow().strftime("%Y-%m-%d")

            existing_df = dt.to_pandas(
                filters=[("partition_date", "=", today)]
            )

            existing_ids = set(existing_df['transaction_id'].astype(str).values)

            self.logger.info(
                f"Existing table (today's partition): {len(existing_df)} rows, "
                f"{len(existing_ids)} unique transaction_ids"
            )

            df_new = df[~df['transaction_id'].astype(str).isin(existing_ids)]

            duplicates_removed = len(df) - len(df_new)

            if duplicates_removed > 0:
                self.logger.warning(
                    f"Removed {duplicates_removed} duplicate records "
                    f"(transaction_ids already exist - likely pipeline re-run)"
                )

            self.logger.info(
                f"After deduplication: {len(df_new)} new records to write"
            )

            return df_new, duplicates_removed

        except TableNotFoundError:
            self.logger.info("Delta table does not exist yet (first run)")
            return df, 0

        except Exception as e:
            self.logger.error(
                f"Unexpected error during deduplication: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise
    
    def _write_to_delta(self, df: pd.DataFrame, silver_path: str) -> None:
        """
        Write DataFrame to Delta Lake using APPEND mode.
        Idempotent - safe to re-run.
        """
        try:
            df = df.reset_index(drop=True)
            
            # Check if table exists
            table_exists = False
            try:
                DeltaTable(silver_path, storage_options=self.storage_options)
                table_exists = True
            except TableNotFoundError:
                table_exists = False
            except Exception as e:
                self.logger.error(
                    f"Error checking table existence: {type(e).__name__}: {e}"
                )
                raise
            
            mode = "append" if table_exists else "overwrite"
            
            write_deltalake(
                silver_path,
                df,
                mode=mode,
                partition_by=["partition_date"],
                storage_options=self.storage_options,
            )
            
            action = "Appended" if table_exists else "Created"
            self.logger.info(
                f"✓ {action} {len(df)} records to Silver Delta table"
            )
            
        except FileNotFoundError as e:
            self.logger.error(f"S3 bucket not found: {e}")
            raise
            
        except PermissionError as e:
            self.logger.error(f"S3 permission denied: {e}")
            raise
            
        except Exception as e:
            self.logger.error(
                f"Failed to write to Delta: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise
