"""Data quality validation using Soda Core."""
from typing import Dict, Any
from deltalake import DeltaTable
from deltalake.exceptions import TableNotFoundError

from src.processors.base_processor import BaseProcessor
from src.utils.database import DuckDBClient
from src.utils.metrics import track_quality_failure


class QualityChecker(BaseProcessor):
    """
    Quality checker processor - validates Silver data using Soda Core.
    Implements the same interface as other processors.
    """
    
    def _validate_config(self) -> None:
        """Validate quality checker configuration."""
        from src.config import get_config
        
        self.silver_bucket = self._get_bucket('silver')
        self.storage_options = self.storage.get_storage_options()
        self.checks_enabled = get_config('data_quality', 'enabled', default=True)
        self.checks_path = get_config('data_quality', 'checks_path')
        self.fail_on_error = get_config(
            'data_quality', 
            'fail_on_error', 
            default=False
        )
        
        if not self.checks_enabled:
            self.logger.info("Quality checks are DISABLED")
        else:
            self.logger.info(
                f"Quality checks enabled: checks_path={self.checks_path}, "
                f"fail_on_error={self.fail_on_error}"
            )
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Run Soda Core quality checks on Silver data.
        
        Returns:
            {
                "status": "success" | "failed" | "skipped",
                "checks_failed": int,
                "failed_checks": List[str]
            }
        """
        if not self.checks_enabled:
            self.logger.info("Quality checks disabled, skipping")
            return {
                "status": "skipped",
                "reason": "disabled_in_config"
            }
        
        try:
            # Load Delta table
            silver_path = f"s3://{self.silver_bucket}/sales"
            
            try:
                dt = DeltaTable(silver_path, storage_options=self.storage_options)
                df = dt.to_pandas()
                
                self.logger.info(
                    f"✓ Loaded Silver table for validation "
                    f"(version={dt.version()}, rows={len(df)})"
                )
                
            except TableNotFoundError:
                self.logger.warning("Silver table does not exist yet")
                return {
                    "status": "skipped",
                    "reason": "table_not_found"
                }
            except Exception as e:
                self.logger.error(f"Could not load Delta table: {e}")
                raise
            
            if len(df) == 0:
                self.logger.info("Table empty, skipping checks")
                return {
                    "status": "skipped",
                    "reason": "empty_table"
                }
            
            self.logger.info(f"Running quality checks on {len(df)} records")
            
            # Run Soda checks
            result = self._run_soda_checks(df)
            
            return result
            
        except Exception as e:
            self._handle_error(e, "quality_check_execution")
    
    def _run_soda_checks(self, df) -> Dict[str, Any]:
        """
        Execute Soda Core checks on the DataFrame.
        Returns a result dict for process().
        """
        try:
            with DuckDBClient() as conn:
                conn.register('sales_silver', df)
                from soda.scan import Scan
                scan = Scan()
                scan.add_duckdb_connection(conn)
                scan.set_data_source_name("duckdb")
                scan.add_sodacl_yaml_files(self.checks_path)
                scan.set_scan_definition_name("silver_quality_check")
                scan.execute()

                total_checks = scan.get_logs_text().count("checks PASSED") + scan.get_logs_text().count("checks FAILED")
                if total_checks == 0:
                    # Fallback: count from scan._checks if available
                    total_checks = len(getattr(scan, '_checks', []))

                if scan.has_check_fails():
                    failed_checks = scan.get_checks_fail()
                    for check in failed_checks:
                        check_name = getattr(check, 'name', 'unknown_check')
                        track_quality_failure(check_name=check_name)
                        self.logger.warning(f"Quality check failed: {check_name}")
                    
                    result = {
                        "status": "failed",
                        "checks_run": total_checks if total_checks > 0 else len(failed_checks),
                        "checks_failed": len(failed_checks),
                        "failed_checks": [getattr(check, 'name', 'unknown_check') for check in failed_checks]
                    }
                    
                    if self.fail_on_error:
                        raise ValueError(
                            f"Quality validation failed: {len(failed_checks)} check(s) failed. "
                            f"Failed checks: {result['failed_checks']}"
                        )
                    else:
                        self.logger.warning(
                            f"Quality checks failed but fail_on_error=False. Continuing pipeline. "
                            f"Failed checks: {result['failed_checks']}"
                        )
                    
                    return result
                else:
                    self.logger.info("✓ All data quality checks passed")
                    return {
                        "status": "success",
                        "checks_run": total_checks if total_checks > 0 else 0,
                        "checks_failed": 0,
                        "failed_checks": []
                    }
        except Exception as e:
            self.logger.error(f"Soda check execution failed: {e}")
            if self.fail_on_error:
                raise
            else:
                self.logger.warning(f"Soda execution error but fail_on_error=False: {e}")
                return {
                    "status": "error",
                    "checks_run": 0,
                    "checks_failed": 0,
                    "failed_checks": [],
                    "error": str(e)
                }

