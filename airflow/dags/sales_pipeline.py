"""
Sales Delta Pipeline DAG - Interface Inheritance Pattern

Design Principles:
- All processors inherit from BaseProcessor
- Consistent error handling and metrics
- Idempotent operations (safe to re-run)
- Parallel Gold aggregations using ThreadPoolExecutor
- Proper task dependencies and branching
"""
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.lineage.entities import File
from airflow.models import DagRun

# Add src to path
sys.path.insert(0, '/opt/airflow')

from src.config import get_config
from src.processors.bronze_processor import BronzeProcessor
from src.processors.silver_processor import SilverProcessor
from src.processors.gold_processor import GoldProcessor
from src.processors.quality_checker import QualityChecker
from src.utils.logger import get_logger

logger = get_logger(__name__)

# DAG configuration
default_args = {
    'owner': get_config('pipeline', 'name', default='airflow'),
    'retries': get_config('pipeline', 'retries', default=1),
    'retry_delay': timedelta(
        minutes=get_config('pipeline', 'retry_delay_minutes', default=1)
    ),
    'execution_timeout': timedelta(minutes=30),
    'email_on_failure': False,
    'email_on_retry': False,
}


@dag(
    dag_id=get_config('pipeline', 'name', default='sales_delta_pipeline'),
    schedule_interval=get_config('pipeline', 'schedule', default='@hourly'),
    start_date=datetime.strptime(
        get_config('pipeline', 'start_date', default='2024-01-01'),
        '%Y-%m-%d'
    ),
    catchup=get_config('pipeline', 'catchup', default=False),
    default_args=default_args,
    tags=["sales", "delta", "incremental", "interface-inheritance"],
    description="Incremental sales data pipeline",
    max_active_runs=1,  # Prevent concurrent runs
    doc_md="""
    # Sales Delta Pipeline
    
    **Architecture**: Bronze → Silver → Gold (Medallion)
    
    **Design Pattern**: Interface Inheritance
    - All processors inherit from `BaseProcessor`
    - Shared authentication, error handling, and metrics
    - Consistent interface across all layers
    
    **Flow**:
    1. **Bronze**: Ingest raw sales data (with dirty data)
    2. **Bronze**: Load dimension tables (idempotent)
    3. **Silver**: Clean, validate, and deduplicate to Delta Lake
    4. **Quality**: Run Soda Core data quality checks
    5. **Gold**: Aggregate business metrics (parallel processing)
    
    **Idempotency**: Safe to re-run at any step
    """
)
def sales_delta_pipeline():
    """
    Sales data pipeline with Bronze-Silver-Gold architecture.
    Uses Interface Inheritance pattern for clean, maintainable code.
    """
    
    @task(
        task_id='ingest_to_bronze',
        inlets=[],
        outlets=[File(url="s3://bronze/sales_raw/")]
    )
    def ingest_sales_data() -> Dict[str, Any]:
        """
        Ingest incremental sales to Bronze layer.
        
        Returns:
            {"key": s3_key, "count": record_count, "status": "success"}
        """
        try:
            processor = BronzeProcessor()
            result = processor.process()
            
            # Skip downstream if no data
            if result.get('count', 0) == 0:
                logger.warning("No sales data generated, skipping pipeline")
                raise AirflowSkipException("No data to process")
            
            logger.info(
                f"✓ Bronze ingestion complete: {result['count']} records → {result['key']}"
            )
            
            return result
            
        except AirflowSkipException:
            raise
        except Exception as e:
            logger.error(f"Bronze ingestion failed: {e}")
            raise
    
    @task
    def load_dimension_tables() -> Dict[str, Any]:
        """
        Load product and customer dimensions (idempotent).
        
        Returns:
            {"status": "loaded" | "skipped", "products": int, "customers": int}
        """
        try:
            processor = BronzeProcessor()
            result = processor.load_dimensions()
            
            if result['status'] == 'skipped':
                logger.info("✓ Dimensions already loaded")
            else:
                logger.info(
                    f"✓ Loaded {result['products']} products and "
                    f"{result['customers']} customers"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Dimension loading failed: {e}")
            raise
    
    @task(
        task_id='transform_to_silver',
        inlets=[File(url="s3://bronze/sales_raw/")],
        outlets=[File(url="s3://silver/sales_cleaned/")]
    )
    def process_to_silver(bronze_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Bronze → Silver with cleaning and deduplication.
        
        Args:
            bronze_result: Output from ingest_sales_data task
        
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
            bronze_key = bronze_result['key']
            
            processor = SilverProcessor()
            result = processor.process(bronze_key=bronze_key)
            
            logger.info(
                f"✓ Silver processing complete: "
                f"{result['records_read']} read → "
                f"{result['records_cleaned']} cleaned → "
                f"{result['records_written']} written "
                f"({result['duplicates_removed']} duplicates removed)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Silver processing failed: {e}")
            raise
    
    @task(
        task_id='validate_quality',
        inlets=File(url="s3://silver/sales_cleaned/")
    )
    def validate_data_quality() -> Dict[str, Any]:
        """
        Run Soda Core quality checks on Silver data.
        
        Returns:
            {
                "status": "success" | "failed" | "skipped",
                "checks_run": int,
                "checks_failed": int
            }
        """
        try:
            checker = QualityChecker()
            result = checker.process()
            
            if result['status'] == 'skipped':
                logger.info(f"✓ Quality checks skipped: {result.get('reason')}")
            elif result['status'] == 'success':
                logger.info("Quality checks passed")
            else:
                logger.warning(
                    f"⚠️ Quality checks failed: "
                    f"{result['checks_failed']}/{result['checks_run']} failed"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            raise
    
    def should_aggregate(**context) -> str:
        """
        Determine if daily aggregation should run.
        Runs on first execution OR at configured hour (default: midnight UTC).
        
        Returns:
            "aggregate_to_gold" or "skip_aggregation"
        """
        
        # Check if this is the first successful run
        dag_id = context['dag'].dag_id
        dag_runs = DagRun.find(dag_id=dag_id, state='success')
        is_first_run = len(dag_runs) == 0
        
        current_hour = datetime.utcnow().hour
        trigger_hour = get_config('aggregation', 'trigger_hour', default=0)
        
        if is_first_run:
            logger.info("✓ First DAG run - triggering Gold aggregation")
            return "aggregate_to_gold"
        elif current_hour == trigger_hour:
            logger.info(
                f"✓ Triggering Gold aggregation (current_hour={current_hour}, "
                f"trigger_hour={trigger_hour})"
            )
            return "aggregate_to_gold"
        else:
            logger.info(
                f"⊘ Skipping Gold aggregation (current_hour={current_hour}, "
                f"trigger_hour={trigger_hour})"
            )
            return "skip_aggregation"
    
    @task(
        task_id='aggregate_to_gold',
        inlets=[File(url="s3://silver/sales_cleaned/")],
        outlets=[
            File(url="s3://gold/daily_sales_summary/"),
            File(url="s3://gold/customer_segments/"),
            File(url="s3://gold/product_performance/")
            ]
        )
    def aggregate_to_gold() -> Dict[str, Any]:
        """
        Create Gold layer business metrics.
        Uses ThreadPoolExecutor for parallel table processing.
        
        Returns:
            {
                "daily_sales_summary": {"status": "success", "rows": 1},
                "product_performance": {"status": "success", "rows": 50},
                "customer_segments": {"status": "success", "rows": 150},
                "overall_status": "success"
            }
        """
        try:
            processor = GoldProcessor()
            result = processor.process()
            
            # Log results for each aggregation
            for table_name, table_result in result.items():
                if table_name == "overall_status":
                    continue
                
                if table_result.get("status") == "success":
                    logger.info(
                        f"✓ {table_name}: {table_result.get('rows', 0)} rows"
                    )
                elif table_result.get("status") == "skipped":
                    logger.info(f"⊘ {table_name}: skipped")
                else:
                    logger.error(
                        f"{table_name}: {table_result.get('error', 'unknown error')}"
                    )
            
            overall_status = result.get("overall_status", "unknown")
            logger.info(f"✓ Gold aggregation complete: {overall_status}")
            
            return result
            
        except Exception as e:
            logger.error(f"Gold aggregation failed: {e}")
            raise
    
    @task
    def skip_aggregation() -> None:
        """
        Placeholder task for when aggregation is skipped.
        Uses AirflowSkipException for proper task state.
        """
        logger.info("⊘ Aggregation skipped (not scheduled time)")
        raise AirflowSkipException("Not aggregation time")
    
    # ========================================
    # DAG Task Flow Definition
    # ========================================
    
    # Bronze layer tasks
    bronze_result = ingest_sales_data()
    dimensions = load_dimension_tables()
    
    # Silver layer task (depends on both Bronze tasks)
    silver_result = process_to_silver(bronze_result)
    
    # Quality validation (depends on Silver)
    quality_result = validate_data_quality()
    
    # Branching logic for Gold aggregation
    branch = BranchPythonOperator(
        task_id="should_aggregate",
        python_callable=should_aggregate,
        doc_md="""
        Determines if Gold aggregation should run based on current hour.
        Configured via `aggregation.trigger_hour` in pipeline_config.yaml.
        """
    )
    
    # Gold layer tasks
    gold_result = aggregate_to_gold()
    skip_task = skip_aggregation()
    
    # ========================================
    # Task Dependencies
    # ========================================
    
    # Bronze → Silver (both Bronze tasks must complete)
    [bronze_result, dimensions] >> silver_result
    
    # Silver → Quality
    silver_result >> quality_result
    
    # Quality → Branch
    quality_result >> branch
    
    # Branch → Gold or Skip
    branch >> [gold_result, skip_task]


# Instantiate the DAG
sales_delta_pipeline_dag = sales_delta_pipeline()
