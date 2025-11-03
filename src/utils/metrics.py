"""Custom metrics for pipeline observability."""
import time
import os
from functools import wraps
from typing import Callable, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
from src.utils.logger import get_logger
from src.config import get_config

logger = get_logger(__name__)

# Pushgateway configuration
PUSHGATEWAY_URL = get_config('observability', 'prometheus_pushgateway', default='prometheus-pushgateway:9091')
JOB_NAME = get_config('observability', 'job_name', default='sales_pipeline')

# Metrics registry
registry = CollectorRegistry()

# Define custom metrics
RECORDS_PROCESSED = Counter(
    'pipeline_records_processed_total',
    'Total number of records processed',
    ['layer', 'table'],
    registry=registry
)

DATA_QUALITY_FAILURES = Counter(
    'pipeline_data_quality_failures_total',
    'Total number of data quality check failures',
    ['check_name'],
    registry=registry
)

DELTA_TABLE_SIZE = Gauge(
    'pipeline_delta_table_rows',
    'Number of rows in Delta table',
    ['table_name'],
    registry=registry
)

TASK_STATUS = Counter(
    'pipeline_task_status_total',
    'Task execution status',
    ['layer', 'operation', 'status'],
    registry=registry
)


def push_metrics(grouping_key: Optional[dict] = None):
    """Push metrics to Pushgateway with optional grouping key.
    
    Args:
        grouping_key: Dictionary of labels to use as grouping key.
                     This prevents last-write-wins by keeping metrics separate.
    """
    try:
        if grouping_key:
            push_to_gateway(
                PUSHGATEWAY_URL, 
                job=JOB_NAME, 
                registry=registry,
                grouping_key=grouping_key
            )
            logger.info(f"Pushed metrics to {PUSHGATEWAY_URL} (grouping: {grouping_key})")
        else:
            push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
            logger.info(f"Pushed metrics to {PUSHGATEWAY_URL}")
    except Exception as e:
        logger.error(f"Failed to push metrics: {e}", exc_info=True)
        # Don't raise - we don't want metric failures to break the pipeline


def track_records(layer: str, table: str, count: int):
    """Track number of records processed."""
    RECORDS_PROCESSED.labels(layer=layer, table=table).inc(count)
    logger.info(f"Processed {count} records in {layer}/{table}")
    
    # Use layer and table as grouping key to prevent overwrites
    push_metrics(grouping_key={'layer': layer, 'table': table})


def track_quality_failure(check_name: str):
    """Track data quality failures."""
    DATA_QUALITY_FAILURES.labels(check_name=check_name).inc()
    logger.warning(f"Quality check failed: {check_name}")
    
    # Use check_name as grouping key
    push_metrics(grouping_key={'check': check_name})


def track_table_size(table_name: str, row_count: int):
    """Track Delta table size."""
    DELTA_TABLE_SIZE.labels(table_name=table_name).set(row_count)
    logger.info(f"Table {table_name} has {row_count} rows")

    push_metrics()
