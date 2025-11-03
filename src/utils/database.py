"""DuckDB utilities."""
import duckdb
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DuckDBClient:
    """DuckDB client with S3 configuration."""
    
    def __init__(self):
        self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        self.conn = self._create_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.conn.close()
    
    def _create_connection(self):
        """Create configured DuckDB connection."""
        conn = duckdb.connect(database=":memory:")
        
        # Install and load extensions
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")
        
        # Configure S3
        endpoint = get_config('s3', 'endpoint')
        access_key = get_config('s3', 'access_key')
        secret_key = get_config('s3', 'secret_key')
        region = get_config('s3', 'region')
        
        conn.execute(f"SET s3_region = '{region}';")
        conn.execute(f"SET s3_access_key_id = '{access_key}';")
        conn.execute(f"SET s3_secret_access_key = '{secret_key}';")
        conn.execute(f"SET s3_endpoint = '{endpoint}';")
        conn.execute("SET s3_url_style = 'path';")
        conn.execute("SET s3_use_ssl = false;")
        
        logger.info("DuckDB connection configured")
        return conn
