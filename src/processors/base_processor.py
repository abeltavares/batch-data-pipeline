"""Base processor implementing shared logic for all data layers."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.utils.storage import StorageClient
from src.utils.logger import get_logger
from src.utils.metrics import track_records, push_metrics
from src.config import get_config


class BaseProcessor(ABC):
    """
    Abstract base class for all processors.
    Enforces interface contract and handles common operations.
    
    Design Pattern: Interface Inheritance
    - Shared authentication/storage logic
    - Consistent error handling
    - Standardized metric tracking
    """
    
    def __init__(self):
        self.storage = StorageClient()
        self.logger = get_logger(self.__class__.__name__)
        self._validate_config()
        self.logger.info(f"✓ Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate processor-specific configuration.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Main processing logic - must be implemented by subclasses.
        
        Returns:
            Dict with processing results (status, counts, metadata)
        """
        pass
    
    def _track_metrics(
        self, 
        layer: str, 
        table: str, 
        count: int,
        additional_labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Shared metric tracking logic."""
        track_records(layer=layer, table=table, count=count)
        
        grouping_key = {'layer': layer, 'table': table}
        if additional_labels:
            grouping_key.update(additional_labels)
        
        push_metrics(grouping_key=grouping_key)
        self.logger.info(f"✓ Tracked {count} records for {layer}/{table}")
    
    def _handle_error(self, error: Exception, context: str) -> None:
        """
        Standardized error handling across all processors.
        
        Args:
            error: The exception that occurred
            context: Description of where the error happened
        """
        self.logger.error(
            f"Error in {context}: {type(error).__name__}: {error}",
            exc_info=True
        )
        # Future: Add alerting logic here (PagerDuty, Slack, etc.)
        raise
    
    def _get_bucket(self, layer: str) -> str:
        """Get bucket name for a specific layer."""
        bucket = get_config('storage', f'{layer}_bucket')
        if not bucket:
            raise ValueError(f"Bucket not configured for layer: {layer}")
        return bucket
