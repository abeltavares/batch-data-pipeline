"""Bronze layer: Raw data ingestion with no validation."""
from datetime import datetime
from typing import Dict, Any
from src.processors.base_processor import BaseProcessor
from src.generators.sales_generator import SalesDataGenerator
from src.models.schemas import Product, Customer


class BronzeProcessor(BaseProcessor):
    """
    Bronze layer processor - ingests raw data without validation.
    Accepts dirty data intentionally for downstream cleaning.
    """
    
    def _validate_config(self) -> None:
        """Validate Bronze-specific configuration."""
        self.bronze_bucket = self._get_bucket('bronze')
        self.generator = SalesDataGenerator()
        self.logger.info(f"Bronze bucket: {self.bronze_bucket}")
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Ingest sales data to Bronze layer.
        
        Returns:
            {
                "key": "s3_key_path",
                "count": record_count,
                "timestamp": "2024-01-01T12:00:00"
            }
        """
        try:
            # Generate sales data (with intentional dirty data)
            sales_data = self.generator.generate_sales()
            
            if not sales_data:
                self.logger.warning("No sales data generated")
                return {"key": None, "count": 0, "status": "skipped"}
            
            # Create timestamped key for incremental loading
            timestamp = datetime.utcnow()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            key = f"sales/sales_{timestamp_str}.json"
            
            # Upload to Bronze (NO VALIDATION - accept dirty data)
            self.storage.upload_json(
                bucket=self.bronze_bucket,
                key=key,
                data=sales_data
            )
            
            # Track metrics
            self._track_metrics(
                layer='bronze',
                table='sales',
                count=len(sales_data)
            )
            
            self.logger.info(
                f"✓ Ingested {len(sales_data)} sales records to {key}"
            )
            
            return {
                "key": key,
                "count": len(sales_data),
                "timestamp": timestamp.isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            self._handle_error(e, "bronze_sales_ingestion")
    
    def load_dimensions(self) -> Dict[str, Any]:
        """
        Load dimension tables (products and customers).
        Idempotent - skips if already loaded.
        
        Returns:
            {"status": "loaded" | "skipped", "products": int, "customers": int}
        """
        try:
            # Check if dimensions already exist (idempotency)
            products_key = "dimensions/products.json"
            customers_key = "dimensions/customers.json"
            
            if self.storage.object_exists(self.bronze_bucket, products_key):
                self.logger.info("✓ Dimensions already loaded, skipping")
                return {
                    "status": "skipped",
                    "reason": "already_exists"
                }
            
            # Generate dimension data
            products = self.generator.generate_products()
            customers = self.generator.generate_customers()
            
            # Validate dimensions (unlike sales, dimensions should be clean)
            validated_products = [Product(**p).dict() for p in products]
            validated_customers = [Customer(**c).dict() for c in customers]
            
            # Upload to Bronze
            self.storage.upload_json(
                bucket=self.bronze_bucket,
                key=products_key,
                data=validated_products
            )
            
            self.storage.upload_json(
                bucket=self.bronze_bucket,
                key=customers_key,
                data=validated_customers
            )
            
            self.logger.info(
                f"✓ Loaded {len(products)} products and {len(customers)} customers"
            )
            
            return {
                "status": "loaded",
                "products": len(products),
                "customers": len(customers)
            }
            
        except Exception as e:
            self._handle_error(e, "dimension_loading")
