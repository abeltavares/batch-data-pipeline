"""S3/MinIO storage utilities."""
import boto3
import json
from typing import Any, Dict
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StorageClient:
    """S3/MinIO storage client."""
    
    def __init__(self):
        self.endpoint_url = f"http://{get_config('s3', 'endpoint')}"
        self.access_key = get_config('s3', 'access_key')
        self.secret_key = get_config('s3', 'secret_key')
        
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
    
    def upload_json(self, bucket: str, key: str, data: Any) -> None:
        """Upload JSON data to S3."""
        try:
            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(data).encode("utf-8")
            )
            logger.info(f"Uploaded to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Failed to upload to s3://{bucket}/{key}: {e}")
            raise
    
    def upload_parquet(self, bucket: str, key: str, data: bytes) -> None:
        """Upload Parquet data to S3."""
        try:
            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data
            )
            logger.info(f"Uploaded to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Failed to upload to s3://{bucket}/{key}: {e}")
            raise
    
    def object_exists(self, bucket: str, key: str) -> bool:
        """Check if object exists in S3."""
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    
    def get_storage_options(self) -> Dict[str, str]:
        """Get storage options for Delta Lake."""
        return {
            "AWS_ACCESS_KEY_ID": self.access_key,
            "AWS_SECRET_ACCESS_KEY": self.secret_key,
            "AWS_ENDPOINT_URL": self.endpoint_url,
            "AWS_REGION": get_config('s3', 'region'),
            "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
            "AWS_ALLOW_HTTP": "true",
        }
