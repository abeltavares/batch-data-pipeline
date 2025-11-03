"""Sales data generator with dirty data simulation."""
from faker import Faker
from faker_commerce import Provider
import random
from datetime import datetime, timedelta
from typing import List, Dict
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SalesDataGenerator:
    """Generate synthetic sales data with realistic data quality issues."""

    def __init__(self):
        # Use different seed each time for variety
        self.fake = Faker()
        self.fake.add_provider(Provider)
        Faker.seed(int(datetime.utcnow().timestamp()))
        random.seed(int(datetime.utcnow().timestamp()))
        self.num_products = get_config('data_generation', 'num_products', default=50)
        self.num_customers = get_config('data_generation', 'num_customers', default=1000)
        self.num_stores = get_config('data_generation', 'num_stores', default=20)

    def generate_sales(self, num_records: int = None) -> List[Dict]:
        """Generate incremental sales transactions with realistic data quality issues."""
        if num_records is None:
            num_records = get_config('data_generation', 'sales_records_per_run', default=100)

        sales = []
        base_time = datetime.utcnow()

        for i in range(num_records):
            quantity = random.randint(1, 10)
            unit_price = round(random.uniform(10.0, 500.0), 2)

            # Base sale record
            sale = {
                "transaction_id": self.fake.uuid4(),  # Always unique
                "customer_id": int(random.randint(1, self.num_customers)),
                "product_id": random.randint(1, self.num_products),
                "quantity": quantity,
                "unit_price": unit_price,
                "total_amount": round(quantity * unit_price, 2),
                "transaction_timestamp": (
                    base_time - timedelta(minutes=random.randint(0, 1440))  # Last 24 hours
                ).isoformat(),
                "payment_method": random.choice([
                    "credit_card", "debit_card", "cash", "digital_wallet"
                ]),
                "store_id": random.randint(1, self.num_stores),
                "status": random.choice(["completed", "pending", "cancelled"]),
            }

            # Introduce realistic data quality issues (20% of records)
            if random.random() < 0.20:
                issue_type = random.choice([
                    'null_customer',
                    'invalid_product',
                    'negative_amount',
                    'future_date',
                    'invalid_status',
                    'zero_quantity',
                    'mismatched_total',
                    'invalid_payment',
                    'null_timestamp'
                ])

                if issue_type == 'null_customer':
                    sale['customer_id'] = None

                elif issue_type == 'invalid_product':
                    # Product ID outside valid range
                    sale['product_id'] = random.randint(9000, 9999)

                elif issue_type == 'negative_amount':
                    sale['total_amount'] = -abs(sale['total_amount'])

                elif issue_type == 'future_date':
                    sale['transaction_timestamp'] = (
                        base_time + timedelta(days=random.randint(1, 30))
                    ).isoformat()

                elif issue_type == 'invalid_status':
                    sale['status'] = random.choice(['UNKNOWN', 'ERROR', '', None])

                elif issue_type == 'zero_quantity':
                    sale['quantity'] = 0

                elif issue_type == 'mismatched_total':
                    # Total doesn't match quantity * unit_price
                    sale['total_amount'] = round(random.uniform(1.0, 1000.0), 2)

                elif issue_type == 'invalid_payment':
                    sale['payment_method'] = random.choice(['', None, 'INVALID', 'bitcoin'])

                elif issue_type == 'null_timestamp':
                    sale['transaction_timestamp'] = None

                logger.debug(f"Injected data quality issue: {issue_type}")

            sales.append(sale)

        logger.info(f"Generated {len(sales)} sales transactions (with ~20% dirty data)")
        return sales

    def generate_products(self) -> List[Dict]:
        """Generate product catalog."""
        products = []
        for i in range(1, self.num_products + 1):
            products.append({
                "product_id": i,
                "product_name": self.fake.ecommerce_name(),
                "category": self.fake.ecommerce_category(),
                "base_price": round(random.uniform(10.0, 500.0), 2),
                "stock_quantity": random.randint(0, 1000),
            })

        logger.info(f"Generated {len(products)} products")
        return products

    def generate_customers(self) -> List[Dict]:
        """Generate customer dimension."""
        customers = []
        for i in range(1, self.num_customers + 1):
            customers.append({
                "customer_id": int(i),
                "customer_name": self.fake.name(),
                "email": self.fake.email(),
                "phone": self.fake.phone_number(),
                "city": self.fake.city(),
                "country": self.fake.country(),
                "signup_date": self.fake.date_between(
                    start_date="-2y", end_date="today"
                ).isoformat(),
                "customer_segment": random.choice(["Premium", "Regular", "New"]),
            })

        logger.info(f"Generated {len(customers)} customers")
        return customers
