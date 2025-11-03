"""Pydantic models for data validation."""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class Sale(BaseModel):
    """Sales transaction schema."""
    transaction_id: str
    customer_id: int = Field(gt=0)
    product_id: int = Field(gt=0)
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0)
    total_amount: float = Field(gt=0)
    transaction_timestamp: str
    payment_method: str
    store_id: int = Field(gt=0)
    status: str

    @validator('total_amount')
    def validate_total(cls, v, values):
        """Ensure total_amount matches quantity * unit_price."""
        expected = values.get('quantity', 0) * values.get('unit_price', 0)
        if abs(v - expected) > 0.01:
            raise ValueError(f'Total amount mismatch: {v} != {expected}')
        return v
    
    @validator('payment_method')
    def validate_payment_method(cls, v):
        """Validate payment method."""
        valid_methods = ['credit_card', 'debit_card', 'cash', 'digital_wallet']
        if v not in valid_methods:
            raise ValueError(f'Invalid payment method: {v}')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        """Validate transaction status."""
        valid_statuses = ['completed', 'pending', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f'Invalid status: {v}')
        return v


class Product(BaseModel):
    """Product dimension schema."""
    product_id: int = Field(gt=0)
    product_name: str
    category: str
    base_price: float = Field(gt=0)
    stock_quantity: int = Field(ge=0)


class Customer(BaseModel):
    """Customer dimension schema."""
    customer_id: int = Field(gt=0)
    customer_name: str
    email: str
    phone: str
    city: str
    country: str
    signup_date: str
    customer_segment: str
    
    @validator('customer_segment')
    def validate_segment(cls, v):
        """Validate customer segment."""
        valid_segments = ['Premium', 'Regular', 'New']
        if v not in valid_segments:
            raise ValueError(f'Invalid segment: {v}')
        return v
