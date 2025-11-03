#!/bin/bash
set -e

# Create marquez database if it doesn't exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE marquez'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'marquez')\gexec
EOSQL

echo "✓ Marquez database created"
