#!/bin/sh
set -e

echo "⏳ Waiting for Metabase to start..."

# Wait for Metabase to be ready
until curl -sf http://metabase:3000/api/health > /dev/null 2>&1; do
    printf '.'
    sleep 3
done

echo ""
echo "✓ Metabase is ready!"

# Get setup token
echo "🔑 Getting setup token..."
SETUP_TOKEN=""
MAX_RETRIES=30
RETRY_COUNT=0

while [ -z "$SETUP_TOKEN" ] || [ "$SETUP_TOKEN" = "null" ]; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to get setup token after $MAX_RETRIES attempts"
        exit 1
    fi
    
    sleep 2
    RESPONSE=$(curl -sf http://metabase:3000/api/session/properties 2>/dev/null || echo '{}')
    SETUP_TOKEN=$(echo "$RESPONSE" | grep -o '"setup-token":"[^"]*"' | cut -d'"' -f4)
    RETRY_COUNT=$((RETRY_COUNT + 1))
    printf '.'
done

echo ""
echo "✓ Setup token obtained"

# Initial setup
echo "👤 Creating admin user..."
SESSION_ID=$(curl -sf -X POST http://metabase:3000/api/setup \
    -H "Content-Type: application/json" \
    -d "{
        \"token\": \"$SETUP_TOKEN\",
        \"user\": {
            \"email\": \"admin@duckmesh.com\",
            \"first_name\": \"Admin\",
            \"last_name\": \"User\",
            \"password\": \"DuckMesh2025!\"
        },
        \"prefs\": {
            \"allow_tracking\": false,
            \"site_name\": \"DuckMesh Sales Analytics\"
        }
    }" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

echo "✓ Admin user created"

# Login to get session
echo "🔐 Logging in..."
SESSION=$(curl -sf -X POST http://metabase:3000/api/session \
    -H "Content-Type: application/json" \
    -d '{
        "username": "admin@duckmesh.com",
        "password": "DuckMesh2025!"
    }' | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$SESSION" ]; then
    echo "Login failed"
    exit 1
fi

echo "✓ Logged in successfully"

# Add Trino database connection
echo "🔌 Adding Trino database connection..."
curl -sf -X POST http://metabase:3000/api/database \
    -H "Content-Type: application/json" \
    -H "X-Metabase-Session: $SESSION" \
    -d '{
        "engine": "starburst",
        "name": "DuckMesh Analytics (Trino)",
        "details": {
            "host": "trino",
            "port": 8081,
            "catalog": "delta",
            "schema": "analytics",
            "user": "metabase",
            "ssl": false
        },
        "is_full_sync": true,
        "auto_run_queries": true
    }' > /dev/null

DATABASE_ID=$(curl -s -H "X-Metabase-Session: $SESSION" http://metabase:3000/api/database | jq -r '.data[0].id')

echo "✓ Trino connection added"

echo ""
echo "📊 Creating Sales Analytics collection..."

# Create Sales Analytics collection with vibrant color
COLLECTION_RESPONSE=$(curl -sf -X POST http://metabase:3000/api/collection \
    -H "Content-Type: application/json" \
    -H "X-Metabase-Session: $SESSION" \
    -d '{
        "name": "Sales Analytics",
        "description": "DuckMesh sales pipeline analytics",
        "color": "#7C3AED"
    }')

COLLECTION_ID=$(echo "$COLLECTION_RESPONSE" | grep -o '"id":[0-9]*' | cut -d':' -f2)

if [ -z "$COLLECTION_ID" ]; then
    echo "Failed to create collection"
    exit 1
fi

echo "✓ Collection created (ID: $COLLECTION_ID)"

echo "📈 Creating questions with enhanced styling..."

# Helper function to create a styled card
create_card() {
  local CARD_NAME="$1"
  local CARD_QUERY="$2"
  local CARD_DISPLAY="$3"
  local VIZ_SETTINGS="$4"

  local PAYLOAD=$(cat <<JSON
{
  "name": "$CARD_NAME",
  "collection_id": $COLLECTION_ID,
  "display": "$CARD_DISPLAY",
  "dataset_query": {
    "type": "native",
    "native": { "query": "$CARD_QUERY" },
    "database": $DATABASE_ID
  },
  "visualization_settings": $VIZ_SETTINGS
}
JSON
)

  for TRY in 1 2 3; do
    RESP=$(curl -s -X POST "http://metabase:3000/api/card" \
      -H "Content-Type: application/json" \
      -H "X-Metabase-Session: $SESSION" \
      -d "$PAYLOAD")

    local CARD_ID
    CARD_ID=$(echo "$RESP" | jq -r '.id // empty')

    if [ -n "$CARD_ID" ]; then
      echo "✓ Created card: $CARD_NAME (ID: $CARD_ID)" 1>&2
      echo "$CARD_ID"
      return 0
    fi

    echo "⚠️ Attempt $TRY failed for $CARD_NAME, retrying..." 1>&2
    sleep $((TRY * 2))
  done

  echo "Failed to create card: $CARD_NAME" 1>&2
  return 1
}

# Scalar cards with enhanced styling
CARD4_ID=$(create_card "Total Revenue Today" \
  "SELECT total_revenue FROM daily_sales_summary ORDER BY partition_date DESC LIMIT 1" \
  "scalar" \
  '{
    "scalar.field": "total_revenue",
    "scalar.switch_positive_negative": false,
    "column_settings": {
      "[\"name\",\"total_revenue\"]": {
        "number_style": "currency",
        "currency": "USD",
        "currency_style": "symbol",
        "scale": 1
      }
    }
  }')

CARD5_ID=$(create_card "Total Transactions Today" \
  "SELECT total_transactions FROM daily_sales_summary ORDER BY partition_date DESC LIMIT 1" \
  "scalar" \
  '{
    "scalar.field": "total_transactions",
    "column_settings": {
      "[\"name\",\"total_transactions\"]": {
        "number_style": "decimal",
        "decimals": 0
      }
    }
  }')

CARD6_ID=$(create_card "Unique Customers Today" \
  "SELECT unique_customers FROM daily_sales_summary ORDER BY partition_date DESC LIMIT 1" \
  "scalar" \
  '{
    "scalar.field": "unique_customers",
    "column_settings": {
      "[\"name\",\"unique_customers\"]": {
        "number_style": "decimal",
        "decimals": 0
      }
    }
  }')

CARD7_ID=$(create_card "Completion Rate Today" \
  "SELECT completion_rate / 100.0 AS completion_rate FROM daily_sales_summary ORDER BY partition_date DESC LIMIT 1" \
  "scalar" \
  '{
    "scalar.field": "completion_rate",
    "column_settings": {
      "[\"name\",\"completion_rate\"]": {
        "number_style": "percent",
        "decimals": 1,
        "scale": 1
      }
    }
  }')

# Line chart with smooth curves and gradient
CARD1_ID=$(create_card "Daily Revenue Trend" \
  "SELECT partition_date, total_revenue FROM daily_sales_summary ORDER BY partition_date" \
  "line" \
  '{
    "graph.dimensions": ["partition_date"],
    "graph.metrics": ["total_revenue"],
    "line.interpolate": "cardinal",
    "line.marker_enabled": true,
    "graph.colors": ["#7C3AED"],
    "graph.show_values": false,
    "graph.y_axis.auto_range": true,
    "graph.y_axis.scale": "linear",
    "column_settings": {
      "[\"name\",\"total_revenue\"]": {
        "number_style": "currency",
        "currency": "USD",
        "currency_style": "symbol"
      }
    }
  }')

# Bar chart with gradient colors
CARD2_ID=$(create_card "Top 10 Products by Revenue" \
  "SELECT product_name, revenue FROM product_performance WHERE CAST(data_date AS DATE) = CURRENT_DATE ORDER BY revenue DESC LIMIT 10" \
  "bar" \
  '{
    "graph.dimensions": ["product_name"],
    "graph.metrics": ["revenue"],
    "graph.colors": ["#10B981"],
    "graph.show_values": true,
    "graph.label_value_formatting": "compact",
    "column_settings": {
      "[\"name\",\"revenue\"]": {
        "number_style": "currency",
        "currency": "USD",
        "currency_style": "symbol",
        "number_separators": ".,",
        "decimals": 0
      }
    }
  }')

# Donut chart with custom colors
CARD3_ID=$(create_card "Customer Value Distribution" \
  "SELECT value_tier, COUNT(*) as customers FROM customer_segments WHERE CAST(data_date AS DATE) = CURRENT_DATE GROUP BY value_tier" \
  "pie" \
  '{
    "pie.dimension": "value_tier",
    "pie.metric": "customers",
    "pie.show_legend": true,
    "pie.show_total": true,
    "pie.slice_threshold": 1,
    "pie.colors": {
      "High Value": "#EF4444",
      "Low Value": "#FBBF24", 
      "Medium Value": "#10B981"
    }
  }')

# Horizontal bar chart with vibrant colors
CARD8_ID=$(create_card "Revenue by Category" \
  "SELECT category, SUM(revenue) as revenue FROM product_performance WHERE CAST(data_date AS DATE) = CURRENT_DATE GROUP BY category ORDER BY revenue DESC" \
  "row" \
  '{
    "graph.dimensions": ["category"],
    "graph.metrics": ["revenue"],
    "graph.colors": ["#F59E0B"],
    "graph.show_values": true,
    "graph.label_value_formatting": "compact",
    "column_settings": {
      "[\"name\",\"revenue\"]": {
        "number_style": "currency",
        "currency": "USD",
        "currency_style": "symbol",
        "decimals": 0
      }
    }
  }')

echo ""
echo "📊 Creating enhanced dashboard..."

# Create dashboard with better styling
DASHBOARD_RESPONSE=$(curl -s -X POST http://metabase:3000/api/dashboard \
  -H "Content-Type: application/json" \
  -H "X-Metabase-Session: $SESSION" \
  -d "{
    \"name\": \"Sales Performance Dashboard\",
    \"description\": \"Real-time sales analytics and KPIs\",
    \"collection_id\": $COLLECTION_ID
  }")

DASHBOARD_ID=$(echo "$DASHBOARD_RESPONSE" | jq -r '.id // empty')

if [ -z "$DASHBOARD_ID" ]; then
    echo "Failed to create dashboard"
    exit 1
fi

echo "✓ Dashboard created (ID: $DASHBOARD_ID)"
echo "📍 Adding styled cards to dashboard..."

# Enhanced layout with card styling
curl -s -X PUT "http://metabase:3000/api/dashboard/$DASHBOARD_ID" \
  -H "Content-Type: application/json" \
  -H "X-Metabase-Session: $SESSION" \
  -d "{
    \"name\": \"Sales Performance Dashboard\",
    \"description\": \"Real-time sales analytics and KPIs\",
    \"collection_id\": $COLLECTION_ID,
    \"width\": \"full\",
    \"dashcards\": [
      {
        \"id\": -1, 
        \"card_id\": $CARD4_ID, 
        \"row\": 0, 
        \"col\": 0, 
        \"size_x\": 6, 
        \"size_y\": 3,
        \"visualization_settings\": {
          \"card.title\": \"Total Revenue Today\"
        }
      },
      {
        \"id\": -2, 
        \"card_id\": $CARD5_ID, 
        \"row\": 0, 
        \"col\": 6, 
        \"size_x\": 6, 
        \"size_y\": 3,
        \"visualization_settings\": {
          \"card.title\": \"Total Transactions Today\"
        }
      },
      {
        \"id\": -3, 
        \"card_id\": $CARD6_ID, 
        \"row\": 0, 
        \"col\": 12, 
        \"size_x\": 6, 
        \"size_y\": 3,
        \"visualization_settings\": {
          \"card.title\": \"Unique Customers Today\"
        }
      },
      {
        \"id\": -4, 
        \"card_id\": $CARD7_ID, 
        \"row\": 0, 
        \"col\": 18, 
        \"size_x\": 6, 
        \"size_y\": 3,
        \"visualization_settings\": {
          \"card.title\": \"Completion Rate Today\"
        }
      },
      {
        \"id\": -5, 
        \"card_id\": $CARD1_ID, 
        \"row\": 3, 
        \"col\": 0, 
        \"size_x\": 24, 
        \"size_y\": 6,
        \"visualization_settings\": {
          \"card.title\": \"Daily Revenue Trend\"
        }
      },
      {
        \"id\": -6, 
        \"card_id\": $CARD2_ID, 
        \"row\": 9, 
        \"col\": 0, 
        \"size_x\": 8, 
        \"size_y\": 6,
        \"visualization_settings\": {
          \"card.title\": \"Top 10 Products by Revenue\"
        }
      },
      {
        \"id\": -7, 
        \"card_id\": $CARD8_ID, 
        \"row\": 9, 
        \"col\": 8, 
        \"size_x\": 8, 
        \"size_y\": 6,
        \"visualization_settings\": {
          \"card.title\": \"Revenue by Category\"
        }
      },
      {
        \"id\": -8, 
        \"card_id\": $CARD3_ID, 
        \"row\": 9, 
        \"col\": 16, 
        \"size_x\": 8, 
        \"size_y\": 6,
        \"visualization_settings\": {
          \"card.title\": \"Customer Value Distribution\"
        }
      }
    ]
  }" > /tmp/dashboard_response.json

echo "✓ Dashboard layout configured"

echo ""
echo "  ╔╦╗╦ ╦╔═╗╦╔═╔╦╗╔═╗╔═╗╦ ╦"
echo "   ║║║ ║║  ╠╩╗║║║║╣ ╚═╗╠═╣"
echo "  ═╩╝╚═╝╚═╝╩ ╩╩ ╩╚═╝╚═╝╩ ╩"
echo ""
echo "  ★彡 ⚡ SALES ANALYTICS ⚡ ★彡"
echo ""
echo "  ✅ Dashboard Setup Complete!"
echo ""
echo "  🌐 URL:      http://localhost:3000"
echo "  👤 Email:    admin@duckmesh.com"
echo "  🔑 Password: DuckMesh2025!"
echo ""
echo "  📊 Database: DuckMesh Analytics (Trino)"
echo "  📁 Schema:   delta.analytics"
echo ""
echo "  ★彡 ⚡ ★彡 ⚡ ★彡 ⚡ ★彡 ⚡"
echo ""