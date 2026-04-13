DATE = 'date'
STORE_ID = 'store_id'
PRODUCT_ID = 'product_id'
PRODUCT_NAME = 'product_name'
CATEGORY = 'category'
INVENTORY_LEVEL = 'inventory_level'
UNITS_SOLD = 'units_sold'
PRICE = 'price'
SEASONALITY = 'seasonality'

TARGET = UNITS_SOLD
TARGET_NEXT_DAY = 'target_units_next_day'
PREDICTED_DEMAND = 'predicted_demand'
REORDER_POINT = 'reorder_point'
REORDER_ALERT = 'reorder_alert'
ACTUAL_DEMAND = 'actual_demand'

LEGACY_TO_CANONICAL = {
    'Date': DATE,
    'Store ID': STORE_ID,
    'Product ID': PRODUCT_ID,
    'Product': PRODUCT_NAME,
    'Inventory Level': INVENTORY_LEVEL,
    'Inventory': INVENTORY_LEVEL,
    'Units Sold': UNITS_SOLD,
    'Price': PRICE,
    'Seasonality': SEASONALITY,
    'Category': SEASONALITY,
}


def normalize_column_names(df):
    out = df.copy()
    renames = {}
    for old_name, new_name in LEGACY_TO_CANONICAL.items():
        if old_name in out.columns:
            renames[old_name] = new_name
    if renames:
        out = out.rename(columns=renames)
    return out


def ensure_training_columns(df):
    out = normalize_column_names(df)
    if INVENTORY_LEVEL not in out.columns and 'inventory' in out.columns:
        out = out.rename(columns={'inventory': INVENTORY_LEVEL})

    for col in ('demand_forecast', 'Demand Forecast'):
        if col in out.columns:
            out = out.drop(columns=[col])

    has_cat = CATEGORY in out.columns
    has_season = SEASONALITY in out.columns
    if has_cat and has_season:
        out = out.drop(columns=[CATEGORY])
    elif has_cat:
        out = out.rename(columns={CATEGORY: SEASONALITY})
    return out
