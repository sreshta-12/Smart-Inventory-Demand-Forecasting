import base64
import io
import sys
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.columns import (
    DATE,
    INVENTORY_LEVEL,
    PREDICTED_DEMAND,
    PRICE,
    PRODUCT_ID,
    PRODUCT_NAME,
    REORDER_ALERT,
    REORDER_POINT,
    SEASONALITY,
    STORE_ID,
    UNITS_SOLD,
    ensure_training_columns,
)
from src.data_loader import CSV_SIMPLE, load_data
from src.forecast_config import FORECAST_HORIZON_DAYS
from src.forecast_future import forecast_calendar_series
from src.pipeline import ACTUAL_NEXT_DAY, run_batch_predict
from src.predict import load_artifacts
from src.reorder import DEFAULT_LEAD_TIME_DAYS

model, scaler, feature_cols, scaled_columns = load_artifacts()

FORECAST_TAIL_PER_SERIES = 200
FORECAST_MAX_GROUPS = 100
HISTORY_DAYS_ON_FORECAST_CHART = 120
DASHBOARD_RECORDS_HISTORY_DAYS = 200
RAW_SLICE_COLUMNS = [DATE, STORE_ID, PRODUCT_ID, UNITS_SOLD, INVENTORY_LEVEL, PRODUCT_NAME, PRICE, SEASONALITY]

app = dash.Dash(__name__)
app.title = 'Inventory Demand Forecasting'

CARD_STYLE = {
    'flex': '1',
    'minWidth': '160px',
    'maxWidth': '240px',
    'padding': '16px',
    'margin': '8px',
    'borderRadius': '10px',
    'backgroundColor': '#f5f7fa',
    'boxShadow': '0 1px 4px rgba(0,0,0,0.08)',
    'textAlign': 'center',
}
GRAPH_SECTION_TITLE = {
    'textAlign': 'center',
    'fontSize': '18px',
    'fontWeight': 'bold',
    'margin': '8px auto 4px',
    'maxWidth': '1000px',
}


def json_for_json_store(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return str(val.date())
    return val


def dates_as_strings(series):
    t = pd.to_datetime(series, errors='coerce')
    out = []
    for x in t:
        out.append(x.strftime('%Y-%m-%d') if pd.notna(x) else '')
    return out


def nums_as_plain_list(series):
    arr = np.asarray(pd.to_numeric(series, errors='coerce'), dtype=float)
    plain = []
    for v in arr.flat:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            plain.append(None)
        else:
            plain.append(float(v))
    return plain


def build_payload_from_df(df: pd.DataFrame, persist_forecasts: bool=False):
    df = ensure_training_columns(df.copy())
    slice_scored, _df_out = run_batch_predict(
        df, model, scaler, feature_cols, scaled_columns,
        lead_time_days=DEFAULT_LEAD_TIME_DAYS,
        persist_db=persist_forecasts,
    )
    if slice_scored.empty:
        return (None, 'Not enough complete rows after feature engineering (lags require history).')

    n_products_full = int(slice_scored[PRODUCT_ID].nunique())
    show = slice_scored.copy()
    show[DATE] = pd.to_datetime(show[DATE], errors='coerce')
    show = show.dropna(subset=[DATE])
    last_day = show[DATE].max()
    show = show[show[DATE] >= last_day - pd.Timedelta(days=DASHBOARD_RECORDS_HISTORY_DAYS)]

    grp_key = [STORE_ID, PRODUCT_ID]
    grouped = sorted(show.groupby(grp_key, sort=False), key=lambda x: len(x[1]), reverse=True)
    chunks = []
    for _key, g in grouped[:FORECAST_MAX_GROUPS]:
        if len(g) >= 35:
            chunks.append(g.tail(FORECAST_TAIL_PER_SERIES))

    if chunks:
        raw_concat = pd.concat(chunks, ignore_index=True)
    else:
        raw_concat = show.iloc[:0]

    rs_cols = [c for c in RAW_SLICE_COLUMNS if c in raw_concat.columns]
    if rs_cols and UNITS_SOLD in rs_cols:
        raw_slim = raw_concat[rs_cols]
    else:
        raw_slim = raw_concat

    raw_records = []
    for rec in raw_slim.to_dict(orient='records'):
        row = {k: json_for_json_store(v) for k, v in rec.items()}
        if DATE in row and row[DATE] is not None:
            row[DATE] = str(pd.to_datetime(row[DATE]).date())
        raw_records.append(row)

    calendar_forecast_agg = None
    if len(raw_concat) >= 50:
        cal_tbl = seven_day_forecast_table(raw_concat)
        if cal_tbl is not None and not cal_tbl.empty:
            calendar_forecast_agg = []
            for r in cal_tbl.to_dict(orient='records'):
                rr = {k: json_for_json_store(v) for k, v in r.items()}
                calendar_forecast_agg.append(rr)

    if INVENTORY_LEVEL in show.columns:
        inv_col = INVENTORY_LEVEL
        inv_vals = show[inv_col]
    else:
        inv_vals = 0

    if PRODUCT_NAME in show.columns:
        names = show[PRODUCT_NAME].astype(str)
    else:
        names = show[PRODUCT_ID].astype(str)

    out_df = pd.DataFrame({
        STORE_ID: show[STORE_ID].astype(str),
        PRODUCT_ID: show[PRODUCT_ID].astype(str),
        PRODUCT_NAME: names,
        DATE: show[DATE].astype(str),
        INVENTORY_LEVEL: inv_vals,
        PREDICTED_DEMAND: show[PREDICTED_DEMAND],
        REORDER_POINT: pd.to_numeric(show[REORDER_POINT], errors='coerce'),
        REORDER_ALERT: show[REORDER_ALERT],
        UNITS_SOLD: pd.to_numeric(show.get(UNITS_SOLD, 0), errors='coerce').fillna(0),
        ACTUAL_NEXT_DAY: pd.to_numeric(show.get(ACTUAL_NEXT_DAY), errors='coerce'),
    })

    dmin = pd.to_datetime(out_df[DATE], errors='coerce').min()
    dmax = pd.to_datetime(out_df[DATE], errors='coerce').max()

    table_records = []
    for r in out_df.to_dict('records'):
        row = {k: json_for_json_store(v) for k, v in r.items()}
        if DATE in row and row[DATE] is not None:
            row[DATE] = str(row[DATE])
        table_records.append(row)

    meta = {
        'n_products': n_products_full,
        'product_ids': sorted(slice_scored[PRODUCT_ID].astype(str).unique().tolist()),
        'date_min': str(dmin.date()) if pd.notna(dmin) else None,
        'date_max': str(dmax.date()) if pd.notna(dmax) else None,
        'forecast_horizon_days': FORECAST_HORIZON_DAYS,
    }
    payload = {
        'records': table_records,
        'raw_slice': raw_records,
        'calendar_forecast_agg': calendar_forecast_agg,
        'meta': meta,
    }
    return (payload, None)


def product_dropdown_options(payload: dict):
    pids = (payload.get('meta') or {}).get('product_ids')
    base = [{'label': 'All Products', 'value': 'ALL'}]
    if pids:
        return base + [{'label': p, 'value': p} for p in pids]
    rest = sorted({r[PRODUCT_ID] for r in payload['records']})
    return base + [{'label': p, 'value': p} for p in rest]


def reorder_gap_units(snap: pd.DataFrame) -> int:
    if snap.empty:
        return 0
    if REORDER_POINT not in snap.columns or INVENTORY_LEVEL not in snap.columns:
        return 0
    inv = pd.to_numeric(snap[INVENTORY_LEVEL], errors='coerce').fillna(0.0)
    rop = pd.to_numeric(snap[REORDER_POINT], errors='coerce').fillna(0.0)
    gap = (rop - inv).clip(lower=0.0)
    return int(np.rint(float(gap.sum())))


def filter_by_product(df: pd.DataFrame, product_value: str) -> pd.DataFrame:
    out = df.copy()
    out[DATE] = pd.to_datetime(out[DATE], errors='coerce')
    out = out.dropna(subset=[DATE])
    if product_value and product_value != 'ALL':
        out = out[out[PRODUCT_ID].astype(str) == str(product_value)]
    return out


def chart_sales_trend(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or UNITS_SOLD not in df.columns:
        fig.update_layout(title='Sales Trend Over Time')
        return fig
    tmp = df.copy()
    daily = tmp.groupby(DATE, as_index=False)[UNITS_SOLD].sum().sort_values(DATE)
    if daily.empty:
        fig.update_layout(title='Sales Trend Over Time')
        return fig
    fig.add_trace(go.Scatter(
        x=dates_as_strings(daily[DATE]),
        y=nums_as_plain_list(daily[UNITS_SOLD]),
        mode='lines',
        name='Sales quantity',
        line=dict(color='#1565c0', width=2),
    ))
    fig.update_layout(
        title='Sales Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Sales quantity',
        height=400,
        margin=dict(l=55, r=25, t=55, b=50),
        showlegend=False,
        hovermode='x unified',
    )
    return fig


def chart_inventory_bars(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title='Current Inventory Levels')
        return fig
    d = df.copy()
    d[DATE] = pd.to_datetime(d[DATE], errors='coerce')
    d = d.dropna(subset=[DATE])
    last = d[DATE].max()
    snap = d[d[DATE] == last]
    if snap.empty:
        fig.update_layout(title='Current Inventory Levels')
        return fig
    g = snap.groupby(PRODUCT_NAME, as_index=False)[INVENTORY_LEVEL].sum().sort_values(INVENTORY_LEVEL, ascending=True)
    fig.add_trace(go.Bar(
        x=nums_as_plain_list(g[INVENTORY_LEVEL]),
        y=g[PRODUCT_NAME].astype(str).tolist(),
        orientation='h',
        marker_color='#5c6bc0',
    ))
    h = max(360, 28 * len(g) + 100)
    fig.update_layout(
        title='Current Inventory Levels',
        xaxis_title='Stock level',
        yaxis_title='Product',
        height=h,
        margin=dict(l=140, r=30, t=55, b=45),
        showlegend=False,
    )
    return fig


def raw_slice_to_df(raw_records: list | None, product_value: str) -> pd.DataFrame:
    if not raw_records:
        return pd.DataFrame()
    raw_df = pd.DataFrame(raw_records)
    raw_df[DATE] = pd.to_datetime(raw_df[DATE], errors='coerce')
    raw_df = raw_df.dropna(subset=[DATE])
    if product_value and product_value != 'ALL' and PRODUCT_ID in raw_df.columns:
        raw_df = raw_df[raw_df[PRODUCT_ID].astype(str) == str(product_value)]
    return raw_df


def seven_day_forecast_table(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    if raw_df.empty or len(raw_df) < 50:
        return None
    cal = forecast_calendar_series(
        raw_df, model, scaler, feature_cols, scaled_columns,
        n=FORECAST_HORIZON_DAYS,
        max_groups=FORECAST_MAX_GROUPS,
    )
    cal = cal.dropna(subset=['mean_predicted_demand'])
    if cal.empty:
        return None
    return cal


def layout_with_legend_at_bottom():
    return dict(
        margin=dict(t=16, l=55, r=30, b=88),
        legend=dict(orientation='h', yanchor='top', y=-0.2, x=0.5, xanchor='center'),
    )


def chart_actual_vs_pred(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    leg = layout_with_legend_at_bottom()
    if df.empty or ACTUAL_NEXT_DAY not in df.columns:
        fig.update_layout(**leg, height=440)
        return fig
    d = df.copy()
    d[DATE] = pd.to_datetime(d[DATE], errors='coerce')
    d = d.dropna(subset=[DATE])
    last_hist = d[DATE].max()
    d = d[d[DATE] >= last_hist - pd.Timedelta(days=HISTORY_DAYS_ON_FORECAST_CHART)]
    act = pd.to_numeric(d[ACTUAL_NEXT_DAY], errors='coerce')
    pred = pd.to_numeric(d[PREDICTED_DEMAND], errors='coerce')
    d['_a'] = act
    d['_p'] = pred
    d = d.dropna(subset=['_a', '_p'])
    if not d.empty:
        d['_day'] = d[DATE].dt.normalize()
        g = d.groupby('_day', as_index=False).agg({'_a': 'mean', '_p': 'mean'})
        ya = np.rint(g['_a'].astype(float)).astype(int)
        yp = np.rint(g['_p'].astype(float)).astype(int)
        xd = dates_as_strings(g['_day'])
        fig.add_trace(go.Scatter(
            x=xd, y=ya.astype(int).tolist(),
            mode='lines', name='Actual (next-day units)',
            line=dict(color='#0d47a1', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=xd, y=yp.astype(int).tolist(),
            mode='lines', name='Predicted (1-day ahead)',
            line=dict(color='#e65100', width=2, dash='dash'),
        ))
    fig.update_layout(**leg, xaxis_title='Date', yaxis_title='Units (whole)', height=440, hovermode='x unified')
    return fig


def chart_next_week(raw_records: list | None, product_value: str, precomputed_cal_records: list | None=None):
    fig = go.Figure()
    leg = layout_with_legend_at_bottom()
    pv = product_value or 'ALL'
    cal = None
    if pv == 'ALL' and precomputed_cal_records:
        cal = pd.DataFrame(precomputed_cal_records)
        if not cal.empty and 'forecast_date' in cal.columns:
            cal = cal.copy()
            cal['forecast_date'] = pd.to_datetime(cal['forecast_date'], errors='coerce')
            cal = cal.dropna(subset=['forecast_date'])
    if cal is None or cal.empty:
        raw_df = raw_slice_to_df(raw_records, pv)
        cal = seven_day_forecast_table(raw_df)
    if cal is None or cal.empty:
        fig.update_layout(**leg, height=440)
        return fig
    cal = cal.copy()
    cal['forecast_date'] = pd.to_datetime(cal['forecast_date'])
    mean = np.rint(cal['mean_predicted_demand'].astype(float)).astype(int)
    std = cal.get('std_predicted_demand', pd.Series(0.0, index=cal.index)).astype(float).fillna(0)
    upper = np.rint(mean + 1.96 * std).astype(int)
    lower = np.maximum(np.rint(mean - 1.96 * std).astype(int), 0)
    xd = dates_as_strings(cal['forecast_date'])
    x_poly = xd + xd[::-1]
    y_poly = upper.astype(int).tolist() + lower.astype(int).tolist()[::-1]
    fig.add_trace(go.Scatter(
        x=x_poly, y=y_poly, fill='toself',
        fillcolor='rgba(46, 125, 50, 0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% band (across series)',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=xd,
        y=mean.astype(int).tolist(),
        mode='lines+markers',
        name=f'Mean forecast (next {FORECAST_HORIZON_DAYS} days)',
        line=dict(color='#2e7d32', width=2),
    ))
    fig.update_layout(**leg, xaxis_title='Date', yaxis_title='Units (whole)', height=440, hovermode='x unified')
    return fig


def calendar_table_block(cal_df: pd.DataFrame | None, *, product_scope: str) -> html.Div:
    if cal_df is None or cal_df.empty:
        return html.P(
            'No 7-day calendar forecast table yet (need loaded data and enough history).',
            style={'textAlign': 'center', 'color': '#666'},
        )
    d = cal_df.copy()
    if 'forecast_date' in d.columns:
        d['forecast_date'] = pd.to_datetime(d['forecast_date'], errors='coerce')
    if 'horizon_day' in d.columns:
        d = d.sort_values('horizon_day')
    th_style = {
        'textAlign': 'left',
        'padding': '10px 12px',
        'backgroundColor': '#e8f5e9',
        'fontWeight': 'bold',
        'borderBottom': '1px solid #c8e6c9',
    }
    num_th = {**th_style, 'textAlign': 'right'}
    body_rows = []
    for _, row in d.iterrows():
        fd = row.get('forecast_date')
        if hasattr(fd, 'strftime'):
            fd_s = fd.strftime('%Y-%m-%d')
        else:
            fd_s = str(fd)[:10]
        mu = row.get('mean_predicted_demand', np.nan)
        sd = row.get('std_predicted_demand', np.nan)
        ns = row.get('n_series', '')
        hz = row.get('horizon_day', '')
        if pd.notna(mu):
            mu_i = int(np.rint(float(mu)))
        else:
            mu_i = '—'
        if pd.notna(sd):
            sd_s = f'{float(sd):.2f}'
        else:
            sd_s = '—'
        ns_num = pd.to_numeric(ns, errors='coerce')
        if pd.notna(ns_num):
            ns_cell = str(int(ns_num))
        else:
            ns_cell = '—'
        cell = {'padding': '8px 12px', 'borderBottom': '1px solid #eee'}
        body_rows.append(html.Tr([
            html.Td(str(hz), style=cell),
            html.Td(fd_s, style=cell),
            html.Td(str(mu_i), style={**cell, 'textAlign': 'right'}),
            html.Td(sd_s, style={**cell, 'textAlign': 'right'}),
            html.Td(ns_cell, style={**cell, 'textAlign': 'right'}),
        ]))
    if product_scope == 'ALL':
        subtitle = 'All products (mean across store–product series).'
    else:
        subtitle = f'Single product: {product_scope} (one series in aggregate).'
    tbl = html.Table([
        html.Thead(html.Tr([
            html.Th('Day #', style=th_style),
            html.Th('Forecast date', style=th_style),
            html.Th('Mean predicted demand (units)', style=num_th),
            html.Th('Std across series', style=num_th),
            html.Th('# series', style=num_th),
        ])),
        html.Tbody(body_rows),
    ], style={
        'width': '100%',
        'maxWidth': '640px',
        'margin': '0 auto',
        'borderCollapse': 'collapse',
        'fontSize': '14px',
    })
    return html.Div([
        html.P(subtitle, style={'textAlign': 'center', 'fontSize': '12px', 'color': '#555', 'marginBottom': '8px'}),
        html.Div(tbl, style={'overflowX': 'auto'}),
    ])


def reorder_table_html(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.P('No data.', style={'textAlign': 'center', 'color': '#666'})
    d = df.copy()
    d[DATE] = pd.to_datetime(d[DATE], errors='coerce')
    d = d.dropna(subset=[DATE])
    last = d[DATE].max()
    d = d[d[DATE] > last - pd.Timedelta(days=14)]
    d = d.sort_values([STORE_ID, PRODUCT_ID, DATE]).groupby([STORE_ID, PRODUCT_ID], group_keys=False).tail(14)
    d = d.sort_values(DATE)
    if REORDER_POINT not in d.columns:
        d[REORDER_POINT] = np.nan
    th_style = {
        'textAlign': 'left',
        'padding': '10px 12px',
        'backgroundColor': '#fff',
        'fontWeight': 'bold',
        'borderBottom': '1px solid #e0e0e0',
    }
    num_th = {**th_style, 'textAlign': 'right'}
    rows_out = []
    for _, row in d.iterrows():
        need = 'Reorder' in str(row.get(REORDER_ALERT, ''))
        if need:
            row_bg = '#fce4ec'
        else:
            row_bg = '#fff'
        if need:
            acolor = '#c62828'
        else:
            acolor = '#1b5e20'
        alert_cell = html.Td(
            str(row.get(REORDER_ALERT, '')),
            style={
                'padding': '8px 12px',
                'color': acolor,
                'fontWeight': 'bold',
                'backgroundColor': row_bg,
            },
        )
        base_cell = {'padding': '8px 12px', 'backgroundColor': row_bg}
        pid = str(row[PRODUCT_ID])
        if pd.notna(row[DATE]):
            ds = row[DATE].strftime('%Y-%m-%d')
        else:
            ds = ''
        inv_txt = f'{float(row[INVENTORY_LEVEL]):,.0f}'
        if pd.notna(row.get(PREDICTED_DEMAND)):
            pred_txt = f'{int(round(float(row[PREDICTED_DEMAND])))}'
        else:
            pred_txt = '—'
        if pd.notna(row.get(REORDER_POINT)):
            rop_txt = f'{int(round(float(row[REORDER_POINT])))}'
        else:
            rop_txt = '—'
        rows_out.append(html.Tr([
            html.Td(pid, style=base_cell),
            html.Td(ds, style=base_cell),
            html.Td(inv_txt, style={**base_cell, 'textAlign': 'right'}),
            html.Td(pred_txt, style={**base_cell, 'textAlign': 'right'}),
            html.Td(rop_txt, style={**base_cell, 'textAlign': 'right'}),
            alert_cell,
        ]))
    tbl = html.Table([
        html.Thead(html.Tr([
            html.Th('product_id', style=th_style),
            html.Th('date', style=th_style),
            html.Th('inventory_level', style=num_th),
            html.Th('predicted_demand', style=num_th),
            html.Th('reorder_point', style=num_th),
            html.Th('reorder_alert', style={**th_style, 'textAlign': 'center'}),
        ])),
        html.Tbody(rows_out),
    ], style={
        'width': '100%',
        'maxWidth': '920px',
        'margin': '0 auto',
        'borderCollapse': 'collapse',
        'fontSize': '14px',
        'fontFamily': "Georgia, 'Times New Roman', serif",
    })
    return html.Div(tbl, style={'overflowX': 'auto'})


def make_layout():
    hdr = html.H1(
        'Smart Inventory Demand Forecasting',
        style={'textAlign': 'center', 'marginBottom': '16px'},
    )
    load_btn = html.Button(
        'Load from database / default CSV',
        id='btn-load-default',
        n_clicks=0,
        style={
            'padding': '8px 16px',
            'borderRadius': '6px',
            'border': '1px solid #2e7d32',
            'backgroundColor': '#e8f5e9',
            'cursor': 'pointer',
            'marginRight': '8px',
        },
    )
    reload_btn = html.Button(
        'Reload',
        id='btn-reload-db',
        n_clicks=0,
        style={'padding': '8px 16px', 'borderRadius': '6px', 'border': '1px solid #ccc', 'cursor': 'pointer'},
    )
    btn_row = html.Div([load_btn, reload_btn], style={'textAlign': 'center', 'marginBottom': '12px'})
    startup_timer = dcc.Interval(id='startup-load', interval=1, n_intervals=0, max_intervals=1)
    poll_timer = dcc.Interval(id='live-poll', interval=20000, n_intervals=0)
    product_dd = dcc.Dropdown(
        id='product-filter',
        options=[{'label': 'All Products', 'value': 'ALL'}],
        value='ALL',
        clearable=False,
        style={'width': '100%', 'maxWidth': '420px', 'margin': '0 auto'},
    )
    product_block = html.Div([
        html.Label('Product', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '6px'}),
        product_dd,
    ], style={'maxWidth': '440px', 'margin': '0 auto 20px', 'textAlign': 'center'})
    upload_box = html.Div(
        style={'width': '92%', 'maxWidth': '720px', 'margin': '12px auto'},
        children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and drop or ', html.A('select CSV')]),
                style={
                    'width': '100%',
                    'padding': '14px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '6px',
                    'textAlign': 'center',
                },
                multiple=False,
            ),
        ],
    )
    status = html.Div(id='upload-status', style={'textAlign': 'center', 'margin': '8px', 'color': 'green'})
    store = dcc.Store(id='results-store')
    last_ct = dcc.Store(id='last-row-count', data=None)
    kpis = html.Div(
        id='kpi-row',
        style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'justifyContent': 'center',
            'maxWidth': '1100px',
            'margin': '16px auto',
        },
    )
    g1 = dcc.Graph(id='graph-sales-trend', style={'maxWidth': '1000px', 'margin': '0 auto 24px'})
    g2 = dcc.Graph(id='graph-inventory-levels', style={'maxWidth': '1000px', 'margin': '0 auto 24px'})
    perf_block = html.Div([
        html.H3('Model Performance (Actual vs Predicted)', style=GRAPH_SECTION_TITLE),
        dcc.Graph(id='graph-demand-forecast', style={'maxWidth': '1000px', 'margin': '0 auto 8px'}),
    ], style={'maxWidth': '1000px', 'margin': '0 auto 24px'})
    week_block = html.Div([
        html.H3('Next 7-Day Demand Forecast', style=GRAPH_SECTION_TITLE),
        dcc.Graph(id='graph-next-7-forecast', style={'maxWidth': '1000px', 'margin': '0 auto 8px'}),
    ], style={'maxWidth': '1000px', 'margin': '0 auto 24px'})
    cal_wrap = html.Div([
        html.H3('Next 7-day forecast — exact values (table)', style=GRAPH_SECTION_TITLE),
        html.Div(id='calendar-forecast-table'),
    ], style={'maxWidth': '1000px', 'margin': '0 auto 24px'})
    reorder_wrap = html.Div([
        html.H3('Reorder alerts', style={'textAlign': 'center', 'marginBottom': '12px'}),
        html.Div(id='reorder-table'),
    ], style={'maxWidth': '720px', 'margin': '0 auto 32px'})
    return html.Div([
        hdr,
        btn_row,
        startup_timer,
        poll_timer,
        product_block,
        upload_box,
        status,
        store,
        last_ct,
        kpis,
        g1,
        g2,
        perf_block,
        week_block,
        cal_wrap,
        reorder_wrap,
    ])


app.layout = make_layout()


def empty_after_failed_load(last_count):
    return ('', None, [{'label': 'All Products', 'value': 'ALL'}], 'ALL', True, last_count)


@app.callback(
    Output('upload-status', 'children'),
    Output('results-store', 'data'),
    Output('product-filter', 'options'),
    Output('product-filter', 'value'),
    Output('product-filter', 'disabled'),
    Output('last-row-count', 'data'),
    Input('upload-data', 'contents'),
    Input('btn-load-default', 'n_clicks'),
    Input('btn-reload-db', 'n_clicks'),
    Input('startup-load', 'n_intervals'),
    Input('live-poll', 'n_intervals'),
    State('upload-data', 'filename'),
    State('last-row-count', 'data'),
    prevent_initial_call=True,
)
def on_data_in(contents, n_def, n_rel, _n_startup, n_live, filename, last_count):
    ctx = dash.callback_context
    who = ctx.triggered[0]['prop_id'].split('.')[0]

    if who == 'startup-load':
        try:
            df = load_data()
        except Exception as e:
            return (f'Startup load error: {e}', *empty_after_failed_load(None)[1:])
        payload, err = build_payload_from_df(df, persist_forecasts=False)
        if err:
            return (err, None, *empty_after_failed_load(None)[2:])
        opts = product_dropdown_options(payload)
        try:
            if CSV_SIMPLE.exists():
                cnt = len(pd.read_csv(CSV_SIMPLE))
            else:
                cnt = None
        except Exception:
            cnt = None
        return ('Loaded default dataset.', payload, opts, 'ALL', False, cnt)

    if who == 'live-poll':
        if not CSV_SIMPLE.exists():
            raise PreventUpdate
        try:
            cur = len(pd.read_csv(CSV_SIMPLE))
        except Exception:
            raise PreventUpdate
        if last_count is not None and cur == last_count:
            raise PreventUpdate
        df = load_data()
        payload, err = build_payload_from_df(df, persist_forecasts=True)
        if err:
            raise PreventUpdate
        opts = product_dropdown_options(payload)
        return (f'Auto-refreshed ({cur} rows in CSV).', payload, opts, 'ALL', False, cur)

    if who in ('btn-load-default', 'btn-reload-db'):
        try:
            df = load_data()
        except Exception as e:
            return (f'Load error: {e}', None, [{'label': 'All Products', 'value': 'ALL'}], 'ALL', True, last_count)
        save_db = who == 'btn-load-default'
        payload, err = build_payload_from_df(df, persist_forecasts=save_db)
        if err:
            return (err, None, [{'label': 'All Products', 'value': 'ALL'}], 'ALL', True, last_count)
        opts = product_dropdown_options(payload)
        msg = 'Loaded data.'
        if save_db:
            msg += ' Predictions saved to database.'
        try:
            if CSV_SIMPLE.exists():
                cnt = len(pd.read_csv(CSV_SIMPLE))
            else:
                cnt = last_count
        except Exception:
            cnt = last_count
        return (msg, payload, opts, 'ALL', False, cnt)

    if who == 'upload-data':
        if contents is None:
            raise PreventUpdate
        _, b64 = contents.split(',')
        raw = base64.b64decode(b64)
        try:
            df = pd.read_csv(io.StringIO(raw.decode('utf-8')))
        except Exception:
            return ('Error reading file.', None, [{'label': 'All Products', 'value': 'ALL'}], 'ALL', True, last_count)
        df = ensure_training_columns(df)
        if STORE_ID not in df.columns:
            return ('CSV must include store_id.', None, [{'label': 'All Products', 'value': 'ALL'}], 'ALL', True, last_count)
        payload, err = build_payload_from_df(df, persist_forecasts=False)
        if err:
            return (err, None, [{'label': 'All Products', 'value': 'ALL'}], 'ALL', True, last_count)
        opts = product_dropdown_options(payload)
        return (f"File '{filename}' processed.", payload, opts, 'ALL', False, last_count)

    raise PreventUpdate


def kpi_row_placeholder():
    c = CARD_STYLE
    return html.Div([
        html.Div([
            html.Div('Total Products', style={'fontSize': '13px', 'color': '#666'}),
            html.Div('—', style={'fontSize': '22px', 'fontWeight': 'bold'}),
        ], style=c),
        html.Div([
            html.Div('Total Sales (Last 30 days)', style={'fontSize': '13px', 'color': '#666'}),
            html.Div('—', style={'fontSize': '22px', 'fontWeight': 'bold'}),
        ], style=c),
        html.Div([
            html.Div('Reorder needed (units)', style={'fontSize': '13px', 'color': '#666'}),
            html.Div('—', style={'fontSize': '22px', 'fontWeight': 'bold'}),
        ], style=c),
        html.Div([
            html.Div('Total predicted demand (next 7 days)', style={'fontSize': '13px', 'color': '#666'}),
            html.Div('—', style={'fontSize': '22px', 'fontWeight': 'bold'}),
        ], style=c),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})


@app.callback(
    Output('kpi-row', 'children'),
    Output('graph-sales-trend', 'figure'),
    Output('graph-inventory-levels', 'figure'),
    Output('graph-demand-forecast', 'figure'),
    Output('graph-next-7-forecast', 'figure'),
    Output('reorder-table', 'children'),
    Output('calendar-forecast-table', 'children'),
    Input('product-filter', 'value'),
    Input('results-store', 'data'),
)
def on_product_view(product_value, data):
    blank = go.Figure()
    empty_kpis = kpi_row_placeholder()

    if not data or 'records' not in data:
        return (
            empty_kpis,
            blank,
            blank,
            blank,
            blank,
            html.P('Load data to view the dashboard.', style={'textAlign': 'center'}),
            html.P(
                'Load data to see the table of exact 7-day mean forecast values.',
                style={'textAlign': 'center', 'color': '#666'},
            ),
        )

    df_full = pd.DataFrame(data['records'])
    df_full[DATE] = pd.to_datetime(df_full[DATE], errors='coerce')
    df_full = df_full.dropna(subset=[DATE])
    df = filter_by_product(df_full, product_value or 'ALL')

    if df.empty:
        return (
            empty_kpis,
            blank,
            blank,
            blank,
            blank,
            html.P('No rows for this product.', style={'textAlign': 'center'}),
            calendar_table_block(None, product_scope=product_value or 'ALL'),
        )

    meta = data.get('meta') or {}
    n_products = int(meta.get('n_products') or df_full[PRODUCT_ID].nunique())
    if product_value and product_value != 'ALL':
        n_products = 1

    dmax = df[DATE].max()
    from_day = dmax - pd.Timedelta(days=30)
    if UNITS_SOLD in df.columns:
        sales_30 = float(df.loc[df[DATE] >= from_day, UNITS_SOLD].sum())
    else:
        sales_30 = float('nan')

    snap = df[df[DATE] == dmax]
    reorder_needed_units = reorder_gap_units(snap)
    pv = product_value or 'ALL'
    raw_for_kpi = raw_slice_to_df(data.get('raw_slice'), pv)

    cal_tbl = None
    if pv == 'ALL' and data.get('calendar_forecast_agg'):
        cal_tbl = pd.DataFrame(data['calendar_forecast_agg'])
    if cal_tbl is None or cal_tbl.empty:
        cal_tbl = seven_day_forecast_table(raw_for_kpi)

    pred_7d = 0
    try:
        if cal_tbl is not None and cal_tbl['mean_predicted_demand'].notna().any():
            pred_7d = int(np.rint(cal_tbl['mean_predicted_demand'].astype(float).fillna(0).sum()))
        elif not snap.empty:
            pred_7d = int(np.rint(float(pd.to_numeric(snap[PREDICTED_DEMAND], errors='coerce').fillna(0).sum())))
    except Exception:
        if not snap.empty:
            pred_7d = int(np.rint(float(pd.to_numeric(snap[PREDICTED_DEMAND], errors='coerce').fillna(0).sum())))

    c = CARD_STYLE
    if sales_30 == sales_30:
        sales_txt = f'{sales_30:,.0f}'
    else:
        sales_txt = '—'
    if reorder_needed_units > 0:
        rcolor = '#c62828'
    else:
        rcolor = '#2e7d32'

    kpi = html.Div([
        html.Div([
            html.Div('Total Products', style={'fontSize': '13px', 'color': '#666'}),
            html.Div(str(n_products), style={'fontSize': '22px', 'fontWeight': 'bold'}),
        ], style=c),
        html.Div([
            html.Div('Total Sales (Last 30 days)', style={'fontSize': '13px', 'color': '#666'}),
            html.Div(sales_txt, style={'fontSize': '22px', 'fontWeight': 'bold'}),
        ], style=c),
        html.Div([
            html.Div('Reorder needed (units)', style={'fontSize': '13px', 'color': '#666'}),
            html.Div(f'{reorder_needed_units:,}', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': rcolor}),
        ], style=c),
        html.Div([
            html.Div('Total predicted demand (next 7 days)', style={'fontSize': '13px', 'color': '#666'}),
            html.Div(f'{pred_7d:,}', style={'fontSize': '22px', 'fontWeight': 'bold'}),
        ], style=c),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})

    fig_sales = chart_sales_trend(df)
    fig_inv = chart_inventory_bars(df)
    fig_perf = chart_actual_vs_pred(df)
    if pv == 'ALL':
        pre_cal = data.get('calendar_forecast_agg')
    else:
        pre_cal = None
    fig_next7 = chart_next_week(data.get('raw_slice'), pv, pre_cal)
    tbl = reorder_table_html(df)
    if cal_tbl is not None and not cal_tbl.empty:
        cal_div = calendar_table_block(cal_tbl, product_scope=pv)
    else:
        cal_div = calendar_table_block(None, product_scope=pv)

    return (kpi, fig_sales, fig_inv, fig_perf, fig_next7, tbl, cal_div)


if __name__ == '__main__':
    app.run(debug=True)
