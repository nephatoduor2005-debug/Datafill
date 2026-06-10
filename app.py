import os
import uuid
import io
import base64
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html as plotly_to_html
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.interpolate import interp1d, CubicSpline
import requests

app = Flask(__name__)
app.secret_key = 'datafill-production-secret-key-change-this'
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------- Per‑session storage ----------
client_data = {}
stock_cache = {}
CACHE_LIMIT_SECONDS = 300

# Fix yfinance 403 with a proper session
yf_session = requests.Session()
yf_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# ---------- Session Helpers ----------
def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_data():
    sid = get_session_id()
    if sid not in client_data:
        client_data[sid] = {
            'is_timeseries': False,
            'date_col': None,
            'price_col': None,
            'original_missing_dates': [],
            'raw_file_path': None,
            'columns': [],
            'selected_x': None,
            'selected_y': None,
            'data_loaded': False,
            'client_name': '',
            'ticker': None,
        }
    return client_data[sid]

# ============================================================
# INTERPOLATION ENGINE
# ============================================================
def newton_gregory_forward(x_table, y_table, x):
    n = len(y_table)
    h = x_table[1] - x_table[0]
    if not np.allclose(np.diff(x_table), h, rtol=1e-5):
        raise ValueError("x-values must be equally spaced")
    u = (x - x_table[0]) / h
    diff = np.zeros((n, n))
    diff[:, 0] = y_table
    for i in range(1, n):
        for j in range(n - i):
            diff[j, i] = diff[j+1, i-1] - diff[j, i-1]
    result = y_table[0]
    u_term = 1.0
    factorial = 1
    terms = 0
    forward_diffs = [y_table[0]]
    for i in range(1, n):
        u_term *= (u - (i - 1))
        factorial *= i
        term = (u_term / factorial) * diff[0, i]
        result += term
        forward_diffs.append(diff[0, i])
        terms += 1
        if abs(diff[0, i]) < 1e-12:
            break
    return result, u, forward_diffs, terms, h

def newton_gregory_backward(x_table, y_table, x):
    n = len(y_table)
    h = x_table[1] - x_table[0]
    if not np.allclose(np.diff(x_table), h, rtol=1e-5):
        raise ValueError("x-values must be equally spaced")
    v = (x - x_table[-1]) / h
    diff = np.zeros((n, n))
    diff[:, 0] = y_table
    for i in range(1, n):
        for j in range(n - i):
            diff[j, i] = diff[j+1, i-1] - diff[j, i-1]
    backward_diffs = [y_table[-1]]
    for i in range(1, n):
        backward_diffs.append(diff[n-1-i, i])
    result = y_table[-1]
    v_term = 1.0
    factorial = 1
    terms = 0
    for i in range(1, n):
        v_term *= (v + (i - 1))
        factorial *= i
        term = (v_term / factorial) * backward_diffs[i]
        result += term
        terms += 1
        if abs(backward_diffs[i]) < 1e-12:
            break
    return result, v, backward_diffs, terms, h

def interpolate_ng(x_table, y_table, x, method='auto'):
    n = len(y_table)
    h = x_table[1] - x_table[0]
    u = (x - x_table[0]) / h
    extrap_warning = u < 0 or u > n-1
    if method == 'forward':
        result, u, diffs, terms, step = newton_gregory_forward(x_table, y_table, x)
        return result, u, diffs, terms, step, 'forward', extrap_warning
    elif method == 'backward':
        v = (x - x_table[-1]) / h
        result, v, diffs, terms, step = newton_gregory_backward(x_table, y_table, x)
        return result, v, diffs, terms, step, 'backward', extrap_warning
    else:  # auto
        if u <= n/2:
            result, u, diffs, terms, step = newton_gregory_forward(x_table, y_table, x)
            return result, u, diffs, terms, step, 'forward', extrap_warning
        else:
            result, v, diffs, terms, step = newton_gregory_backward(x_table, y_table, x)
            return result, v, diffs, terms, step, 'backward', extrap_warning

def interpolate_linear(x_table, y_table, x):
    f = interp1d(x_table, y_table, kind='linear', fill_value='extrapolate')
    return float(f(x)), 0, [], 0, 0, 'linear', False

def interpolate_spline(x_table, y_table, x):
    cs = CubicSpline(x_table, y_table, extrapolate=True)
    return float(cs(x)), 0, [], 0, 0, 'cubic_spline', False

def universal_interpolate(x_table, y_table, x, method='auto', allow_extrap=True):
    if method in ['ng_auto', 'ng_forward', 'ng_backward']:
        h = x_table[1] - x_table[0]
        if not np.allclose(np.diff(x_table), h, rtol=1e-5):
            raise ValueError("x-values are not equally spaced – use Linear or Spline instead.")
        ng_method = method.replace('ng_', '')
        return interpolate_ng(x_table, y_table, x, ng_method)
    elif method == 'linear':
        return interpolate_linear(x_table, y_table, x)
    elif method == 'spline':
        return interpolate_spline(x_table, y_table, x)
    else:  # auto
        try:
            h = x_table[1] - x_table[0]
            if np.allclose(np.diff(x_table), h, rtol=1e-5):
                return interpolate_ng(x_table, y_table, x, 'auto')
            else:
                return interpolate_linear(x_table, y_table, x)
        except:
            return interpolate_linear(x_table, y_table, x)

# ============================================================
# DIFFERENCE TABLE
# ============================================================
def build_difference_table(x_vals, y_vals, max_rows=8, max_diffs=6):
    n = len(y_vals)
    diff = np.zeros((n, n))
    diff[:, 0] = y_vals
    for i in range(1, n):
        for j in range(n - i):
            diff[j, i] = diff[j+1, i-1] - diff[j, i-1]
    headers = ['x', 'f(x)'] + [f'Δ{i}f' for i in range(1, max_diffs+1)]
    rows = []
    for i in range(min(n, max_rows)):
        row = [f"{x_vals[i]:.4f}", f"{y_vals[i]:.4f}"]
        for j in range(1, min(max_diffs+1, n-i)):
            row.append(f"{diff[i, j]:.6f}")
        while len(row) < len(headers):
            row.append("")
        rows.append(row)
    return headers, rows

# ============================================================
# TIME‑SERIES HELPERS
# ============================================================
def prepare_time_series(df, date_col, price_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[price_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df['index_num'] = range(len(df))
    return df[['index_num', date_col, price_col]]

def date_to_fractional_index(df, target_date):
    target = pd.to_datetime(target_date)
    prev = df[df['Date'] <= target].iloc[-1] if any(df['Date'] <= target) else df.iloc[0]
    next_ = df[df['Date'] >= target].iloc[0] if any(df['Date'] >= target) else df.iloc[-1]
    if prev['Date'] == target:
        return prev['index_num'], True
    total = (next_['Date'] - prev['Date']).days
    if total == 0:
        return prev['index_num'], False
    frac = (target - prev['Date']).days / total
    return prev['index_num'] + frac * (next_['index_num'] - prev['index_num']), False

# ============================================================
# CHART GENERATION (Plotly)
# ============================================================
def generate_chart(df, x_col='x', y_col='y', chart_type='bar', dark=True):
    x = df[x_col].tolist()
    y = df[y_col].tolist()
    fig = go.Figure()
    
    if chart_type == 'bar':
        fig.add_trace(go.Bar(
            x=x, y=y,
            marker_color='#6c5ce7',
            marker_line_color='#4a00e0',
            marker_line_width=1.5,
            name='Data'
        ))
    elif chart_type == 'line':
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            line=dict(color='#6c5ce7', width=2.5),
            marker=dict(color='#4a00e0', size=6),
            name='Data'
        ))
    elif chart_type == 'area':
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#6c5ce7', width=2),
            fillcolor='rgba(108,92,231,0.3)',
            name='Data'
        ))
    elif chart_type == 'scatter':
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(color='#6c5ce7', size=8, opacity=0.7),
            name='Data'
        ))
    
    # Highlight interpolated points if available
    if 'interpolated' in df.columns:
        interp_df = df[df['interpolated'] == True]
        if not interp_df.empty:
            fig.add_trace(go.Scatter(
                x=interp_df[x_col].tolist(),
                y=interp_df[y_col].tolist(),
                mode='markers',
                marker=dict(color='#f59e0b', size=12, symbol='star'),
                name='Filled Points'
            ))
    
    template = 'plotly_dark' if dark else 'plotly_white'
    fig.update_layout(
        title=f"{chart_type.title()} Chart: {y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template=template,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return plotly_to_html(fig, full_html=False, include_plotlyjs='cdn')

# ============================================================
# SUMMARY STATISTICS
# ============================================================
def generate_summary(df, y_col):
    y_clean = df[y_col].dropna()
    missing = df[y_col].isna().sum()
    return {
        'count': len(y_clean),
        'mean': round(y_clean.mean(), 4) if len(y_clean) > 0 else None,
        'median': round(y_clean.median(), 4) if len(y_clean) > 0 else None,
        'std': round(y_clean.std(), 4) if len(y_clean) > 1 else None,
        'min': round(y_clean.min(), 4) if len(y_clean) > 0 else None,
        'max': round(y_clean.max(), 4) if len(y_clean) > 0 else None,
        'missing': missing,
        'total': len(df)
    }

# ============================================================
# MATPLOTLIB CHART (for PNG download)
# ============================================================
def create_matplotlib_chart(df, x_col, y_col, date_x=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[x_col], df[y_col], 'o-', color='#4f46e5', linewidth=2,
            markersize=8 if not date_x else 5,
            markerfacecolor='#7c3aed', markeredgecolor='white', markeredgewidth=1.5)
    ax.set_title(f"DataFill — {y_col} vs {x_col}", fontsize=16, fontweight='bold', color='#1a1a2e')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    ax.text(0.99, 0.01, 'DataFill', transform=ax.transAxes,
            fontsize=10, color='#ccc', ha='right', va='bottom', style='italic')
    if date_x:
        plt.xticks(rotation=45)
    plt.tight_layout()
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close(fig)
    return img_bytes

# ============================================================
# ROUTES
# ============================================================

# ---------- Landing Page ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    get_session_id()
    if request.method == 'POST':
        # File upload
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            try:
                df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
                if len(df.columns) < 2:
                    raise ValueError("File must have at least 2 columns")
                sid = get_session_id()
                raw_path = os.path.join(UPLOAD_FOLDER, f'raw_{sid}.csv')
                df.to_csv(raw_path, index=False)
                data = get_data()
                data['raw_file_path'] = raw_path
                data['columns'] = list(df.columns)
                data['selected_x'] = df.columns[0]
                data['selected_y'] = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                data['is_timeseries'] = False
                return redirect(url_for('data'))
            except Exception as e:
                return render_template('index.html', error=str(e))

        # Manual data entry
        elif 'manual_data' in request.form and request.form['manual_data'].strip():
            raw_text = request.form['manual_data'].strip()
            try:
                lines = raw_text.split('\n')
                if not lines:
                    raise ValueError("No data entered")
                header = lines[0].split(',') if ',' in lines[0] else lines[0].split('\t')
                rows = []
                for line in lines[1:]:
                    if line.strip():
                        vals = line.split(',') if ',' in line else line.split('\t')
                        rows.append(vals)
                df = pd.DataFrame(rows, columns=header)
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                raise ValueError("Could not parse manual data. Use comma or tab separated lines. First line = column names.")
            if len(df.columns) < 2:
                raise ValueError("You must have at least 2 columns (X and Y).")
            sid = get_session_id()
            raw_path = os.path.join(UPLOAD_FOLDER, f'raw_{sid}.csv')
            df.to_csv(raw_path, index=False)
            data = get_data()
            data['raw_file_path'] = raw_path
            data['columns'] = list(df.columns)
            data['selected_x'] = df.columns[0]
            data['selected_y'] = df.columns[1]
            data['is_timeseries'] = False
            data['client_name'] = request.form.get('client_name', '').strip()
            return redirect(url_for('data'))
    return render_template('index.html')

# ---------- Data Workspace ----------
@app.route('/data', methods=['GET', 'POST'])
def data():
    sid = get_session_id()
    data = get_data()
    if not data.get('raw_file_path') or not os.path.exists(data.get('raw_file_path', '')):
        return redirect(url_for('index'))

    table_html = chart_html = error_msg = None
    interp_result = interp_x = None
    method_used = None
    accuracy_note = warning_msg = None
    diff_headers = diff_rows = None
    selected_method = 'ng_auto'
    is_timeseries = data.get('is_timeseries', False)
    columns = data.get('columns', [])
    selected_x = data.get('selected_x', '')
    selected_y = data.get('selected_y', '')
    extrap_warning = False

    if request.method == 'POST':
        try:
            # Column mapping confirmation
            if 'confirm_columns' in request.form:
                x_col = request.form.get('x_col', selected_x)
                y_col = request.form.get('y_col', selected_y)
                data['selected_x'] = x_col
                data['selected_y'] = y_col
                df = pd.read_csv(data['raw_file_path'])
                try:
                    pd.to_datetime(df[x_col])
                    is_date = True
                except:
                    is_date = False

                if is_date:
                    data['is_timeseries'] = True
                    data['date_col'] = x_col
                    data['price_col'] = y_col
                    display_df = df[[x_col, y_col]].dropna(subset=[y_col]).sort_values(x_col)
                    display_df[x_col] = pd.to_datetime(display_df[x_col])
                    table_html = display_df.head(1000).to_html(classes='table', index=False)
                    fig = px.line(display_df, x=x_col, y=y_col,
                                  title=f"DataFill — {y_col} vs {x_col}", markers=True)
                    fig.update_layout(template="plotly_white")
                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    ts_df = prepare_time_series(df, x_col, y_col)
                    processed_path = os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv')
                    ts_df.to_csv(processed_path, index=False)
                    diff_headers, diff_rows = build_difference_table(ts_df['index_num'].tolist(), ts_df[y_col].tolist())
                else:
                    data['is_timeseries'] = False
                    df = df[[x_col, y_col]].dropna(subset=[y_col]).sort_values(x_col)
                    processed_path = os.path.join(UPLOAD_FOLDER, f'data_{sid}.csv')
                    df.to_csv(processed_path, index=False)
                    table_html = df.head(1000).to_html(classes='table', index=False)
                    fig = px.line(df, x=x_col, y=y_col,
                                  title=f"DataFill — {y_col} vs {x_col}", markers=True)
                    fig.update_layout(template="plotly_white")
                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    try:
                        h = df[x_col].iloc[1] - df[x_col].iloc[0]
                        if np.allclose(np.diff(df[x_col]), h, rtol=1e-5):
                            diff_headers, diff_rows = build_difference_table(df[x_col].tolist(), df[y_col].tolist())
                    except:
                        pass
                data['data_loaded'] = True

            # Interpolation request
            elif 'interp_x' in request.form and request.form['interp_x'].strip():
                raw_x = request.form['interp_x'].strip()
                interp_method = request.form.get('method', 'ng_auto')
                days_offset = request.form.get('days_offset', '').strip()
                allow_extrap = request.form.get('allow_extrap', '0') == '1'
                selected_method = interp_method
                is_ts = data.get('is_timeseries', False)

                if is_ts:
                    df_ts = pd.read_csv(os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv'))
                    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
                    if days_offset:
                        target_date = df_ts['Date'].max() + timedelta(days=int(days_offset))
                    else:
                        target_date = pd.to_datetime(raw_x)
                    frac_idx, exact = date_to_fractional_index(df_ts, target_date)
                    x_vals = df_ts['index_num'].tolist()
                    y_vals = df_ts[data['price_col']].tolist()
                    result, param, diffs, terms, step, method_used, extrap_warning = interpolate_ng(
                        x_vals, y_vals, frac_idx, 'auto')
                    interp_result = result
                    interp_x = target_date.strftime('%Y-%m-%d')
                    accuracy_note = f"Method: Newton‑Gregory (index {frac_idx:.4f})"
                else:
                    interp_x = float(raw_x)
                    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, f'data_{sid}.csv'))
                    x_vals = df[data['selected_x']].tolist()
                    y_vals = df[data['selected_y']].tolist()
                    result, _, _, _, _, method_used, extrap_warning = universal_interpolate(
                        x_vals, y_vals, interp_x, interp_method, allow_extrap)
                    interp_result = result
                    accuracy_note = f"Method: {method_used.replace('_',' ').title()}"
                    if method_used in ['forward', 'backward']:
                        accuracy_note += " (equally spaced)"
                    if extrap_warning:
                        warning_msg = "⚠️ Extrapolation (outside known range) – accuracy not guaranteed."

        except Exception as e:
            error_msg = str(e)

    # GET: rebuild display
    if request.method == 'GET' and data.get('data_loaded'):
        is_ts = data['is_timeseries']
        if is_ts and os.path.exists(os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv')):
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv'))
            df['Date'] = pd.to_datetime(df['Date'])
            price_col = data['price_col']
            table_html = df.head(1000).to_html(classes='table', index=False)
            fig = px.line(df, x='Date', y=price_col,
                          title=f"DataFill — {price_col} vs Date", markers=True)
            fig.update_layout(template="plotly_white")
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            diff_headers, diff_rows = build_difference_table(df['index_num'].tolist(), df[price_col].tolist())
        elif not is_ts and os.path.exists(os.path.join(UPLOAD_FOLDER, f'data_{sid}.csv')):
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, f'data_{sid}.csv'))
            x_col = data['selected_x']
            y_col = data['selected_y']
            table_html = df.head(1000).to_html(classes='table', index=False)
            fig = px.line(df, x=x_col, y=y_col,
                          title=f"DataFill — {y_col} vs {x_col}", markers=True)
            fig.update_layout(template="plotly_white")
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            try:
                h = df[x_col].iloc[1] - df[x_col].iloc[0]
                if np.allclose(np.diff(df[x_col]), h, rtol=1e-5):
                    diff_headers, diff_rows = build_difference_table(df[x_col].tolist(), df[y_col].tolist())
            except:
                pass

    return render_template('data.html',
                           table=table_html, chart=chart_html,
                           result=interp_result, interp_x=interp_x,
                           method_used=method_used,
                           error=error_msg, accuracy=accuracy_note,
                           warning=warning_msg,
                           diff_headers=diff_headers, diff_rows=diff_rows,
                           selected_method=selected_method,
                           is_timeseries=is_timeseries,
                           columns=columns,
                           selected_x=selected_x, selected_y=selected_y,
                           ticker=data.get('ticker', ''),
                           original_missing_dates=data.get('original_missing_dates', []))

# ---------- Stock Fetch ----------
@app.route('/fetch_stock', methods=['POST'])
def fetch_stock():
    ticker = request.form.get('ticker', 'AAPL').upper().strip()
    period = request.form.get('period', '1mo')
    cache_key = (ticker, period)
    now = datetime.now()

    if cache_key in stock_cache:
        cached = stock_cache[cache_key]
        if (now - cached['timestamp']).total_seconds() < CACHE_LIMIT_SECONDS:
            df = cached['data'].copy()
        else:
            del stock_cache[cache_key]
            df = None
    else:
        df = None

    if df is None:
        try:
            raw = yf.download(ticker, period=period, interval='1d', auto_adjust=False, session=yf_session)
            if raw.empty:
                raise ValueError(f"No data found for {ticker}")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[['Close']].copy()
            df = df.reset_index()
            date_col = None
            for col in df.columns:
                if col.lower() in ['date', 'datetime']:
                    date_col = col
                    break
            if date_col is None:
                date_col = df.columns[0]
            df = df.rename(columns={date_col: 'Date'})
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            stock_cache[cache_key] = {'data': df.copy(), 'timestamp': now}
        except Exception as e:
            return render_template('index.html', error=f"Stock fetch failed: {str(e)}")

    np.random.seed(42)
    n = len(df)
    n_missing = max(1, int(n * 0.2))
    all_indices = df.index.tolist()
    missing_idx = np.random.choice(all_indices, size=n_missing, replace=False)
    missing_dates = df.loc[missing_idx, 'Date'].dt.strftime('%Y-%m-%d').tolist()
    df_clean = df.drop(missing_idx).reset_index(drop=True)

    sid = get_session_id()
    processed_path = os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv')
    df_clean.to_csv(processed_path, index=False)

    data = get_data()
    data['is_timeseries'] = True
    data['date_col'] = 'Date'
    data['price_col'] = 'Close'
    data['ticker'] = ticker
    data['original_missing_dates'] = missing_dates
    data['selected_x'] = 'Date'
    data['selected_y'] = 'Close'
    data['columns'] = ['Date', 'Close']
    data['raw_file_path'] = processed_path
    data['data_loaded'] = True

    return redirect(url_for('data'))

# ---------- Fill Missing (stock gaps) ----------
@app.route('/fill_missing', methods=['GET'])
def fill_missing():
    sid = get_session_id()
    data = get_data()
    if not data.get('is_timeseries') or not os.path.exists(os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv')):
        return "No time‑series data.", 400
    missing_dates = data.get('original_missing_dates', [])
    if not missing_dates:
        return "No missing dates.", 400

    df_ts = pd.read_csv(os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv'))
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    price_col = data['price_col']
    x_vals = df_ts['index_num'].tolist()
    y_vals = df_ts[price_col].tolist()

    filled_rows = []
    for d_str in missing_dates:
        d = pd.to_datetime(d_str)
        frac_idx, exact = date_to_fractional_index(df_ts, d)
        if exact:
            price = df_ts[df_ts['Date'] == d][price_col].values[0]
        else:
            price, _, _, _, _, _, _ = interpolate_ng(x_vals, y_vals, frac_idx, 'auto')
        filled_rows.append({'Date': d_str, 'Estimated_Price': round(price, 4)})

    filled_df = pd.DataFrame(filled_rows)
    original = df_ts[['Date', price_col]].copy()
    original.columns = ['Date', 'Actual_Price']
    original['Date'] = pd.to_datetime(original['Date'])
    filled_df['Date'] = pd.to_datetime(filled_df['Date'])
    full = original.merge(filled_df, on='Date', how='outer').sort_values('Date')
    return send_file(io.BytesIO(full.to_csv(index=False).encode()),
                     mimetype='text/csv', as_attachment=True, download_name='datafill_filled_missing.csv')

# ---------- Full Calendar Fill ----------
@app.route('/fill_gaps')
def fill_gaps():
    sid = get_session_id()
    data = get_data()
    if not data.get('is_timeseries') or not os.path.exists(os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv')):
        return "No time‑series data.", 400
    df_ts = pd.read_csv(os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv'))
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    price_col = data.get('price_col', 'Close')
    start = df_ts['Date'].min()
    end = df_ts['Date'].max()
    all_dates = pd.date_range(start, end, freq='D')
    x_vals = df_ts['index_num'].tolist()
    y_vals = df_ts[price_col].tolist()
    filled = []
    for d in all_dates:
        frac_idx, exact = date_to_fractional_index(df_ts, d)
        if exact:
            price = df_ts[df_ts['Date'] == d][price_col].values[0]
        else:
            price, _, _, _, _, _, _ = interpolate_ng(x_vals, y_vals, frac_idx, 'auto')
        filled.append({'Date': d.strftime('%Y-%m-%d'), 'Estimated_Price': round(price, 4)})
    return send_file(io.BytesIO(pd.DataFrame(filled).to_csv(index=False).encode()),
                     mimetype='text/csv', as_attachment=True, download_name='datafill_clean_timeseries.csv')

# ---------- PNG Download ----------
@app.route('/download/png')
def download_png():
    sid = get_session_id()
    data = get_data()
    if data.get('is_timeseries'):
        path = os.path.join(UPLOAD_FOLDER, f'ts_{sid}.csv')
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        return send_file(create_matplotlib_chart(df, 'Date', data['price_col'], date_x=True),
                         mimetype='image/png', as_attachment=True, download_name='datafill_chart.png')
    else:
        path = os.path.join(UPLOAD_FOLDER, f'data_{sid}.csv')
        df = pd.read_csv(path)
        x_col, y_col = df.columns[0], df.columns[1]
        return send_file(create_matplotlib_chart(df, x_col, y_col),
                         mimetype='image/png', as_attachment=True, download_name='datafill_chart.png')

# ---------- Report Page ----------
@app.route('/report')
def report_page():
    return render_template('report.html')

# ---------- API: Upload Any ----------
@app.route('/upload_any', methods=['POST'])
def upload_any():
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
    except Exception as e:
        return jsonify({'error': f'Failed to read file: {str(e)}'}), 400
    if len(df.columns) < 2:
        return jsonify({'error': 'File must have at least 2 columns'}), 400
    sid = get_session_id()
    raw_path = os.path.join(UPLOAD_FOLDER, f'raw_{sid}.csv')
    df.to_csv(raw_path, index=False)
    data = get_data()
    data['raw_file_path'] = raw_path
    data['columns'] = list(df.columns)
    data['selected_x'] = df.columns[0]
    data['selected_y'] = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return jsonify({'filepath': raw_path, 'columns': list(df.columns),
                    'auto_x': df.columns[0], 'auto_y': df.columns[1] if len(df.columns) > 1 else df.columns[0]})

# ---------- API: Process Any (with Report) ----------
@app.route('/process_any', methods=['POST'])
def process_any_data():
    req_data = request.get_json()
    if not req_data:
        return jsonify({'error': 'Request must be JSON'}), 400
    filepath = req_data.get('filepath')
    x_col = req_data.get('x_col')
    y_col = req_data.get('y_col')
    interp_x = req_data.get('interp_x', None)
    method = req_data.get('method', 'ng_auto')
    chart_type = req_data.get('chart_type', 'bar')

    if not filepath or not x_col or not y_col:
        return jsonify({'error': 'Missing filepath, x_col, or y_col'}), 400

    try:
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to read file: {str(e)}'}), 400

    if x_col not in df.columns or y_col not in df.columns:
        return jsonify({'error': f'Columns not found. Available: {list(df.columns)}'}), 400

    # Keep copy, rename internally
    df = df[[x_col, y_col]].copy()
    df = df.dropna(subset=[y_col]).sort_values(x_col)
    df = df.rename(columns={x_col: 'x', y_col: 'y'})

    # Detect time‑series
    try:
        pd.to_datetime(df['x'])
        is_timeseries = True
        df['x'] = pd.to_datetime(df['x'])
        df = df.sort_values('x').reset_index(drop=True)
        df['index_num'] = range(len(df))
    except:
        is_timeseries = False
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df = df.dropna(subset=['x'])

    # Difference table
    if is_timeseries:
        x_vals = df['index_num'].tolist()
    else:
        x_vals = df['x'].tolist()
    y_vals = df['y'].tolist()
    diff_headers, diff_rows = build_difference_table(x_vals, y_vals)

    # Interpolation
    interp_result = None
    interp_x_display = None
    method_used = None
    accuracy_note = None
    warning_msg = None

    if interp_x is not None and str(interp_x).strip():
        try:
            if is_timeseries:
                target_date = pd.to_datetime(interp_x)
                frac_idx, exact = date_to_fractional_index(df.rename(columns={'x': 'Date'}), target_date)
                result, param, diffs, terms, step, method_used, extrap = interpolate_ng(x_vals, y_vals, frac_idx, 'auto')
                interp_result = result
                interp_x_display = target_date.strftime('%Y-%m-%d')
                accuracy_note = f"Newton‑Gregory (index {frac_idx:.4f})"
            else:
                x_target = float(interp_x)
                result, _, _, _, _, method_used, extrap = universal_interpolate(x_vals, y_vals, x_target, method)
                interp_result = result
                interp_x_display = x_target
                accuracy_note = f"{method_used.replace('_',' ').title()}"
                if extrap:
                    warning_msg = "⚠️ Extrapolation – accuracy not guaranteed."
        except Exception as e:
            warning_msg = f"Interpolation failed: {str(e)}"

    # Summary statistics
    summary = generate_summary(df, 'y')

    # Chart
    chart_html = generate_chart(df, 'x', 'y', chart_type=chart_type, dark=True)

    # Preview table
    preview_df = df[['x', 'y']].head(100)
    preview_html = preview_df.to_html(classes='table', index=False)

    # Build report HTML
    html_report = f"""
    <div style="font-family: 'Inter', sans-serif; max-width:900px; margin:0 auto; padding:20px; background:#f8fafc; border-radius:16px;">
        <div style="background:linear-gradient(135deg,#6c5ce7,#a855f7); color:white; padding:30px; border-radius:14px; margin-bottom:25px;">
            <h2 style="margin:0; font-size:1.8em;">📊 DataFill Report</h2>
            <p style="margin:8px 0 0; opacity:0.9;">Columns: {x_col} → {y_col}</p>
        </div>
        <div style="display:flex; gap:15px; flex-wrap:wrap; margin-bottom:25px;">
            <div style="flex:1; min-width:150px; background:white; padding:20px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:2em; font-weight:700; color:#6c5ce7;">{summary['total']}</div>
                <div style="color:#666;">Total Rows</div>
            </div>
            <div style="flex:1; min-width:150px; background:white; padding:20px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:2em; font-weight:700; color:#059669;">{summary['missing']}</div>
                <div style="color:#666;">Missing</div>
            </div>
            <div style="flex:1; min-width:150px; background:white; padding:20px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:2em; font-weight:700; color:#d97706;">{summary['mean'] or 'N/A'}</div>
                <div style="color:#666;">Mean</div>
            </div>
        </div>
        {f'''<div style="background:linear-gradient(135deg,#eef2ff,#f5f3ff); border-left:4px solid #6c5ce7; padding:18px 22px; border-radius:12px; margin-bottom:20px;">
            <p style="font-size:1.1em;">f({interp_x_display}) = <strong style="font-size:1.3em; color:#6c5ce7;">{interp_result:.6f}</strong></p>
            <p style="color:#666; margin:0;">{accuracy_note}</p>
            {f'<p style="color:#b45309; margin:5px 0 0;">{warning_msg}</p>' if warning_msg else ''}
        </div>''' if interp_result is not None else ''}
        <div style="background:white; padding:20px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05); margin-bottom:20px;">
            <h4 style="margin:0 0 15px;">📊 Chart</h4>
            {chart_html}
        </div>
        <div style="background:white; padding:20px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05); margin-bottom:20px;">
            <h4 style="margin:0 0 15px;">🔢 Forward Difference Table</h4>
            <div style="overflow-x:auto;">
                <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
                    <thead><tr>{"".join(f"<th style='background:#f1f5f9; padding:10px 14px; text-align:center;'>{h}</th>" for h in diff_headers)}</tr></thead>
                    <tbody>{"".join("<tr>" + "".join(f"<td style='padding:8px 14px; border-bottom:1px solid #eee; text-align:center;'>{cell}</td>" for cell in row) + "</tr>" for row in diff_rows)}</tbody>
                </table>
            </div>
        </div>
        <div style="background:white; padding:20px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h4 style="margin:0 0 15px;">📋 Data Preview</h4>
            <div style="overflow-x:auto;">{preview_html}</div>
        </div>
        <div style="text-align:center; margin-top:25px;">
            <button onclick="window.print()" style="padding:12px 24px; background:#6c5ce7; color:white; border:none; border-radius:10px; font-weight:600; cursor:pointer;">🖨️ Print / Save as PDF</button>
        </div>
    </div>
    """

    return jsonify({
        'html': html_report,
        'stats': {
            'total_rows': len(df),
            'missing_values': summary['missing'],
            'diff_rows': len(diff_rows),
            'interpolated_value': interp_result,
            'interp_x': interp_x_display,
            'method_used': method_used.replace('_',' ').title() if method_used else None,
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
