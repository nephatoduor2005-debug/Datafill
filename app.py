from flask import Flask, render_template, request, send_file, redirect, url_for, session
import pandas as pd
import plotly.express as px
import numpy as np
import io
import os
import uuid
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.interpolate import interp1d, CubicSpline

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change to a random string in production

# ---------- Per‑session storage ----------
client_data = {}       # key = session_id, value = dict like old stored_data
stock_cache = {}
CACHE_LIMIT_SECONDS = 300

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
        }
    return client_data[sid]

# ---------- Interpolation functions (unchanged) ----------
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
    if method == 'forward':
        if u < 0 or u > n-1:
            raise ValueError(f"x={x} outside table range")
        return newton_gregory_forward(x_table, y_table, x) + ('forward',)
    elif method == 'backward':
        v = (x - x_table[-1]) / h
        if v > 0 or v < -(n-1):
            raise ValueError(f"x={x} outside table range")
        return newton_gregory_backward(x_table, y_table, x) + ('backward',)
    else:
        if u < 0 or u > n-1:
            raise ValueError(f"x={x} outside table range")
        if u <= n/2:
            return newton_gregory_forward(x_table, y_table, x) + ('forward',)
        else:
            return newton_gregory_backward(x_table, y_table, x) + ('backward',)

def interpolate_linear(x_table, y_table, x):
    f = interp1d(x_table, y_table, kind='linear', fill_value='extrapolate')
    return float(f(x)), 0, [], 0, 0, 'linear'

def interpolate_spline(x_table, y_table, x):
    cs = CubicSpline(x_table, y_table, extrapolate=True)
    return float(cs(x)), 0, [], 0, 0, 'cubic_spline'

def universal_interpolate(x_table, y_table, x, method='auto'):
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
    else:
        try:
            h = x_table[1] - x_table[0]
            if np.allclose(np.diff(x_table), h, rtol=1e-5):
                return interpolate_ng(x_table, y_table, x, 'auto')
            else:
                return interpolate_linear(x_table, y_table, x)
        except:
            return interpolate_linear(x_table, y_table, x)

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

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    # Ensure session exists
    get_session_id()

    if request.method == 'POST':
        # --- File Upload ---
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            try:
                df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
                if len(df.columns) < 2:
                    raise ValueError("File must have at least 2 columns")
                # Save under session
                sid = get_session_id()
                raw_path = f'temp_raw_{sid}.csv'
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

        # --- Manual Data Entry ---
        elif 'manual_data' in request.form and request.form['manual_data'].strip():
            raw_text = request.form['manual_data'].strip()
            # Try comma or tab separated
            try:
                lines = raw_text.split('\n')
                if not lines:
                    raise ValueError("No data entered")
                # Use first line as header
                header = lines[0].split(',') if ',' in lines[0] else lines[0].split('\t')
                rows = []
                for line in lines[1:]:
                    if line.strip():
                        vals = line.split(',') if ',' in line else line.split('\t')
                        rows.append(vals)
                df = pd.DataFrame(rows, columns=header)
                # Convert to numeric if possible
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                raise ValueError("Could not parse manual data. Use comma or tab separated lines. First line = column names.")
            if len(df.columns) < 2:
                raise ValueError("You must have at least 2 columns (X and Y).")
            sid = get_session_id()
            raw_path = f'temp_raw_{sid}.csv'
            df.to_csv(raw_path, index=False)
            data = get_data()
            data['raw_file_path'] = raw_path
            data['columns'] = list(df.columns)
            data['selected_x'] = df.columns[0]
            data['selected_y'] = df.columns[1]
            data['is_timeseries'] = False
            # Store optional client name
            data['client_name'] = request.form.get('client_name', '').strip()
            return redirect(url_for('data'))

    return render_template('index.html')

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

    if request.method == 'POST':
        try:
            # Column mapping confirmation
            if 'confirm_columns' in request.form:
                x_col = request.form.get('x_col', selected_x)
                y_col = request.form.get('y_col', selected_y)
                data['selected_x'] = x_col
                data['selected_y'] = y_col
                df = pd.read_csv(data['raw_file_path'])

                # Detect date
                try:
                    pd.to_datetime(df[x_col])
                    is_date = True
                except:
                    is_date = False

                if is_date:
                    data['is_timeseries'] = True
                    data['date_col'] = x_col
                    data['price_col'] = y_col
                    display_df = df[[x_col, y_col]].copy()
                    display_df[x_col] = pd.to_datetime(display_df[x_col])
                    table_html = display_df.head(1000).to_html(classes='table', index=False)
                    fig = px.line(display_df.dropna(), x=x_col, y=y_col,
                                  title=f"DataFill — {y_col} vs {x_col}", markers=True)
                    fig.update_layout(template="plotly_white")
                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    ts_df = prepare_time_series(df, x_col, y_col)
                    processed_path = f'temp_ts_{sid}.csv'
                    ts_df.to_csv(processed_path, index=False)
                    diff_headers, diff_rows = build_difference_table(ts_df['index_num'].tolist(), ts_df[y_col].tolist())
                else:
                    data['is_timeseries'] = False
                    summary = df.groupby(x_col)[y_col].sum().reset_index()
                    processed_path = f'temp_data_{sid}.csv'
                    summary.to_csv(processed_path, index=False)
                    table_html = summary.head(1000).to_html(classes='table', index=False)
                    fig = px.line(summary, x=x_col, y=y_col,
                                  title=f"DataFill — {y_col} vs {x_col}", markers=True)
                    fig.update_layout(template="plotly_white")
                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    try:
                        h = summary[x_col].iloc[1] - summary[x_col].iloc[0]
                        if np.allclose(np.diff(summary[x_col]), h, rtol=1e-5):
                            diff_headers, diff_rows = build_difference_table(summary[x_col].tolist(), summary[y_col].tolist())
                    except:
                        pass
                data['data_loaded'] = True

            # Interpolation request
            elif 'interp_x' in request.form and request.form['interp_x'].strip():
                raw_x = request.form['interp_x'].strip()
                interp_method = request.form.get('method', 'ng_auto')
                selected_method = interp_method
                is_ts = data.get('is_timeseries', False)

                if is_ts:
                    df_ts = pd.read_csv(f'temp_ts_{sid}.csv')
                    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
                    target_date = pd.to_datetime(raw_x)
                    frac_idx, exact = date_to_fractional_index(df_ts, target_date)
                    x_vals = df_ts['index_num'].tolist()
                    y_vals = df_ts[data['price_col']].tolist()
                    result, param, diffs, terms, step, method_used = interpolate_ng(
                        x_vals, y_vals, frac_idx, 'auto')
                    interp_result = result
                    interp_x = target_date.strftime('%Y-%m-%d')
                    accuracy_note = f"Method: Newton‑Gregory (index {frac_idx:.4f})"
                else:
                    interp_x = float(raw_x)
                    df = pd.read_csv(f'temp_data_{sid}.csv')
                    x_vals = df[data['selected_x']].tolist()
                    y_vals = df[data['selected_y']].tolist()
                    result, _, _, _, _, method_used = universal_interpolate(x_vals, y_vals, interp_x, interp_method)
                    interp_result = result
                    accuracy_note = f"Method: {method_used.replace('_',' ').title()}"
                    if method_used in ['forward', 'backward']:
                        accuracy_note += " (equally spaced)"

        except Exception as e:
            error_msg = str(e)

    # On GET request, rebuild display if already processed
    if request.method == 'GET' and data.get('data_loaded'):
        is_ts = data['is_timeseries']
        if is_ts and os.path.exists(f'temp_ts_{sid}.csv'):
            df = pd.read_csv(f'temp_ts_{sid}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            price_col = data['price_col']
            table_html = df.head(1000).to_html(classes='table', index=False)
            fig = px.line(df, x='Date', y=price_col,
                          title=f"DataFill — {price_col} vs Date", markers=True)
            fig.update_layout(template="plotly_white")
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            diff_headers, diff_rows = build_difference_table(df['index_num'].tolist(), df[price_col].tolist())
        elif not is_ts and os.path.exists(f'temp_data_{sid}.csv'):
            df = pd.read_csv(f'temp_data_{sid}.csv')
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
                           table=table_html,
                           chart=chart_html,
                           result=interp_result,
                           interp_x=interp_x,
                           method_used=method_used,
                           error=error_msg,
                           accuracy=accuracy_note,
                           warning=warning_msg,
                           diff_headers=diff_headers,
                           diff_rows=diff_rows,
                           selected_method=selected_method,
                           is_timeseries=is_timeseries,
                           columns=columns,
                           selected_x=selected_x,
                           selected_y=selected_y,
                           ticker=data.get('ticker', ''),
                           original_missing_dates=data.get('original_missing_dates', []))

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
            raw = yf.download(ticker, period=period, interval='1d', auto_adjust=False)
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
    processed_path = f'temp_ts_{sid}.csv'
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
    data['raw_file_path'] = processed_path  # ensures /data can load it
    data['data_loaded'] = True

    return redirect(url_for('data'))

@app.route('/fill_missing', methods=['GET'])
def fill_missing():
    sid = get_session_id()
    data = get_data()
    if not data.get('is_timeseries') or not os.path.exists(f'temp_ts_{sid}.csv'):
        return "No time‑series data.", 400
    missing_dates = data.get('original_missing_dates', [])
    if not missing_dates:
        return "No missing dates.", 400

    df_ts = pd.read_csv(f'temp_ts_{sid}.csv')
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
            price, _, _, _, _, _ = interpolate_ng(x_vals, y_vals, frac_idx, 'auto')
        filled_rows.append({'Date': d_str, 'Estimated_Price': round(price, 4)})

    filled_df = pd.DataFrame(filled_rows)
    original = df_ts[['Date', price_col]].copy()
    original.columns = ['Date', 'Actual_Price']
    original['Date'] = pd.to_datetime(original['Date'])
    filled_df['Date'] = pd.to_datetime(filled_df['Date'])
    full = original.merge(filled_df, on='Date', how='outer')
    full = full.sort_values('Date')
    csv_data = full.to_csv(index=False)
    return send_file(io.BytesIO(csv_data.encode()), mimetype='text/csv',
                     as_attachment=True, download_name='datafill_filled_missing.csv')

@app.route('/fill_gaps')
def fill_gaps():
    sid = get_session_id()
    data = get_data()
    if not data.get('is_timeseries') or not os.path.exists(f'temp_ts_{sid}.csv'):
        return "No time‑series data.", 400
    df_ts = pd.read_csv(f'temp_ts_{sid}.csv')
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
            price, _, _, _, _, _ = interpolate_ng(x_vals, y_vals, frac_idx, 'auto')
        filled.append({'Date': d.strftime('%Y-%m-%d'), 'Estimated_Price': round(price, 4)})
    filled_df = pd.DataFrame(filled)
    return send_file(io.BytesIO(filled_df.to_csv(index=False).encode()),
                     mimetype='text/csv', as_attachment=True,
                     download_name='datafill_clean_timeseries.csv')

@app.route('/download/csv')
def download_csv():
    sid = get_session_id()
    data = get_data()
    if data.get('is_timeseries'):
        path = f'temp_ts_{sid}.csv'
    else:
        path = f'temp_data_{sid}.csv'
    if not os.path.exists(path):
        return "No data yet", 400
    return send_file(path, as_attachment=True, download_name='datafill_results.csv')

@app.route('/download/png')
def download_png():
    sid = get_session_id()
    data = get_data()
    if data.get('is_timeseries'):
        path = f'temp_ts_{sid}.csv'
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        return send_file(
            create_matplotlib_chart(df, 'Date', data['price_col'], date_x=True),
            mimetype='image/png', as_attachment=True, download_name='datafill_chart.png')
    else:
        path = f'temp_data_{sid}.csv'
        df = pd.read_csv(path)
        x_col, y_col = df.columns[0], df.columns[1]
        return send_file(
            create_matplotlib_chart(df, x_col, y_col),
            mimetype='image/png', as_attachment=True, download_name='datafill_chart.png')

if __name__ == '__main__':
    app.run(debug=True)
