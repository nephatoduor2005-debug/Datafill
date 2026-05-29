from flask import Flask, render_template, request, send_file
import pandas as pd
import plotly.express as px
import numpy as np
import io
import os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf

app = Flask(__name__)

# ============================================
#  DataFill – Fill the gaps. Find the pattern.
#  Newton‑Gregory Interpolation + Stocks
# ============================================

stock_cache = {}
CACHE_LIMIT_SECONDS = 300

stored_data = {
    'is_timeseries': False,
    'date_col': None,
    'price_col': None,
    'original_missing_dates': []
}

# ---------- Newton‑Gregory Core ----------
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

def interpolate(x_table, y_table, x, method='auto'):
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
    table_html = chart_html = error_msg = None
    interp_result = interp_x = None
    method_used = param_value = None
    accuracy_note = warning_msg = None
    diff_headers = diff_rows = None
    method_selected = 'auto'
    is_timeseries = stored_data.get('is_timeseries', False)

    if request.method == 'POST':
        try:
            # --- File Upload ---
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
                if len(df.columns) < 2:
                    raise ValueError("File must have at least 2 columns")
                
                date_col = None
                for col in df.columns:
                    if 'date' in col.lower():
                        date_col = col
                        break
                if date_col is None:
                    try:
                        pd.to_datetime(df.iloc[:,0])
                        date_col = df.columns[0]
                    except:
                        pass
                
                if date_col:
                    price_col = df.columns[1] if df.columns[1] != date_col else df.columns[2]
                    display_df = df[[date_col, price_col]].copy()
                    display_df[date_col] = pd.to_datetime(display_df[date_col])
                    table_html = display_df.to_html(classes='table', index=False)
                    
                    fig = px.line(display_df.dropna(), x=date_col, y=price_col,
                                  title=f"DataFill — {price_col} vs {date_col}", markers=True)
                    fig.update_layout(template="plotly_white")
                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    
                    ts_df = prepare_time_series(df, date_col, price_col)
                    ts_df.to_csv('temp_ts_data.csv', index=False)
                    
                    stored_data['is_timeseries'] = True
                    stored_data['date_col'] = date_col
                    stored_data['price_col'] = price_col
                    stored_data['original_missing_dates'] = []
                    
                    x_vals = ts_df['index_num'].tolist()
                    y_vals = ts_df[price_col].tolist()
                    diff_headers, diff_rows = build_difference_table(x_vals, y_vals)
                else:
                    stored_data['is_timeseries'] = False
                    x_col, y_col = df.columns[0], df.columns[1]
                    summary = df.groupby(x_col)[y_col].sum().reset_index()
                    summary.to_csv('temp_data.csv', index=False)
                    
                    fig = px.line(summary, x=x_col, y=y_col,
                                  title=f"DataFill — {y_col} vs {x_col}", markers=True)
                    fig.update_layout(template="plotly_white")
                    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    table_html = summary.to_html(classes='table', index=False)
                    
                    x_vals = summary[x_col].tolist()
                    y_vals = summary[y_col].tolist()
                    diff_headers, diff_rows = build_difference_table(x_vals, y_vals)

            # --- Interpolation Request ---
            if 'interp_x' in request.form and request.form['interp_x'].strip():
                raw_x = request.form['interp_x'].strip()
                method_selected = request.form.get('method', 'auto')
                is_ts = stored_data.get('is_timeseries', False)
                
                if is_ts:
                    try:
                        target_date = pd.to_datetime(raw_x)
                    except:
                        raise ValueError("Please enter a valid date (YYYY-MM-DD)")
                    
                    df_ts = pd.read_csv('temp_ts_data.csv')
                    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
                    price_col = stored_data['price_col']
                    
                    frac_idx, exact = date_to_fractional_index(df_ts, target_date)
                    x_vals = df_ts['index_num'].tolist()
                    y_vals = df_ts[price_col].tolist()
                    
                    result, param, diffs, terms, step, method_used = interpolate(
                        x_vals, y_vals, frac_idx, method_selected)
                    interp_result = result
                    interp_x = target_date.strftime('%Y-%m-%d')
                    param_value = param
                    
                    accuracy_note = (
                        f"DataFill used <strong>{method_used.title()}</strong> | "
                        f"Index: {frac_idx:.4f} | h=1 trading day | Terms: {terms}"
                    )
                    if exact:
                        accuracy_note += " (exact trading day)"
                    else:
                        accuracy_note += " (interpolated date)"
                    
                    diff_headers, diff_rows = build_difference_table(x_vals, y_vals)
                else:
                    interp_x = float(raw_x)
                    if not os.path.exists('temp_data.csv'):
                        raise ValueError("Please upload a CSV file first")
                    df = pd.read_csv('temp_data.csv')
                    x_vals = df.iloc[:,0].tolist()
                    y_vals = df.iloc[:,1].tolist()
                    
                    result, param, diffs, terms, step, method_used = interpolate(
                        x_vals, y_vals, interp_x, method_selected)
                    interp_result = result
                    param_value = param
                    
                    accuracy_note = (
                        f"DataFill used <strong>{method_used.title()}</strong> | "
                        f"h = {step:.4f} | Terms: {terms} | Parameter = {param:.4f}"
                    )
                    if method_used == 'forward' and param < 0:
                        warning_msg = "⚠️ Extrapolating before first data point."
                    elif method_used == 'backward' and param > 0:
                        warning_msg = "⚠️ Extrapolating beyond last data point."
                    
                    diff_headers, diff_rows = build_difference_table(x_vals, y_vals)

        except Exception as e:
            error_msg = str(e)

    return render_template('index.html',
                           table=table_html,
                           chart=chart_html,
                           result=interp_result,
                           interp_x=interp_x,
                           method_used=method_used,
                           param=param_value,
                           error=error_msg,
                           accuracy=accuracy_note,
                           warning=warning_msg,
                           diff_headers=diff_headers,
                           diff_rows=diff_rows,
                           selected_method=method_selected,
                           is_timeseries=is_timeseries,
                           ticker=stored_data.get('ticker', ''),
                           original_missing_dates=stored_data.get('original_missing_dates', []))


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
            # Detect date column name
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
            return render_template('index.html',
                                   error=f"Stock fetch failed: {str(e)}",
                                   table=None,
                                   chart=None,
                                   result=None,
                                   interp_x=None,
                                   method_used=None,
                                   param=None,
                                   accuracy=None,
                                   warning=None,
                                   diff_headers=None,
                                   diff_rows=None,
                                   selected_method='auto',
                                   is_timeseries=False,
                                   ticker='',
                                   original_missing_dates=[])

    np.random.seed(42)
    n = len(df)
    n_missing = max(1, int(n * 0.2))
    all_indices = df.index.tolist()
    missing_idx = np.random.choice(all_indices, size=n_missing, replace=False)
    missing_dates = df.loc[missing_idx, 'Date'].dt.strftime('%Y-%m-%d').tolist()
    df_clean = df.drop(missing_idx).reset_index(drop=True)

    df_clean.to_csv('temp_ts_data.csv', index=False)

    stored_data['is_timeseries'] = True
    stored_data['date_col'] = 'Date'
    stored_data['price_col'] = 'Close'
    stored_data['ticker'] = ticker
    stored_data['original_missing_dates'] = missing_dates

    table_html = df_clean.to_html(classes='table', index=False)

    fig = px.line(df_clean, x='Date', y='Close',
                  title=f"DataFill — Close of {ticker} (gaps introduced)", markers=True)
    fig.update_layout(template="plotly_white")
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    ts_df = prepare_time_series(df_clean, 'Date', 'Close')
    ts_df.to_csv('temp_ts_data.csv', index=False)

    x_vals = ts_df['index_num'].tolist()
    y_vals = ts_df['Close'].tolist()
    diff_headers, diff_rows = build_difference_table(x_vals, y_vals)

    return render_template('index.html',
                           table=table_html,
                           chart=chart_html,
                           result=None,
                           interp_x=None,
                           method_used=None,
                           param=None,
                           error=None,
                           accuracy=None,
                           warning=None,
                           diff_headers=diff_headers,
                           diff_rows=diff_rows,
                           selected_method='auto',
                           is_timeseries=True,
                           ticker=ticker,
                           original_missing_dates=missing_dates)


@app.route('/fill_missing', methods=['GET'])
def fill_missing():
    if not os.path.exists('temp_ts_data.csv'):
        return "No time‑series data. Fetch stock or upload a file first.", 400
    missing_dates = stored_data.get('original_missing_dates', [])
    if not missing_dates:
        return "No missing dates to fill.", 400

    df_ts = pd.read_csv('temp_ts_data.csv')
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    price_col = stored_data['price_col']

    x_vals = df_ts['index_num'].tolist()
    y_vals = df_ts[price_col].tolist()

    filled_rows = []
    for d_str in missing_dates:
        d = pd.to_datetime(d_str)
        frac_idx, exact = date_to_fractional_index(df_ts, d)
        if exact:
            price = df_ts[df_ts['Date'] == d][price_col].values[0]
        else:
            price, _, _, _, _, _ = interpolate(x_vals, y_vals, frac_idx, method='auto')
        filled_rows.append({'Date': d_str, 'Estimated_Price': round(price, 4)})

    filled_df = pd.DataFrame(filled_rows)
    original = df_ts[['Date', price_col]].copy()
    original.columns = ['Date', 'Actual_Price']

    # Fix date types
    original['Date'] = pd.to_datetime(original['Date'])
    filled_df['Date'] = pd.to_datetime(filled_df['Date'])

    # Merge
    full = original.merge(filled_df, on='Date', how='outer')
    full = full.sort_values('Date')

    csv_data = full.to_csv(index=False)
    return send_file(io.BytesIO(csv_data.encode()), mimetype='text/csv',
                     as_attachment=True, download_name='datafill_filled_missing.csv')


@app.route('/fill_gaps')
def fill_gaps():
    if not os.path.exists('temp_ts_data.csv'):
        return "No time‑series data. Upload a file or fetch stock first.", 400

    df_ts = pd.read_csv('temp_ts_data.csv')
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    price_col = stored_data.get('price_col', 'Close')

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
            price, _, _, _, _, _ = interpolate(x_vals, y_vals, frac_idx, method='auto')
        filled.append({'Date': d.strftime('%Y-%m-%d'), 'Estimated_Price': round(price, 4)})

    filled_df = pd.DataFrame(filled)
    csv_data = filled_df.to_csv(index=False)
    return send_file(io.BytesIO(csv_data.encode()), mimetype='text/csv',
                     as_attachment=True, download_name='datafill_clean_timeseries.csv')


@app.route('/download/csv')
def download_csv():
    if stored_data.get('is_timeseries'):
        if not os.path.exists('temp_ts_data.csv'):
            return "No data yet", 400
        return send_file('temp_ts_data.csv', as_attachment=True, download_name='datafill_results.csv')
    else:
        if not os.path.exists('temp_data.csv'):
            return "No data yet", 400
        return send_file('temp_data.csv', as_attachment=True, download_name='datafill_results.csv')


@app.route('/download/png')
def download_png():
    if stored_data.get('is_timeseries'):
        if not os.path.exists('temp_ts_data.csv'):
            return "No data yet", 400
        df = pd.read_csv('temp_ts_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return send_file(
            create_matplotlib_chart(df, 'Date', stored_data['price_col'], date_x=True),
            mimetype='image/png', as_attachment=True, download_name='datafill_chart.png')
    else:
        if not os.path.exists('temp_data.csv'):
            return "No data yet", 400
        df = pd.read_csv('temp_data.csv')
        x_col, y_col = df.columns[0], df.columns[1]
        return send_file(
            create_matplotlib_chart(df, x_col, y_col),
            mimetype='image/png', as_attachment=True, download_name='datafill_chart.png')


if __name__ == '__main__':
    app.run(debug=True)
