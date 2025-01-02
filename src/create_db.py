import sqlite3
import pandas as pd
import numpy as np
from helper import get_paths

def create_db():
    ### creates synthetic_data.db 
    db_fpath = get_paths()["sql"]
    conn     = sqlite3.connect(db_fpath)
    cursor   = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ohlc (
        date TEXT PRIMARY KEY,
        open REAL,
        high REAL,
        low REAL,
        close REAL
    )
    ''')

    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='B')
    data = {
        'date' : dates,
        'open' : np.random.uniform(75, 120, len(dates)),
        'high' : np.random.uniform(120, 130, len(dates)),
        'low'  : np.random.uniform(50, 70, len(dates)),
        'close': np.random.uniform(80, 120, len(dates))
        }

    ohlc_df = pd.DataFrame(data)
    ohlc_df.to_sql('ohlc', conn, if_exists='replace', index=False)

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fxrates (
        date TEXT PRIMARY KEY,
        usd_to_eur REAL,
        usd_to_gbp REAL,
        usd_to_jpy REAL
    )
    ''')

    fx_data = {
        'date'      : dates,
        'usd_to_eur': np.random.uniform(0.8, 1.2, len(dates)),
        'usd_to_gbp': np.random.uniform(0.7, 0.9, len(dates)),
        'usd_to_jpy': np.random.uniform(110, 180, len(dates))
        }
    fxrates_df = pd.DataFrame(fx_data)
    fxrates_df.to_sql('fxrates', conn, if_exists='replace', index=False)

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS treasury_yields (
        date TEXT PRIMARY KEY,
        yield_5_year REAL,
        yield_7_year REAL,
        yield_10_year REAL
    )
    ''')

    treasury_data = {
        'date'         : dates,
        'yield_5_year' : np.random.uniform(1.2, 4.1, len(dates)),
        'yield_7_year' : np.random.uniform(1.3, 4.5, len(dates)),
        'yield_10_year': np.random.uniform(1.4, 4.5, len(dates))
        }
    treasury_yields_df = pd.DataFrame(treasury_data)
    treasury_yields_df.to_sql('treasury_yields', conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()

    print(f"Database created successfully at: {db_fpath}")

if __name__ == '__main__':
    create_db()