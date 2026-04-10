"""
P2-ETF-SIGNATURE-ENGINE  ·  calendar_utils.py
NYSE next trading day utility.
"""

from __future__ import annotations
from datetime import date, timedelta
import pandas as pd

try:
    import pandas_market_calendars as mcal
    _NYSE     = mcal.get_calendar("NYSE")
    _USE_MCal = True
except ImportError:
    _USE_MCal = False


def next_trading_day(from_date: date | str | None = None) -> str:
    if from_date is None:
        from_date = date.today()
    elif isinstance(from_date, str):
        from_date = pd.Timestamp(from_date).date()

    if _USE_MCal:
        start = pd.Timestamp(from_date) + pd.Timedelta(days=1)
        end   = start + pd.Timedelta(days=14)
        sched = _NYSE.schedule(start_date=start, end_date=end)
        if len(sched) > 0:
            return sched.index[0].strftime("%Y-%m-%d")

    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")
