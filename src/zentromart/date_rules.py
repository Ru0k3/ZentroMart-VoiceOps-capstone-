"""Date calculations using the capstone's fixed reference and calendar rules."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable


REFERENCE_DATE = date(2026, 10, 20)


def add_business_days(
    start: date,
    count: int,
    holidays: Iterable[str] = (),
) -> date:
    """Add Monday–Saturday business days, skipping Sundays and listed holidays.

    The start date is not counted. Return windows should use ``add_calendar_days``
    instead, as specified by the dataset.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    holiday_dates = {date.fromisoformat(value) for value in holidays}
    current = start
    remaining = count
    while remaining:
        current += timedelta(days=1)
        if current.weekday() != 6 and current not in holiday_dates:
            remaining -= 1
    return current


def add_calendar_days(start: date, count: int) -> date:
    """Add calendar days, including Sundays and holidays."""
    if count < 0:
        raise ValueError("count must be non-negative")
    return start + timedelta(days=count)
