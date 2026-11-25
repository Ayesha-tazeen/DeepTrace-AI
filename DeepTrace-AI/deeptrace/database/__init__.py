"""DeepTrace database package."""

from deeptrace.database.db import get_report, init_db, list_reports, save_report

__all__ = ["init_db", "save_report", "get_report", "list_reports"]
