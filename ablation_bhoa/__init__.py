"""Paket eksperimen studi ablasi Lévy Flight x Local Search pada BHOA untuk seleksi fitur NIDS."""
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

__version__ = "0.1.0"
