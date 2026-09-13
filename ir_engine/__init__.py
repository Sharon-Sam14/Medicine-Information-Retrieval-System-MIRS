# ir_engine/__init__.py
# Medicine Information Retrieval Engine — Sharon's Core IR Module
# Exposes the top-level search interface for Django/Sam to consume.

from .search_engine import MedicineSearchEngine

__all__ = ["MedicineSearchEngine"]
