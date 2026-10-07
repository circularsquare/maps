"""Kyrgyzstan's rules for build_model.py: Kazakhstan's (rules/kz.py has the reasoning)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("rules_kz_shared", Path(__file__).with_name("kz.py"))
_kz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kz)

looks_like_service = _kz.looks_like_service
SERVICE_IF_ALL_ROUTES_ARE = _kz.SERVICE_IF_ALL_ROUTES_ARE
