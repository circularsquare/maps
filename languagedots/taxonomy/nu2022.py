"""Niue, 2022 census household languages (sources/nu_census.py, sources/nu.md).

Households named Vagahau Niue, English and/or "Other languages" (several allowed); the mentions
are shared across the island's residents (spec §3.6). Vagahau Niue is Niuean (austronesian.oceanic.
niuean, drawn under Polynesian); "Other languages" is the unnamed remainder, on `other` (the
ethnicity table shows Tuvaluans, Samoans, Tongans and Fijians among residents, but the household
table names none of their languages).
"""
NAMES = {
    "Vagahau Niue": "austronesian.oceanic.niuean",
    "English": "indoeuropean.germanic.english",
    "Other languages": "other",
}


def resolve(label):
    return NAMES[label]
