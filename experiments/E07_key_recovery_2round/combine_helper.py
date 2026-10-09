"""Tiny shared helper: the `--field` string tools/cheap_rows.py expects."""


def field_key(F):
    return f"2^{F.n}" if F.char == 2 else str(F.q)
