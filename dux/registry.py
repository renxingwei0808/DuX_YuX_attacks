"""Instance-name -> cipher class registry (R5, W15-A).

`experiments/` and `tools/` scripts take an `--instance` name; from R5 on that
name can select either cipher family.  The mapping is by NAME PREFIX:

    dux-*    / toy-*      ->  dux.cipher.DuX        (DuX, Wu et al., DCC 94:140)
    yu2x-*   / yupx-*
             / yuxtoy-*   ->  yux.cipher.YuX        (YuX, Liu et al., TIT 70(5))

Nothing else in `dux/` changes: this module is additive, it imports `yux`
lazily so that `import dux` never depends on `yux`, and `get_cipher` forwards
its keyword arguments (alpha, rounds, ...) untouched.
"""
from __future__ import annotations

DUX_PREFIXES = ("dux-", "toy-")
YUX_PREFIXES = ("yu2x-", "yupx-", "yuxtoy-")


def cipher_family(instance: str) -> str:
    """'dux' or 'yux' for an instance name; raises on an unknown prefix."""
    if instance.startswith(YUX_PREFIXES):
        return "yux"
    if instance.startswith(DUX_PREFIXES):
        return "dux"
    raise KeyError(f"unknown instance {instance!r}: expected one of "
                   f"{DUX_PREFIXES + YUX_PREFIXES} as a prefix")


def instances(family: str | None = None) -> dict:
    """All known instance specs, optionally restricted to one family."""
    from .params import INSTANCES as DUX_INSTANCES
    out = {}
    if family in (None, "dux"):
        out.update({k: dict(v, family="dux") for k, v in DUX_INSTANCES.items()})
    if family in (None, "yux"):
        from yux.params import INSTANCES as YUX_INSTANCES
        out.update({k: dict(v, family="yux") for k, v in YUX_INSTANCES.items()})
    return out


def get_cipher(instance: str, **kw):
    """The cipher object for `instance` (DuX or YuX), forwarding **kw."""
    fam = cipher_family(instance)
    if fam == "yux":
        from yux.cipher import YuX
        return YuX(instance, **kw)
    from .cipher import DuX
    kw.pop("ks_literal", None)          # DuX has no such option
    return DuX(instance, **kw)
