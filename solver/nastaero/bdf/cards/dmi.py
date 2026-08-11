# DMI 직접 행렬 입력 카드 — W2GJ(캠버/비틀림 초기 다운워시) 등 실행렬 조립
"""DMI (Direct Matrix Input) card parser.

Assembles rectangular real matrices from a header card plus one or
more column cards.

Header entry::

    DMI  NAME  0  FORM  TIN  TOUT  --  M  N

Column entry::

    DMI  NAME  J  I1  A(I1,J)  A(I1+1,J)  ...

Within a column entry an integer field restarts the row index (sparse
columns); real fields are stored at the running row index. This is the
subset used by W2GJ decks (FORM=2 rectangular, real single/double);
complex matrices (TIN=3/4) are not supported.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..field_parser import nastran_float, nastran_int


def _is_int_field(s: str) -> bool:
    """True if the field is an integer (row-index restart marker)."""
    try:
        int(s)
        return True
    except ValueError:
        return False


class DMI:
    """One named DMI matrix, filled incrementally by column cards."""

    def __init__(self, name: str, form: int, tin: int, tout: int,
                 m: int, n: int) -> None:
        self.name = name
        self.form = form
        self.tin = tin
        self.tout = tout
        self.m = m
        self.n = n
        self.matrix = np.zeros((m, n))

    @classmethod
    def header_from_fields(cls, fields: List[str]) -> "DMI":
        name = fields[1].strip().upper()
        form = nastran_int(fields[3])
        tin = nastran_int(fields[4])
        tout = nastran_int(fields[5]) if len(fields) > 5 else 0
        m = nastran_int(fields[7])
        n = nastran_int(fields[8])
        if tin in (3, 4):
            raise ValueError(f"DMI {name}: complex matrices (TIN={tin}) "
                             "not supported")
        return cls(name, form, tin, tout, m, n)

    def fill_column(self, fields: List[str]) -> None:
        """Apply one column card: DMI NAME J I1 A(I1,J) ..."""
        j = nastran_int(fields[2])
        if not (1 <= j <= self.n):
            raise ValueError(f"DMI {self.name}: column {j} outside 1..{self.n}")
        i = None
        for f in fields[3:]:
            s = f.strip() if isinstance(f, str) else str(f)
            if not s:
                continue
            if _is_int_field(s):
                i = int(s)
                continue
            val = nastran_float(s)
            if i is None or not (1 <= i <= self.m):
                raise ValueError(
                    f"DMI {self.name}: row index {i} outside 1..{self.m}")
            self.matrix[i - 1, j - 1] = val
            i += 1
