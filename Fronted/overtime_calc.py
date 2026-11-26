"""
Overtime pay computed in code, so the same facts always give the same numbers.

Left to the model, the arithmetic varied between runs (one answer derived the
hourly wage, the next refused; one wrote 「約 2.33 倍（4/3 倍）」; one cited
施行細則 §11 — the 基本工資 definition — as the source of the hourly formula).

Rules (勞動部 examples reproduced in tests/test_overtime_calc.py):
- 平日每小時工資額 = 月薪 ÷ 30 ÷ 8 (行政院勞工委員會 90.6.26 台90勞動2字第0026202號函;
  an agreement in the labor contract takes precedence).
- 平日延長工時 (勞基法 §24 I): first 2 h × 4/3, next 2 h × 5/3; §32 caps a day at
  12 h including normal hours, i.e. 4 h of overtime.
- 休息日 (§24 II): first 2 h × 4/3, hours 3–8 × 5/3; hours 9–12 × 8/3
  (勞動部勞動條2字第1050030466號書函).
- 國定假日／特別休假 (§39) and 例假 (§40, emergencies only): an extra day's wage
  for up to 8 h, however few hours were worked; hours beyond 8 per §24 I.
Pure Python, no Streamlit, so it can be unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

HOURS_PER_MONTH = 240           # 30 days × 8 hours
HOURLY_SOURCE = "行政院勞工委員會 90 年 6 月 26 日台 90 勞動 2 字第 0026202 號函（月薪 ÷ 30 ÷ 8；勞動契約另有約定者從其約定）"

DAY_TYPES = ("平日", "休息日", "國定假日", "例假日")
_ALIASES = {"國假": "國定假日", "休假日": "國定假日", "特休": "國定假日", "例假": "例假日",
            "工作日": "平日", "上班日": "平日"}


@dataclass
class OvertimeResult:
    day_type: str
    hours: float
    hourly: float
    hourly_note: str
    lines: list[str] = field(default_factory=list)
    total: int = 0
    warnings: list[str] = field(default_factory=list)
    basis: list[str] = field(default_factory=list)

    def as_prompt(self) -> str:
        out = ["【試算結果】（由程式依下列規則計算，回答時直接使用這些數字，不要自行重算）",
               f"- 情況：{self.day_type}出勤 {self.hours:g} 小時",
               f"- 平日每小時工資額：{self.hourly_note}"]
        out += [f"- {l}" for l in self.lines]
        out.append(f"- 合計：約新臺幣 {self.total:,} 元（四捨五入至元）")
        out += [f"- 注意：{w}" for w in self.warnings]
        out.append(f"- 依據：{'；'.join(self.basis)}")
        return "\n".join(out)


def normalize_day_type(s: str | None) -> str | None:
    if not s:
        return None
    s = str(s).strip()
    s = _ALIASES.get(s, s)
    return s if s in DAY_TYPES else None


def _money(x) -> int:
    return int(Fraction(x).limit_denominator(10**6) + Fraction(1, 2))   # round half up


def overtime_pay(day_type: str | None, hours, monthly_wage=None, hourly_wage=None) -> OvertimeResult | None:
    """None when the facts are incomplete or out of range (the model then
    explains the rule without numbers)."""
    day = normalize_day_type(day_type)
    try:
        h = Fraction(str(hours))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if day is None or h <= 0 or h > 24:
        return None

    if monthly_wage:
        try:
            m = Fraction(str(monthly_wage))
        except (ValueError, ZeroDivisionError):
            return None
        if m <= 0:
            return None
        hr = m / HOURS_PER_MONTH
        note = f"月薪 {_money(m):,} 元 ÷ 30 ÷ 8 ≈ {float(hr):,.2f} 元"
        basis = [HOURLY_SOURCE]
        holiday_day_pay = hr * 8
    elif hourly_wage:
        try:
            hr = Fraction(str(hourly_wage))
        except (ValueError, ZeroDivisionError):
            return None
        if hr <= 0:
            return None
        note = f"時薪 {float(hr):,.2f} 元（使用者提供）"
        basis = []
        holiday_day_pay = None
    else:
        return None

    r = OvertimeResult(day, float(h), float(hr), note, basis=basis)
    total = Fraction(0)

    def seg(label, n, mult, mult_txt):
        nonlocal total
        if n <= 0:
            return
        amt = hr * n * mult
        total += amt
        r.lines.append(f"{label}：{float(hr):,.2f} × {n:g} 小時 × {mult_txt} ≈ {_money(amt):,} 元")

    if day == "平日":
        seg("前 2 小時", min(h, 2), Fraction(4, 3), "4/3")
        seg("第 3–4 小時", min(max(h - 2, 0), 2), Fraction(5, 3), "5/3")
        if h > 4:
            r.warnings.append("平日延長工時每日最多 4 小時（連同正常工時不得超過 12 小時，勞基法第 32 條），超過部分已違法，未列入試算。")
        r.basis.append("勞動基準法第 24 條第 1 項")
    elif day == "休息日":
        seg("前 2 小時", min(h, 2), Fraction(4, 3), "4/3")
        seg("第 3–8 小時", min(max(h - 2, 0), 6), Fraction(5, 3), "5/3")
        seg("第 9–12 小時", min(max(h - 8, 0), 4), Fraction(8, 3), "8/3")
        if h > 12:
            r.warnings.append("休息日出勤連同延長工時每日不得超過 12 小時，超過部分未列入試算。")
        r.basis += ["勞動基準法第 24 條第 2 項", "勞動部勞動條 2 字第 1050030466 號書函（第 9–12 小時）"]
    else:  # 國定假日 / 例假日
        if holiday_day_pay is None:
            return None      # hourly workers: «加倍發給» depends on the agreement; don't guess
        total += holiday_day_pay
        r.lines.append(f"加發一日工資（8 小時內，不論實際時數）：{float(hr):,.2f} × 8 ≈ {_money(holiday_day_pay):,} 元")
        seg("超過 8 小時的第 9–10 小時", min(max(h - 8, 0), 2), Fraction(4, 3), "4/3")
        seg("第 11–12 小時", min(max(h - 10, 0), 2), Fraction(5, 3), "5/3")
        if day == "例假日":
            r.warnings.append("例假日原則上不得出勤，只有天災、事變或突發事件才可要求，且事後應給予補假休息（勞動基準法第 40 條）。")
            r.basis.append("勞動基準法第 40 條")
        else:
            r.basis.append("勞動基準法第 39 條")
        if h > 8:
            r.basis.append("勞動基準法第 24 條第 1 項（超過 8 小時部分）")
        if h > 12:
            r.warnings.append("出勤連同延長工時每日不得超過 12 小時，超過部分未列入試算。")

    r.total = _money(total)
    return r
