"""勞動基準法一般月薪制加班費概算。

目前只估算平日及休息日加班；例假日、國定假日、變形工時、
部分工時及工資項目認定等情形，需另依個案與最新法規核對。
"""
from __future__ import annotations


def calculate_overtime(monthly_wage: float, hours: float, day_type: str) -> dict:
    """Return estimated overtime pay and a tier-by-tier breakdown."""
    wage = float(monthly_wage)
    overtime_hours = float(hours)
    if wage <= 0:
        raise ValueError("月薪必須大於 0。")
    if overtime_hours <= 0:
        raise ValueError("加班時數必須大於 0。")
    if overtime_hours > 12:
        raise ValueError("單次試算最多輸入 12 小時，請確認工時及適用規則。")

    hourly_wage = wage / 240.0
    if day_type == "平日":
        tiers = [(2.0, 4 / 3, "前 2 小時"), (2.0, 5 / 3, "第 3–4 小時")]
    elif day_type == "休息日":
        tiers = [
            (2.0, 4 / 3, "第 1–2 小時"),
            (6.0, 5 / 3, "第 3–8 小時"),
            (4.0, 8 / 3, "第 9–12 小時"),
        ]
    else:
        raise ValueError("目前僅支援平日及休息日；例假日、國定假日請另行確認。")

    remaining = overtime_hours
    breakdown = []
    total = 0.0
    for cap, multiplier, label in tiers:
        used = min(remaining, cap)
        if used <= 0:
            break
        amount = hourly_wage * multiplier * used
        breakdown.append({"label": label, "hours": used, "multiplier": multiplier, "amount": amount})
        total += amount
        remaining -= used

    return {
        "monthly_wage": wage,
        "hourly_wage": hourly_wage,
        "hours": overtime_hours,
        "day_type": day_type,
        "breakdown": breakdown,
        "total": total,
    }
