"""一般月薪制加班費概算，不判斷實際出勤是否合法。"""
from fractions import Fraction
from decimal import Decimal, InvalidOperation


def _positive(value, label):
    if isinstance(value, bool): raise ValueError(f'{label}格式錯誤。')
    try: d=Decimal(str(value))
    except (InvalidOperation, ValueError): raise ValueError(f'{label}格式錯誤。') from None
    if not d.is_finite() or d <= 0 or d > Decimal('100000000') or d.as_tuple().exponent < -8:
        raise ValueError(f'{label}必須為合理且大於零的有限數值。')
    return Fraction(d)


def calculate_overtime(monthly_wage: float, hours: float, day_type: str) -> dict:
    wage=_positive(monthly_wage,'月薪'); h=_positive(hours,'加班時數')
    if day_type=='平日':
        tiers=[(2,Fraction(4,3),'前 2 小時'),(2,Fraction(5,3),'第 3–4 小時')]
        if h>4: raise ValueError('一般每日正常工時 8 小時，平日加班逾 4 小時不適用本試算；特殊情形需另外核對。')
    elif day_type=='休息日':
        tiers=[(2,Fraction(4,3),'第 1–2 小時'),(6,Fraction(5,3),'第 3–8 小時'),(4,Fraction(8,3),'第 9–12 小時')]
        if h>12: raise ValueError('休息日逾 12 小時不適用本試算。')
    else: raise ValueError('目前只支援一般月薪制平日及休息日。')
    hourly=wage/240; remaining=h; total=Fraction(0); breakdown=[]
    for cap,m,label in tiers:
        used=min(remaining,cap)
        if used<=0: break
        amount=hourly*m*used; total+=amount; remaining-=used
        breakdown.append(dict(label=label,hours=float(used),multiplier=float(m),fraction=str(m),amount=float(amount)))
    if remaining: raise ValueError('超出支援的試算範圍。')
    return dict(monthly_wage=float(wage),hourly_wage=float(hourly),hours=float(h),day_type=day_type,breakdown=breakdown,total=float(total),source='https://calcr2.mol.gov.tw/Monthly')
