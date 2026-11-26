"""Overtime calculator vs. 勞動部 published examples.

    python -m unittest tests.test_overtime_calc -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Fronted"))
from overtime_calc import overtime_pay  # noqa: E402


class TestOvertime(unittest.TestCase):
    def test_hourly_from_monthly(self):
        # 勞動部: 月薪 36,000 → 每日 1,200 → 每小時 150
        self.assertAlmostEqual(overtime_pay("平日", 1, monthly_wage=36000).hourly, 150)

    def test_rest_day_8h_mol_example(self):
        # 勞動部: 150×2×4/3 + 150×6×5/3 = 1,900
        self.assertEqual(overtime_pay("休息日", 8, monthly_wage=36000).total, 1900)

    def test_rest_day_hours_9_to_12(self):
        # 第 9–12 小時 × 8/3: 1,900 + 150×2×8/3 = 2,700
        self.assertEqual(overtime_pay("休息日", 10, monthly_wage=36000).total, 2700)

    def test_holiday_extra_day_even_if_short(self):
        # 勞動部: 國定假日出勤未滿 8 小時仍加發一日工資 = 1,200
        self.assertEqual(overtime_pay("國定假日", 3, monthly_wage=36000).total, 1200)
        self.assertEqual(overtime_pay("國定假日", 8, monthly_wage=36000).total, 1200)

    def test_holiday_over_8h(self):
        # 1,200 + 150×2×4/3 = 1,600
        self.assertEqual(overtime_pay("國定假日", 10, monthly_wage=36000).total, 1600)

    def test_weekday(self):
        # 150×2×4/3 + 150×1×5/3 = 400 + 250 = 650
        self.assertEqual(overtime_pay("平日", 3, monthly_wage=36000).total, 650)
        # 月薪 5 萬, 3 小時: 208.33×2×4/3 + 208.33×5/3 = 902.8 → 903
        self.assertEqual(overtime_pay("平日", 3, monthly_wage=50000).total, 903)

    def test_weekday_over_cap_warns(self):
        r = overtime_pay("平日", 6, monthly_wage=36000)
        self.assertEqual(r.total, 900)          # only 4 h counted
        self.assertTrue(r.warnings)

    def test_incomplete_or_invalid(self):
        self.assertIsNone(overtime_pay(None, 3, monthly_wage=36000))
        self.assertIsNone(overtime_pay("平日", None, monthly_wage=36000))
        self.assertIsNone(overtime_pay("平日", 3))
        self.assertIsNone(overtime_pay("平日", -1, monthly_wage=36000))
        self.assertIsNone(overtime_pay("國定假日", 3, hourly_wage=200))   # don't guess for hourly workers

    def test_aliases_and_hourly(self):
        self.assertEqual(overtime_pay("國假", 8, monthly_wage=36000).total, 1200)
        self.assertEqual(overtime_pay("平日", 2, hourly_wage=150).total, 400)


if __name__ == "__main__":
    unittest.main()
