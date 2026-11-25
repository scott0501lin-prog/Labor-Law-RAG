import unittest
from Fronted.labor_calculator import calculate_overtime


class OvertimeCalculatorTests(unittest.TestCase):
    def test_weekday_first_two_hours(self):
        result = calculate_overtime(36000, 2, "平日")
        self.assertAlmostEqual(result["hourly_wage"], 150)
        self.assertAlmostEqual(result["total"], 400)

    def test_weekday_four_hours(self):
        result = calculate_overtime(36000, 4, "平日")
        self.assertAlmostEqual(result["total"], 900)

    def test_restdays_nine_hours(self):
        result = calculate_overtime(36000, 9, "休息日")
        self.assertEqual(len(result["breakdown"]), 3)
        self.assertAlmostEqual(result["total"], 2300)

    def test_reject_unsupported_day_type(self):
        with self.assertRaises(ValueError):
            calculate_overtime(36000, 2, "國定假日")


if __name__ == "__main__":
    unittest.main()
