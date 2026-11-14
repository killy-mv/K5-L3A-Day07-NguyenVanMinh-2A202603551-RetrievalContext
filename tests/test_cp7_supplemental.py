import unittest
from scripts.cp7_supplemental_38 import CASES

class SupplementTest(unittest.TestCase):
    def test_only_provider_failed_cases_allowed(self):
        self.assertEqual(CASES,{'dense':('D7-T02','D7-T04'),'hybrid':('D7-T03',)})

if __name__=='__main__': unittest.main()
