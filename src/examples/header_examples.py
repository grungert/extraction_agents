from typing import List, Dict

def _get_header_examples() -> List[Dict]:
    """
    Return shared header examples as a list of raw table + JSON output pairs.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """
    raw_examples = [
        # Example 1
        ("""| ODDO BHF Fund DAILY PRICING | Unnamed: 1 | Unnamed: 2 | Unnamed: 3 | Unnamed: 4 | Unnamed: 5 | Unnamed: 6 | Unnamed: 7 | Unnamed: 8 | Unnamed: 9 | Unnamed: 10 | Unnamed: 11 | Unnamed: 12 | Unnamed: 13 | Unnamed: 14 | Unnamed: 15 | Unnamed: 16 | Unnamed: 17 | Unnamed: 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| Sender: | ODDO BHF Asset Management SAS | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| Email: | lux_nav@oddo-bhf.com | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| NaN | NaN | NaN | NaN | NaN | NaN | NaN | NAV per share | % change on | NAV per share | % change on | NaN | NaN | NaN | NaN | NaN | NaN | NaN | Total Fund Assets |
| Fund Name | ISIN Code | WKN | NaN | NaN | Curr | NAV per share | Prior day | prior day | Prior Month | prior month | Total NAV | Shares Outstanding | NaN | NaN | TIS | NaN | NaN | in base |
| NaN | NaN | NaN | NaN | NaN | NaN | 2025-01-28 00:00:00 | 2025-01-27 00:00:00 | NaN | 2024-12-31 00:00:00 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | currency |
| ODDO BHF Euro High Yield Bond CI | LU0115288721 | 940818 | NaN | NaN | EUR | 36.804 | 36.775 | 0.000789 | 36.783 | 0.000571 | 159247523.58 | 4326929.04 | NaN | NaN | 0 | NaN | NaN | 835941068.04 |
""",
        {
            "HeaderStartLine": 8,
            "HeaderEndLine": 10,
            "ContentStartLine": 11,
            "ValidationConfidence": 0.95
        }),

        # Example 2
        ("""| CODE ISIN | FCP | DATE | COUPON | CREDIT D'IMPÔT | VL APRES DETACH. | Coefficient | Ouvrant à PL | NET du PL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FR0007436969 | UFF AVENIR SECURITE | 1989-12-27 | 5.526277 | 0.030490 | 75.418053 | 1.073275 | NR | 5.526277 |
| FR0007437124 | UFF AVENIR DIVERSIFIE | 1989-12-27 | 1.300390 | 0.650957 | 90.638563 | 1.014347 | NR | 1.300390 |
| FR0010180786 | UFF AVENIR FRANCE | 1989-12-27 | 1.265327 | 0.411612 | 106.699067 | 1.011859 | NR | 1.265327 |
| FR0007437124 | UFF AVENIR DIVERSIFIE | 1990-12-26 | 0.039637 | 0.019818 | 74.645137 | 1.000531 | NR | 0.039637 |
| FR0007436969 | UFF AVENIR SECURITE | 1990-12-28 | 7.650430 | 0.038112 | 77.384646 | 1.098862 | NR | 7.650430 |
| FR0010180786 | UFF AVENIR FRANCE | 1991-12-15 | 1.129647 | 0.564061 | 90.239147 | 1.012518 | NR | 1.129647 |
| FR0007436969 | UFF AVENIR SECURITE | 1991-12-23 | 4.733542 | 0.007622 | 78.860352 | 1.060024 | NR | 4.733542 |
| FR0007437124 | UFF AVENIR DIVERSIFIE | 1991-12-27 | 1.684562 | 0.246967 | 80.869630 | 1.020831 | NR | 1.684562 |
| FR0007436969 | UFF AVENIR SECURITE | 1992-12-15 | 3.096240 | 0.001524 | 85.450723 | 1.036234 | NR | 3.096240 |""",
        {
            "HeaderStartLine": 0,
            "HeaderEndLine": 0,
            "ContentStartLine": 1,
            "ValidationConfidence": 0.92
        }),

        # Example 3
        ("""| Jyske Invest Balanced Strategy (GBP) CL | DK0060238194 | GBP | 173.5499 | 62414 | 10831943.4586 | 2025-01-30 00:00:00 |
| Jyske Invest Balanced Strategy USD | DK0060656197 | USD | 145.6897 | 47670 | 6945027.999 | 2025-01-30 00:00:00 |
| Jyske Invest Stable Strategy USD | DK0060729259 | USD | 130.2118 | 42006 | 5469676.8708 | 2025-01-30 00:00:00 |
| Jyske Invest Stable Strategy GBP | DK0060729333 | GBP | 120.2343 | 35561 | 4275651.9423 | 2025-01-30 00:00:00 |
| Jyske Invest High Yield Corporate Bonds CL | DK0016262728 | EUR | 242.3 | 223484 | 54150173.2 | 2025-01-30 00:00:00 |
| Jyske Invest Global Equities CL | DK0016259930 | USD | 518.992 | 23243 | 12062931.056 | 2025-01-30 00:00:00 |
| Jyske Invest Stable Strategy EUR | DK0016262058 | EUR | 188.758967 | 294096 | 55513257.158832 | 2025-01-30 00:00:00 |""",
        {
            "HeaderStartLine": None,
            "HeaderEndLine": None,
            "ContentStartLine": 1,
            "ValidationConfidence": 0.97
        })
    ]

    return [{"table": table, "json": metadata} for table, metadata in raw_examples]