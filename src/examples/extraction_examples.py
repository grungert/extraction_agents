from typing import List, Dict, Any, Type, Union
import json
from ..models import BaseExtraction, EXTRACTION_MODELS

def get_characteristic_examples() -> List[Dict[str, Any]]:
    """
    Return examples for CharacteristicsModel extraction.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """
    raw_examples = [
        # Example 1
        (""" | """,
        {
            "Strategy": "",
            "AssetManager": "",
            "PortfolioManager": "",
            "HostingCountry": "",
            "LegalStatus": "",
            "UnderLegalStatus": "",
            "InceptionDate": "",
            "DistributionPolicy": "",
            "PaymentFrequency": ""
        }),
        # Example 2
         (""" | """,
        {
            "Strategy": "",
            "AssetManager": "",
            "PortfolioManager": "",
            "HostingCountry": "",
            "LegalStatus": "",
            "UnderLegalStatus": "",
            "InceptionDate": "",
            "DistributionPolicy": "",
            "PaymentFrequency": ""
        }), 
        # Example 3
        (""" | """,
        {
            "Strategy": "",
            "AssetManager": "",
            "PortfolioManager": "",
            "HostingCountry": "",
            "LegalStatus": "",
            "UnderLegalStatus": "",
            "InceptionDate": "",
            "DistributionPolicy": "",
            "PaymentFrequency": ""
        })
    ]
    return [{"table": table, "json": metadata} for table, metadata in raw_examples]

def get_corporateact_examples() -> List[Dict[str, Any]]:
    """
    Return examples for CorporateActionModel extraction.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """
    raw_examples = [
        # Example 1
        (""" | Name of the Fund      | Evalution   | Currency   |   WKN |   Valor | ISIN         |   Redemption Price |   Offering Price |    NAV |   Distribution |   Distribution-Date |   Reinvestment Price | FV-Currency   | FV-Date   |   TrancheVolume |   Split-date |   Split-Factor |   ISIN-FundSavingPlan |   TIS |   Number-of-shares-Tranche |   Number-of-shares-Funds |   FundsVolume |   interim profit |   interim profit % |   Liquidation-Date |
             |:----------------------|:------------|:-----------|------:|--------:|:-------------|-------------------:|-----------------:|-------:|---------------:|--------------------:|---------------------:|:--------------|:----------|----------------:|-------------:|---------------:|----------------------:|------:|---------------------------:|-------------------------:|--------------:|-----------------:|-------------------:|-------------------:|
             | Active-Aktien (R) A   | 30.01.25    | EUR        |   nan |     nan | AT0000796446 |             173.13 |           173.13 | 173.13 |            nan |                 nan |               173.13 | EUR           | 30.01.25  |     4.2048e+06  |          nan |            nan |                   nan |   nan |                    24286.6 |                   522426 |   1.15662e+08 |              nan |                nan |                nan |
             | Active-Aktien (R) T   | 30.01.25    | EUR        |   nan |     nan | AT0000796453 |             243.37 |           243.37 | 243.37 |            nan |                 nan |               243.37 | EUR           | 30.01.25  |     8.38151e+07 |          nan |            nan |                   nan |   nan |                   344382   |                   522426 |   1.15662e+08 |              nan |                nan |                nan |
             | Active-Aktien (R) VTA | 30.01.25    | EUR        |   nan |     nan | AT0000712575 |             272.65 |           272.65 | 272.65 |            nan |                 nan |               272.65 | EUR           | 30.01.25  |     3.79211e+06 |          nan |            nan |                   nan |   nan |                    13908.3 |                   522426 |   1.15662e+08 |              nan |                nan |                nan |
             | Active-Aktien (RZ) A  | 30.01.25    | EUR        |   nan |     nan | AT0000A1V4N3 |             148.11 |           148.11 | 148.11 |            nan |                 nan |               148.11 | EUR           | 30.01.25  |     4.30316e+06 |          nan |            nan |                   nan |   nan |                    29052.6 |                   522426 |   1.15662e+08 |              nan |                nan |                nan |
             | Active-Aktien (RZ) T  | 30.01.25    | EUR        |   nan |     nan | AT0000A1V4M5 |             176.41 |           176.41 | 176.41 |            nan |                 nan |               176.41 | EUR           | 30.01.25  |     1.95465e+07 |          nan |            nan |                   nan |   nan |                   110796   |                   522426 |   1.15662e+08 |              nan |                nan |                nan |""",
        { 
            "Currency": "",
            "Type": "Dividend",
            "Value": "Distribution",
            "ExecutionDate": "Distribution-Date",
            "PaymentDate": "",
            "RecordDate": "",
            "DistributionRate": "",
            "validation_confidence": 0.98
        }),
        # Example 2
         ("""|   DISPLAY_NAME                      | Instrument Identifier/ISIN code     | WKN OFFICIAL_CODE   |   NAV LAST |   Subsription GEN_VAL1 |   Belgian Asset-Test - Gültig ab   |   Belgian Asset-Test - Gültig bis   |   TIS Grenze (in Scope)   |   TIS-Ratio   |   Redemption GEN_Val5 |   Subscription Ask |   Imm 1038 |   Imm Value 1032 |   TID 1038 |   TID Value 1032 | Div 1039   |   Div Value 1033 | Div Pay Date 800   | Ex Div Date 995     |     NET_CHANGE |   Currency   |   Datum             |     Share Class Volume |     Fund Volume |     Shares outstanding per shareclass |     Redemption price |   Calculation of Equilazation?   |   Aktiengewinn KStG in Prozent Share profit KStG |     Leerfeld |     BE-TIS |   Kapitalbeteilungsquote in % Equity Participation rate in % |   Anteil am Gesamtvermögen in % share of total assets in % |   Immobilienquote in % Real Estate rate in % |
             |:------------------------------------|:------------------------------------|:--------------------|-----------:|-----------------------:|-----------------------------------:|------------------------------------:|--------------------------:|--------------:|----------------------:|-------------------:|-----------:|-----------------:|-----------:|-----------------:|:-----------|-----------------:|:-------------------|:--------------------|---------------:|:-------------|:--------------------|-----------------------:|----------------:|--------------------------------------:|---------------------:|:---------------------------------|-------------------------------------------------:|-------------:|-----------:|-------------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------:|
             | Tungsten TRYCON AI Global Markets B | LU0451958135.LUF                    | HAFX28              |     120.7  |                 124.32 |                                nan |                                 nan |                       nan |           nan |                120.7  |             124.32 |        nan |              nan |        nan |              nan | Div        |                0 | Div Pay Date       | 2023-09-30 00:00:00 |         0.1577 | EUR          | 2025-01-30 00:00:00 |            4.6933e+06  |     1.74192e+08 |                               38883.5 |               120.7  | YES                              |                                              nan |          nan |        nan |                                                            0 |                                                     2.6943 |                                            0 |
             | Tungsten TRYCON AI Global Markets C | LU0451958309.LUF                    | HAFX29              |     132.89 |                 134.22 |                                nan |                                 nan |                       nan |           nan |                132.89 |             134.22 |        nan |              nan |        nan |              nan | Div        |                0 | Div Pay Date       | 2023-09-30 00:00:00 |         0.1658 | EUR          | 2025-01-30 00:00:00 |            4.51641e+07 |     1.74192e+08 |                              339865   |               132.89 | YES                              |                                              nan |          nan |        nan |                                                            0 |                                                    25.9278 |                                            0 |
             | Tungsten TRYCON AI Global Markets D | LU1251115991.LUF                    | HAFX70              |     124.09 |                 124.09 |                                nan |                                 nan |                       nan |           nan |                124.09 |             124.09 |        nan |              nan |        nan |              nan | Div        |                0 | Div Pay Date       | 2023-09-30 00:00:00 |         0.1695 | USD          | 2025-01-30 00:00:00 |            1.38478e+07 |     1.74192e+08 |                              111598   |               124.09 | YES                              |                                              nan |          nan |        nan |                                                            0 |                                                     7.6297 |                                            0 |
             | Tungsten TRYCON AI Global Markets E | LU1578228022.LUF                    | HAFX78              |     108.31 |                 108.31 |                                nan |                                 nan |                       nan |           nan |                108.31 |             108.31 |        nan |              nan |        nan |              nan | Div        |                0 | Div Pay Date       | 2022-09-30 00:00:00 |         0.1572 | CHF          | 2025-01-30 00:00:00 |            4.6452e+06  |     1.74192e+08 |                               42888   |               108.31 | YES                              |                                              nan |          nan |        nan |                                                            0 |                                                     2.8234 |                                            0 |
             | Tungsten TRYCON AI Global Markets H | LU2480924161.LUF                    | A3DMSG              |   10827    |               10827    |                                nan |                                 nan |                       nan |           nan |              10827    |           10827    |        nan |              nan |        nan |              nan | Div        |                0 | Div Pay Date       | 2023-09-30 00:00:00 |         0.1609 | EUR          | 2025-01-30 00:00:00 |            1.06126e+08 |     1.74192e+08 |                                9802   |             10827    | YES                              |                                              nan |          nan |        nan |                                                            0 |                                                    60.9247 |                                            0 | """,
        {
            "Currency": "",
            "Type": "Dividend",
            "Value": "Div Value",
            "ExecutionDate": "Ex Div Date",
            "PaymentDate": "Div Pay Date",
            "RecordDate": "",
            "DistributionRate": "",
            "validation_confidence": 0.97
        }), 
        # Example 3
        (""" |     Adviser |   Fund Name                       | SSB Fund #   | Inception Date      |   Denomination   |   ISIN       |     Per Share Rate |   Re-Invest Net Asset Value Per Share |
             |------------:|:----------------------------------|:-------------|:--------------------|:-----------------|:-------------|-------------------:|--------------------------------------:|
             |         nan | PIMCO Capital Securities Fund     | nan          | nan                 | nan              | nan          |           nan      |                                nan    |
             |       14603 | Class M Retail Income II          | PNE3H        | 2013-12-23 00:00:00 | USD              | IE00BH3X8443 |             0.0431 |                                  9.22 |
             |       14607 | Class M Retail (Hedged) Income II | PNE3J        | 2013-12-23 00:00:00 | SGD              | IE00BH3X8559 |             0.0408 |                                  8.78 |
             |       14603 | Class M HKD (Unhedged) Income     | PNE310       | 2017-07-28 00:00:00 | HKD              | IE00BF0F6D43 |             0.0352 |                                 10.2  |
             |       14603 | Class M Retail                    | PNE324       | 2024-05-09 00:00:00 | GBP              | IE0002Z7IO91 |             0.0357 |                                 10.32 |""",
        {
            "Currency": "",
            "Type": "Dividend",
            "Value": "",
            "ExecutionDate": "Ex Date",
            "PaymentDate": "Payable Date",
            "RecordDate": "Record Date",
            "DistributionRate": "Per Share Rate",
            "validation_confidence": 0.94
            })
    ]
    
    return [{"table": table, "json": metadata} for table, metadata in raw_examples]

def get_marketcap_examples() -> List[Dict[str, Any]]:
    """
    Return examples for MarketCapModel extraction.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """  
    raw_examples = [
        # Example 1
        ("""| Devise   | Libellé                              | Date VL    |     VL |   Encours Net |    Nb Parts | ISIN         |   Encours_Global |
            |:---------|:-------------------------------------|:-----------|-------:|--------------:|------------:|:-------------|-----------------:|
            | EUR      | LF Multimmo – Part LF Philosophale 2 | 10/01/2025 | 224.98 |   3.06819e+08 | 1.36373e+06 | FR0013488020 |      5.46153e+08 |""",
        {
            "ReferenceDate": "Date VL",
            "CompartmentCurrency": "Devise",
            "CompartmentAssetValue": "Encours_Global",
            "ShareAssetValue": "Encours Net",
            "Number of Shares": "Nb Parts",
            "validation_confidence": 0.92
        }),
        # Example 2
         ("""|   InvestOne Number | Sedol   | ISIN         | MEXID   | Price Date   | VP Time   | Fund Name                         | Share Class Name        | Currency   |   Basis | Inc/Acc   |   Unrounded Mid Price |   Unrounded NAV Price |   Published NAV Price |   Equalisation |   Move |   Move% |   ISC% |   Yield% |   Base CCY Share Class NAV |   Share Class Units In Issue |   XD Marker |
             |-------------------:|:--------|:-------------|:--------|:-------------|:----------|:----------------------------------|:------------------------|:-----------|--------:|:----------|----------------------:|----------------------:|----------------------:|---------------:|-------:|--------:|-------:|---------:|---------------------------:|-----------------------------:|------------:|
             |            3000138 | B1JMNF9 | GB00B1JMNF99 | OYOPUS  | 30/01/2025   | 22:30     | WS Lancaster Global Equity Fund   | R Income                | GBP        |     nan | INC       |              5506.69  |              5506.69  |               5506.69 |       0.801816 |  38.73 |    0.71 |      2 |   0.7891 |                4.7593e+06  |                      86427.5 |         nan |
             |            3000138 | B54RK12 | GB00B54RK123 | CGODOU  | 30/01/2025   | 22:30     | WS Lancaster Global Equity Fund   | I Accumulation          | GBP        |     nan | ACC       |               284.161 |               284.161 |                284.16 |       0.157107 |   2    |    0.71 |      2 |   1.2665 |                4.96412e+07 |                  1.74694e+07 |         nan |
             |            3000138 | B717BM7 | GB00B717BM70 | CGODOP  | 30/01/2025   | 22:30     | WS Lancaster Global Equity Fund   | I Income                | GBP        |     nan | INC       |               247.557 |               247.557 |                247.56 |       0.136813 |   1.75 |    0.71 |      2 |   1.2847 |                9.21077e+06 |                  3.72067e+06 |         nan |
             |            3000138 | BM8ND72 | GB00BM8ND727 | LIAATQ  | 30/01/2025   | 22:30     | WS Lancaster Global Equity Fund   | R Accumulation          | GBP        |     nan | ACC       |               105.986 |               105.986 |                105.99 |       0.015318 |   0.75 |    0.71 |      5 |   0.7822 |                1.41205e+06 |                  1.33231e+06 |         nan |
             |            3000135 | B55NGS8 | GB00B55NGS86 | CGCOUA  | 30/01/2025   | 22:30     | WS Lancaster Absolute Return Fund | Sterling I Accumulation | GBP        |     nan | ACC       |               462.054 |               462.054 |                462.05 |       0.954156 |   2.85 |    0.62 |      2 |   1.5443 |                2.04612e+08 |                  4.4283e+07  |         nan |""",
        {
            "ReferenceDate": "Price Date ",
            "CompartmentCurrency": "Currency",
            "CompartmentAssetValue": "Base CCY Share Class NAV",
            "ShareAssetValue": " ",
            "Number of Shares": "Share Class Units In Issue",
            "validation_confidence": 0.92
        }), 
        # Example 3
        ("""| SYCOMORE ASSET MANAGEMENT - Valeurs liquidatives Libellé   |   ISIN       |   Devise   |     Actif Part |     Nb Parts |     VL |     Perf. Jour |     Perf 2025 |   Date              |     Actif Net Global |
            |:-----------------------------------------------------------|:-------------|:-----------|---------------:|-------------:|-------:|---------------:|--------------:|:--------------------|---------------------:|
            | SYCOMORE SELECTION RESPONSABLE A                           | FR0013076452 | EUR        |       48730885 |       260466 | 187.09 |         0.0061 |        0.0569 | 2025-01-29 00:00:00 |            796447880 |
            | SYCOMORE SELECTION RESPONSABLE I                           | FR0010971705 | EUR        |      480224643 |       803142 | 597.93 |         0.0061 |        0.0572 | 2025-01-29 00:00:00 |            796447880 |
            | SYCOMORE SELECTION RESPONSABLE I USD H                     | FR0013320314 | USD        |          39470 |          229 | 172.36 |         0.0058 |        0.0577 | 2025-01-29 00:00:00 |            796447880 |
            | SYCOMORE SELECTION RESPONSABLE ID                          | FR0012719524 | EUR        |       14315238 |        26645 | 537.25 |         0.0061 |        0.0566 | 2025-01-29 00:00:00 |            796447880 |
            | SYCOMORE SELECTION RESPONSABLE ID2                         | FR0013277175 | EUR        |          13641 |           99 | 138.1  |         0.006  |        0.0571 | 2025-01-29 00:00:00 |            796447880 |""",
        {
            "ReferenceDate": "Date",
            "CompartmentCurrency": "Devise",
            "CompartmentAssetValue": "Actif Part",
            "ShareAssetValue": "Nb Parts",
            "Number of Shares": "",
            "validation_confidence": 0.98
        })
]
    
    return [{"table": table, "json": metadata} for table, metadata in raw_examples]

def get_valorisation_examples() -> List[Dict[str, Any]]:
    """
    Return examples for ValorisationModel extraction.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """
    raw_examples = [
        # Example 1
        ("""| Devise   | Libellé                              | Date VL    |     VL |   Encours Net |    Nb Parts | ISIN         |   Encours_Global |
            |:---------|:-------------------------------------|:-----------|-------:|--------------:|------------:|:-------------|-----------------:|
            | EUR      | LF Multimmo – Part LF Philosophale 2 | 10/01/2025 | 224.98 |   3.06819e+08 | 1.36373e+06 | FR0013488020 |      5.46153e+08 |""",
        {
        "Nav": "VL",
        "NavDate": "Date VL",
        "validation_confidence": 0.95
        }),
        # Example 2
         ("""| Name                                   | Price Date   |   Price | Currency   | ISIN         |   Price in SEK |   Price in EUR |   Price in USD |   1 Day |   1 Mth |   3 Mths |   YTD |   Inception |   1 Year |   3 Years |   5 Years |
             |:---------------------------------------|:-------------|--------:|:-----------|:-------------|---------------:|---------------:|---------------:|--------:|--------:|---------:|------:|------------:|---------:|----------:|----------:|
             | East Capital Baltic Property Investors | 2024-12-31   |   319.6 | EUR        | SE0011788439 |        3671.74 |          319.6 |         332.48 |       0 |    0.01 |    -0.99 |  0.03 |        0.16 |    -0.04 |     -0.02 |     -0.03 |""",
        {
        "Nav": "Price",
        "NavDate": "Price Date",
        "validation_confidence": 0.93
        }), 
        # Example 3
        ("""| Date de publication   | Code ISIN     | Devise     | FCP                                |   Valeur Liquidative |   Actif Net |   Nombre de Parts |
            |:----------------------|:--------------|:-----------|:-----------------------------------|---------------------:|------------:|------------------:|
            | 2025-01-23 00:00:00   | FR0011365873  | EUR        | UFF ACTIONS EUROPE EVOLUTIF - M    |              1531.7  | 7.54548e+07 |             49262 |
            | 2025-01-23 00:00:00   | FR001400PEV5  | EUR        | UFF ACTIONS EUROPE EVOLUTIF - A    |               137.16 | 9.19273e+07 |            670187 |
            | 2025-01-23 00:00:00   | FR0010286864  | EUR        | UFF ACTIONS EUROPE EVOLUTIF CT - C |               130.38 | 7.52122e+07 |            576869 |
            | 2025-01-23 00:00:00   | FR0012881043  | EUR        | UFF ALLOCATION DEFENSIVE - A       |                94.18 | 7.00557e+07 |            743849 |
            | 2025-01-23 00:00:00   | FR001400BCD7  | EUR        | UFF OBLIG OPPORTUNITES 2025        |               108.52 | 2.82986e+07 |            260752 |""",
        {
            "Nav": "Valeur Liquidative",
            "NavDate": "Date de publication",
            "validation_confidence": 0.94
        })
                ]
    
    return [{"table": table, "json": metadata} for table, metadata in raw_examples]

def get_denomination_examples() -> List[Dict[str, Any]]:
    """
    Return examples for DenominationModel extraction.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """
    raw_examples = [
        # Example 1 - Table with clear denomination fields
        ("""| Fund Name             | ISIN Code    | Currency | NAV   |
            | --------------------- | ------------ | -------- | ----- |
            | UFF AVENIR SECURITE   | FR0007436969 | EUR      | 75.42 |
            | UFF AVENIR DIVERSIFIE | FR0007437124 | EUR      | 90.64 |""",
        {
            "vehicule_name": " ",
            "compartment_name": " ",
            "instrument_name": "Fund Name",
            "share_type": "",
            "validation_confidence": 0.92
        }),
        #Example 2
        ("""| WKN    | ISIN         | NAME                                        | COMPARTMENT                    | CLASS      | CURRENCY   | DATE                |   NAV |   Interim |   BID PRICE |   OFFER PRICE |   Number of shares |   Total NAV |      Totals |   ADDI |   DDI |   Aktiengewinn |   TIS |   TID |   Redemption status |   Distribution status |   Redemption ratio |   Distribution ratio |   Redemption start date |   Redemption en date |   Distribution start date |   Distribution end date |   Immobiliengewinn |   Income Equalization |   Prior NAV |   NAV Change |   Aktiengewinn 2 |   BTIS |
            |:-------|:-------------|:--------------------------------------------|:-------------------------------|:-----------|:-----------|:--------------------|------:|----------:|------------:|--------------:|-------------------:|------------:|------------:|-------:|------:|---------------:|------:|------:|--------------------:|----------------------:|-------------------:|---------------------:|------------------------:|---------------------:|--------------------------:|------------------------:|-------------------:|----------------------:|------------:|-------------:|-----------------:|-------:|
            | A0M2VQ | IE00B1Z6CS11 | Natixis International Funds (Dublin) I plc. | LOOMIS SAYLES HIGH INCOME FUND | H-I/A(EUR) | EUR        | 2025-01-29 00:00:00 | 23.85 |         0 |       23.85 |         23.85 |            309.533 |     7381.84 | 4.55186e+06 |    nan |   nan |            nan |   nan |   nan |                 nan |                   nan |                nan |                  nan |                     nan |                  nan |                       nan |                     nan |                nan |                   nan |       23.87 |        -0.02 |              nan |    nan |
            | A0LFZ0 | IE0003063223 | Natixis International Funds (Dublin) I plc. | LOOMIS SAYLES HIGH INCOME FUND | I/D(USD)   | USD        | 2025-01-29 00:00:00 |  4.75 |         0 |        4.75 |          4.75 |         190671     |   906184    | 4.55186e+06 |    nan |   nan |            nan |   nan |   nan |                 nan |                   nan |                nan |                  nan |                     nan |                  nan |                       nan |                     nan |                nan |                   nan |        4.76 |        -0.01 |              nan |    nan |
            | A0M2VL | IE00B23XDT98 | Natixis International Funds (Dublin) I plc. | LOOMIS SAYLES HIGH INCOME FUND | I/A(USD)   | USD        | 2025-01-29 00:00:00 | 28.12 |         0 |       28.12 |         28.12 |            141.113 |     3968    | 4.55186e+06 |    nan |   nan |            nan |   nan |   nan |                 nan |                   nan |                nan |                  nan |                     nan |                  nan |                       nan |                     nan |                nan |                   nan |       28.14 |        -0.02 |              nan |    nan |
            | A0M2VM | IE00B23XDV11 | Natixis International Funds (Dublin) I plc. | LOOMIS SAYLES HIGH INCOME FUND | I/A(EUR)   | EUR        | 2025-01-29 00:00:00 | 22.84 |         0 |       22.84 |         22.84 |           7850     |   179289    | 4.55186e+06 |    nan |   nan |            nan |   nan |   nan |                 nan |                   nan |                nan |                  nan |                     nan |                  nan |                       nan |                     nan |                nan |                   nan |       22.85 |        -0.01 |              nan |    nan |
            | A1W2GC | IE00B7FTJT25 | Natixis International Funds (Dublin) I plc. | LOOMIS SAYLES HIGH INCOME FUND | R/A(USD)   | USD        | 2025-01-29 00:00:00 | 11.74 |         0 |       11.74 |         11.74 |           4280     |    50258    | 4.55186e+06 |    nan |   nan |            nan |   nan |   nan |                 nan |                   nan |                nan |                  nan |                     nan |                  nan |                       nan |                     nan |                nan |                   nan |       11.75 |        -0.01 |              nan |    nan |""",
        {
        "VehiculeName": "",
        "CompartmentName": "COMPARTMENT",
        "InstrumentName": "",
        "ShareType": "CLASS",
        "validation_confidence": 1.0 
        }),
        #Example3
        ("""| Umbrella_Name                             | Fund_Name                                | Fund_Class                       | Valuation_Date   | BASECurrency   | QUOTCurrency   |   NAV_per_unit |   Previous_NAV_per_unit |   Change_in_NAV_per_unit |   Change_Percent_in_NAV_per_unit |   Client_Offer_Price |   Client_Offer_Percentage |   Income_Aktien |   Total_NAV_per_Class_Base_Ccy |   Total_Net_Assets_per_Class_MIL |   Units_In_Issue |   DIV_Flag |   Cutas |   FT | ISIN         | BloombergTicker   |   VN |   FMSedols | ISESedols   |   EuroClearCedel |   Bundesamt |   Sicovam |   AFCCodes | WKN    |   Annualised_1_Day_Yield |   clio_fund_id |   Distribution_rate_per_share |   Accum_Deemed_Distrib |   Cap_Per_Unit |   Inc_Per_Unit | Fund_Title                               |   IFAST_account_number |   IFAST_account_class |   Income_Zwish |   Unrounded_Quotations_NAV |   Unrounded_Base_NAV |   Exchange_Rate |   ExpenseRatio |   DividendtoMonthMethod |   FundType |   AccrualDate |   MillRate |   DailyDividend |   DividendMonthtodate |   DayYield1 |   DayYield7 |   DayYield30 |   MCH_account_number |   MCH_account_class |   Income_IG |   Total_NAV_Per_Fund |   AG2 |   TIS |   TID |   CUSIP | EndofRow   |
            |:------------------------------------------|:-----------------------------------------|:---------------------------------|:-----------------|:---------------|:---------------|---------------:|------------------------:|-------------------------:|---------------------------------:|---------------------:|--------------------------:|----------------:|-------------------------------:|---------------------------------:|-----------------:|-----------:|--------:|-----:|:-------------|:------------------|-----:|-----------:|:------------|-----------------:|------------:|----------:|-----------:|:-------|-------------------------:|---------------:|------------------------------:|-----------------------:|---------------:|---------------:|:-----------------------------------------|-----------------------:|----------------------:|---------------:|---------------------------:|---------------------:|----------------:|---------------:|------------------------:|-----------:|--------------:|-----------:|----------------:|----------------------:|------------:|------------:|-------------:|---------------------:|--------------------:|------------:|---------------------:|------:|------:|------:|--------:|:-----------|
            | PRINCIPAL GLOBAL OPPORTUNITIES SERIES PLC | PGOS Post Short Duration High Yield Fund | US Dollar Accumulation           | 01/29/2025       | USD            | USD            |          15.89 |                   15.89 |                     0    |                             0    |                  nan |                       nan |               0 |                    2.93252e+07 |                            29.33 |      1.84609e+06 |        nan |     nan |  nan | IE00B95PXD61 | PGPITHU ID        |  nan |        nan | B95PXD6     |              nan |         nan |       nan |        nan | A1W49J |                      nan |            nan |                           nan |                    nan |            nan |            nan | PGOS Post Short Duration High Yield Fund |                    nan |                   nan |              0 |                    15.885  |             15.885   |            1    |            nan |                     nan |        nan |           nan |        nan |             nan |                   nan |         nan |         nan |          nan |                  nan |                 nan |           0 |            179641861 |     0 |     0 |     0 |     nan | EndofRow   |
            | PRINCIPAL GLOBAL OPPORTUNITIES SERIES PLC | PGOS Post Short Duration High Yield Fund | JPY Hedged Accumulation X Shares | 01/29/2025       | USD            | JPY            |        1196.81 |                 1197.42 |                    -0.61 |                            -0.05 |                  nan |                       nan |               0 |                    7.4467e+07  |                            74.47 |      9.64488e+06 |        nan |     nan |  nan | IE00BKM4GT07 | PGPIJHX ID        |  nan |        nan | BKM4GT0     |              nan |         nan |       nan |        nan | A116WT |                      nan |            nan |                           nan |                    nan |            nan |            nan | PGOS Post Short Duration High Yield Fund |                    nan |                   nan |              0 |                  1196.81   |              7.72088 |          155.01 |            nan |                     nan |        nan |           nan |        nan |             nan |                   nan |         nan |         nan |          nan |                  nan |                 nan |           0 |            179641861 |     0 |     0 |     0 |     nan | EndofRow   |
            | PRINCIPAL GLOBAL OPPORTUNITIES SERIES PLC | PGOS Post Short Duration High Yield Fund | USD X ACCUMULATION               | 01/29/2025       | USD            | USD            |          13.04 |                   13.04 |                     0    |                             0    |                  nan |                       nan |               0 |                    7.58497e+07 |                            75.85 |      5.81608e+06 |        nan |     nan |  nan | IE00BFZP8P53 | PGPIUAX ID        |  nan |        nan | BFZP8P5     |              nan |         nan |       nan |        nan | A2N8YA |                      nan |            nan |                           nan |                    nan |            nan |            nan | PGOS Post Short Duration High Yield Fund |                    nan |                   nan |              0 |                    13.0414 |             13.0414  |            1    |            nan |                     nan |        nan |           nan |        nan |             nan |                   nan |         nan |         nan |          nan |                  nan |                 nan |           0 |            179641861 |     0 |     0 |     0 |     nan | EndofRow   |
            | PRINCIPAL GLOBAL OPPORTUNITIES SERIES PLC | PGOS Principal Multi Sector Bond Fund    | I USD Unhedged Acc               | 01/29/2025       | USD            | USD            |          10.74 |                   10.74 |                     0    |                             0    |                  nan |                       nan |               0 |                    2.73032e+08 |                           273.03 |      2.54284e+07 |        nan |     nan |  nan | IE0005GJWVY5 | PRINSUS ID        |  nan |        nan | nan         |              nan |         nan |       nan |        nan | nan    |                      nan |            nan |                           nan |                    nan |            nan |            nan | PGOS Principal Multi Sector Bond Fund    |                    nan |                   nan |              0 |                    10.7373 |             10.7373  |            1    |            nan |                     nan |        nan |           nan |        nan |             nan |                   nan |         nan |         nan |          nan |                  nan |                 nan |           0 |            273031560 |     0 |     0 |     0 |     nan | EndofRow   |
            | PRINCIPAL GLOBAL OPPORTUNITIES SERIES PLC | PGOS Principal Flexible Growth Fund      | I USD Unhedged Acc               | 01/29/2025       | USD            | USD            |          12.71 |                   12.75 |                    -0.04 |                            -0.31 |                  nan |                       nan |               0 |                    8.17866e+07 |                            81.79 |      6.43525e+06 |        nan |     nan |  nan | IE000XFO2ZV6 | PRGTHUS ID        |  nan |        nan | nan         |              nan |         nan |       nan |        nan | nan    |                      nan |            nan |                           nan |                    nan |            nan |            nan | PGOS Principal Flexible Growth Fund      |                    nan |                   nan |              0 |                    12.7092 |             12.7092  |            1    |            nan |                     nan |        nan |           nan |        nan |             nan |                   nan |         nan |         nan |          nan |                  nan |                 nan |           0 |             81786588 |     0 |     0 |     0 |     nan | EndofRow   |""",
        {
            "VehiculeName": "Umbrella_Name",
            "CompartmentName": "",
            "InstrumentName": "Fund_Name",
            "ShareType": "Fund_Class",
            "validation_confidence": 0.94
        })
    ]
    
    return [{"table": table, "json": metadata} for table, metadata in raw_examples]

def get_identifier_examples() -> List[Dict[str, Any]]:
    """
    Return examples for IdentifierModel extraction.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """
    raw_examples = [
        # Example 1
        ("""| Fund name                                  | ISIN code    | Currency   |     NAV |   Number of shares |   Total net assets value | Date                |
            |:-------------------------------------------|:-------------|:-----------|--------:|-------------------:|-------------------------:|:--------------------|
            | Jyske Invest Balanced Strategy (GBP) CL    | DK0060238194 | GBP        | 173.55  |              62414 |              1.08319e+07 | 2025-01-30 00:00:00 |
            | Jyske Invest Balanced Strategy USD         | DK0060656197 | USD        | 145.69  |              47670 |              6.94503e+06 | 2025-01-30 00:00:00 |
            | Jyske Invest Stable Strategy USD           | DK0060729259 | USD        | 130.212 |              42006 |              5.46968e+06 | 2025-01-30 00:00:00 |
            | Jyske Invest Stable Strategy GBP           | DK0060729333 | GBP        | 120.234 |              35561 |              4.27565e+06 | 2025-01-30 00:00:00 |
            | Jyske Invest High Yield Corporate Bonds CL | DK0016262728 | EUR        | 242.3   |             223484 |              5.41502e+07 | 2025-01-30 00:00:00 |""",
        {
            "Code": "ISIN code",
            "CodeType": " ",
            "Currency": "Currency",
            "CIC Code": "",
            "ValidationConfidence": 0.99
        }),
        # Example 2
         ("""| Fund Name                          | ISIN Code      | WKN      | Curr  | NAV per share       | Prior day           | prior day | Prior Month         | prior month | Total NAV      | Shares Outstanding | TIS | in base        |
             |                                    |                |          |       | 2025-01-28 00:00:00 | 2025-01-27 00:00:00 |           | 2024-12-31 00:00:00 |             |                |                    |     | currency       |
             |:-----------------------------------|:---------------|:---------|:------|:--------------------|:--------------------|:----------|:--------------------|:------------|:---------------|:-------------------|:----|:---------------|
             | ODDO BHF Euro High Yield Bond CI   | LU0115288721   | 940818   | EUR   | 36.804              | 36.775              | 0.0008    | 36.783              | 0.0006      | 159247523.58   | 4326929.04         | 0   | 835941068.04   |
             | ODDO BHF Euro High Yield Bond CN   | LU1486847152   | A2DNK1   | EUR   | 120.946             | 120.852             | 0.0008    | 120.909             | 0.0003      | 17823362.14    | 147366.15          | 0   | 835941068.04   |
             | ODDO BHF Euro High Yield Bond CP   | LU0456627131   | A0YDE9   | EUR   | 16.136              | 16.123              | 0.0008    | 16.124              | 0.0007      | 312407251.09   | 19360880.03        | 0   | 835941068.04   |
             | ODDO BHF Euro High Yield Bond CR   | LU0115290974   | 940820   | EUR   | 30.984              | 30.96               | 0.0008    | 30.984              | 0           | 99843747.81    | 3222413.8          | 0   | 835941068.04   |
             | ODDO BHF Euro High Yield Bond DI   | LU0115293481   | 940819   | EUR   | 10.747              | 10.739              | 0.0007    | 10.741              | 0.0006      | 20300764.45    | 1888965.06         | 0   | 835941068.04   |
             | ODDO BHF Euro High Yield Bond DP   | LU0456627214   | A0YDEA   | EUR   | 11.002              | 10.993              | 0.0008    | 10.993              | 0.0008      | 203711130.84   | 18516373.29        | 0   | 835941068.04   |""",
        {
            "Code": "ISIN Code",
            "CodeType": " ",
            "Currency": "Curr",
            "CIC Code": "",
            "validation_confidence": 0.91
        }), 
        # Example 3
        ("""| Devise   | Libellé                              | Date VL    |     VL |   Encours Net |    Nb Parts | ISIN         |   Encours_Global |
            |:---------|:-------------------------------------|:-----------|-------:|--------------:|------------:|:-------------|-----------------:|
            | EUR      | LF Multimmo – Part LF Philosophale 2 | 10/01/2025 | 224.98 |   3.06819e+08 | 1.36373e+06 | FR0013488020 |      5.46153e+08 |""",
        {
        "Code": "ISIN",
        "CodeType": " ",
        "Currency": "Devise",
        "CIC Code": "",
        "validation_confidence": 0.99
        })
    ]
    
    return [{"table": table, "json": metadata} for table, metadata in raw_examples]

def get_model_examples(model_class: Type[BaseExtraction]) -> List[Dict[str, Any]]:
    """
    Get examples for a specific model class.
    
    Args:
        model_class: The model class to get examples for
        
    Returns:
        List of examples for the specified model
    """
    model_name = model_class.__name__
    
    # Return examples based on model name
    if model_name == "CharacteristicsModel":
        return get_characteristic_examples()
    elif model_name == "CorporateActionModel":
        return get_corporateact_examples()
    elif model_name == "MarketCapModel":
        return get_marketcap_examples()
    elif model_name == "ValorisationModel":
        return get_valorisation_examples()
    elif model_name == "DenominationModel":
        return get_denomination_examples()
    elif model_name == "IdentifierModel":
        return get_identifier_examples()
    
    # If no examples are defined for this model, return an empty list
    return []

def get_extraction_examples(model_name_or_class: Union[str, Type[BaseExtraction]]) -> List[Dict[str, Any]]:
    """
    Get examples for a specific model.
    
    Args:
        model_name_or_class: Name of the model or model class to get examples for
        
    Returns:
        List of examples for the specified model
    """
    # Handle both string names and actual model classes
    if isinstance(model_name_or_class, str):
        # Look up the model class by name
        if model_name_or_class == "DenominationModel":
            return get_denomination_examples()
        elif model_name_or_class == "Identifier":
            return get_identifier_examples()
        elif model_name_or_class in EXTRACTION_MODELS:
            model_class = EXTRACTION_MODELS[model_name_or_class]
            return get_model_examples(model_class)
        else:
            return []
    else:
        # It's already a model class
        return get_model_examples(model_name_or_class)

def create_validation_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a validation example from an extraction example.
    
    Args:
        example: The extraction example
        
    Returns:
        A validation example with modified data
    """
    # Copy the example
    validation_example = example.copy()
    
    # Create a modified version of the JSON data
    modified_json = example["json"].copy()
    
    # Modify some fields to demonstrate validation
    for field, value in modified_json.items():
        if field == "ValidationConfidence":
            continue
            
        # For string fields, modify them slightly
        if isinstance(value, str) and value:
            # Uppercase the value to demonstrate case correction
            modified_json[field] = value.upper()
        # For non-null fields that aren't strings, set to None to demonstrate adding missing values
        elif value is not None and not isinstance(value, (int, float, bool)):
            modified_json[field] = None
    
    # Create the validation input
    validation_example["modified_json"] = modified_json
    
    # Create the expected validation output
    validation_example["validation_output"] = {
        "validated_data": example["json"],
        "ValidationConfidence": example["json"].get("ValidationConfidence", 0.9),
        "corrections_made": ["Corrected data based on table context"]
    }
    
    return validation_example

def format_messages(examples: List[Dict[str, Any]], system_message: Dict[str, str], 
                   message_type: str = "extraction") -> List[Dict[str, str]]:
    """
    Format examples for LLM agents.
    
    Args:
        examples: List of examples
        system_message: System message for the LLM
        message_type: Type of formatting to apply ("extraction" or "validation")
        
    Returns:
        List of formatted messages for the LLM
    """
    formatted_examples = []
    
    for idx, example in enumerate(examples, start=1):
        if message_type == "extraction":
            # Simple extraction format
            input_content = f"""
# Example {idx}
# Input Table 
{example["table"]}"""
            output_content = json.dumps(example["json"])
            
        elif message_type == "validation":
            # Create a validation example from the extraction example
            validation_example = create_validation_example(example)
            
            input_content = f"""
# Example {idx}
# Original Table
{example["table"]}

# Extracted Data
```json
{json.dumps(validation_example["modified_json"], indent=2)}
```"""
            output_content = json.dumps(validation_example["validation_output"])
            
        else:
            raise ValueError(f"Unknown message type: {message_type}")

        formatted_examples.extend([
            {"role": "user", "content": input_content},
            {"role": "assistant", "content": output_content}
        ])
    
    return [system_message] + formatted_examples

# Keep these functions for backward compatibility
def format_extraction_messages(examples: List[Dict[str, Any]], system_message: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Format examples for extraction agent (wrapper for format_messages).
    """
    return format_messages(examples, system_message, "extraction")

def format_validation_messages(examples: List[Dict[str, Any]], system_message: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Format examples for validation agent (wrapper for format_messages).
    """
    return format_messages(examples, system_message, "validation")
