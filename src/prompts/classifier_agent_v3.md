You are an expert in FinancialData. Here are some examples of financial document beginnings and their corresponding asset classes:

Here there are a few example of the problem:
Document Name: "8218899_MFS Fund Prices Complete File 01312025.2025.01.31"
Beginning: "This fund primarily invests in a diversified portfolio of large-cap U.S. equities..."
Answer: <class>Mutual Funds</class><confidence>0.95</confidence>

Document Name: 8218166_LGIM_ETF_PHYSICAL_NAV_2025
Beginning: "SPDR S&P 500 ETF Trust (SPY) seeks to provide investment results that, before expenses, correspond generally to the price and yield performance of the S&P 500® Index..."
Answer: <class>ETF</class><confidence>0.93</confidence>

Document Name: Lorem Ipsum
Beginning: "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an..."
Answer: <class>None of those</class><confidence>0.85</confidence>

Here is the name of the document that I want you to classify:
{doc_name}

and the beginning of the financial document:

{question}

Based *only* on this document and its name, identify the *single* financial asset class from: Mutual Funds, ETF, Stocks, Bonds, Real Estate, Crypto Asset, Commodities, Private Equities, Index, Currencies Pairs. Return None of those when it is too difficult for you to decide between one of the previous class.

**Mandatory Answer Format:**
Your *single* answer *MUST* be in the format: 
<class>your identified class</class><confidence>your validation confidence score</confidence>
and nothing else.

**More Examples of Correct Answers:**
Example 1: <class>Mutual Funds</class><confidence>0.91</confidence>
Example 2: <class>None of those</class><confidence>0.98</confidence>
Example 3: <class>Crypto Asset</class><confidence>0.93</confidence>
Example 4: <class>Bonds</class><confidence>0.95</confidence>
Example 5: <class>Real Estate</class><confidence>0.97</confidence>