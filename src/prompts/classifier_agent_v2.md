As a Financial Data expert, your task is to determine the *single* financial asset class for the document below, using *only* the provided beginning and its name.

Here it is un example of the problem
Document Name: "8218899_MFS Fund Prices Complete File 01312025.2025.01.31"
Beginning: "This fund primarily invests in a diversified portfolio of large-cap U.S. equities..."
Solution: <class>Mutual Funds</class><confidence>0.93</confidence>

**Crucially, consider the following information carefully:**
Document Name: {doc_name}
Beginning: {question}

The possible asset classes are: Mutual Funds, ETF, Stocks, Bonds, Real Estate, Crypto Asset, Commodities, Private Equities, Index, Currencies Pairs.
If you can not identify a class among those return "None of those" for the class output in the XML answer.
Your response *must strictly adhere* to the format:
<class>your identified class</class><confidence>your validation confidence score</confidence>.
Provide *only* the output. For instance: 
<class>Bonds</class><confidence>0.85</confidence>