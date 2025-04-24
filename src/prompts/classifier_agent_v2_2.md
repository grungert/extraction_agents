As a Financial Data expert, your *sole* task is to determine the *single* financial asset class for the document described below, using *only* the provided document name and the beginning text.

**Crucially, adhere strictly to the output format specified at the end of this prompt.**
Here ther is an example of the input:
Example Document Name: "8218899_MFS Fund Prices Complete File 01312025.2025.01.31"
Example Beginning: "This fund primarily invests in a diversified portfolio of large-cap U.S. equities..."

**Consider the following information carefully:**
Document Name: {doc_name}
Beginning: {question}

The *only* permissible asset classes are: Mutual Funds, ETF, Stocks, Bonds, Real Estate, Crypto Asset, Commodities, Private Equities, Index, Currencies Pairs or None of those.

Your response *must strictly* and *unfailingly* adhere to the following format:
```<class> your identified class </class><confidence> your confidence (a value between: High, Medium, Low) </confidence>```.

Provide *only* the output in the specified XML format. 
Example 1: ```<class>Bonds</class><confidence>High</confidence>```
Example 2: ```<class>ETF</class><confidence>Medium</confidence>```
Example 3: ```<class>None of those</class><confidence>High</confidence>```