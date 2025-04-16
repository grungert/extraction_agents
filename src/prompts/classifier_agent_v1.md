You are an expert in FinancialData. Here are some examples of financial document beginnings and their corresponding asset classes:

Document: "This fund primarily invests in a diversified portfolio of large-cap U.S. equities..."
Answer: '<sol>Mutual Funds</sol>'

Document: "SPDR S&P 500 ETF Trust (SPY) seeks to provide investment results that, before expenses, correspond generally to the price and yield performance of the S&P 500® Index..."
Answer: '<sol>ETF</sol>'

Document: "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an..."
Answer: '<sol>None of those</sol>'

Here is the name of the document:
{doc_name}

and the beginning of the financial data document:

{question}

Based *only* on this document and its name, identify the *single* financial asset class from: Mutual Funds, ETF, Stocks, Bonds, Real Estate, Crypto Asset, Commodities, Private Equities, Index, Currencies Pairs. Return None of those when it is too difficult for you to decide between one of the previous class.

Your *single* answer *MUST* be in the format: '<sol>asset class</sol>' and nothing else.
**Mandatory Answer Format:**
'<sol> your answer here </sol>'

**More Examples of Correct Answers:**
Example 1: '<sol>Mutual Funds</sol>'
Example 2: '<sol>None of those</sol>'
Example 3: '<sol>Crypto Asset</sol>'
Example 4: '<sol>Bonds</sol>'
Example 5: '<sol>Real Estate</sol>'
