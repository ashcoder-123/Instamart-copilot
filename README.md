# Instamart Customer Intelligence & Support Copilot

## Overview

A capstone project that builds an internal Customer Intelligence & Support Copilot on fully synthetic data (nothing here is real Instamart data). The repository is built in four connected Parts: A turns messy order exports into a trusted SQLite database, B segments customers and predicts lapse, C triages support tickets with guardrails, D answers policy questions and runs a refund agent. All amounts are in INR. All model calls use the offline `mock_llm.py`; no key or network is needed.

Status: **Part A complete.** Parts B, C and D are filled in as they are built.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py              # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
python src/part_a_clean.py           # audit, cleaning, data/clean/*.csv and data/instamart.db
python src/validate.py               # must print ALL CHECKPOINTS PASSED
python src/run_sql.py                # runs the seven queries in sql/queries.sql
python src/pandas_analysis.py        # pandas cross-check and exploratory statistics
```

(Commands for Parts B, C and D are appended here as those Parts are added.)

## Part A

### Audit of the raw data

Shapes: customers 1200 x 5, products 40 x 4, orders 13,084 x 9. Missing values (pandas default reading, so the text `N/A` also counts as missing in order_amount; the unparseable-amount figure below is counted separately on raw strings):

| table | column | missing | % |
|---|---|---|---|
| customers | age | 61 | 5.08 |
| customers | gender | 30 | 2.50 |
| orders | order_amount | 53 (`N/A` text) | 0.41 |
| orders | delivery_minutes | 2,541 | 19.42 |

All other columns have 0 missing. Dtypes: ids, city, signup_date, gender, order_ts, status are text; age and delivery_minutes are float; quantity, discount_pct, price_inr are integers; order_amount is text because of the planted `N/A` / `error` entries.

| audit figure | value | reference |
|---|---|---|
| raw orders rows | 13,084 | 13,084 |
| exact duplicate rows | 119 | 119 |
| status pending rows | 656 | 656 |
| distinct order_id | 12,309 | 12,309 |
| non-numeric order_amount | 120 | 120 |
| negative delivery_minutes | 106 | 106 |
| blank delivery_minutes | 2,541 | 2,541 |
| blank age | 61 | 61 |
| age outside 10-100 | 6 | 6 |
| distinct raw city spellings | 24 | 24 |
| distinct non-blank gender spellings | 10 | 10 |
| blank gender | 30 | 30 |

### Cleaning rules and what each changed

Customers: (a) city stripped and title-cased, 476 values changed, 6 cities remain; (b) gender mapped to male / female / unknown (30 blanks become unknown); (c) 6 impossible ages set to missing, then all 67 missing ages filled with the city median via `groupby().transform("median")`; (d) signup_date converted to datetime.

Orders, in this order:

| rule | action | rows or values affected |
|---|---|---|
| 1 | order_ts to datetime | 13,084 |
| 2 | remove exact duplicate rows | 119 removed (12,965 left) |
| 3 | per order_id keep the latest order_ts (drops the earlier pending record) | 656 removed (12,309 left) |
| 4 | order_amount to numeric; unparseable refilled as round(price x quantity x (1 - discount/100), 2) | 112 refilled |
| 5 | negative delivery_minutes set to missing; completed orders with missing delivery filled with the median of their product category among completed orders; cancelled orders left missing | 105 negatives; 774 completed refilled; 1,199 cancelled left missing |

The raw counts of 120 unparseable amounts and 106 negative times shrink to 112 and 105 because some of those rows were duplicate or pending records removed by rules 2 and 3.

After cleaning: 12,309 rows, 11,110 completed, 1,199 cancelled, 0 pending, 0 missing order_amount, 0 missing delivery among completed, completed revenue INR 2,998,668.80. Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Genders: female 610, male 560, unknown 30. No missing ages; ages 18 to 60.

### Storage and validation

Clean tables are written to `data/clean/*.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; order_ts stored as text `YYYY-MM-DD HH:MM:SS`). `src/validate.py` asserts 14 checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL results (`sql/queries.sql`, run by `src/run_sql.py`)

| query | result | reference |
|---|---|---|
| Q1 WHERE + HAVING | 30 customers | 30 |
| Q2 three-table join | 30 rows, total INR 2,998,668.80, largest Bengaluru / Electronics INR 233,909.45 | same |
| Q3 scalar subquery | 11 products above average price | 11 |
| Q4 LEFT JOIN | 25 customers with no completed order | 25 |
| Q5 CTEs | 550 customers above the average total of INR 2,552.06 | 550 / 2,552.06 |
| Q6 window ranks | 8 rows where RANK() differs from ROW_NUMBER() | 8 |
| Q7 running total | 183 daily rows, last cumulative INR 2,998,668.80 | same |

Q6 top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10).

RANK() and ROW_NUMBER() differ because when two customers in a city have identical total spend, RANK() gives them the same rank and skips the next number, while ROW_NUMBER() always assigns unique consecutive numbers.

### pandas cross-check and exploratory analysis

The pivot table with `margins=True` has grand total INR 2,998,668.80, equal to the SQL total. Per city (completed orders):

| city | sum (INR) | mean (INR) | count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences -283.50 and 645.30, outliers 0 below and 897 above.

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

**What is this data about?** 11,110 completed quick-commerce orders from April to September 2025 across six cities, with amount, quantity, discount and delivery time. **Distributions:** order amounts are strongly right-skewed, mean 269.91 against median 135.00, skewness 4.61, middle half between 64.80 and 297.00. **Outliers:** 897 orders lie above the 645.30 fence and none below; they look like genuine expensive purchases rather than errors, since amounts follow price times quantity. **Relationships:** all correlations are weak; the largest in absolute value is order_amount with discount_pct at -0.095, just ahead of order_amount with quantity at +0.087. A correlation this small does not establish causation.

## Part B

*To be added.*

## Part C

*To be added.*

## Part D

*To be added.*

## Limitations and Next Steps

*To be completed after Parts B to D. Part A note: the raw exports are synthetic, and median imputation of delivery time and age is a simple rule that hides real variation.*
