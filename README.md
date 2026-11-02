# Instamart Customer Intelligence & Support Copilot

## Overview

A capstone project that builds an internal Customer Intelligence & Support Copilot on fully synthetic data (nothing here is real Instamart data). The repository is built in four connected Parts: A turns messy order exports into a trusted SQLite database, B segments customers and predicts lapse, C triages support tickets with guardrails, D answers policy questions and runs a refund agent. All amounts are in INR. All model calls use the offline `mock_llm.py`; no key or network is needed.

Status: **Parts A, B and C complete.** Part D is filled in as it is built.

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
python src/part_b_segments.py        # RFM, K-means, PCA -> outputs/customer_segments.csv
python src/part_b_lapse.py           # lapse model -> outputs/lapse_model.joblib, outputs/retention_list.csv
python src/part_c_triage.py          # ticket triage -> outputs/triage_results.csv, outputs/triage_budget_demo.csv
```

(Commands for Part D are appended here when that Part is added.)

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

Scripts: `src/part_b_segments.py` (tasks 1 to 3) and `src/part_b_lapse.py` (tasks 4 to 8). Only completed orders are used for customer behaviour.

### RFM table

Snapshot 2025-10-01 00:00:00. Recency is `(snapshot - last_completed_order_ts).dt.days`, frequency the number of distinct completed orders, monetary the sum of completed order_amount.

| measure | result | reference |
|---|---|---|
| customers | 1,175 | 1,175 |
| recency median / min / max | 19 / 0 / 180 | 19 / 0 / 180 |
| frequency mean / max | 9.46 / 33 | 9.46 / 33 |
| monetary mean / max (INR) | 2,552.06 / 14,766.55 | 2,552.06 / 14,766.55 |

### K-means (StandardScaler, `n_init=10`, `random_state=42`)

| K | inertia | silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** Inertia falls steeply from K=2 to K=4 (1820.8 to 743.1) and much more slowly afterwards, so the elbow is at 4; the silhouette is also highest at K=4 (0.526); and four segments are few enough for the business to run a distinct campaign for each. The inertia of K=8 is always lower than that of K=2 because every extra centroid can only bring points closer to their nearest centre (with one centroid per customer the inertia would be zero), so a lower inertia alone cannot pick K; the elbow, the silhouette and interpretability have to decide.

### Segment profile (K = 4)

Cluster numbers are only labels with no ordering; names below are derived from each segment's means, not from its number.

| segment_id | name | size | mean recency (days) | mean frequency | mean monetary (INR) | premium | recommended action |
|---|---|---|---|---|---|---|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | no | bundle offers and category nudges to raise frequency |
| 1 | Premium Big Spenders | 138 | 24.12 | 6.77 | 7,129.70 | **yes** | VIP perks, early access to electronics launches, priority support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | no | protect the habit with free-delivery thresholds and referral rewards |
| 3 | At-Risk Fading | 182 | 109.04 | 2.20 | 685.99 | no | win-back coupon and reminder before they lapse |

Premium is the segment with the highest mean monetary value. PCA with two components on the scaled RFM matrix gives `explained_variance_ratio_` of 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium (1,175 rows, 138 premium).

### Lapse problem

Cutoff 2025-09-01. The cohort is every customer with at least one completed order before the cutoff; `lapsed = 1` if the customer has no completed order from 2025-09-01 to 2025-09-30. Features use only orders dated before the cutoff plus the clean customer table: recency (days from cutoff to last completed order), frequency, monetary, average discount_pct of completed orders, cancellation rate (cancelled / all orders with a final status), average delivery_minutes of completed orders, favourite product category, city, age, gender and tenure in days. The segment labels from the clustering were **not** used as features.

| item | result | reference |
|---|---|---|
| cohort | 1,156 customers | 1,156 |
| lapsed | 443 (38.32%) | 443 (38.32%) |
| split (`test_size=0.2`, stratified, `random_state=42`) | 924 train / 232 test | 924 / 232 |
| baseline "never lapsed": test accuracy / lapsed F1 | 61.6% / 0.000 | 61.6% / 0 |

### Preprocessing without leakage

City, gender and favourite category are one-hot encoded with `drop="first"` and numeric features are standardised, both inside a scikit-learn `Pipeline` so they are fitted on the 924 training rows only. The learned mean of frequency in the scaler is 8.1147 versus 8.0285 over the full cohort: they differ because the scaler saw only the training rows and never the test rows. The feature matrix has 0 missing values (asserted in code). **No encoder, scaler or resampler saw test rows, no data from 2025-09-01 onwards entered the features, and the segment labels were not used as features.**

### Tuning and test evaluation (lapsed = positive class)

| model | grid | best parameters | best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100}, cv=5 | C = 100 | 0.651 |
| RandomForest(n_estimators=100, random_state=42) | max_depth {3,5,8} x min_samples_leaf {5,10,20}, cv=5 | depth 8, leaf 10 | 0.681 |

| test set | confusion [[TN, FP],[FN, TP]] | precision | recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[111, 32],[31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30],[31, 58]] | 0.659 | 0.652 | 0.655 |

The reference best C was 1; ours is 100, the edge of the grid, because our feature set differs slightly; the test F1 values are of the stated order and above the 0.55 requirement. **Selected model: random forest**, chosen on cross-validated F1 (0.681 against 0.651), not on the test set, with a test F1 of 0.655. Missing a customer who is about to lapse loses all of their future revenue, while wrongly targeting a loyal customer costs only a coupon or a message, so recall matters at least as much as precision; the forest has the higher precision and F1 at equal recall (0.652), so it wastes fewer offers for the same number of lapsing customers caught. If the retention team cares only about catching lapsers, the balanced logistic regression below is the alternative.

### Class-balance experiment

Training set over-sampled with randomly drawn duplicates of lapsed rows (`resample`, with replacement, `random_state=42`) to 570 and 570 rows; the best logistic regression is refitted on it and tested on the original, unchanged test set.

| model | precision | recall | F1 | confusion |
|---|---|---|---|---|
| logistic regression, un-balanced | 0.644 | 0.652 | 0.648 | [[111, 32],[31, 58]] |
| logistic regression, balanced | 0.589 | 0.854 | 0.697 | [[90, 53],[13, 76]] |

Recall moved most (0.652 to 0.854) while precision did not rise (it fell from 0.644 to 0.589). For the retention team this means over-sampling catches many more lapsing customers (76 instead of 58 of 89) at the price of 21 more wrongly targeted loyal customers.

### Interpretation, persistence and retention list

Top five random-forest features: frequency (0.280), monetary (0.271), recency (0.115), average delivery minutes (0.061) and cancellation rate (0.052). Business reading: how often and how much a customer buys is by far the best signal of whether they stay, recency adds a smaller signal, and delivery experience and cancellations matter only a little.

Signs of the logistic-regression coefficients (standardised features): frequency **negative** (-1.70), more orders means less likely to lapse; recency **positive** (+0.04), a longer gap since the last order means more likely to lapse, although the effect is small once frequency is in the model; cancellation rate **negative** (-0.12), counter-intuitive and small, probably noise or overlap with frequency, so it should not be read as "cancelling keeps customers".

The selected model is saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions and probabilities exactly (checked in code).

`outputs/retention_list.csv` holds, for the 40 test-set customers in the Premium segment, the 10 with the highest predicted lapse probability (columns customer_id, lapse_probability, actually_lapsed). Only **1 of those 10** truly lapsed. The Premium flag comes from whole-window RFM that includes September, the month that defines the lapse label, so it is a post-hoc whole-window label: this retention list is illustrative and not a leakage-free evaluation (the model's own features still use only pre-cutoff orders).

## Part C

Script: `src/part_c_triage.py` (uses the unmodified `mock_llm.py`; no key or network). Tickets are processed in file order, and `mock_llm.reset()` is called immediately before every full run.

### Guardrails and redaction

| check | result | reference |
|---|---|---|
| empty tickets (status `skipped_empty`, never sent to the model) | 4 | 4 |
| long tickets (est_tokens > MAX_INPUT_TOKENS = 400, status `flagged_long`, never sent) | 4, each 734 estimated tokens | 4 / 734 |

Redaction is applied in this order before any prompt is built: email `[\w.+-]+@[\w-]+\.[\w.]+` to `[EMAIL]`, phone `\b[6-9]\d{9}\b` to `[PHONE]`, order id `ORD-\d{6}` to `[ORDER_ID]`. The user message sent to the model is exactly the redacted text.

| type | substitutions over all 164 tickets |
|---|---|
| email | 55 |
| phone | 91 |
| order id | 132 |
| **total** | **278** (reference 278) |

Re-applying the three patterns to the redacted texts finds 0 matches, and an assertion inside the pipeline checks that no prompt (system or user) contains an email, phone or order id before each call (it never fired).

### Prompts

Both contain the word CLASSIFY, list the four categories (billing, delivery, technical, fraud) and demand a JSON-only reply `{"category": "...", "reason": "..."}`. The zero-shot prompt (42 estimated tokens) has no `EXAMPLES:` line. The few-shot prompt (117 estimated tokens) adds an `EXAMPLES:` line followed by four labelled examples, including a fraud example.

### Retries, validation and budget

Every model call goes through a retry function: at most 3 retries per ticket, wait before retry n equal to `0.05 * 2**n * random.uniform(0.9, 1.15)` seconds, one log line per retry (`outputs/triage_retry_log.txt`). A full run over the 156 eligible tickets hit exactly **19** `TransientLLMError`s, all recovered by the first retry (retry number 0, waits between 0.045 and 0.0575 s), and 0 permanent errors.

Each answer is validated: valid JSON with an allowed category is `classified_json`; non-JSON text is split with `re.findall(r"[a-z]+", text.lower())` and accepted as `classified_keyword` if exactly one allowed category word appears; anything else, including valid JSON with a category outside the list ("shipping"), is `flagged_invalid_output` and goes to human review. Nothing is guessed.

| status (one full run, both prompts) | tickets | reference |
|---|---|---|
| classified_json | 135 | 135 |
| classified_keyword | 12 | 12 |
| flagged_invalid_output | 9 | 9 |
| skipped_empty | 4 | 4 |
| flagged_long | 4 | 4 |

### Zero-shot versus few-shot

| run | classified | accuracy | total tokens | tokens per classified ticket |
|---|---|---|---|---|
| zero-shot | 147 | 85.7% (126 of 147) | 10,960 | 74.56 |
| few-shot | 147 | 100% (147 of 147) | 22,660 | 154.15 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

All 21 zero-shot errors are fraud tickets predicted as billing (11) or technical tickets predicted as billing (10). The few-shot prompt costs about 80 extra tokens per classified ticket (154.15 against 74.56, roughly double, 11,700 tokens over the whole run) but removes all 21 errors. That is worth paying whenever a wrong label has a real cost: a fraud ticket labelled billing misses the security team and its 2-hour escalation target, and the 11 fraud misses in the zero-shot run are exactly that. For a low-stakes or very high-volume stream where an occasional wrong queue is cheap, the zero-shot prompt at half the tokens would be enough.

### Token budget demo and final routing

With zero-shot and `TOKEN_BUDGET = 6000` (checked before each call against the running total of `usage.total_tokens`), the run stopped calling the model after 6,059 tokens and **70 tickets** ended `budget_exhausted` (routed to human review); the result is in `outputs/triage_budget_demo.csv`. The count depends on prompt length (this prompt is 42 estimated tokens).

`outputs/triage_results.csv` (164 rows; columns ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used) comes from a few-shot run with `TOKEN_BUDGET = 100000`, routed by these rules using `outputs/customer_segments.csv` from Part B: skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted go to `human_review` (NORMAL); otherwise fraud goes to `security_team` (HIGH); otherwise a premium customer goes to `premium_desk` (HIGH); otherwise `support_queue` (NORMAL). A customer with no row in the segment file counts as not premium.

| route | tickets |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| premium_desk | 11 |
| human_review | 17 (4 empty, 4 long, 9 invalid output, 0 budget) |

**What was rule-validated and what went to humans:** 147 tickets (135 by JSON schema check, 12 by the single-category-word rule) were routed automatically; 17 tickets went to human review and received no automatic category. All model prompts contained only redacted text.

## Part D

*To be added.*

## Limitations and Next Steps

*To be completed after Part D.*

- Part A: the raw exports are synthetic, and median imputation of delivery time and age is a simple rule that hides real variation.
- Part B: the Premium flag is a post-hoc whole-window label (its RFM includes September, which defines the lapse label), so `outputs/retention_list.csv` is illustrative and not a leakage-free evaluation. The test set has only 232 customers, so test F1 differences of a few hundredths between models are within noise. Over-sampling by duplication raises recall at the cost of precision and was not tuned further.
- Part C: the mock classifier is keyword-based, so the 100% few-shot accuracy reflects the mock, not a real language model; redaction uses three fixed regular expressions and would miss names, addresses and card numbers; 9 of 156 eligible tickets (5.8%) needed a human because the mock's answer was unusable.
