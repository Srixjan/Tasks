# Task 7 — Findings: MPLADS Fund Utilization EDA

## Research Question
Which Members of Parliament and states convert their sanctioned MPLADS funds into
completed work most efficiently, and where does the money tend to get stuck?

## State-Level Findings
Averaging utilization rate by state shows Nagaland at the top (~0.60), but this is
based on a very small number of MPs from that state, so it's not a statistically
robust finding — more a data point to note than a strong conclusion. Excluding
small-sample states, Mizoram, Sikkim, and Manipur also rank highly. At the bottom,
several states (Andaman and Nicobar Islands, Lakshadweep, Ladakh, The Dadra and
Nagar Haveli and Daman and Diu) show 0.0 average utilization — meaning the MPs
representing these regions in this dataset show no disbursed funds at all.

## MP-Level Findings
After filtering to MPs with allocated amounts above ₹1.47 crore (to avoid small
allocations producing misleadingly perfect utilization ratios), the top-performing
MP, Dr. Rajesh Mishra (Madhya Pradesh), converts only 45% of allocated funds into
disbursed work. This is a meaningful finding on its own: even the best-performing,
high-allocation MP in the dataset is far from fully utilizing available funds.
At the other end, several MPs with allocations above ₹1.47 crore show 0% utilization —
meaning none of their allocated funds have resulted in any disbursed, completed work
based on this snapshot.

## Category-Level Findings
Grouping the raw Works Sanctioned and Works Completed tables by work category
(Normal/Others, Repair and Renovation, Trust and Society) produced an unexpected
result: total disbursed amount exceeds total sanctioned amount in every single
category, by roughly 2-3x. This is the inverse of what the MP-level and state-level
analysis showed. This isn't a pipeline bug — it's the same root cause documented in
Task 4: some completed works in this snapshot don't have a matching sanctioned
record, because the sanctioned data reflects a different (rolling, multi-year)
time window than the completed data. At the category level, this effect is large
enough to flip the direction of the gap entirely.

## Data Limitations
- 116 MPs show zero sanctioned and zero completed work activity in this dataset.
- 217 MPs (~40%) show completed work with no matching sanctioned record ("orphan"
  completions), consistent with the sanctioned/completed tables being multi-year
  rolling snapshots rather than a perfectly aligned point-in-time pair.
- The category-level gap analysis inherits and amplifies this same limitation —
  disbursed exceeds sanctioned in all three categories, which should not be read
  as "no backlog exists," but as an artifact of snapshot misalignment.
- One row with a blank `state` value and one row with a blank `work_category` value
  were found and dropped before analysis.
- A trailing total/summary row was found at the end of all three raw CSVs and
  dropped during cleaning, as it was not a real data record.

## Next Steps
This cleaned dataset, along with the engineered features (`utilization_rate`,
`sanctioned_backlog`, `completion_ratio`), is now ready to feed into Phase 2 of
the MPLADS Anomaly Detection Platform: a completion-risk classifier built during
the Supervised ML section of the course.