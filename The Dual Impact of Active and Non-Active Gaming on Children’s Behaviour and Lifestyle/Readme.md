# The Dual Impact of Active and Non-Active Gaming on Children’s Behaviour and Lifestyle

Single-notebook analysis that cleans a children’s daily-activity dataset, engineers gaming/physical-activity features, segments participants with K-Means, and visualises how behaviours change across intervention phases.

## What this notebook does
- **Load & clean**
  - Reads the 24-hour recall activity dataset (CSV/XLSX) from `data/`.
  - Converts minutes→hours and merges duplicate hour/minute columns.
  - Keeps core variables: `Active_Video_Game_Hours`, `Non_Active_Video_Game_Hours`, `Sports_Hours`, `Walking_Biking_Hours`, `Household_Chores_Hours`, `Physical_Education_Hours`, plus `Week`/phase labels. :contentReference[oaicite:0]{index=0}

- **Segment (K-Means)**
  - Standardises selected features and fits K-Means (K chosen via elbow method).
  - Adds a `cluster` label to each participant/week. (See elbow chart reference below.) :contentReference[oaicite:1]{index=1}

- **Visualise behaviour**
  - **Cluster characteristics:** average time spent on each activity by cluster. *(Figure 2 in report)* :contentReference[oaicite:2]{index=2}
  - **Weekly trends:** sports activity over time by cluster to track baseline → intervention (weeks 2–6) → washout (week 10). *(Figure 3)* :contentReference[oaicite:3]{index=3}
  - **Active vs non-active gaming:** comparison across clusters. *(Figure 4)* :contentReference[oaicite:4]{index=4}
  - **Phase-wise distributions:** violin plots for active and non-active gaming across phases. *(Figures 5–6)* :contentReference[oaicite:5]{index=5}
  - **Cluster sizes:** share of participants per cluster. *(Figure 7)* :contentReference[oaicite:6]{index=6}

- **Interpret**
  - Summarises how intervention weeks increased active gaming/physical activity for some clusters, but effects tapered in the washout week without sustained support; recommends balanced, personalised interventions. :contentReference[oaicite:7]{index=7}

## Files
- `The Dual Impact of Active and Non-Active Gaming on Children’s Behaviour and Lifestyle.ipynb` — main notebook
- `The Dual Impact of Active and Non-Active Gaming on Children’s Behaviour and Lifestyle.pdf` — short report with figures and discussion (see pages with Figs. 1–7). :contentReference[oaicite:8]{index=8}
