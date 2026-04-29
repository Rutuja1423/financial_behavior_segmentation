import nbformat

notebook_path = "financial_behavior_segmentation_analysis.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

business_insights = {
    "The dataset contains demographic": "Having a rich set of demographic and behavioral data enables precise segmentation, which is critical for personalized financial product recommendations.",
    "The dataset contains small item-level": "Data imputation and handling missing values correctly ensures that no valuable behavioral signal is lost, maximizing the reach of our insights.",
    "Duplicate records should be": "Data integrity is paramount in behavioral analytics. Clean data ensures our recommendations are based on actual distinct user behaviors.",
    "The sample leans towards users": "This highlights our core demographic—young adults who are early in their financial journey and prime candidates for wealth-building features.",
    "The behavioral scales show strong": "Strong reliability guarantees that our behavioral profiles are statistically sound before we use them to target users in a production environment.",
    "There is a moderate negative": "Behavioral spending directly impacts savings outcomes more than income alone. Fintech apps should introduce 'cool-off' nudges during high impulse moments.",
    "Stress spending has a weak": "Emotional spending is a key friction point; users need supportive interventions during high-stress periods rather than punitive alerts.",
    "There is a moderate positive": "Educational content directly translates into stickier budgeting behaviors, proving the ROI of investing in financial education features.",
    "While the 'Student' group has": "Employment type doesn't dictate savings success; behavioral intervention works for everyone. Financial apps shouldn't gatekeep premium savings tools based on employment status alone.",
    "The regression model explains": "Psychological traits strongly predict financial health, allowing for early intervention scoring. *(Note: Since the dataset is synthetic, relationships may appear stronger than real-world data. In real scenarios, additional noise and external factors would likely reduce model performance.)*",
    "PCA reduced the 5 behavioral": "By reducing dimensions, we can build more robust and interpretable models, which means faster inference in real-time personalization engines.",
    "The best silhouette score": "Note: The silhouette score of 0.246 indicates moderate cluster separation, suggesting behavioral overlap between users. Increasing feature engineering or exploring alternative clustering methods (e.g., hierarchical clustering or GMM) could improve segmentation quality. (Tried K = 2–6. Selected K=2 based on silhouette + interpretability).",
    "The K-Means algorithm identified": "Identifying overarching behaviors allows us to design distinct product experiences—one geared toward wealth growth and another toward spending control.",
    "Cluster 0": "Cluster 0 users need spending alerts, category limits, and short-term savings challenges. Cluster 1 users need investment education, goal-based tools, and automated plans.",
}

for cell in nb.cells:
    if cell.cell_type == 'markdown' and '**Interpretation:**' in cell.source:
        lines = cell.source.split('\n')
        
        # Check if we already have Business Insight to avoid duplication
        if '**Business Insight:**' in cell.source:
            continue
            
        # Match the interpretation text to our insights
        matched_insight = None
        for key, insight in business_insights.items():
            if key in cell.source:
                matched_insight = insight
                break
                
        if matched_insight:
            cell.source = cell.source.strip() + f"\n\n**Business Insight:**  \n{matched_insight}"

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook updated successfully with storytelling elements.")
