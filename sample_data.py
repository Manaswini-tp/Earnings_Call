# sample_data.py
# Sample transcript and questions for testing

SAMPLE_TRANSCRIPT = """
Apple Inc. Q3 2024 Earnings Call

Tim Cook, CEO:
"Today Apple is reporting revenue of $85.8 billion, up 5% year over year. 
EPS came in at $1.40, above our guidance range."

"We saw particular strength in Services, which reached an all-time high 
of $24.2 billion. iPhone revenue was $39.3 billion."

Luca Maestri, CFO:
"Gross margin was 44.5% for the quarter. Looking ahead to Q4, we expect 
revenue to be similar to Q3 with year-over-year growth of low single digits."

Key Risks Discussed:
1. Supply chain constraints in Southeast Asia
2. Regulatory pressure from EU Digital Markets Act
3. Slowing consumer demand in China
4. Currency headwinds from strong US dollar

Forward Guidance:
While we expect continued growth in Services, margin pressure from 
increased R&D spending and component costs may partially offset revenue growth.
"""

SAMPLE_QUESTIONS = {
    "extract": "What was the reported EPS and revenue for the quarter?",
    "identify": "List the top 3 risk factors management mentioned.",
    "synthesize": "Management gave revenue guidance for next quarter, but also mentioned margin pressure. What's your net read on forward earnings?"
}

SAMPLE_GROUND_TRUTH = {
    "extract": {
        "eps": "1.40",
        "revenue": "85.8"
    },
    "identify": [
        "supply chain constraints",
        "regulatory pressure",
        "slowing consumer demand"
    ],
    "synthesize_rubric": {
        "factual_grounding": 0.4,
        "reasoning_quality": 0.3,
        "completeness": 0.3
    }
}