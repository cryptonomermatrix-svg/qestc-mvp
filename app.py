import os
import json
import pandas as pd
from datetime import datetime, timedelta
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Metric, Dimension
from openai import OpenAI
import serpapi
from sklearn.cluster import KMeans
import sqlite3
import unittest

# Set API keys (replace with your own; better to use env vars)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your-ga4-credentials.json'  # Path to GA4 service account key
OPENAI_API_KEY = 'your-openai-api-key'
SERPAPI_API_KEY = 'your-serpapi-api-key'

# Initialize clients
ga_client = BetaAnalyticsDataClient()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Database setup (SQLite for storing keywords, reports)
DB_FILE = 'seo_data.db'
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS keywords (keyword TEXT UNIQUE, rank INTEGER, cluster TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS reports (date TEXT, report TEXT)''')
conn.commit()

# Module 1: SEO and Optimization (GA4 Analysis)
def get_ga_report(property_id, start_date, end_date):
    """Pull GA4 report for pages and sessions."""
    request = RunReportRequest(
        property=f'properties/{property_id}',
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name='pagePath')],
        metrics=[Metric(name='activeUsers'), Metric(name='sessions')]
    )
    response = ga_client.run_report(request)
    rows = []
    for row in response.rows:
        rows.append({
            'page': row.dimension_values[0].value,
            'users': int(row.metric_values[0].value),
            'sessions': int(row.metric_values[1].value)
        })
    return pd.DataFrame(rows)

def analyze_ga_metrics(report_df):
    """Use GPT to analyze popular pages and suggest optimizations."""
    most_pages = report_df.sort_values('users', ascending=False).head(10).to_json()
    prompt = f"Analyze these popular pages: {most_pages}. Suggest SEO improvements and key SOPs."
    response = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def crawl_faqs(site_url):
    """Placeholder for FAQ crawling; use GPT to generate/rephrase (integrate web scrape if needed)."""
    prompt = f"Generate 5 FAQs for site: {site_url} related to crypto decoding."
    response = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    faqs = response.choices[0].message.content
    # Rephrase answers
    rephrase_prompt = f"Rephrase these FAQs for better SEO: {faqs}"
    rephrased = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": rephrase_prompt}]).choices[0].message.content
    return rephrased

# Module 2: SERP & Competitor Analysis
def upload_keywords(keywords_list):
    """Upload/store keywords."""
    for kw in keywords_list:
        cursor.execute("INSERT OR IGNORE INTO keywords (keyword) VALUES (?)", (kw,))
    conn.commit()
    return keywords_list

def get_serp_results(keyword):
    """Fetch SERP via serpapi."""
    params = {'q': keyword, 'api_key': SERPAPI_API_KEY, 'num': 10}
    results = serpapi.search(params)
    organic = results.get('organic_results', [])
    return [{'position': r['position'], 'title': r['title'], 'snippet': r['snippet'], 'link': r['link']} for r in organic]

def analyze_serp(keyword):
    """Use GPT to analyze SERP, images, snippets."""
    serp_data = get_serp_results(keyword)
    prompt = f"Analyze SERP for '{keyword}': {json.dumps(serp_data)}. Suggest rank improvements, related keywords, and deduplicate."
    response = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    analysis = response.choices[0].message.content
    # Extract related (mock parse)
    related = [kw.strip() for kw in analysis.split('\n') if kw.startswith('- Related:')]
    # Save report
    save_report(analysis)
    return analysis, related

def save_report(report_content):
    """Save report to DB."""
    date = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("INSERT INTO reports (date, report) VALUES (?, ?)", (date, report_content))
    conn.commit()

# Module 3: Analyze & Save Keyword Performance
def get_ranks(keywords):
    """Get current ranks."""
    ranks = {}
    for kw in keywords:
        serp = get_serp_results(kw)
        ranks[kw] = serp[0]['position'] if serp else None  # Assume top result is target site
    return ranks

def cluster_keywords(keywords, ranks):
    """Use GPT or KMeans for clustering."""
    data = pd.DataFrame({'keyword': keywords, 'rank': [ranks.get(kw, 0) for kw in keywords]})
    # GPT cluster
    prompt = f"Cluster these keywords by theme: {data.to_json()}"
    response = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    clusters = response.choices[0].message.content
    # Alternative: KMeans on embeddings (simplified)
    # For production, use OpenAI embeddings + KMeans
    for kw, cluster in zip(keywords, clusters.split('\n')):  # Mock parse
        cursor.execute("UPDATE keywords SET cluster = ? WHERE keyword = ?", (cluster, kw))
    conn.commit()
    return clusters

# Module 4: Create Report & Rewrite Article
def create_full_report(ga_analysis, serp_analysis, clusters):
    """Compile report."""
    report = f"GA4 Analysis: {ga_analysis}\nSERP Analysis: {serp_analysis}\nClusters: {clusters}"
    save_report(report)
    return report

def rewrite_article(original_text, keywords):
    """Rewrite for SEO."""
    prompt = f"Rewrite this article incorporating keywords {keywords} for better SEO: {original_text}"
    response = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# Update Functionality
def update_data(property_id, start_date=None, end_date=None, keywords=[]):
    """Pull fresh data and update DB."""
    if not start_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    report_df = get_ga_report(property_id, start_date, end_date)
    ga_analysis = analyze_ga_metrics(report_df)
    upload_keywords(keywords)
    for kw in keywords:
        serp_analysis, related = analyze_serp(kw)
        upload_keywords(related)  # Add related
    ranks = get_ranks(keywords)
    clusters = cluster_keywords(keywords, ranks)
    full_report = create_full_report(ga_analysis, serp_analysis, clusters)
    return full_report

# Backtesting
def backtest(property_id, historical_start, historical_end, mock_keywords, mock_serp_data=None):
    """Simulate workflow on historical data."""
    # Pull historical GA4
    historical_df = get_ga_report(property_id, historical_start, historical_end)
    ga_analysis = analyze_ga_metrics(historical_df)
    # Mock SERP if no real historical (use provided or simulate)
    if not mock_serp_data:
        mock_serp_data = {kw: [{'position': 5, 'title': 'Mock', 'snippet': 'Mock'}] for kw in mock_keywords}
    ranks = {kw: mock_serp_data[kw][0]['position'] for kw in mock_keywords}
    clusters = cluster_keywords(mock_keywords, ranks)
    # Simulate optimization: Assume 20% rank improvement
    simulated_ranks = {kw: max(1, ranks[kw] - int(ranks[kw]*0.2)) for kw in mock_keywords}
    # Calculate metrics: e.g., estimated traffic uplift (simplified formula)
    uplift = sum([100 / r for r in simulated_ranks.values()]) - sum([100 / r for r in ranks.values()])
    return f"Backtest Results: Uplift {uplift:.2f}%. GA: {ga_analysis}\nClusters: {clusters}"

# Main Execution Example
if __name__ == "__main__":
    # Example usage: Update
    PROPERTY_ID = 'your-ga4-property-id'  # Replace
    keywords = ['blockchain decoding', 'crypto trading']
    report = update_data(PROPERTY_ID, keywords=keywords)
    print("Updated Report:", report)
    
    # Example: Rewrite
    original = "This is a sample crypto article."
    rewritten = rewrite_article(original, keywords)
    print("Rewritten Article:", rewritten)
    
    # Example: Backtest
    hist_start = '2023-01-01'
    hist_end = '2023-12-31'
    mock_kw = keywords
    backtest_result = backtest(PROPERTY_ID, hist_start, hist_end, mock_kw)
    print("Backtest:", backtest_result)

# Testing
class TestSEOWorkflow(unittest.TestCase):
    def test_get_ga_report(self):
        # Mock property_id for test; in real, use valid
        try:
            df = get_ga_report('invalid-for-test', '2024-01-01', '2024-01-02')
            self.assertIsInstance(df, pd.DataFrame)
        except Exception as e:
            self.assertIn("error", str(e).lower())  # Expect error on invalid

    def test_analyze_ga_metrics(self):
        mock_df = pd.DataFrame({'page': ['/home'], 'users': [100], 'sessions': [200]})
        analysis = analyze_ga_metrics(mock_df)
        self.assertTrue(len(analysis) > 0)

    def test_get_serp_results(self):
        results = get_serp_results('test keyword')
        self.assertIsInstance(results, list)

    def test_cluster_keywords(self):
        kw = ['test1', 'test2']
        ranks = {'test1': 1, 'test2': 2}
        clusters = cluster_keywords(kw, ranks)
        self.assertTrue(len(clusters) > 0)

    def test_backtest(self):
        result = backtest('invalid', '2023-01-01', '2023-01-02', ['test'], {'test': [{'position': 10}]})
        self.assertIn("Uplift", result)

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)  # Run tests
