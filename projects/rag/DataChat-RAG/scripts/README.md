# Database Setup Scripts

This directory contains scripts for setting up and managing the DataChat-RAG database.

## setup_database.py

Creates PostgreSQL schema and seeds sample data for a healthcare AdTech company.

### Database Schema

#### Tables

1. **campaigns** - Advertising campaign information
   - Campaign metadata (name, client, industry, dates, budget, status)
   - Targeting criteria stored as JSON
   - Industry types: healthcare, pharma, medical_device

2. **impressions** - Individual ad impressions
   - Timestamp, device type, geo location, ad placement, cost

3. **clicks** - Click events on ads
   - Links to impressions and campaigns
   - Landing page tracking

4. **conversions** - Conversion events (leads, signups, purchases)
   - Conversion type and value tracking

5. **daily_metrics** - Aggregated daily metrics (for fast queries)
   - Pre-aggregated impressions, clicks, conversions, spend
   - Calculated metrics: CTR, CVR, CPA, CPC, CPM

### Usage

```bash
# Complete setup (create database, tables, and seed data)
python scripts/setup_database.py --all

# Create database only
python scripts/setup_database.py --create-db

# Create tables only
python scripts/setup_database.py --create-tables

# Seed data only
python scripts/setup_database.py --seed

# Drop all tables (CAUTION!)
python scripts/setup_database.py --drop-tables

# Custom number of campaigns and days
python scripts/setup_database.py --all --campaigns 200 --days 60

# Generate raw impression/click events (slower, more realistic)
python scripts/setup_database.py --all --raw-events
```

### Environment Variables

Ensure `.env` file is configured with:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=datachat_rag
DB_USER=postgres
DB_PASSWORD=your-password
```

### Sample Data Characteristics

- **Campaigns**: 100 by default (configurable)
- **Time Period**: 30 days by default (configurable)
- **Clients**: 20 realistic healthcare AdTech company names
- **Industries**: healthcare, pharma, medical_device
- **CTR**: 0.5% - 3% (realistic for healthcare)
- **CVR**: 1% - 5% (realistic for B2B healthcare)
- **Budget Range**: $5,000 - $500,000 per campaign
- **Ad Placements**: banner, video, native, interstitial, sponsored_content
- **Device Types**: mobile, desktop, tablet
- **Geo Locations**: 15 US states

### Query Examples

```sql
-- Top performing campaigns by CPA
SELECT
    c.name,
    c.client_name,
    SUM(dm.spend) as total_spend,
    SUM(dm.conversions) as total_conversions,
    SUM(dm.spend) / NULLIF(SUM(dm.conversions), 0) as cpa
FROM campaigns c
JOIN daily_metrics dm ON c.id = dm.campaign_id
GROUP BY c.id, c.name, c.client_name
ORDER BY cpa ASC
LIMIT 10;

-- Daily trends for last 7 days
SELECT
    date,
    SUM(impressions) as impressions,
    SUM(clicks) as clicks,
    SUM(conversions) as conversions,
    SUM(spend) as spend
FROM daily_metrics
WHERE date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY date
ORDER BY date;

-- Campaign performance by industry
SELECT
    industry,
    COUNT(*) as num_campaigns,
    AVG(ctr) as avg_ctr,
    AVG(cvr) as avg_cvr,
    SUM(spend) as total_spend
FROM campaigns c
JOIN daily_metrics dm ON c.id = dm.campaign_id
GROUP BY industry;
```
