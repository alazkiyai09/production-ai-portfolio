#!/usr/bin/env python3
"""
Database Setup Script for DataChat-RAG

Creates PostgreSQL schema and seeds sample data for a healthcare AdTech company.
"""

import os
import random
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.schema import CreateTable

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', 'postgres')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'datachat_rag')}"
)

Base = declarative_base()


# =============================================================================
# Security Utilities
# =============================================================================

VALID_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def validate_identifier(identifier: str, max_length: int = 63) -> str:
    """
    Validate a SQL identifier (table name, column name, database name) to prevent SQL injection.

    PostgreSQL identifiers must:
    - Start with a letter or underscore
    - Contain only letters, numbers, and underscores
    - Be at most 63 characters long

    Args:
        identifier: The identifier to validate
        max_length: Maximum allowed length (default: 63 for PostgreSQL)

    Returns:
        The validated identifier

    Raises:
        ValueError: If the identifier contains unsafe characters
    """
    if not identifier:
        raise ValueError("Identifier cannot be empty")

    if len(identifier) > max_length:
        raise ValueError(f"Identifier exceeds maximum length of {max_length} characters")

    if not VALID_IDENTIFIER_PATTERN.match(identifier):
        raise ValueError(
            f"Invalid identifier '{identifier}'. "
            "Identifiers must start with a letter or underscore and contain only letters, numbers, and underscores."
        )

    return identifier


# =============================================================================
# SQLAlchemy Models
# =============================================================================

class Campaign(Base):
    """Campaign table for healthcare advertising campaigns."""

    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    client_name = Column(String(255), nullable=False)
    industry = Column(String(50), nullable=False)  # healthcare, pharma, medical_device
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)
    budget = Column(Float, nullable=False)  # in USD
    actual_spend = Column(Float, default=0.0)
    status = Column(String(50), nullable=False, default="active")  # active, paused, completed

    # Targeting criteria stored as JSON
    targeting_criteria = Column(JSON, nullable=True)

    # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    impressions = relationship("Impression", back_populates="campaign", cascade="all, delete-orphan")
    clicks = relationship("Click", back_populates="campaign", cascade="all, delete-orphan")
    conversions = relationship("Conversion", back_populates="campaign", cascade="all, delete-orphan")
    daily_metrics = relationship("DailyMetrics", back_populates="campaign", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Campaign(id={self.id}, name='{self.name}', client='{self.client_name}')>"


class Impression(Base):
    """Individual ad impressions."""

    __tablename__ = "impressions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    device_type = Column(String(50), nullable=False)  # mobile, desktop, tablet
    geo_location = Column(String(100), nullable=False)  # US state or country
    ad_placement = Column(String(100), nullable=False)  # banner, video, native, etc.
    cost = Column(Float, nullable=False)  # CPM cost in USD

    # Relationships
    campaign = relationship("Campaign", back_populates="impressions")
    clicks = relationship("Click", back_populates="impression", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Impression(id={self.id}, campaign_id={self.campaign_id}, timestamp={self.timestamp})>"


class Click(Base):
    """Individual click events on ads."""

    __tablename__ = "clicks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    impression_id = Column(UUID(as_uuid=True), ForeignKey("impressions.id"), nullable=False)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    landing_page = Column(String(500), nullable=True)

    # Relationships
    impression = relationship("Impression", back_populates="clicks")
    campaign = relationship("Campaign", back_populates="clicks")
    conversions = relationship("Conversion", back_populates="click", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Click(id={self.id}, impression_id={self.impression_id}, timestamp={self.timestamp})>"


class Conversion(Base):
    """Conversion events (leads, signups, purchases)."""

    __tablename__ = "conversions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    click_id = Column(UUID(as_uuid=True), ForeignKey("clicks.id"), nullable=False)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    conversion_type = Column(String(100), nullable=False)  # lead, signup, purchase, download
    value = Column(Float, nullable=False, default=0.0)  # Conversion value in USD

    # Relationships
    click = relationship("Click", back_populates="conversions")
    campaign = relationship("Campaign", back_populates="conversions")

    def __repr__(self) -> str:
        return f"<Conversion(id={self.id}, type={self.conversion_type}, value={self.value})>"


class DailyMetrics(Base):
    """Aggregated daily metrics for fast querying."""

    __tablename__ = "daily_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(Date, nullable=False, index=True)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False, index=True)

    # Metrics
    impressions = Column(Integer, nullable=False, default=0)
    clicks = Column(Integer, nullable=False, default=0)
    conversions = Column(Integer, nullable=False, default=0)
    spend = Column(Float, nullable=False, default=0.0)

    # Calculated metrics
    ctr = Column(Float, nullable=False, default=0.0)  # Click-through rate
    cvr = Column(Float, nullable=False, default=0.0)  # Conversion rate
    cpa = Column(Float, nullable=True)  # Cost per acquisition
    cpc = Column(Float, nullable=True)  # Cost per click
    cpm = Column(Float, nullable=True)  # Cost per mille (impressions)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    campaign = relationship("Campaign", back_populates="daily_metrics")

    def __repr__(self) -> str:
        return f"<DailyMetrics(date={self.date}, campaign_id={self.campaign_id}, impressions={self.impressions})>"


# =============================================================================
# Setup Functions
# =============================================================================

def create_database():
    """Create the database if it doesn't exist."""
    # Connect to default postgres database first
    default_url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', 'postgres')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/postgres"
    )

    engine = create_engine(default_url)
    conn = engine.connect()
    conn.execute("commit")

    db_name = os.getenv("DB_NAME", "datachat_rag")

    # Check if database exists (using parameterized query for safety)
    # Note: PostgreSQL doesn't allow parameters for database name in CREATE DATABASE,
    # so we validate the database name against a whitelist of safe characters
    safe_db_name = validate_identifier(db_name)
    result = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :db_name"), {"db_name": safe_db_name})
    exists = result.fetchone() is not None

    if not exists:
        # CREATE DATABASE cannot use parameters in PostgreSQL, but we validated the name
        conn.execute(text(f"CREATE DATABASE \"{safe_db_name}\""))
        print(f"✓ Database '{safe_db_name}' created")
    else:
        print(f"✓ Database '{db_name}' already exists")

    conn.close()
    engine.dispose()


def create_tables():
    """Create all tables in the database."""
    engine = create_engine(DATABASE_URL, echo=False)

    print("Creating tables...")
    Base.metadata.create_all(engine)
    print("✓ All tables created successfully")

    engine.dispose()


def drop_tables():
    """Drop all tables (use with caution!)."""
    engine = create_engine(DATABASE_URL, echo=False)

    response = input("Are you sure you want to drop all tables? (yes/no): ")
    if response.lower() == "yes":
        Base.metadata.drop_all(engine)
        print("✓ All tables dropped")
    else:
        print("Operation cancelled")

    engine.dispose()


# =============================================================================
# Sample Data Generation
# =============================================================================

# Sample data for realistic healthcare AdTech campaigns
HEALTHCARE_CLIENTS = [
    "MedTech Solutions", "PharmaCorp Inc", "HealthFirst Systems",
    "BioGen Laboratories", "CareConnect", "MediCare Plus",
    "HealthWave Analytics", "Pharm Dynamics", "VitalSigns Inc",
    "ClinicalPath Solutions", "MedDevice Innovations", "HealthGenomics",
    "PharmaEdge", "CareNetwork Systems", "BioHealth Analytics",
    "MedStream Solutions", "HealthTech Partners", "PharmaGrowth",
    "VitalCare Systems", "MediSphere Analytics"
]

INDUSTRIES = ["healthcare", "pharma", "medical_device"]

CAMPAIGN_NAME_TEMPLATES = [
    "{client} Brand Awareness Q4",
    "{client} Lead Generation Campaign",
    "{client} Product Launch",
    "{client} Healthcare Professionals Reach",
    "{client} Patient Education Series",
    "{client} Digital Health Summit",
    "{client} Clinical Trials Recruitment",
    "{client} Medical Conference Campaign",
    "{client} Healthcare Provider Network",
    "{client} Telemedicine Promotion"
]

AD_PLACEMENTS = ["banner", "video", "native", "interstitial", "sponsored_content"]

DEVICE_TYPES = ["mobile", "desktop", "tablet"]

GEO_LOCATIONS = [
    "California", "New York", "Texas", "Florida", "Illinois",
    "Pennsylvania", "Ohio", "Georgia", "North Carolina", "Michigan",
    "New Jersey", "Virginia", "Washington", "Arizona", "Massachusetts"
]

CONVERSION_TYPES = ["lead", "signup", "download", "consultation", "trial_request"]

TARGETING_TEMPLATES = {
    "healthcare": {
        "audience": ["healthcare_professionals", "patients", "caregivers"],
        "specialties": ["primary_care", "cardiology", "oncology", "pediatrics"],
        "age_groups": ["25-34", "35-44", "45-54", "55-64", "65+"],
    },
    "pharma": {
        "audience": ["physicians", "pharmacists", "healthcare_administrators"],
        "specialties": ["oncology", "neurology", "cardiology", "rare_diseases"],
        "practice_types": ["hospital", "clinic", "private_practice"],
    },
    "medical_device": {
        "audience": ["surgeons", "hospital_administrators", "procurement_managers"],
        "device_types": ["diagnostic", "therapeutic", "surgical", "monitoring"],
        "facility_sizes": ["small", "medium", "large", "teaching_hospital"],
    }
}

CAMPAIGN_STATUSES = ["active", "paused", "completed"]


def generate_targeting_criteria(industry: str) -> Dict[str, Any]:
    """Generate realistic targeting criteria based on industry."""
    template = TARGETING_TEMPLATES[industry]

    return {
        "audience": random.sample(template["audience"], k=random.randint(1, len(template["audience"]))),
        "specialties": random.sample(template.get("specialties", ["general"]), k=random.randint(1, 2)),
        "age_groups": random.sample(template.get("age_groups", ["25-54"]), k=random.randint(1, 3)),
        "gender": random.choice(["all", "male", "female"]),
        "income_levels": random.sample(["low", "medium", "high", "affluent"], k=random.randint(1, 3)),
    }


def generate_campaigns(session, num_campaigns: int = 100, days_back: int = 90) -> List[Campaign]:
    """Generate campaign records with realistic data."""
    campaigns = []
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    print(f"Generating {num_campaigns} campaigns...")

    for i in range(num_campaigns):
        client = random.choice(HEALTHCARE_CLIENTS)
        industry = random.choice(INDUSTRIES)

        # Create campaign name
        name_template = random.choice(CAMPAIGN_NAME_TEMPLATES)
        name = name_template.format(client=client) + f" {random.randint(1, 2024)}"

        # Date range (campaigns last 30-90 days)
        campaign_start = start_date + timedelta(days=random.randint(0, 60))
        duration_days = random.randint(30, 90)
        campaign_end = campaign_start + timedelta(days=duration_days)

        # Budget: $5,000 to $500,000
        budget = random.randint(5000, 500000)

        # Status based on dates
        if campaign_end < end_date:
            status = random.choice(["completed", "paused"])
        elif campaign_start > end_date:
            status = "paused"
        else:
            status = random.choice(["active", "paused"])

        campaign = Campaign(
            name=name,
            client_name=client,
            industry=industry,
            start_date=campaign_start,
            end_date=campaign_end if status == "completed" else None,
            budget=budget,
            actual_spend=budget * random.uniform(0.3, 1.2) if status == "completed" else 0,
            status=status,
            targeting_criteria=generate_targeting_criteria(industry)
        )

        campaigns.append(campaign)
        session.add(campaign)

    session.commit()
    print(f"✓ Generated {len(campaigns)} campaigns")

    return campaigns


def generate_impressions_and_clicks(session, campaigns: List[Campaign], days_to_generate: int = 30):
    """Generate impression and click data for campaigns.

    This generates raw impression/click data for a subset of campaigns.
    For performance, we generate daily_metrics instead for full history.
    """
    print(f"Generating impressions and clicks for {days_to_generate} days...")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_to_generate)

    # Use a smaller subset of campaigns for raw event data
    sample_campaigns = random.sample(campaigns, min(20, len(campaigns)))

    total_impressions = 0
    total_clicks = 0

    for campaign in sample_campaigns:
        # Skip if campaign wasn't active in this period
        campaign_start = max(campaign.start_date, start_date)
        campaign_end = campaign.end_date or end_date
        campaign_end = min(campaign_end, end_date)

        if campaign_start > campaign_end:
            continue

        # Generate data for each day
        current_date = campaign_start
        while current_date <= campaign_end:
            # Daily impressions: 1,000 to 100,000 per day
            daily_impressions = random.randint(1000, 100000)

            # CTR: 0.5% to 3% (typical for healthcare)
            ctr = random.uniform(0.005, 0.03)
            daily_clicks = int(daily_impressions * ctr)

            # Generate impressions throughout the day
            for _ in range(daily_impressions):
                # Distribute impressions across hours
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                timestamp = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)

                # Cost per impression (CPM basis): $2 to $50 CPM
                cpm = random.uniform(2.0, 50.0)
                cost = cpm / 1000

                impression = Impression(
                    campaign_id=campaign.id,
                    timestamp=timestamp,
                    device_type=random.choice(DEVICE_TYPES),
                    geo_location=random.choice(GEO_LOCATIONS),
                    ad_placement=random.choice(AD_PLACEMENTS),
                    cost=cost
                )
                session.add(impression)
                total_impressions += 1

            # Generate clicks (subset of impressions)
            impression_ids = []  # We'll get these after commit
            for _ in range(daily_clicks):
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                timestamp = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)

                landing_pages = [
                    f"https://{campaign.client_name.lower().replace(' ', '')}.com/learn-more",
                    f"https://{campaign.client_name.lower().replace(' ', '')}.com/contact",
                    f"https://{campaign.client_name.lower().replace(' ', '')}.com/trial",
                    f"https://{campaign.client_name.lower().replace(' ', '')}.com/demo",
                ]

                # For simplicity, we'll associate clicks with campaign_id directly
                # In production, you'd link to actual impression_ids
                click = Click(
                    campaign_id=campaign.id,
                    impression_id=None,  # Would link to actual impression in production
                    timestamp=timestamp,
                    landing_page=random.choice(landing_pages)
                )
                session.add(click)
                total_clicks += 1

            current_date += timedelta(days=1)

        # Commit in batches to avoid memory issues
        session.commit()

    print(f"✓ Generated {total_impressions:,} impressions and {total_clicks:,} clicks")


def generate_conversions(session, campaigns: List[Campaign], days_to_generate: int = 30):
    """Generate conversion data from clicks."""
    print(f"Generating conversions for {days_to_generate} days...")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_to_generate)

    sample_campaigns = random.sample(campaigns, min(15, len(campaigns)))
    total_conversions = 0

    for campaign in sample_campaigns:
        # Get clicks for this campaign
        clicks = session.query(Click).filter(
            Click.campaign_id == campaign.id,
            Click.timestamp >= datetime.combine(start_date, datetime.min.time()),
            Click.timestamp <= datetime.combine(end_date, datetime.max.time())
        ).all()

        # CVR: 1% to 5% (typical for healthcare B2B)
        cvr = random.uniform(0.01, 0.05)
        num_conversions = max(1, int(len(clicks) * cvr))

        # Sample clicks to convert
        sampled_clicks = random.sample(clicks, min(num_conversions, len(clicks)))

        for click in sampled_clicks:
            # Conversion happens within 24 hours of click
            delay_minutes = random.randint(5, 1440)
            conv_timestamp = click.timestamp + timedelta(minutes=delay_minutes)

            # Conversion value: $50 to $5,000 depending on type
            conv_type = random.choice(CONVERSION_TYPES)
            if conv_type == "lead":
                value = random.uniform(50, 500)
            elif conv_type == "trial_request":
                value = random.uniform(500, 2000)
            elif conv_type == "consultation":
                value = random.uniform(200, 1000)
            else:
                value = random.uniform(10, 100)

            conversion = Conversion(
                click_id=click.id,
                campaign_id=campaign.id,
                timestamp=conv_timestamp,
                conversion_type=conv_type,
                value=value
            )
            session.add(conversion)
            total_conversions += 1

        session.commit()

    print(f"✓ Generated {total_conversions:,} conversions")


def generate_daily_metrics(session, campaigns: List[Campaign], days_to_generate: int = 30):
    """Generate aggregated daily metrics for all campaigns."""
    print(f"Generating daily metrics for {days_to_generate} days...")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_to_generate)

    total_metrics = 0

    for campaign in campaigns:
        # Determine active date range
        campaign_start = max(campaign.start_date, start_date)
        campaign_end = campaign.end_date or end_date
        campaign_end = min(campaign_end, end_date)

        if campaign_start > campaign_end:
            continue

        # Base metrics that vary by industry and campaign
        base_daily_impressions = random.randint(5000, 50000)
        base_ctr = random.uniform(0.005, 0.03)  # 0.5% to 3%
        base_cvr = random.uniform(0.01, 0.05)  # 1% to 5%
        base_cpm = random.uniform(5.0, 40.0)

        current_date = campaign_start
        while current_date <= campaign_end:
            # Add daily variation (weekends have lower traffic)
            day_of_week = current_date.weekday()
            weekend_factor = 0.7 if day_of_week >= 5 else 1.0

            # Add random variation
            daily_variation = random.uniform(0.8, 1.2)

            # Calculate metrics
            impressions = int(base_daily_impressions * weekend_factor * daily_variation)
            clicks = int(impressions * base_ctr * daily_variation)
            conversions = max(0, int(clicks * base_cvr * daily_variation))

            # Calculate spend
            spend = (impressions / 1000) * base_cpm * daily_variation

            # Calculate rates
            ctr = (clicks / impressions * 100) if impressions > 0 else 0
            cvr = (conversions / clicks * 100) if clicks > 0 else 0
            cpa = (spend / conversions) if conversions > 0 else None
            cpc = (spend / clicks) if clicks > 0 else None
            cpm = (spend / impressions * 1000) if impressions > 0 else None

            metric = DailyMetrics(
                date=current_date,
                campaign_id=campaign.id,
                impressions=impressions,
                clicks=clicks,
                conversions=conversions,
                spend=round(spend, 2),
                ctr=round(ctr, 2),
                cvr=round(cvr, 2),
                cpa=round(cpa, 2) if cpa else None,
                cpc=round(cpc, 2) if cpc else None,
                cpm=round(cpm, 2) if cpm else None
            )
            session.add(metric)
            total_metrics += 1

            current_date += timedelta(days=1)

        session.commit()

    print(f"✓ Generated {total_metrics:,} daily metric records")


def seed_database(
    num_campaigns: int = 100,
    days_to_generate: int = 30,
    generate_raw_events: bool = False
):
    """Seed the database with sample data."""
    engine = create_engine(DATABASE_URL, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        print("\n" + "="*60)
        print("SEEDING DATABASE WITH SAMPLE DATA")
        print("="*60 + "\n")

        # Clear existing data
        print("Clearing existing data...")
        session.query(DailyMetrics).delete()
        session.query(Conversion).delete()
        session.query(Click).delete()
        session.query(Impression).delete()
        session.query(Campaign).delete()
        session.commit()
        print("✓ Existing data cleared\n")

        # Generate campaigns
        campaigns = generate_campaigns(session, num_campaigns=num_campaigns)

        # Generate raw event data (optional - slower)
        if generate_raw_events:
            generate_impressions_and_clicks(session, campaigns, days_to_generate)
            generate_conversions(session, campaigns, days_to_generate)

        # Generate aggregated metrics (always)
        generate_daily_metrics(session, campaigns, days_to_generate)

        # Print summary statistics
        print("\n" + "="*60)
        print("DATABASE SUMMARY")
        print("="*60)

        campaign_count = session.query(func.count(Campaign.id)).scalar()
        metrics_count = session.query(func.count(DailyMetrics.id)).scalar()

        if generate_raw_events:
            impression_count = session.query(func.count(Impression.id)).scalar()
            click_count = session.query(func.count(Click.id)).scalar()
            conversion_count = session.query(func.count(Conversion.id)).scalar()

            print(f"Campaigns:       {campaign_count:,}")
            print(f"Impressions:     {impression_count:,}")
            print(f"Clicks:          {click_count:,}")
            print(f"Conversions:     {conversion_count:,}")
        else:
            print(f"Campaigns:       {campaign_count:,}")

        print(f"Daily Metrics:   {metrics_count:,}")

        # Calculate aggregate stats
        total_spend = session.query(func.sum(DailyMetrics.spend)).scalar() or 0
        total_impressions = session.query(func.sum(DailyMetrics.impressions)).scalar() or 0
        total_clicks = session.query(func.sum(DailyMetrics.clicks)).scalar() or 0
        total_conversions = session.query(func.sum(DailyMetrics.conversions)).scalar() or 0

        print(f"\nAggregate Stats:")
        print(f"  Total Spend:      ${total_spend:,.2f}")
        print(f"  Total Impressions: {total_impressions:,}")
        print(f"  Total Clicks:      {total_clicks:,}")
        print(f"  Total Conversions: {total_conversions:,}")
        print(f"  Avg CTR:          {(total_clicks/total_impressions*100) if total_impressions > 0 else 0:.2f}%")
        print(f"  Avg CVR:          {(total_conversions/total_clicks*100) if total_clicks > 0 else 0:.2f}%")

        print("\n" + "="*60)
        print("✓ DATABASE SEEDED SUCCESSFULLY")
        print("="*60 + "\n")

    except Exception as e:
        session.rollback()
        print(f"\n✗ Error seeding database: {e}")
        raise
    finally:
        session.close()
        engine.dispose()


def export_schema():
    """Export the database schema to a file for documentation."""
    engine = create_engine(DATABASE_URL, echo=False)

    schema_file = "docs/database_schema.sql"
    os.makedirs(os.path.dirname(schema_file), exist_ok=True)

    with open(schema_file, 'w') as f:
        for table in Base.metadata.sorted_tables:
            f.write(f"-- Table: {table.name}\n")
            f.write(str(CreateTable(table).compile(engine)))
            f.write("\n\n")

    print(f"✓ Schema exported to {schema_file}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for database setup."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup DataChat-RAG database")
    parser.add_argument("--create-db", action="store_true", help="Create the database")
    parser.add_argument("--create-tables", action="store_true", help="Create tables")
    parser.add_argument("--drop-tables", action="store_true", help="Drop all tables")
    parser.add_argument("--seed", action="store_true", help="Seed with sample data")
    parser.add_argument("--campaigns", type=int, default=100, help="Number of campaigns to generate")
    parser.add_argument("--days", type=int, default=30, help="Number of days of data to generate")
    parser.add_argument("--raw-events", action="store_true", help="Generate raw impression/click events (slower)")
    parser.add_argument("--all", action="store_true", help="Run complete setup (create + seed)")

    args = parser.parse_args()

    try:
        if args.all:
            create_database()
            create_tables()
            seed_database(
                num_campaigns=args.campaigns,
                days_to_generate=args.days,
                generate_raw_events=args.raw_events
            )
        else:
            if args.create_db:
                create_database()
            if args.create_tables:
                create_tables()
            if args.drop_tables:
                drop_tables()
            if args.seed:
                seed_database(
                    num_campaigns=args.campaigns,
                    days_to_generate=args.days,
                    generate_raw_events=args.raw_events
                )

        if not any([args.create_db, args.create_tables, args.drop_tables, args.seed, args.all]):
            parser.print_help()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
