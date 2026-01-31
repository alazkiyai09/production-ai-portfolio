-- StreamProcess-Pipeline Database Initialization Script
-- Run automatically on PostgreSQL container start

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity searches

-- Create database schema if not exists (handled by SQLAlchemy)
-- This script can be used for initial data seeding

-- Sample data for testing (commented out in production)
-- INSERT INTO campaigns (id, name, budget) VALUES
-- (1, 'Test Campaign 1', 10000),
-- (2, 'Test Campaign 2', 20000);

-- Create indexes for common queries (these are also created by SQLAlchemy)
-- CREATE INDEX IF NOT EXISTS idx_events_timestamp ON event_records(timestamp);
-- CREATE INDEX IF NOT EXISTS idx_events_campaign_id ON event_records(campaign_id);
-- CREATE INDEX IF NOT EXISTS idx_events_event_type ON event_records(event_type);
