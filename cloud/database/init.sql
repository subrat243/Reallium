-- Database initialization script

-- Create database
CREATE DATABASE IF NOT EXISTS deepfake_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_user_created ON detections(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);

-- Create function for updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_threat_intelligence_updated_at BEFORE UPDATE ON threat_intelligence
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
