/*
 * Cache Manager Implementation
 */

#include "cache_manager.h"
#include <ctime>
#include <sstream>
#include <iostream>

namespace deepfake {
namespace offline {

CacheManager::CacheManager(const std::string& db_path)
    : db_(nullptr), db_path_(db_path) {}

CacheManager::~CacheManager() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool CacheManager::Initialize() {
    // Open database
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    // Create tables
    return CreateTables();
}

bool CacheManager::StoreResult(const std::string& file_hash,
                              const InferenceResult& result) {
    // Get current timestamp
    time_t now = time(nullptr);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    // Prepare SQL
    std::stringstream sql;
    sql << "INSERT OR REPLACE INTO cache_results "
        << "(file_hash, authenticity_score, is_authentic, is_deepfake, "
        << "confidence, inference_time_ms, timestamp, synced) VALUES ("
        << "'" << file_hash << "', "
        << result.authenticity_score << ", "
        << (result.is_authentic ? 1 : 0) << ", "
        << (result.is_deepfake ? 1 : 0) << ", "
        << result.confidence << ", "
        << result.inference_time_ms << ", "
        << "'" << timestamp << "', 0);";
    
    return ExecuteSQL(sql.str());
}

bool CacheManager::GetResult(const std::string& file_hash,
                            InferenceResult& result) {
    std::stringstream sql;
    sql << "SELECT authenticity_score, is_authentic, is_deepfake, "
        << "confidence, inference_time_ms FROM cache_results "
        << "WHERE file_hash = '" << file_hash << "';";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql.str().c_str(), -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        return false;
    }
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result.authenticity_score = sqlite3_column_double(stmt, 0);
        result.is_authentic = sqlite3_column_int(stmt, 1) == 1;
        result.is_deepfake = sqlite3_column_int(stmt, 2) == 1;
        result.confidence = sqlite3_column_double(stmt, 3);
        result.inference_time_ms = sqlite3_column_int64(stmt, 4);
        
        sqlite3_finalize(stmt);
        return true;
    }
    
    sqlite3_finalize(stmt);
    return false;
}

std::vector<CachedResult> CacheManager::GetUnsyncedResults() {
    std::vector<CachedResult> results;
    
    const char* sql = "SELECT file_hash, authenticity_score, is_authentic, "
                     "is_deepfake, confidence, inference_time_ms, timestamp "
                     "FROM cache_results WHERE synced = 0;";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        return results;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        CachedResult cached;
        cached.file_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        cached.result.authenticity_score = sqlite3_column_double(stmt, 1);
        cached.result.is_authentic = sqlite3_column_int(stmt, 2) == 1;
        cached.result.is_deepfake = sqlite3_column_int(stmt, 3) == 1;
        cached.result.confidence = sqlite3_column_double(stmt, 4);
        cached.result.inference_time_ms = sqlite3_column_int64(stmt, 5);
        cached.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        cached.synced = false;
        
        results.push_back(cached);
    }
    
    sqlite3_finalize(stmt);
    return results;
}

bool CacheManager::MarkSynced(const std::string& file_hash) {
    std::stringstream sql;
    sql << "UPDATE cache_results SET synced = 1 "
        << "WHERE file_hash = '" << file_hash << "';";
    
    return ExecuteSQL(sql.str());
}

int CacheManager::ClearOldEntries(int days_old) {
    std::stringstream sql;
    sql << "DELETE FROM cache_results "
        << "WHERE julianday('now') - julianday(timestamp) > " << days_old << ";";
    
    ExecuteSQL(sql.str());
    
    return sqlite3_changes(db_);
}

std::map<std::string, int> CacheManager::GetStatistics() {
    std::map<std::string, int> stats;
    
    // Total entries
    const char* sql_total = "SELECT COUNT(*) FROM cache_results;";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db_, sql_total, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats["total"] = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // Unsynced entries
    const char* sql_unsynced = "SELECT COUNT(*) FROM cache_results WHERE synced = 0;";
    
    if (sqlite3_prepare_v2(db_, sql_unsynced, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats["unsynced"] = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    return stats;
}

bool CacheManager::CreateTables() {
    const char* sql = 
        "CREATE TABLE IF NOT EXISTS cache_results ("
        "file_hash TEXT PRIMARY KEY,"
        "authenticity_score REAL NOT NULL,"
        "is_authentic INTEGER NOT NULL,"
        "is_deepfake INTEGER NOT NULL,"
        "confidence REAL NOT NULL,"
        "inference_time_ms INTEGER NOT NULL,"
        "timestamp TEXT NOT NULL,"
        "synced INTEGER DEFAULT 0"
        ");";
    
    return ExecuteSQL(sql);
}

bool CacheManager::ExecuteSQL(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    
    return true;
}

} // namespace offline
} // namespace deepfake
