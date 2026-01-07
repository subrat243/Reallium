/*
 * Cache Manager for Offline Detection Results
 */

#ifndef CACHE_MANAGER_H
#define CACHE_MANAGER_H

#include "../core/inference_engine.h"
#include <string>
#include <map>
#include <vector>
#include <sqlite3.h>

namespace deepfake {
namespace offline {

struct CachedResult {
    std::string file_hash;
    InferenceResult result;
    std::string timestamp;
    bool synced;
};

class CacheManager {
public:
    CacheManager(const std::string& db_path = "deepfake_cache.db");
    ~CacheManager();
    
    /**
     * Initialize cache database
     * @return true if successful
     */
    bool Initialize();
    
    /**
     * Store detection result
     * @param file_hash File hash
     * @param result Detection result
     * @return true if successful
     */
    bool StoreResult(const std::string& file_hash,
                    const InferenceResult& result);
    
    /**
     * Get cached result
     * @param file_hash File hash
     * @param result Output result
     * @return true if found
     */
    bool GetResult(const std::string& file_hash,
                  InferenceResult& result);
    
    /**
     * Get all unsynced results
     * @return Vector of unsynced results
     */
    std::vector<CachedResult> GetUnsyncedResults();
    
    /**
     * Mark result as synced
     * @param file_hash File hash
     * @return true if successful
     */
    bool MarkSynced(const std::string& file_hash);
    
    /**
     * Clear old cache entries
     * @param days_old Delete entries older than this
     * @return Number of entries deleted
     */
    int ClearOldEntries(int days_old = 30);
    
    /**
     * Get cache statistics
     * @return Statistics map
     */
    std::map<std::string, int> GetStatistics();

private:
    sqlite3* db_;
    std::string db_path_;
    
    /**
     * Create database tables
     */
    bool CreateTables();
    
    /**
     * Execute SQL query
     */
    bool ExecuteSQL(const std::string& sql);
};

} // namespace offline
} // namespace deepfake

#endif // CACHE_MANAGER_H
