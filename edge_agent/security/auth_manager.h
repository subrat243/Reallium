/*
 * Authentication Manager for Edge Devices
 */

#ifndef AUTH_MANAGER_H
#define AUTH_MANAGER_H

#include <string>
#include <map>
#include <vector>

namespace deepfake {
namespace security {

enum class UserRole {
    VIEWER,
    OPERATOR,
    ADMIN
};

struct UserCredentials {
    std::string user_id;
    std::string username;
    std::string password_hash;
    UserRole role;
    bool mfa_enabled;
    std::string mfa_secret;
    bool is_active;
};

class AuthManager {
public:
    AuthManager();
    ~AuthManager();
    
    /**
     * Initialize authentication system
     * @return true if successful
     */
    bool Initialize();
    
    /**
     * Authenticate user
     * @param username Username
     * @param password Password
     * @param mfa_token MFA token (if enabled)
     * @return User ID if successful, empty string otherwise
     */
    std::string Authenticate(const std::string& username,
                            const std::string& password,
                            const std::string& mfa_token = "");
    
    /**
     * Check if user has required role
     * @param user_id User ID
     * @param required_role Required role
     * @return true if authorized
     */
    bool CheckAuthorization(const std::string& user_id,
                           UserRole required_role);
    
    /**
     * Generate MFA secret
     * @return MFA secret
     */
    std::string GenerateMFASecret();
    
    /**
     * Verify MFA token
     * @param secret MFA secret
     * @param token User-provided token
     * @return true if valid
     */
    bool VerifyMFAToken(const std::string& secret,
                       const std::string& token);
    
    /**
     * Hash password
     * @param password Plain password
     * @return Hashed password
     */
    std::string HashPassword(const std::string& password);
    
    /**
     * Verify password
     * @param password Plain password
     * @param hash Stored hash
     * @return true if match
     */
    bool VerifyPassword(const std::string& password,
                       const std::string& hash);

private:
    std::map<std::string, UserCredentials> users_;
    
    /**
     * Load users from secure storage
     */
    bool LoadUsers();
    
    /**
     * Save users to secure storage
     */
    bool SaveUsers();
};

} // namespace security
} // namespace deepfake

#endif // AUTH_MANAGER_H
