/*
 * Firmware Verifier - Secure boot and integrity checking
 */

#ifndef FIRMWARE_VERIFIER_H
#define FIRMWARE_VERIFIER_H

#include <string>
#include <vector>

namespace deepfake {
namespace security {

struct FirmwareInfo {
    std::string version;
    std::string build_date;
    std::string signature;
    std::string hash;
};

class FirmwareVerifier {
public:
    FirmwareVerifier();
    ~FirmwareVerifier();
    
    /**
     * Verify firmware integrity
     * @return true if firmware is valid
     */
    bool VerifyIntegrity();
    
    /**
     * Verify firmware signature
     * @param public_key_path Path to public key
     * @return true if signature is valid
     */
    bool VerifySignature(const std::string& public_key_path);
    
    /**
     * Check for tampering
     * @return true if no tampering detected
     */
    bool CheckTampering();
    
    /**
     * Get firmware information
     * @return Firmware info
     */
    FirmwareInfo GetFirmwareInfo();
    
    /**
     * Compute hash of firmware
     * @return SHA256 hash
     */
    std::string ComputeFirmwareHash();

private:
    FirmwareInfo firmware_info_;
    
    /**
     * Load firmware info from secure storage
     */
    bool LoadFirmwareInfo();
    
    /**
     * Verify code signing certificate
     */
    bool VerifyCodeSigningCert(const std::string& cert_path);
};

} // namespace security
} // namespace deepfake

#endif // FIRMWARE_VERIFIER_H
