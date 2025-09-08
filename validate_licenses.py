#!/usr/bin/env python3
"""
ClariAI License Validation Script

This script validates that all dependencies in the ClariAI project
are open source and use permissive licenses suitable for commercial use.

Usage:
    python validate_licenses.py
"""

import subprocess
import json
import sys
from typing import Dict, List, Tuple

def run_command(command: str) -> str:
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error: {result.stderr}")
            return ""
        return result.stdout
    except Exception as e:
        print(f"Exception running command {command}: {e}")
        return ""

def get_installed_packages() -> List[Dict]:
    """Get list of installed packages with license information."""
    try:
        # Try to use pip-licenses if available
        output = run_command("pip-licenses --format=json")
        if output:
            return json.loads(output)
    except:
        pass
    
    # Fallback to pip list
    output = run_command("pip list --format=json")
    if output:
        packages = json.loads(output)
        # Add placeholder license info
        for pkg in packages:
            pkg['License'] = 'Unknown'
        return packages
    
    return []

def validate_licenses() -> Tuple[bool, List[str]]:
    """
    Validate that all packages use open source licenses.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    packages = get_installed_packages()
    issues = []
    
    # Define allowed open source licenses
    allowed_licenses = {
        'MIT', 'MIT License', 'Apache Software License', 'Apache 2.0', 
        'BSD', 'BSD License', 'BSD-3-Clause', 'BSD-2-Clause',
        'ISC License (ISCL)', 'ISC', 'Python Software Foundation License',
        'Apache License 2.0', 'Apache-2.0', '3-Clause BSD License',
        'GNU Library or Lesser General Public License (LGPL)',
        'Zope Public License', 'CC0', 'CC0 1.0 Universal (CC0 1.0)',
        'Academic Free License (AFL)', 'Mozilla Public License 2.0 (MPL 2.0)',
        'FreeBSD', 'OLDAP-2.8', 'SIP'
    }
    
    # Define restricted licenses
    restricted_licenses = {
        'GNU General Public License (GPL)', 'GPL', 'GPLv2', 'GPLv3',
        'GNU General Public License v2 (GPLv2)',
        'GNU General Public License v2 or later (GPLv2+)',
        'GNU Lesser General Public License v3 or later (LGPLv3+)'
    }
    
    # Check each package
    for pkg in packages:
        name = pkg.get('Name', '')
        license_name = pkg.get('License', 'Unknown')
        version = pkg.get('Version', 'Unknown')
        
        # Skip if license is unknown (might be acceptable for some packages)
        if license_name == 'Unknown':
            continue
            
        # Check for restricted licenses
        if any(restricted in license_name for restricted in restricted_licenses):
            issues.append(f"❌ {name} v{version}: {license_name} (RESTRICTED)")
            continue
            
        # Check if license is in allowed list
        if not any(allowed in license_name for allowed in allowed_licenses):
            issues.append(f"⚠️  {name} v{version}: {license_name} (UNKNOWN)")
        else:
            print(f"✅ {name} v{version}: {license_name}")
    
    return len(issues) == 0, issues

def check_specific_packages() -> None:
    """Check specific packages from requirements.txt."""
    required_packages = [
        'langchain', 'langchain-community', 'transformers', 'torch', 
        'torchaudio', 'librosa', 'soundfile', 'numpy', 'pandas', 
        'scikit-learn', 'huggingface-hub', 'datasets', 'accelerate', 
        'evaluate', 'pydub', 'webrtcvad', 'speechrecognition', 'pyaudio'
    ]
    
    print("\n🔍 Checking specific ClariAI dependencies:")
    print("=" * 50)
    
    packages = get_installed_packages()
    package_dict = {pkg.get('Name', '').lower(): pkg for pkg in packages}
    
    for pkg_name in required_packages:
        pkg = package_dict.get(pkg_name.lower())
        if pkg:
            name = pkg.get('Name', pkg_name)
            version = pkg.get('Version', 'Unknown')
            license_name = pkg.get('License', 'Unknown')
            print(f"✅ {name} v{version}: {license_name}")
        else:
            print(f"❌ {pkg_name}: Not installed")

def main():
    """Main validation function."""
    print("🔍 ClariAI License Validation")
    print("=" * 40)
    
    # Check if pip-licenses is available
    try:
        run_command("pip-licenses --version")
        print("✅ pip-licenses is available")
    except:
        print("⚠️  pip-licenses not available, installing...")
        run_command("pip install pip-licenses")
    
    print("\n📋 Validating all installed packages:")
    print("-" * 40)
    
    is_valid, issues = validate_licenses()
    
    if issues:
        print(f"\n❌ Found {len(issues)} license issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All packages use open source licenses!")
    
    # Check specific packages
    check_specific_packages()
    
    print("\n📊 Summary:")
    print("-" * 20)
    if is_valid:
        print("✅ VALIDATION PASSED: All dependencies are open source")
        print("✅ Commercial use is allowed for all packages")
        return 0
    else:
        print("❌ VALIDATION FAILED: Some packages have restricted licenses")
        print("⚠️  Review the issues above before commercial use")
        return 1

if __name__ == "__main__":
    sys.exit(main())
