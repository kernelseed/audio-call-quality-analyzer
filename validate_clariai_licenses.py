#!/usr/bin/env python3
"""
ClariAI Specific License Validation Script

This script validates ONLY the dependencies specified in requirements.txt
to ensure they are all open source and suitable for commercial use.

Usage:
    python validate_clariai_licenses.py
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

def get_package_license(package_name: str) -> Tuple[str, str]:
    """Get license information for a specific package."""
    try:
        # Try to get license info using pip show
        output = run_command(f"pip show {package_name}")
        if output:
            lines = output.split('\n')
            version = "Unknown"
            license_name = "Unknown"
            
            for line in lines:
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                elif line.startswith("License:"):
                    license_name = line.split(":", 1)[1].strip()
            
            return version, license_name
    except:
        pass
    
    return "Unknown", "Unknown"

def validate_clariai_dependencies() -> Tuple[bool, List[str]]:
    """
    Validate that all ClariAI dependencies use open source licenses.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    # Define ClariAI specific dependencies from requirements.txt
    clariai_dependencies = [
        'langchain',
        'langchain-community', 
        'transformers',
        'torch',
        'torchaudio',
        'librosa',
        'soundfile',
        'numpy',
        'pandas',
        'scikit-learn',
        'huggingface-hub',
        'datasets',
        'accelerate',
        'evaluate',
        'pydub',
        'webrtcvad',
        'speechrecognition',
        'pyaudio'
    ]
    
    issues = []
    
    # Define allowed open source licenses
    allowed_licenses = {
        'MIT', 'MIT License', 'Apache Software License', 'Apache 2.0', 
        'BSD', 'BSD License', 'BSD-3-Clause', 'BSD-2-Clause',
        'ISC License (ISCL)', 'ISC', 'Python Software Foundation License',
        'Apache License 2.0', 'Apache-2.0', '3-Clause BSD License'
    }
    
    # Define restricted licenses
    restricted_licenses = {
        'GNU General Public License (GPL)', 'GPL', 'GPLv2', 'GPLv3',
        'GNU General Public License v2 (GPLv2)',
        'GNU General Public License v2 or later (GPLv2+)',
        'GNU Lesser General Public License v3 or later (LGPLv3+)',
        'LGPL', 'LGPLv2', 'LGPLv3'
    }
    
    print("🔍 Validating ClariAI Dependencies")
    print("=" * 50)
    
    # Check each ClariAI dependency
    for package_name in clariai_dependencies:
        version, license_name = get_package_license(package_name)
        
        if version == "Unknown":
            issues.append(f"❌ {package_name}: Package not installed")
            continue
            
        # Check for restricted licenses
        if any(restricted in license_name for restricted in restricted_licenses):
            issues.append(f"❌ {package_name} v{version}: {license_name} (RESTRICTED)")
            continue
            
        # Check if license is in allowed list
        if not any(allowed in license_name for allowed in allowed_licenses):
            issues.append(f"⚠️  {package_name} v{version}: {license_name} (UNKNOWN)")
        else:
            print(f"✅ {package_name} v{version}: {license_name}")
    
    return len(issues) == 0, issues

def main():
    """Main validation function."""
    print("🔍 ClariAI License Validation")
    print("=" * 40)
    print("Validating ONLY ClariAI dependencies from requirements.txt")
    print()
    
    is_valid, issues = validate_clariai_dependencies()
    
    print("\n📊 Results:")
    print("-" * 20)
    
    if issues:
        print(f"❌ Found {len(issues)} issues with ClariAI dependencies:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ All ClariAI dependencies are open source!")
    
    print("\n📋 Summary:")
    print("-" * 20)
    if is_valid:
        print("✅ VALIDATION PASSED: All ClariAI dependencies are open source")
        print("✅ Commercial use is allowed for all ClariAI packages")
        print("✅ No restricted licenses found in ClariAI dependencies")
        return 0
    else:
        print("❌ VALIDATION FAILED: Some ClariAI dependencies have issues")
        print("⚠️  Review the issues above before commercial use")
        return 1

if __name__ == "__main__":
    sys.exit(main())
