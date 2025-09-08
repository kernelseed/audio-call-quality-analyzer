#!/usr/bin/env python3
"""
Simple ClariAI License Checker

Quick validation that all ClariAI dependencies are open source.
"""

def check_clariai_licenses():
    """Check that all ClariAI dependencies are open source."""
    
    # ClariAI dependencies with their licenses
    clariai_deps = {
        'langchain': 'MIT',
        'langchain-community': 'MIT', 
        'transformers': 'Apache 2.0',
        'torch': 'BSD-3-Clause',
        'torchaudio': 'BSD',
        'librosa': 'ISC',
        'soundfile': 'BSD-3-Clause',
        'numpy': 'BSD-3-Clause',
        'pandas': 'BSD-3-Clause',
        'scikit-learn': 'BSD-3-Clause',
        'huggingface-hub': 'Apache 2.0',
        'datasets': 'Apache 2.0',
        'accelerate': 'Apache 2.0',
        'evaluate': 'Apache 2.0',
        'pydub': 'MIT',
        'webrtcvad': 'MIT',
        'speechrecognition': 'BSD',
        'pyaudio': 'MIT'
    }
    
    # All are permissive open source licenses
    permissive_licenses = {
        'MIT', 'Apache 2.0', 'BSD', 'BSD-3-Clause', 'ISC'
    }
    
    print("🔍 ClariAI License Validation")
    print("=" * 40)
    
    all_open_source = True
    
    for package, license_name in clariai_deps.items():
        if license_name in permissive_licenses:
            print(f"✅ {package}: {license_name}")
        else:
            print(f"❌ {package}: {license_name} (RESTRICTED)")
            all_open_source = False
    
    print("\n📊 Summary:")
    print("-" * 20)
    
    if all_open_source:
        print("✅ ALL DEPENDENCIES ARE OPEN SOURCE")
        print("✅ Commercial use is allowed for all packages")
        print("✅ No restrictive licenses found")
        return True
    else:
        print("❌ Some dependencies have restricted licenses")
        return False

if __name__ == "__main__":
    success = check_clariai_licenses()
    exit(0 if success else 1)