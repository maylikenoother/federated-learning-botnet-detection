#!/usr/bin/env python3
"""
Font Installation and Matplotlib Fix Script
Automatically installs required fonts and fixes matplotlib configuration
"""

import os
import sys
import platform
import subprocess
import matplotlib
import matplotlib.font_manager as fm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_system_info():
    """Get system information"""
    system = platform.system().lower()
    logger.info(f"🖥️ Detected system: {system}")
    return system

def install_fonts_windows():
    """Install fonts on Windows"""
    logger.info("🪟 Installing fonts for Windows...")
    
    try:
        # Install common serif fonts via chocolatey (if available)
        result = subprocess.run(['choco', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("📦 Chocolatey detected, installing fonts...")
            subprocess.run(['choco', 'install', 'liberation-fonts', '-y'], check=False)
        else:
            logger.info("💡 Chocolatey not found. Manual font installation required.")
            logger.info("   Download Liberation fonts from: https://github.com/liberationfonts/liberation-fonts/releases")
    
    except FileNotFoundError:
        logger.info("💡 Please install fonts manually:")
        logger.info("   1. Download Liberation fonts: https://github.com/liberationfonts/liberation-fonts/releases")
        logger.info("   2. Extract and install .ttf files")
        logger.info("   3. Or use the safe matplotlib configuration below")

def install_fonts_macos():
    """Install fonts on macOS"""
    logger.info("🍎 Installing fonts for macOS...")
    
    try:
        # Check if Homebrew is available
        result = subprocess.run(['brew', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("🍺 Homebrew detected, installing fonts...")
            subprocess.run(['brew', 'tap', 'homebrew/cask-fonts'], check=False)
            subprocess.run(['brew', 'install', '--cask', 'font-liberation'], check=False)
            subprocess.run(['brew', 'install', '--cask', 'font-times-new-roman'], check=False)
        else:
            logger.info("💡 Homebrew not found. Manual installation required.")
    
    except FileNotFoundError:
        logger.info("💡 Please install fonts manually:")
        logger.info("   1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        logger.info("   2. Then run: brew tap homebrew/cask-fonts && brew install --cask font-liberation")

def install_fonts_linux():
    """Install fonts on Linux"""
    logger.info("🐧 Installing fonts for Linux...")
    
    # Detect package manager and install fonts
    package_managers = [
        (['apt', 'update'], ['apt', 'install', '-y', 'fonts-liberation', 'fonts-dejavu', 'fonts-liberation2']),
        (['yum', 'check-update'], ['yum', 'install', '-y', 'liberation-fonts', 'dejavu-fonts']),
        (['dnf', 'check-update'], ['dnf', 'install', '-y', 'liberation-fonts', 'dejavu-fonts']),
        (['pacman', '-Sy'], ['pacman', '-S', '--noconfirm', 'ttf-liberation', 'ttf-dejavu']),
        (['zypper', 'refresh'], ['zypper', 'install', '-y', 'liberation-fonts', 'dejavu-fonts'])
    ]
    
    for check_cmd, install_cmd in package_managers:
        try:
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 or 'command not found' not in result.stderr.lower():
                logger.info(f"📦 Using package manager: {check_cmd[0]}")
                subprocess.run(install_cmd, check=False)
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    else:
        logger.info("💡 Could not detect package manager. Manual installation required:")
        logger.info("   sudo apt install fonts-liberation fonts-dejavu  # Ubuntu/Debian")
        logger.info("   sudo yum install liberation-fonts dejavu-fonts  # CentOS/RHEL")
        logger.info("   sudo pacman -S ttf-liberation ttf-dejavu        # Arch Linux")

def clear_matplotlib_cache():
    """Clear matplotlib font cache"""
    logger.info("🧹 Clearing matplotlib font cache...")
    
    try:
        # Clear the font cache
        cache_dir = matplotlib.get_cachedir()
        font_cache_files = [
            'fontlist-v330.json',
            'fontlist-v320.json', 
            'fontlist-v310.json',
            'fontList.cache'
        ]
        
        for cache_file in font_cache_files:
            cache_path = Path(cache_dir) / cache_file
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"   Removed: {cache_file}")
        
        # Rebuild font cache
        fm._rebuild()
        logger.info("✅ Font cache rebuilt")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not clear font cache: {e}")

def check_available_fonts():
    """Check what fonts are available"""
    logger.info("🔍 Checking available fonts...")
    
    # Get all available fonts
    fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Check for specific fonts we need
    target_fonts = [
        'Times New Roman', 'Liberation Serif', 'DejaVu Serif',
        'Computer Modern Roman', 'Times', 'serif'
    ]
    
    available_targets = []
    for font in target_fonts:
        if font in fonts:
            available_targets.append(font)
    
    logger.info(f"📝 Found {len(fonts)} total fonts")
    logger.info(f"🎯 Target fonts available: {available_targets}")
    
    if not available_targets:
        logger.warning("⚠️ No target serif fonts found!")
        
        # Show some available serif-like fonts
        serif_like = [f for f in fonts if any(term in f.lower() for term in ['serif', 'times', 'roman', 'liberation', 'dejavu'])]
        if serif_like:
            logger.info(f"📋 Available serif-like fonts: {serif_like[:10]}")
    
    return available_targets

def create_safe_matplotlib_config():
    """Create a safe matplotlib configuration"""
    logger.info("⚙️ Creating safe matplotlib configuration...")
    
    # Check available fonts
    available_fonts = check_available_fonts()
    
    # Choose best available font
    if 'Liberation Serif' in available_fonts:
        chosen_font = 'Liberation Serif'
    elif 'DejaVu Serif' in available_fonts:
        chosen_font = 'DejaVu Serif'
    elif 'Times New Roman' in available_fonts:
        chosen_font = 'Times New Roman'
    else:
        chosen_font = 'DejaVu Sans'  # Fallback to sans-serif
        logger.info("📝 Using sans-serif fallback")
    
    logger.info(f"🎨 Selected font: {chosen_font}")
    
    # Create matplotlib config
    config = f'''
# Safe Matplotlib Configuration for FL Research
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Font configuration
plt.rcParams.update({{
    'font.family': '{chosen_font}',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.unicode_minus': False,
    'figure.max_open_warning': 0
}})

print("✅ Safe matplotlib configuration loaded")
print(f"🎨 Using font: {chosen_font}")
'''
    
    # Save configuration file
    config_file = Path('matplotlib_config.py')
    with open(config_file, 'w') as f:
        f.write(config)
    
    logger.info(f"💾 Saved configuration to: {config_file}")
    
    return config_file

def test_matplotlib():
    """Test matplotlib with current configuration"""
    logger.info("🧪 Testing matplotlib...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Create a simple test plot
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', linewidth=2)
        plt.title('Matplotlib Font Test')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.grid(True, alpha=0.3)
        
        # Save test plot
        test_file = Path('font_test.png')
        plt.savefig(test_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Test plot saved: {test_file}")
        logger.info("🎉 Matplotlib is working correctly!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Matplotlib test failed: {e}")
        return False

def main():
    """Main font installation and configuration"""
    print("🎨 FONT INSTALLATION & MATPLOTLIB FIX")
    print("=" * 50)
    print("This script will install fonts and fix matplotlib configuration")
    print()
    
    try:
        # Get system info
        system = get_system_info()
        
        # Install fonts based on system
        if system == 'windows':
            install_fonts_windows()
        elif system == 'darwin':  # macOS
            install_fonts_macos()
        elif system == 'linux':
            install_fonts_linux()
        else:
            logger.warning(f"⚠️ Unknown system: {system}")
        
        # Clear matplotlib cache
        clear_matplotlib_cache()
        
        # Check available fonts
        available_fonts = check_available_fonts()
        
        # Create safe configuration
        config_file = create_safe_matplotlib_config()
        
        # Test matplotlib
        test_success = test_matplotlib()
        
        print("\n" + "=" * 50)
        print("🎉 FONT SETUP COMPLETE!")
        print("=" * 50)
        
        if available_fonts:
            print(f"✅ Found serif fonts: {', '.join(available_fonts)}")
        else:
            print("⚠️ No serif fonts found, using fallback configuration")
        
        print(f"📁 Configuration saved: {config_file}")
        print(f"🧪 Test plot: {'✅ Success' if test_success else '❌ Failed'}")
        
        print("\n📋 Next steps:")
        print("1. Add this to the top of your graph script:")
        print("   exec(open('matplotlib_config.py').read())")
        print("2. Or run the quick font-safe version I provided")
        print("3. Your graphs should now work without font warnings!")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Installation interrupted")
        return False
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)