
# Safe Matplotlib Configuration for FL Research
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Font configuration
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
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
})

print("✅ Safe matplotlib configuration loaded")
print(f"🎨 Using font: DejaVu Serif")
