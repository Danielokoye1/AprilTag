import csv

import matplotlib.pyplot as plt
import numpy as np

csv_file = 'results_horizontal_32pixel_updated.csv'
title = 'Wiener Deconvolution Results for 32-Pixel Horizontal Blur and Position Error'
out_file = csv_file.replace('.csv', '.png')

# Convert data to float
def to_float(value):
    value = value.strip()
    if value in ('', 'None', 'nan'):
        return np.nan
    return float(value)


data = {}
with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for column, value in row.items():
            data.setdefault(column, []).append(to_float(value))
data = {column: np.array(values) for column, values in data.items()}

sigma = data['Noise Sigma']

# Plot data
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
panels = [
    ('PSNR', 'PSNR (dB)'),
    ('SSIM', 'SSIM'),
    ('Position Error', 'Position error (pixels)'),]

for ax, (metric, ylabel) in zip(axes, panels):
    blurred = data[f'{metric} Blurred']
    restored = data[f'{metric} Restored']
    ax.plot(sigma, blurred, 'o--', label='Blurred')
    ax.plot(sigma, restored, 's-', label='Wiener restored')
    ax.set_xlabel('Noise std dev')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(alpha=0.3)

# Mark noise levels where the tag was not detected
ax = axes[2]
for key, marker_color, label in [
    ('Position Error Blurred', 'C0', 'Blurred: not detected'),
    ('Position Error Restored', 'C1', 'Restored: not detected'),
]:
    missed = np.isnan(data[key])
    if missed.any():
        ax.plot(sigma[missed], np.zeros(missed.sum()), 'x', color=marker_color,
                markersize=10, markeredgewidth=2, label=label)
ax.set_ylim(bottom=-0.5)

for ax in axes:
    ax.legend()
fig.suptitle(title)
fig.tight_layout()
fig.savefig(out_file, dpi=150)
print(f'Saved {out_file}')
plt.show()