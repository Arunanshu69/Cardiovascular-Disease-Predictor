import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the GA history CSV
csv_path = 'results/ga_history.csv'
df = pd.read_csv(csv_path)

# Create the convergence plot
plt.figure(figsize=(10, 6))
plt.plot(df['generation'], df['best_score'], label='Best Score', linewidth=2, color='blue')
plt.plot(df['generation'], df['avg_score'], label='Average Score', linewidth=2, color='orange', linestyle='--')

plt.xlabel('Generation', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Genetic Algorithm Convergence Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save the plot to results folder
output_path = 'results/ga_convergence_curve.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Convergence curve saved to: {output_path}")
