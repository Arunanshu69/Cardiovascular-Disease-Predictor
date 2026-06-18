import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the PSO history CSV
csv_path = 'results/pso_history.csv'
df = pd.read_csv(csv_path)

# Create the convergence plot
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['best_score'], label='Best Score', linewidth=2, color='green')
plt.plot(df['iteration'], df['avg_score'], label='Average Score', linewidth=2, color='orange', linestyle='--')

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Particle Swarm Optimization Convergence Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save the plot to results folder
output_path = 'results/pso_convergence_curve.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Convergence curve saved to: {output_path}")
