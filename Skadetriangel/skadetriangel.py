# ------------------------------
# Aktuarieprojekt: Skadetriangel
# ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D  # For custom legend

# ------------------------------
# 1. Skapa fiktiv skadetriangel
# ------------------------------

# Triangel: rader = origin year, kolumner = development year
triangle = pd.DataFrame({
    1: [100, 120, 130, 140],
    2: [150, 170, 190, np.nan],
    3: [180, 210, np.nan, np.nan],
    4: [200, np.nan, np.nan, np.nan]
}, index=[2018, 2019, 2020, 2021])

triangle_original = triangle.copy()  # Spara originaldata för visualisering


# ------------------------------
# 2. Beräkna utvecklingsfaktorer
# ------------------------------

development_factors = []
cols = triangle.columns
for i in range(len(cols)-1):
    col_current = triangle[cols[i]]
    col_next = triangle[cols[i+1]]
    mask = col_current.notna() & col_next.notna()
    factor = col_next[mask].sum() / col_current[mask].sum()
    development_factors.append(factor)

# ------------------------------
# 3. Fyll i prediktioner
# ------------------------------

for row in triangle.index:
    for j in range(len(cols)-1):
        if pd.isna(triangle.loc[row, cols[j+1]]):
            triangle.loc[row, cols[j+1]] = triangle.loc[row, cols[j]] * development_factors[j]


print("Slutlig triangel med prediktioner:")
print(triangle)
print("\nUtvecklingsfaktorer:", development_factors)

# ------------------------------
# 4. Visualiseringar
# ------------------------------

# Linjediagram per origin year (observerat och predikterat)
for year, row in triangle.iterrows():
    plt.plot(triangle.columns.to_numpy(), row.to_numpy(), marker='o', label=str(year))

    nan_mask = triangle_original.loc[year].isna() # sätter True på rader med nan, False annars
    plt.scatter(triangle.columns[nan_mask], row[nan_mask], color='black', zorder=5)


# Add custom legend entry for black dots
black_dot = Line2D([0], [0], marker='o', color='w', label='Predicted values',
                    markerfacecolor='black', markersize=8)


plt.xlabel('Development year')
plt.ylabel('Cumulative claims')
plt.title('Claims development by origin year')
plt.legend(handles=[*plt.gca().get_legend_handles_labels()[0], black_dot],
           labels=[*plt.gca().get_legend_handles_labels()[1], 'Predicted values'],
           title='Origin year')
plt.grid(True)
plt.show()
