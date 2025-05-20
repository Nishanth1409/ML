import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax1)
ax1.set_title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig("2_1.png")
plt.show()
pairplot = sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']], diag_kind="kde")
pairplot.fig.suptitle("Pair Plot of Selected Features", y=1.02)  # Add title
plt.tight_layout()
pairplot.savefig("2_2.png")
plt.show()
