<<<<<<< HEAD
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
=======
# import matplotlib
# matplotlib.use('TkAgg') # Use TkAgg backend
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing
# # Load the dataset
# df = fetch_california_housing(as_frame=True).frame
# # Set up the grid layout for plots (2 rows, 1 column)
# fig, axes = plt.subplots(2, 1, figsize=(12, 12))
# # Heatmap of correlation matrix (Top plot)
# sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0])
# axes[0].set_title("Correlation Matrix Heatmap")
# # Pair plot for selected features (Bottom plot)
# sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']], diag_kind="kde")
# plt.subplots_adjust(hspace=0.4) # Adjust space between subplots
# # Show the combined figure
# plt.show()
# 
# 
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the dataset
df = fetch_california_housing(as_frame=True).frame

# First figure: Correlation Matrix Heatmap
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax1)
ax1.set_title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig("2_1.png")  # Save first figure
plt.show()

# Second figure: Pair Plot
pairplot = sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']], diag_kind="kde")
pairplot.fig.suptitle("Pair Plot of Selected Features", y=1.02)  # Add title
plt.tight_layout()
pairplot.savefig("2_2.png")  # Save second figure
plt.show()
>>>>>>> 18dd9f31c199c0697a429e9a044efe14f494798d
