import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 生成示例數據
np.random.seed(42)
data = {
    'x': np.random.rand(100) * 100,
    'y': np.random.rand(100) * 100
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 數據標準化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 訓練 DBSCAN 模型
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(df_scaled)

# 獲取聚類結果
df['cluster'] = dbscan.labels_

# 繪製聚類結果
plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', hue='cluster', data=df, palette='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title='Cluster')
plt.savefig('dbscan_clustering.png')
plt.show()

# 輸出聚類結果
print("Cluster Labels:")
print(df['cluster'].value_counts())