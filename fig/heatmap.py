import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os # 导入os模块来组合路径

def plot_heatmap(ax, filepath, title, cmap):
    """
    在一个指定的matplotlib axes上绘制一个三角热力图，并进行自定义格式化。
    """
    # --- 数据读取和处理 ---
    Task_list=["1","2","3","4","5","6"]
    if title.endswith("order2"):
        Task_list = ["6","5","4","3","2","1"]
    elif title.endswith("order3"):
        Task_list = ["3","1","6","4","2","5"]
    try:
        df = pd.read_csv(filepath, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
    except FileNotFoundError:
        ax.set_title(f'{title}\n(File Not Found)', color='red', fontsize=16)
        ax.axis('off') # 隐藏坐标轴
        print(f"错误：找不到文件 '{filepath}'，已跳过。")
        return
    except Exception as e:
        ax.set_title(f'{title}\n(Error Reading File)', color='red', fontsize=16)
        ax.axis('off')
        print(f"读取文件 '{filepath}' 时发生错误: {e}")
        return

    # --- 核心绘图逻辑 ---
    mask = np.zeros_like(df, dtype=bool)
    mask[np.tril_indices_from(mask, k=-1)] = True

    # 修改点 1: 设置 cbar=False 来禁用颜色条
    sns.heatmap(df, mask=mask, annot=True, fmt=".2f", cmap=cmap, 
                linewidths=.5, ax=ax, cbar=False, square=True, annot_kws={'size': 24})
# cbar_kws={'shrink': 0.8}
    # --- 格式化坐标轴 ---
    # 修改点 2: 将X轴移动到顶部，Y轴移动到右侧
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    
    # 修改点 3: 隐藏刻度杠 (小黑线)
    ax.tick_params(length=0)

    # --- 设置子图的标题和坐标轴标签 ---
    # ax.set_title(title, fontsize=24, pad=20)
    task_labels = [f'T{i}' for i in Task_list]
    
    # 修改点 4: 调整对齐方式以适应新的标签位置
    ax.set_xticklabels(task_labels, rotation=0, ha='center', fontsize=30)
    ax.set_yticklabels(task_labels, rotation=0, va='center', fontsize=30)
    
    ax.set_xlabel(title, fontsize=30, labelpad=20)
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')


# --- 主程序 (无需修改) ---

# 1. 设置文件路径、orders和metrics
base_dir = "save_test/Qwen2.5-7B-Instruct/lora"
method = "CPPO"
orders = [f'{method}-order1', f'{method}-order2', f'{method}-order3']
metrics = {
    'BLEU-4': 'bleu-4_matrix.csv',
    'ROUGE-L': 'rouge-L_matrix.csv',
    'LLM-Judge': 'LLM-judge_matrix.csv'
}

# 2. 创建自定义颜色映射
colors = ["#F2F8FD", "#083471"] 
custom_cmap = LinearSegmentedColormap.from_list('custom_gradient', colors)

# 3. 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 4. 创建一个包含3行3列的子图网格
fig, axes = plt.subplots(3, 3, figsize=(30, 27))

# 5. 使用嵌套循环遍历网格并调用绘图函数
for row_idx, order_name in enumerate(orders):
    for col_idx, (metric_title, metric_filename) in enumerate(metrics.items()):
        ax = axes[row_idx, col_idx]
        full_path = os.path.join(base_dir, order_name, metric_filename)
        suffix = order_name.replace("-"," with ")
        plot_title = f'{metric_title} of {suffix}'
        plot_heatmap(ax, full_path, plot_title, custom_cmap)

# 6. 调整整体布局，防止标题和标签重叠
plt.tight_layout(pad=4.0)

# 7. 保存整个图像
output_filename = f'heatmap-{method}.png'
plt.savefig(output_filename, dpi=300)

print(f"\n包含九个热力图的网格图像已成功保存为 '{output_filename}'")
# plt.show()