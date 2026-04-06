import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import json
import os

plt.rcParams['font.family'] = 'serif'
    
    # 2. 在衬线字体中，指定 'Times New Roman' 作为首选
    #    Matplotlib 会依次尝试列表中的字体，直到找到一个可用的
plt.rcParams['font.serif'] = ['Times New Roman']

# --- 数据加载 ---
def load_data(file_path):
    """安全地加载JSON文件。"""
    if not os.path.exists(file_path):
        print(f"警告: 在 {file_path} 未找到文件")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"读取 {file_path} 时出错: {e}")
        return None

def load_proj_data():
    """加载th=0.9情况下改变proj的数据"""
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    bwt_bleu, last_bleu, avg_bleu = [], [], []
    bwt_rouge, last_rouge, avg_rouge = [], [], []
    bwt_llm, last_llm = [], []
    
    base_test_dir = 'save_test/Qwen2.5-7B-Instruct/lora/DPO+ours-0707-th=0.9-proj'
    
    for threshold in thresholds:
        test_dir_path = f"{base_test_dir}={threshold}/metric.json"
        metric = load_data(test_dir_path)
        
        bwt_bleu.append(metric.get('bleu-4', {}).get('bwt', np.nan) if metric else np.nan)
        last_bleu.append(metric.get('bleu-4', {}).get('last', np.nan) if metric else np.nan)
        avg_bleu.append(metric.get('bleu-4', {}).get('avg', np.nan) if metric else np.nan)
        
        bwt_rouge.append(metric.get('rouge-L', {}).get('bwt', np.nan) if metric else np.nan)
        last_rouge.append(metric.get('rouge-L', {}).get('last', np.nan) if metric else np.nan)
        avg_rouge.append(metric.get('rouge-L', {}).get('avg', np.nan) if metric else np.nan)
        
        bwt_llm.append(metric.get('LLM-judge', {}).get('bwt', np.nan) if metric else np.nan)
        last_llm.append(metric.get('LLM-judge', {}).get('last', np.nan) if metric else np.nan)
    
    return thresholds, (bwt_bleu, last_bleu, avg_bleu), (bwt_rouge, last_rouge, avg_rouge), (bwt_llm, last_llm)

def load_th_data():
    """加载proj=0.5情况下改变th的数据"""
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    bwt_bleu, last_bleu, avg_bleu = [], [], []
    bwt_rouge, last_rouge, avg_rouge = [], [], []
    bwt_llm, last_llm = [], []
    
    base_test_dir = 'save_test/Qwen2.5-7B-Instruct/lora/DPO+ours-0707-th'
    
    for threshold in thresholds:
        test_dir_path = f"{base_test_dir}={threshold}-proj=0.5/metric.json"
        metric = load_data(test_dir_path)
        
        bwt_bleu.append(metric.get('bleu-4', {}).get('bwt', np.nan) if metric else np.nan)
        last_bleu.append(metric.get('bleu-4', {}).get('last', np.nan) if metric else np.nan)
        avg_bleu.append(metric.get('bleu-4', {}).get('avg', np.nan) if metric else np.nan)
        
        bwt_rouge.append(metric.get('rouge-L', {}).get('bwt', np.nan) if metric else np.nan)
        last_rouge.append(metric.get('rouge-L', {}).get('last', np.nan) if metric else np.nan)
        avg_rouge.append(metric.get('rouge-L', {}).get('avg', np.nan) if metric else np.nan)
        
        bwt_llm.append(metric.get('LLM-judge', {}).get('bwt', np.nan) if metric else np.nan)
        last_llm.append(metric.get('LLM-judge', {}).get('last', np.nan) if metric else np.nan)
    
    return thresholds, (bwt_bleu, last_bleu, avg_bleu), (bwt_rouge, last_rouge, avg_rouge), (bwt_llm, last_llm)

def get_ylims(metric_type, data_source):
    """根据指标类型和数据源设置y轴范围"""
    if metric_type == 'bleu':
        if data_source == 'proj':  # th=0.9改变proj的情况
            return (-7, 2), (22, 31)
        else:  # proj=0.5改变th的情况
            return (-4, 1), (26, 31)
    elif metric_type == 'rouge':
        if data_source == 'proj':  # th=0.9改变proj的情况
            return (-3, 2), (22, 27)
        else:  # proj=0.5改变th的情况
            return (-1, 2), (24, 28)
    elif metric_type == 'llm':
        if data_source == 'proj':  # th=0.9改变proj的情况
            return (-9, 3), (47, 60)
        else:  # proj=0.5改变th的情况
            return (-4, 2), (51, 58)

def plot_broken_axis_subplot(gs_position, fig, thresholds, bwt, last, avg, title, ylabel, xlabel, 
                           metric_type, data_source, has_avg=True):
    """在指定位置创建带有断裂y轴的子图"""
    # 使用matplotlib.gridspec.GridSpecFromSubplotSpec来创建子网格
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    inner_gs = GridSpecFromSubplotSpec(2, 1, gs_position, height_ratios=[1, 1], hspace=0.15)
    
    ax_top = fig.add_subplot(inner_gs[0])
    ax_bottom = fig.add_subplot(inner_gs[1], sharex=ax_top)
    
    # 获取y轴范围
    bottom_ylim, top_ylim = get_ylims(metric_type, data_source)
    
    colors = {'bwt': '#9C4084', 'last': '#3F81B4', 'avg': '#6EBD87'}
    
    # 在两个轴上都绘制数据
    for ax in (ax_top, ax_bottom):
        ax.plot(thresholds, bwt, marker="o", label="BWT ↑", color=colors['bwt'], 
                linewidth=1.5, markersize=6)
        ax.plot(thresholds, last, marker="s", label="Last ↑", color=colors['last'], 
                linewidth=1.5, markersize=6)
        if has_avg and avg is not None:
            ax.plot(thresholds, avg, marker="^", label="AP ↑", color=colors['avg'], 
                    linewidth=1.5, markersize=6)
        
        ax.set_xticks(thresholds)
        ax.set_xticklabels([str(t) for t in thresholds], rotation=45, ha='right')
        ax.set_xlim(-0.05, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelsize=22)
    
    # 设置y轴范围
    ax_top.set_ylim(top_ylim)
    ax_bottom.set_ylim(bottom_ylim)
    
    # 隐藏连接处的边框
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_bottom.xaxis.tick_bottom()
    ax_bottom.tick_params(axis='x', labelsize=22)
    
    # 添加断裂标记
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    # 设置标题和标签
    # ax_top.set_title(title, fontsize=21, pad=10)
    # ax.tick_params(axis='x', labelsize=20)
    # ax.tick_params(axis='y', labelsize=18)
    ax_bottom.set_xlabel(xlabel, fontsize=25)
    
    # 在中间位置添加y轴标签
    fig.text(gs_position.get_position(fig).x0 - 0.035, 
             (gs_position.get_position(fig).y0 + gs_position.get_position(fig).y1) / 2, 
             ylabel, va='center', ha='center', rotation='vertical', fontsize=25)
    
    return ax_top, ax_bottom

def plot_combined_metrics():
    """创建包含6个子图的综合图表，每个子图都有断裂y轴"""
    # 加载数据
    proj_thresholds, proj_bleu, proj_rouge, proj_llm = load_proj_data()
    th_thresholds, th_bleu, th_rouge, th_llm = load_th_data()
    
    # 创建图形
    fig = plt.figure(figsize=(24, 12))
    
    # 创建主GridSpec，为共享图例留出空间
    main_gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.2, 
                      top=0.88, bottom=0.08, left=0.08, right=0.96)
    
    # 第一行：th=0.9情况下改变proj
    plot_broken_axis_subplot(main_gs[0, 0], fig, proj_thresholds, proj_bleu[0], proj_bleu[1], proj_bleu[2], 
                           "", 
                           "BLEU-4 Score", "(a) BLEU-4 vs. λ (θ=0.9)", 'bleu', 'proj')
    
    plot_broken_axis_subplot(main_gs[0, 1], fig, proj_thresholds, proj_rouge[0], proj_rouge[1], proj_rouge[2], 
                           "", 
                           "ROUGE-L Score", "(b) ROUGE-L vs. λ (θ=0.9)", 'rouge', 'proj')
    
    plot_broken_axis_subplot(main_gs[0, 2], fig, proj_thresholds, proj_llm[0], proj_llm[1], None, 
                           "", 
                           "LLM-Judge Score", "(c) LLM-Judge vs. λ (θ=0.9)", 'llm', 'proj', has_avg=False)
    
    # 第二行：proj=0.5情况下改变th
    plot_broken_axis_subplot(main_gs[1, 0], fig, th_thresholds, th_bleu[0], th_bleu[1], th_bleu[2], 
                           "", 
                           "BLEU-4 Score", "(d) BLEU-4 vs. θ (λ=0.5)", 'bleu', 'th')
    
    plot_broken_axis_subplot(main_gs[1, 1], fig, th_thresholds, th_rouge[0], th_rouge[1], th_rouge[2], 
                           "", 
                           "ROUGE-L Score", "(e) ROUGE-L vs. θ (λ=0.5)", 'rouge', 'th')
    
    plot_broken_axis_subplot(main_gs[1, 2], fig, th_thresholds, th_llm[0], th_llm[1], None, 
                           "", 
                           "LLM-Judge Score", "(f) LLM-Judge vs. θ (λ=0.5)", 'llm', 'th', has_avg=False)
    
    # 创建共享图例
    colors = {'bwt': '#9C4084', 'last': '#3F81B4', 'avg': '#6EBD87'}
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=colors['bwt'], label='BWT ↑', 
                   linewidth=3, markersize=12),
        plt.Line2D([0], [0], marker='s', color=colors['last'], label='Last ↑', 
                   linewidth=3, markersize=12),
        plt.Line2D([0], [0], marker='^', color=colors['avg'], label='AP ↑', 
                   linewidth=3, markersize=12)
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.68, 0.96), 
               ncol=3, frameon=False, fontsize=28)
    
    # 保存图片
    plt.savefig("hypeparameter-detailed-0406.png", dpi=300, bbox_inches='tight')

# --- 主程序执行 ---
if __name__ == "__main__":
    plot_combined_metrics()