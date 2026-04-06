import os
import json
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'serif'
    
    # 2. 在衬线字体中，指定 'Times New Roman' 作为首选
    #    Matplotlib 会依次尝试列表中的字体，直到找到一个可用的
plt.rcParams['font.serif'] = ['Times New Roman']

# (可选，但推荐) 为了让数学公式的字体也和正文匹配
# 'stix' 是一种与 Times New Roman 风格非常接近的数学字体
# plt.rcParams['mathtext.fontset'] = 'stix' 

# 解决负号显示问题 (通常在西文字体下不是问题，但保留是个好习惯)
# plt.rcParams['axes.unicode_minus'] = False

def plot_baseline_metrics(method_paths,
                          orders=('order1', 'order2', 'order3'),
                          metric_names=('bleu-4', 'rouge-L'),
                          stages=('bwt', 'last', 'avg'),
                          colors=None,
                          y_limits=None,
                          bar_width=None,
                          figsize=None,
                          annot_fontsize=9,
                          legend_pad=0.8):
    """
    method_paths: list of 完整目录，每个目录名形如 `<baseline>-<order>`，
                  目录下直接有 metric.json
    orders:       要比较的顺序名称列表，比如 ('order1','order2','order3')
    metric_names: 要绘的 metric，比如 ['bleu-4','rouge-L']
    stages:       要绘的阶段，比如 ['bwt','last','avg']
    colors:       可选，每个 baseline 的柱子颜色列表
    y_limits:     可选，为每个 stage 显式指定 y 轴范围
    bar_width:    可选，显式指定每根柱子的宽度
    """
    # 1. 读取并拆分 baseline vs order
    all_data = {}
    for p in method_paths:
        base = os.path.basename(p.rstrip('/'))
        matched = False
        for ord_name in orders:
            suffix = f"-{ord_name}"
            if base.endswith(suffix):
                baseline = base[:-len(suffix)]
                order = ord_name
                matched = True
                break
        if not matched:
            raise ValueError(f"路径 {p} 无法匹配任何 order（{orders}）")
        fn = os.path.join(p, 'metric.json')
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"{fn} 不存在")
        with open(fn, 'r', encoding='utf-8') as f:
            met = json.load(f)
        all_data.setdefault(baseline, {})[order] = met

    methods = sorted(all_data.keys())
    n_methods = len(methods)
    n_orders = len(orders)

    # 2. 处理颜色
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [prop_cycle[i % len(prop_cycle)] for i in range(n_methods)]
    elif len(colors) < n_methods:
        raise ValueError(f"colors 长度 {len(colors)} 少于 baseline 数量 {n_methods}")

    # 3. 处理 y_limits
    if y_limits is None:
        y_limits_map = {}
    elif isinstance(y_limits, (list, tuple)):
        if len(y_limits) != len(stages):
            raise ValueError("如果 y_limits 是列表，其长度必须等于 stages 的长度")
        y_limits_map = {st: y_limits[i] for i, st in enumerate(stages)}
    elif isinstance(y_limits, dict):
        y_limits_map = y_limits
    else:
        raise TypeError("y_limits 必须是 dict 或 list/tuple")

    # 4. 准备子图 grid
    fig, axes = plt.subplots(len(stages), len(metric_names),
                             figsize=(6 * len(metric_names), 4 * len(stages)),
                             constrained_layout=True)
    if axes.ndim == 1:
        axes = axes.reshape(-1, len(metric_names))

    # 5. 计算 bar_width 和 offsets
    if bar_width is None:
        total_w = 0.8
        bw = total_w / n_methods
        offsets = np.arange(n_methods) * bw - (total_w - bw) / 2
    else:
        bw = bar_width
        offsets = (np.arange(n_methods) - (n_methods-1)/2) * bw

    x = np.arange(n_orders)

    # 6. 绘制各子图
    for i, stage in enumerate(stages):
        for j, metric in enumerate(metric_names):
            ax = axes[i][j]
            for m_idx, method in enumerate(methods):
                vals = [
                    all_data[method].get(ord_name, {}).get(metric, {}).get(stage.lower(), 0.0)
                    for ord_name in orders
                ]
                ax.bar(x + offsets[m_idx], vals,
                       width=bw,
                       color=colors[m_idx],
                       edgecolor='black',
                       label=method if i==0 and j==0 else "")
                # --- 主要改动：移除了在这里显示数值的逻辑 ---
                # for k, v in enumerate(vals):
                #     ax.text(x[k] + offsets[m_idx],
                #             v + max(vals)*0.02,
                #             f"{v:.2f}",
                #             ha='center', va='bottom', fontsize=annot_fontsize)
            metricname = "LLM-Judge"
            if metric == "bleu-4":
                metricname = "BLEU-4"
            elif metric == "rouge-L":
                metricname = "ROUGE-L"
            stagename = stage
            if stage == "Avg":
                stagename = "AP"
            ax.set_title(f"{stagename} for {metricname}", fontsize=28)
            ax.set_xticks(x)
            # ax.tick_params(axis='x', labelsize=22)
            ax.tick_params(axis='y', labelsize=32)
            ax.set_xticklabels(orders, fontsize=32)
            ax.set_ylabel(stagename, fontsize=28)

            key = (stage, metric)
            if key in y_limits_map:
                ymin, ymax = y_limits_map[key]
                ax.set_ylim(ymin, ymax)
            elif stage in y_limits_map:
                ymin, ymax = y_limits_map[stage]
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_ylim(0, None)

    # 7. 添加共享图例：水平放置于最上方中央
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               ncol=n_methods,
               bbox_to_anchor=(0.52, 1.09),
               fontsize=28
               )
    fig.subplots_adjust(top=legend_pad, bottom=0.12, hspace=0.4)

    # 8. 保存
    fig.savefig('order.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


if __name__ == '__main__':
    method_dirs = [
        'save_test/Qwen2.5-7B-Instruct/lora/Ours-order1',
        'save_test/Qwen2.5-7B-Instruct/lora/Ours-order2',
        'save_test/Qwen2.5-7B-Instruct/lora/Ours-order3',
        'save_test/Qwen2.5-7B-Instruct/lora/ER-order1',
        'save_test/Qwen2.5-7B-Instruct/lora/ER-order2',
        'save_test/Qwen2.5-7B-Instruct/lora/ER-order3',
        'save_test/Qwen2.5-7B-Instruct/lora/CPPO-order1',
        'save_test/Qwen2.5-7B-Instruct/lora/CPPO-order2',
        'save_test/Qwen2.5-7B-Instruct/lora/CPPO-order3',
    ]
    orders = ['order1', 'order2', 'order3']
    metrics = ['bleu-4', 'rouge-L','LLM-judge']
    stages = ['BWT', 'Last', 'Avg']

    y_limits = {
        # 'BWT':  (-12,  10),
        # 'Last': (0,  38),
        # 'Avg':  (0,  33),
        ('BWT','bleu-4'):(-12,10),
        ('BWT','rouge-L'):(-6,3),
        ('BWT','LLM-judge'):(-12,5),
        ('Last','bleu-4'):(0,35),
        ('Last','rouge-L'):(0,33),
        ('Last','LLM-judge'):(0,68),
        ('Avg','bleu-4'):(0,35),
        ('Avg','rouge-L'):(0,33),
        ('Avg','LLM-judge'):(0,68),
    }
    custom_colors = ['#B8DFB9', '#76BDE5', '#E7E6D6']  # 各 baseline 颜色
    plot_baseline_metrics(
        method_dirs,
        orders=orders,
        metric_names=metrics,
        stages=stages,
        colors=custom_colors,
        y_limits=y_limits,
        bar_width=0.24,
        annot_fontsize=14,
        legend_pad=0.8,
        # figsize=(20, 15),
    )