import os
import json
import pandas as pd
import numpy as np

def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def calculate_metric_statistics(metric_values, datasets):
    """
    计算单个指标的统计信息
    """
    N = len(datasets)
    
    # 对角线值 (在对应数据集上训练后在该数据集上的性能)
    diagonal = [metric_values[i][i] for i in range(N)]
    
    # 每行平均值 (训练到第i个数据集后的平均性能)
    row_avg = [sum(metric_values[i]) / len(metric_values[i]) for i in range(N)]
    
    # 累计平均值 (从第1个到第i个数据集的row_avg的平均)
    cumulative_avg = [sum(row_avg[:i+1]) / (i+1) for i in range(N)]
    
    # 后向迁移 (BWT - Backward Transfer)
    bwt = [0.0]  # 第一个数据集没有后向迁移
    for i in range(1, N):
        # 计算前i个数据集在训练到第i个数据集后vs训练后的性能差异
        diff = sum(metric_values[i][j] - metric_values[j][j] for j in range(i)) / i
        bwt.append(diff)
    
    return {
        'diagonal': [round(x, 2) for x in diagonal],
        'row_avg': [round(x, 2) for x in row_avg], 
        'cumulative_avg': [round(x, 2) for x in cumulative_avg],
        'bwt': [round(x, 2) for x in bwt]
    }

def load_all_metrics(metric_dir, datasets, metric_fields):
    """
    加载所有指标数据
    """
    all_metrics = {field: [] for field in metric_fields}
    
    for i, tr in enumerate(datasets):
        metric_rows = {field: [] for field in metric_fields}
        
        for j in range(i + 1):
            path = f"{metric_dir}/{tr}/{datasets[j]}/predict_results.json"
            data = load_data(path)
            # print(data)
            
            for field in metric_fields:
                metric_rows[field].append(data[field])
        
        for field in metric_fields:
            all_metrics[field].append(metric_rows[field])
    
    return all_metrics

def create_metric_matrix(metric_dir, datasets, metric_config, all_metrics):
    """
    创建指标统计矩阵
    """
    matrix = {}
    cols = datasets + ["bwt", "last", "avg"]
    
    for display_name, field_name in metric_config.items():
        stats = calculate_metric_statistics(all_metrics[field_name], datasets)
        
        # 组合最终结果：对角线值 + [最终bwt, 最终row_avg, 最终cumulative_avg]
        final_values = (stats['diagonal'] + 
                       [stats['bwt'][-1], stats['row_avg'][-1], stats['cumulative_avg'][-1]])
        
        matrix[display_name] = dict(zip(cols, final_values))
    
    return matrix

def save_metrics_to_csv(metric_dir, datasets, metric_config, all_metrics):
    """
    保存指标数据到CSV文件
    """
    N = len(datasets)
    
    for display_name, field_name in metric_config.items():
        # 1. 创建上三角矩阵CSV
        matrix_data = np.full((N, N), np.nan)  # 用NaN填充下三角
        
        for i in range(N):
            for j in range(i + 1):  # 只填充上三角和对角线
                matrix_data[j, i] = all_metrics[field_name][i][j]
        
        # 创建DataFrame
        matrix_df = pd.DataFrame(matrix_data, 
                               index=datasets,  # 行名：测试集
                               columns=datasets)  # 列名：训练集
        
        # 保存上三角矩阵
        matrix_path = os.path.join(metric_dir, f"{display_name}_matrix.csv")
        matrix_df.to_csv(matrix_path, float_format='%.2f')
        print(f"Saved {display_name} matrix to {matrix_path}")
        
        # 2. 计算并保存统计信息
        stats = calculate_metric_statistics(all_metrics[field_name], datasets)
        
        # 创建统计信息DataFrame
        stats_data = {
            'dataset': datasets,
            'diagonal': stats['diagonal'],
            'last': stats['row_avg'],
            'avg': stats['cumulative_avg'], 
            'bwt': stats['bwt']
        }
        
        stats_df = pd.DataFrame(stats_data)
        
        # 保存统计信息
        stats_path = os.path.join(metric_dir, f"{display_name}_stats.csv")
        stats_df.to_csv(stats_path, index=False, float_format='%.2f')
        print(f"Saved {display_name} stats to {stats_path}")

def create_combined_stats_csv(metric_dir, datasets, metric_config, all_metrics):
    """
    创建综合统计信息CSV文件，包含所有指标的统计数据
    """
    combined_data = []
    
    for display_name, field_name in metric_config.items():
        stats = calculate_metric_statistics(all_metrics[field_name], datasets)
        
        for i, dataset in enumerate(datasets):
            combined_data.append({
                'metric': display_name,
                'dataset': dataset,
                'stage': i + 1,
                'diagonal': stats['diagonal'][i],
                'last': stats['row_avg'][i],
                'avg': stats['cumulative_avg'][i],
                'bwt': stats['bwt'][i]
            })
    
    combined_df = pd.DataFrame(combined_data)
    
    # 保存综合统计信息
    combined_path = os.path.join(metric_dir, "combined_stats.csv")
    combined_df.to_csv(combined_path, index=False, float_format='%.2f')
    print(f"Saved combined stats to {combined_path}")
    
    return combined_df

def create_summary_table(metric_dir, datasets, metric_config, all_metrics):
    """
    创建汇总表，每个指标一行，包含最终的bwt, last, avg值
    """
    summary_data = []
    
    for display_name, field_name in metric_config.items():
        stats = calculate_metric_statistics(all_metrics[field_name], datasets)
        
        summary_data.append({
            'metric': display_name,
            'final_bwt': stats['bwt'][-1],
            'final_last': stats['row_avg'][-1],
            'final_avg': stats['cumulative_avg'][-1],
            'avg_diagonal': np.mean(stats['diagonal'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存汇总表
    summary_path = os.path.join(metric_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False, float_format='%.2f')
    print(f"Saved summary to {summary_path}")
    
    return summary_df

def process_method(method_name, model_name, datasets, metric_config):
    """
    针对每个method_name进行BERTScore计算并保存结果
    """
    metric_dir = f"save_test/{model_name}/lora/{method_name}"
    order = metric_dir.split("order")[-1]
    datasets = [
        "Capybara-Preferences", "HC3", "hh-rlhf-harmless-base",
        "hh-rlhf-helpful-base", "safe-rlhf", "TruthfulQA",
    ]
    if order == "2":
        datasets = ["TruthfulQA","safe-rlhf","hh-rlhf-helpful-base", "hh-rlhf-harmless-base","HC3","Capybara-Preferences"]
    elif order == "3":
        datasets = ["hh-rlhf-harmless-base","Capybara-Preferences", "TruthfulQA", "hh-rlhf-helpful-base","HC3", "safe-rlhf"]
    # 提取所有需要的字段
    metric_fields = list(metric_config.values())
    
    # 加载所有指标数据
    all_metrics = load_all_metrics(metric_dir, datasets, metric_fields)
    
    # 计算指标矩阵
    matrix = create_metric_matrix(metric_dir, datasets, metric_config, all_metrics)
    
    # 保存结果
    output_path = os.path.join(metric_dir, "metric.json")
    save_data(matrix, output_path)
    print(f"Saved matrix to {output_path}")
    
    # 可选：保存CSV文件
    save_metrics_to_csv(metric_dir, datasets, metric_config, all_metrics)
    create_combined_stats_csv(metric_dir, datasets, metric_config, all_metrics)
    create_summary_table(metric_dir, datasets, metric_config, all_metrics)
    
    # 打印结果预览
    print("\n指标矩阵预览:")
    for metric_name, values in matrix.items():
        print(f"\n{metric_name}:")
        for col, val in values.items():
            print(f"  {col}: {val}")

def main():
    # 配置参数
    model_name = "Qwen2.5-7B-Instruct"
    method_names = ['DPO+CPPO-w-replay-order1']
    datasets = [
        "Capybara-Preferences", "HC3",
        "hh-rlhf-harmless-base", "hh-rlhf-helpful-base", 
        "safe-rlhf", "TruthfulQA"
    ]
    
    # 指标配置：{显示名称: JSON字段名称}
    metric_config = {
        "bleu-4": "predict_bleu-4",
        "rouge-L": "predict_rouge-l", 
        "LLM-judge": "LLM-judge"
    }
    
    # 对于每个method_name，执行相同的处理流程
    for method_name in method_names:
        print(f"\nProcessing method: {method_name}")
        process_method(method_name, model_name, datasets, metric_config)

if __name__ == "__main__":
    main()

