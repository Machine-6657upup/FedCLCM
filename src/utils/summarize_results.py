#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验结果汇总和分析脚本
用于自动提取、整理和可视化实验结果
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResultAnalyzer:
    def __init__(self, result_dir):
        self.result_dir = Path(result_dir)
        self.results = defaultdict(dict)
        
    def parse_log_file(self, log_path):
        """解析单个日志文件，提取关键指标"""
        results = {
            'accuracy': [],
            'asr': [],  # Attack Success Rate
            'loss': [],
            'best_acc': 0.0,
            'best_asr': 0.0,
            'convergence_round': -1
        }
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 提取测试准确率
                acc_matches = re.findall(r'Test Accuracy[:\s]+([0-9.]+)', content)
                if acc_matches:
                    results['accuracy'] = [float(x) for x in acc_matches]
                    results['best_acc'] = max(results['accuracy'])
                
                # 提取ASR
                asr_matches = re.findall(r'ASR[:\s]+([0-9.]+)', content)
                if asr_matches:
                    results['asr'] = [float(x) for x in asr_matches]
                    results['best_asr'] = max(results['asr']) if results['asr'] else 0.0
                
                # 提取损失
                loss_matches = re.findall(r'Loss[:\s]+([0-9.]+)', content)
                if loss_matches:
                    results['loss'] = [float(x) for x in loss_matches]
                
                # 估算收敛轮数（准确率达到最高值95%时的轮数）
                if results['accuracy']:
                    threshold = results['best_acc'] * 0.95
                    for i, acc in enumerate(results['accuracy']):
                        if acc >= threshold:
                            results['convergence_round'] = i + 1
                            break
                            
        except Exception as e:
            print(f"解析文件失败 {log_path}: {e}")
            
        return results
    
    def collect_results(self, pattern='*.log'):
        """收集所有实验结果"""
        log_files = list(self.result_dir.glob(pattern))
        print(f"找到 {len(log_files)} 个日志文件")
        
        for log_file in log_files:
            exp_name = log_file.stem
            print(f"正在解析: {exp_name}")
            self.results[exp_name] = self.parse_log_file(log_file)
            
        return self.results
    
    def generate_comparison_table(self, metrics=['best_acc', 'best_asr', 'convergence_round']):
        """生成对比表格"""
        data = []
        for exp_name, result in self.results.items():
            row = {'实验名称': exp_name}
            for metric in metrics:
                if metric in result:
                    if metric == 'best_acc' or metric == 'best_asr':
                        row[metric] = f"{result[metric]*100:.2f}%"
                    else:
                        row[metric] = result[metric]
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 重命名列
        column_names = {
            'best_acc': '最佳准确率',
            'best_asr': 'ASR↓',
            'convergence_round': '收敛轮数'
        }
        df.rename(columns=column_names, inplace=True)
        
        return df
    
    def plot_convergence_curves(self, save_path=None):
        """绘制收敛曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准确率曲线
        ax1 = axes[0]
        for exp_name, result in self.results.items():
            if result['accuracy']:
                ax1.plot(result['accuracy'], label=exp_name, linewidth=2)
        ax1.set_xlabel('训练轮数', fontsize=12)
        ax1.set_ylabel('测试准确率', fontsize=12)
        ax1.set_title('准确率收敛曲线', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ASR曲线
        ax2 = axes[1]
        for exp_name, result in self.results.items():
            if result['asr']:
                ax2.plot(result['asr'], label=exp_name, linewidth=2)
        ax2.set_xlabel('训练轮数', fontsize=12)
        ax2.set_ylabel('攻击成功率 (ASR)', fontsize=12)
        ax2.set_title('ASR变化曲线', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线已保存至: {save_path}")
        else:
            plt.savefig(self.result_dir / 'convergence_curves.png', dpi=300, bbox_inches='tight')
            print(f"收敛曲线已保存至: {self.result_dir / 'convergence_curves.png'}")
            
        plt.close()
    
    def plot_hyperparameter_sensitivity(self, param_name, metric='best_acc'):
        """绘制超参数敏感性曲线"""
        # 提取参数值和对应的性能
        param_results = defaultdict(list)
        
        for exp_name, result in self.results.items():
            # 从实验名称中提取参数值
            match = re.search(f'{param_name}([0-9.]+)', exp_name)
            if match:
                param_value = float(match.group(1))
                metric_value = result.get(metric, 0)
                if isinstance(metric_value, str):
                    metric_value = float(metric_value.strip('%')) / 100
                param_results[param_value].append(metric_value)
        
        if not param_results:
            print(f"未找到关于参数 {param_name} 的结果")
            return
        
        # 计算均值和标准差
        param_values = sorted(param_results.keys())
        mean_values = [np.mean(param_results[p]) for p in param_values]
        std_values = [np.std(param_results[p]) for p in param_values]
        
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, mean_values, 'o-', linewidth=2, markersize=8)
        plt.fill_between(param_values, 
                        np.array(mean_values) - np.array(std_values),
                        np.array(mean_values) + np.array(std_values),
                        alpha=0.3)
        
        plt.xlabel(f'{param_name}', fontsize=12)
        plt.ylabel(f'{metric}', fontsize=12)
        plt.title(f'{param_name} 敏感性分析', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        save_path = self.result_dir / f'sensitivity_{param_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"敏感性曲线已保存至: {save_path}")
        plt.close()
    
    def generate_latex_table(self, df, caption="实验结果对比"):
        """生成LaTeX表格代码"""
        latex_code = df.to_latex(index=False, escape=False, 
                                  caption=caption,
                                  label='tab:results')
        
        latex_file = self.result_dir / 'results_table.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        print(f"LaTeX表格已保存至: {latex_file}")
        return latex_code
    
    def generate_summary_report(self):
        """生成综合报告"""
        report = []
        report.append("=" * 60)
        report.append("实验结果汇总报告")
        report.append("=" * 60)
        report.append(f"\n实验总数: {len(self.results)}")
        report.append(f"结果目录: {self.result_dir}")
        report.append("\n" + "-" * 60)
        
        # 找出最佳性能
        best_acc_exp = max(self.results.items(), 
                          key=lambda x: x[1].get('best_acc', 0))
        best_asr_exp = min(self.results.items(), 
                          key=lambda x: x[1].get('best_asr', 1))
        
        report.append("\n关键发现:")
        report.append(f"  最高准确率: {best_acc_exp[1]['best_acc']*100:.2f}% ({best_acc_exp[0]})")
        report.append(f"  最低ASR: {best_asr_exp[1]['best_asr']*100:.2f}% ({best_asr_exp[0]})")
        
        # 详细结果
        report.append("\n" + "-" * 60)
        report.append("\n详细结果:")
        for exp_name, result in sorted(self.results.items()):
            report.append(f"\n  {exp_name}:")
            report.append(f"    准确率: {result.get('best_acc', 0)*100:.2f}%")
            report.append(f"    ASR: {result.get('best_asr', 0)*100:.2f}%")
            report.append(f"    收敛轮数: {result.get('convergence_round', 'N/A')}")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_file = self.result_dir / 'summary_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n报告已保存至: {report_file}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description='实验结果汇总和分析')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='结果目录路径')
    parser.add_argument('--pattern', type=str, default='*.log',
                       help='日志文件匹配模式')
    parser.add_argument('--generate_plots', action='store_true',
                       help='生成图表')
    parser.add_argument('--generate_latex', action='store_true',
                       help='生成LaTeX表格')
    parser.add_argument('--hyperparam', type=str, default=None,
                       help='超参数敏感性分析的参数名称，如: beta, adveps')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ResultAnalyzer(args.result_dir)
    
    # 收集结果
    print("正在收集实验结果...")
    analyzer.collect_results(args.pattern)
    
    # 生成对比表格
    print("\n生成对比表格...")
    df = analyzer.generate_comparison_table()
    print("\n" + df.to_string(index=False))
    
    # 保存为CSV
    csv_path = Path(args.result_dir) / 'results_summary.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果表格已保存至: {csv_path}")
    
    # 生成图表
    if args.generate_plots:
        print("\n生成收敛曲线...")
        analyzer.plot_convergence_curves()
    
    # 超参数敏感性分析
    if args.hyperparam:
        print(f"\n生成 {args.hyperparam} 敏感性曲线...")
        analyzer.plot_hyperparameter_sensitivity(args.hyperparam)
    
    # 生成LaTeX表格
    if args.generate_latex:
        print("\n生成LaTeX表格...")
        analyzer.generate_latex_table(df)
    
    # 生成综合报告
    print("\n生成综合报告...")
    analyzer.generate_summary_report()
    
    print("\n✅ 分析完成！")


if __name__ == '__main__':
    main()


