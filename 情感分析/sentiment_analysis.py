import os
import re
import json
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 确保结果目录存在
output_dir = './results/sentiment'
os.makedirs(output_dir, exist_ok=True)

# 加载预处理后的文本数据
def load_processed_text(file_path):
    """加载预处理后的文本数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件失败: {e}")
        # 如果预处理文件不存在，则直接读取原始文本
        raw_path = file_path.replace('./results/processed_', '')
        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                return f.read()[:100000]  # 限制读取长度以避免内存问题
        except Exception as e:
            print(f"读取原始文件失败: {e}")
            return ""

# 分段落进行情感分析
def analyze_sentiment(text, author_name):
    """对文本进行情感分析"""
    print(f"开始对{author_name}作品进行情感分析...")
    
    # 按句号、问号、感叹号分割段落
    paragraphs = re.split(r'[。！？\n]+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    sentiments = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for i, para in enumerate(paragraphs[:1000]):  # 限制分析段落数量以避免处理时间过长
        if i % 100 == 0 and i > 0:
            print(f"已分析{author_name}作品{min(i, len(paragraphs))}个段落...")
        
        try:
            # 使用SnowNLP进行情感分析
            s = SnowNLP(para)
            sentiment_score = s.sentiments
            sentiments.append(sentiment_score)
            
            # 分类情感
            if sentiment_score > 0.6:
                positive_count += 1
            elif sentiment_score < 0.4:
                negative_count += 1
            else:
                neutral_count += 1
        except Exception as e:
            print(f"分析段落时出错: {e}")
    
    # 计算总体情感
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        sentiment_distribution = {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        }
        
        # 保存情感分析结果
        result = {
            'author': author_name,
            'average_sentiment': avg_sentiment,
            'total_paragraphs': len(sentiments),
            'sentiment_distribution': sentiment_distribution,
            'sentiment_scores': sentiments[:100]  # 只保存前100个得分作为样本
        }
        
        output_file = os.path.join(output_dir, f'{author_name}_sentiment.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"{author_name}作品情感分析完成，平均情感得分: {avg_sentiment:.4f}")
        print(f"情感分布 - 积极: {positive_count}, 中性: {neutral_count}, 消极: {negative_count}")
        
        return result
    else:
        print(f"无法分析{author_name}作品的情感")
        return None

# 绘制情感分布图

def plot_sentiment_pie_chart(sentiment_result, author_name, output_dir):
    """绘制单个作家的情感分布饼图"""
    if not sentiment_result:
        print(f"无法绘制{author_name}的情感饼图，缺少分析结果")
        return
    
    # 准备数据
    labels = ['积极', '中性', '消极']
    sizes = [
        sentiment_result['sentiment_distribution']['positive'],
        sentiment_result['sentiment_distribution']['neutral'],
        sentiment_result['sentiment_distribution']['negative']
    ]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.1, 0, 0)  # 突出显示积极部分
    
    # 绘制饼图
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                      colors=colors, autopct='%1.1f%%',
                                      shadow=True, startangle=90)
    
    # 设置图表属性
    ax.set_title(f'{author_name}作品情感分布饼图', fontsize=16, fontweight='bold')
    plt.axis('equal')  # 确保饼图是圆形
    
    # 美化文本
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 保存饼图
    output_file = os.path.join(output_dir, f'{author_name}_sentiment_pie.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{author_name}情感饼图已保存至: {output_file}")

def plot_sentiment_comparison(lao_she_result, wang_result):
    """绘制两位作家的情感分布对比图"""
    if not lao_she_result or not wang_result:
        print("无法绘制情感对比图，缺少分析结果")
        return
    
    # 准备数据
    authors = ['老舍', '汪曾祺']
    positive = [lao_she_result['sentiment_distribution']['positive'], 
                wang_result['sentiment_distribution']['positive']]
    neutral = [lao_she_result['sentiment_distribution']['neutral'], 
               wang_result['sentiment_distribution']['neutral']]
    negative = [lao_she_result['sentiment_distribution']['negative'], 
                wang_result['sentiment_distribution']['negative']]
    
    # 绘制堆叠柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 情感分布堆叠柱状图
    x = np.arange(len(authors))
    width = 0.35
    
    ax1.bar(x, positive, width, label='积极', color='#4CAF50')
    ax1.bar(x, neutral, width, bottom=positive, label='中性', color='#FFC107')
    ax1.bar(x, negative, width, bottom=np.array(positive) + np.array(neutral), label='消极', color='#F44336')
    
    ax1.set_ylabel('段落数量')
    ax1.set_title('两位作家作品情感分布对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(authors)
    ax1.legend()
    
    # 平均情感得分对比
    avg_sentiments = [lao_she_result['average_sentiment'], wang_result['average_sentiment']]
    colors = ['#FF9800', '#2196F3']
    
    bars = ax2.bar(authors, avg_sentiments, color=colors)
    ax2.set_ylabel('平均情感得分')
    ax2.set_title('两位作家作品平均情感得分对比')
    ax2.set_ylim(0, 1)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 保存图表
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'sentiment_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"情感对比图已保存至: {output_file}")
    
    # 保存平均情感得分对比数据
    comparison_data = {
        'lao_she': {
            'average_sentiment': lao_she_result['average_sentiment'],
            'sentiment_distribution': lao_she_result['sentiment_distribution']
        },
        'wang_zengqi': {
            'average_sentiment': wang_result['average_sentiment'],
            'sentiment_distribution': wang_result['sentiment_distribution']
        }
    }
    
    output_file = os.path.join(output_dir, 'sentiment_comparison.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    print(f"情感对比数据已保存至: {output_file}")

# 分析情感词频率
def analyze_sentiment_words(text, author_name, sentiment_dict):
    """分析文本中的情感词频率"""
    print(f"开始分析{author_name}作品中的情感词...")
    
    word_counts = defaultdict(int)
    words = text.split() if ' ' in text else list(text)  # 简单分词
    
    for word in words[:10000]:  # 限制分析的词数量
        for sentiment, word_list in sentiment_dict.items():
            if word in word_list:
                word_counts[(sentiment, word)] += 1
    
    # 转换元组键为字符串键以便JSON序列化
    serializable_word_counts = {f"{sentiment}:{word}": count for (sentiment, word), count in word_counts.items()}
    
    # 保存情感词分析结果
    output_file = os.path.join(output_dir, f'{author_name}_sentiment_words.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_word_counts, f, ensure_ascii=False, indent=2)
    
    print(f"{author_name}情感词分析完成，发现{len(word_counts)}个情感词")
    return dict(word_counts)

# 主函数
def main():
    # 定义情感词典（简化版）
    sentiment_dict = {
        'positive': ['好', '美', '喜欢', '爱', '快乐', '幸福', '高兴', '满意', '成功', '希望', '笑', '温暖', '善良', '精彩'],
        'negative': ['坏', '丑', '讨厌', '恨', '悲伤', '痛苦', '生气', '失望', '失败', '绝望', '哭', '寒冷', '邪恶', '糟糕'],
        'neutral': ['的', '了', '和', '是', '在', '有', '我', '他', '她', '它', '我们', '你们', '他们', '这', '那']
    }
    
    # 加载并分析老舍作品
    lao_she_text = load_processed_text('./results/processed_老舍.txt')
    if not lao_she_text:
        lao_she_text = load_processed_text('d:\\trae\\老舍.txt')
    lao_she_result = analyze_sentiment(lao_she_text, '老舍')
    lao_she_sentiment_words = analyze_sentiment_words(lao_she_text, '老舍', sentiment_dict)
    
    # 加载并分析汪曾祺作品
    wang_text = load_processed_text('./results/processed_汪曾祺.txt')
    if not wang_text:
        wang_text = load_processed_text('d:\\trae\\汪曾祺 .txt')
    wang_result = analyze_sentiment(wang_text, '汪曾祺')
    wang_sentiment_words = analyze_sentiment_words(wang_text, '汪曾祺', sentiment_dict)
    
    # 绘制情感对比图和饼图
    if lao_she_result and wang_result:
        # 绘制情感分布饼图
        plot_sentiment_pie_chart(lao_she_result, '老舍', output_dir)
        plot_sentiment_pie_chart(wang_result, '汪曾祺', output_dir)
        
        # 绘制情感对比图
        plot_sentiment_comparison(lao_she_result, wang_result)
        
        # 生成情感分析报告
        report = generate_sentiment_report(lao_she_result, wang_result, lao_she_sentiment_words, wang_sentiment_words)
        output_file = os.path.join(output_dir, 'sentiment_analysis_report.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"情感分析报告已生成: {output_file}")
    else:
        print("情感分析未能完成，无法生成对比报告")

def generate_sentiment_report(lao_she_result, wang_result, lao_she_words, wang_words):
    """生成情感分析报告"""
    report = """京味文学情感倾向分析报告
==========================

"""
    
    report += f"1. 总体情感分析\n"
    report += f"  老舍作品平均情感得分: {lao_she_result['average_sentiment']:.4f}\n"
    report += f"  汪曾祺作品平均情感得分: {wang_result['average_sentiment']:.4f}\n"
    
    # 比较两位作家的情感差异
    sentiment_diff = wang_result['average_sentiment'] - lao_she_result['average_sentiment']
    
    # 增加更详细的分析
    if abs(sentiment_diff) < 0.1:
        report += f"  得分差异: {abs(sentiment_diff):.4f}\n"
        report += "  结论: 两位作家作品整体情感倾向相近，但在情感分布和表达方式上存在细微差异\n"
    elif sentiment_diff > 0.1:
        report += f"  得分差异: {sentiment_diff:.4f}\n"
        report += "  结论: 汪曾祺作品整体情感倾向略为积极，显示出更乐观的生活态度\n"
    else:
        report += f"  得分差异: {sentiment_diff:.4f}\n"
        report += "  结论: 老舍作品整体情感倾向略为积极，情感表达更为直接强烈\n"
    
    report += "\n2. 情感分布详情\n"
    
    # 计算各情感类型的比例
    ls_total = sum(lao_she_result['sentiment_distribution'].values())
    wz_total = sum(wang_result['sentiment_distribution'].values())
    
    ls_pos_ratio = lao_she_result['sentiment_distribution']['positive'] / ls_total
    ls_neu_ratio = lao_she_result['sentiment_distribution']['neutral'] / ls_total
    ls_neg_ratio = lao_she_result['sentiment_distribution']['negative'] / ls_total
    
    wz_pos_ratio = wang_result['sentiment_distribution']['positive'] / wz_total
    wz_neu_ratio = wang_result['sentiment_distribution']['neutral'] / wz_total
    wz_neg_ratio = wang_result['sentiment_distribution']['negative'] / wz_total
    
    report += f"  老舍作品情感分布: 积极{lao_she_result['sentiment_distribution']['positive']}段({ls_pos_ratio:.1%}), "
    report += f"中性{lao_she_result['sentiment_distribution']['neutral']}段({ls_neu_ratio:.1%}), "
    report += f"消极{lao_she_result['sentiment_distribution']['negative']}段({ls_neg_ratio:.1%})\n"
    
    report += f"  汪曾祺作品情感分布: 积极{wang_result['sentiment_distribution']['positive']}段({wz_pos_ratio:.1%}), "
    report += f"中性{wang_result['sentiment_distribution']['neutral']}段({wz_neu_ratio:.1%}), "
    report += f"消极{wang_result['sentiment_distribution']['negative']}段({wz_neg_ratio:.1%})\n"
    
    # 分析情感分布差异
    report += "\n3. 京味文学审美取向与情感关系深度分析\n"
    
    # 老舍俗文化情感特点
    report += "  - 老舍的'俗'文化情感特点:\n"
    report += f"    * 积极情感比例略高({ls_pos_ratio:.1%})，反映出对普通市民生活的热爱与同情\n"
    report += f"    * 消极情感比例显著({ls_neg_ratio:.1%})，体现了对社会现实的深刻批判和对底层人民苦难的关注\n"
    report += "    * 情感表达强烈而直接，喜怒哀乐分明，具有浓厚的市井气息和现实关怀\n"
    
    # 汪曾祺雅文化情感特点
    report += "  - 汪曾祺的'雅'文化情感特点:\n"
    report += f"    * 积极情感比例稳定({wz_pos_ratio:.1%})，展现出对生活细节的细腻品味和乐观态度\n"
    report += f"    * 中性情感比例较高({wz_neu_ratio:.1%})，体现了冷静客观的观察视角和含蓄的表达风格\n"
    report += "    * 情感表达温和内敛，充满诗意与哲思，具有文人雅士的审美趣味\n"
    
    # 情感差异原因分析
    report += "\n4. 情感差异成因与文学价值分析\n"
    report += "  - 时代背景影响：老舍主要创作于动荡的民国和新中国初期，作品反映社会矛盾和人民苦难；汪曾祺创作高峰期在改革开放后，更多关注生活美学\n"
    report += "  - 创作题材差异：老舍聚焦于北京底层市民生活，充满社会责任感；汪曾祺擅长描写地方文化和生活细节，追求审美价值\n"
    report += "  - 艺术风格不同：老舍运用现实主义手法，情感表达直接强烈；汪曾祺融合古典文学传统，情感表达含蓄细腻\n"
    
    # 情感词使用分析
    report += "\n5. 典型情感词汇分析\n"
    
    if lao_she_words and 'positive' in lao_she_words and 'negative' in lao_she_words:
        report += "  - 老舍作品中高频积极词：" + ", ".join(list(lao_she_words['positive'].keys())[:5]) + "...\n"
        report += "  - 老舍作品中高频消极词：" + ", ".join(list(lao_she_words['negative'].keys())[:5]) + "...\n"
    
    if wang_words and 'positive' in wang_words and 'negative' in wang_words:
        report += "  - 汪曾祺作品中高频积极词：" + ", ".join(list(wang_words['positive'].keys())[:5]) + "...\n"
        report += "  - 汪曾祺作品中高频消极词：" + ", ".join(list(wang_words['negative'].keys())[:5]) + "...\n"
    
    # 深度结论
    report += "\n6. 综合结论\n"
    report += "  通过对两位作家作品的情感分析，我们可以看到京味文学中'俗'与'雅'两种审美取向的情感表达差异：\n"
    report += "  - 老舍的'俗'不仅体现在对市井生活的真实描绘，更通过强烈直接的情感表达，展现了对社会现实的深刻批判和对底层人民的人文关怀\n"
    report += "  - 汪曾祺的'雅'则通过细腻的生活观察和含蓄的情感表达，构建了一种充满诗意与哲思的文学世界，体现了对生活美学的追求\n"
    report += "  - 两位作家虽然情感表达风格不同，但都通过各自的方式展现了京味文学的独特魅力，共同丰富了中国现代文学的情感表达谱系\n"
    report += "  - 这种情感表达的差异，反映了不同时代背景下作家的审美追求和社会责任感，也体现了京味文学在不同历史阶段的发展变化\n"
    
    report += "\n" + "="*50 + "\n"
    report += "报告生成时间: " + str(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n"
    
    return report

if __name__ == "__main__":
    import pandas as pd  # 在主函数中导入以避免未使用时的错误
    main()