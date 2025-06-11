import re
import ast
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    """解析日志文件，提取有效数据"""
    data = []
    pattern = re.compile(r"^{.*}$")  # 匹配大括号包裹的字典行
    
    with open(log_path, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if pattern.match(line):
                try:
                    # 将单引号JSON转换为双引号JSON并解析
                    dict_str = line.replace("'", '"')
                    entry = ast.literal_eval(dict_str)
                    data.append(entry)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing line {i}: {e}")
                    continue
    return data

def plot_metrics(data, smoothing_window=50):
    """绘制三个关键指标的变化曲线"""
    if not data:
        raise ValueError("No valid data found in log file")
        
    steps = range(len(data))
    print(f"Found {len(data)} valid data entries")
    print("Sample data:", data[0])
    
    def smooth(scalars, weight=0.9):
        """指数移动平均平滑"""
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    check_answer = smooth([d.get('rewards/checkAnswer/mean', 0) for d in data])
    match_format = smooth([d.get('rewards/matchFormat/mean', 0) for d in data])
    reward = smooth([d.get('reward', 0) for d in data])
    
    plt.figure(figsize=(16, 9), dpi=120)
    # plt.plot(steps, check_answer, 'b', linewidth=1.5, alpha=0.5,
    #          label='Check Answer Reward')
    plt.plot(steps, match_format, 'g', linewidth=1.5, alpha=0.7,
             label='Match Format Reward')
    # plt.plot(steps, reward, 'r', linewidth=2, alpha=0.7,
    #          label='Total Reward')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Reward Value', fontsize=12)
    plt.title(f'Training Rewards Over Steps ({smoothing_window} windowed)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rewards_0611_F.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    log_data = parse_log_file('grpo_0611_F.log')
    plot_metrics(log_data)