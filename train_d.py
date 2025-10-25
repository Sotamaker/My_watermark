import torch
import torch.nn as nn
import torch.optim as optim
import threading
import argparse
import sys

# 定义一个更大的模型
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.fc1 = nn.Linear(8192, 18192)
        self.fc2 = nn.Linear(18192, 18192)
        self.fc3 = nn.Linear(18192, 8192)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 每张 GPU 的任务
def gpu_task(gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    model = LargeModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    input_tensor = torch.randn(8192, 8192).to(device)

    print(f"[GPU {gpu_id}] 正在执行任务...")
    while True:
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        optimizer.step()

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, required=True,
                        help='使用的 GPU 序号，例如 --gpus 0 1 2')
    args = parser.parse_args()

    gpu_ids = args.gpus

    # 检查 GPU 合法性
    available_gpus = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= available_gpus:
            print(f"❌ 无效的 GPU 编号: {gpu_id}")
            sys.exit(1)

    print(f"✅ 使用以下 GPU 卡: {gpu_ids}")

    threads = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(target=gpu_task, args=(gpu_id,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
