import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import json

# 设置随机种子和设备
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# =====================
# 数据预处理模块
# =====================

class TextDataset(Dataset):
    def __init__(self, text, vocab):
        self.vocab = vocab
        self.text_indices = [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

    def __len__(self):
        return len(self.text_indices) - 1  # 输入和目标序列需错开一位

    def __getitem__(self, idx):
        input_seq = self.text_indices[:idx + 1]  # 动态生成输入序列（从头到当前索引）
        target_seq = self.text_indices[1:idx + 2]  # 动态生成目标序列（从第2个词到当前索引+1）
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def build_vocab(text, min_freq=1):
    """
    构建词汇表
    """
    words = text.split()
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    vocab = {word: idx for idx, (word, freq) in enumerate(word_freq.items()) if freq >= min_freq}
    vocab["<PAD>"] = len(vocab)
    vocab["<UNK>"] = len(vocab)

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    return vocab, reverse_vocab

def collate_fn(batch, vocab):
    """
    动态填充批次中的序列，使其长度一致。
    """
    batch_size = len(batch)
    max_len = max(len(seq[0]) for seq in batch)  # 找到当前批次中最长序列的长度
    padded_inputs = torch.full((batch_size, max_len), vocab["<PAD>"], dtype=torch.long)
    padded_targets = torch.full((batch_size, max_len), vocab["<PAD>"], dtype=torch.long)

    for i, (input_seq, target_seq) in enumerate(batch):
        padded_inputs[i, :len(input_seq)] = input_seq
        padded_targets[i, :len(target_seq)] = target_seq

    return padded_inputs, padded_targets

def load_text_from_file(file_path):
    """
    从外部文件加载文本数据
    :param file_path: 文本文件路径
    :return: 读取的文本数据
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# =====================
# 词汇表管理模块
# =====================

def save_vocab(vocab, file_path="vocab.json"):
    """
    保存词汇表到JSON文件
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"词汇表已保存至 {file_path}")

def load_vocab(file_path="vocab.json"):
    """
    从JSON文件加载词汇表
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # 确保索引是整数类型
        vocab = {word: int(idx) for word, idx in vocab.items()}
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        print(f"已加载词汇表，共 {len(vocab)} 个词汇")
        return vocab, reverse_vocab
    else:
        print(f"未找到词汇表文件 {file_path}")
        return None, None

# =====================
# 模型定义模块
# =====================

class GenerativeDenoisingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(GenerativeDenoisingModel, self).__init__()

        # 1. 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. 动态去噪模块
        self.row_transform = nn.Linear(embed_dim, hidden_dim)
        self.dim_transform = nn.Linear(hidden_dim, hidden_dim)

        self.denoise_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # 3. 输出层
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, mask=None):
        # 1. 嵌入层
        embedded_seq = self.embedding(input_seq)  # [batch_size, seq_len, embed_dim]

        # 2. 隐空间初始化
        hidden_space = self.row_transform(embedded_seq)  # [batch_size, seq_len, hidden_dim]
        hidden_space = self.dim_transform(hidden_space)  # [batch_size, seq_len, hidden_dim]

        # 3. 动态去噪
        for denoise_layer in self.denoise_layers:
            signal = denoise_layer(hidden_space)  # [batch_size, seq_len, hidden_dim]
            hidden_space = hidden_space - signal
            hidden_space = hidden_space + torch.relu(signal)

        # 4. 输出层
        logits = self.output_layer(hidden_space)  # [batch_size, seq_len, vocab_size]
        return logits

# =====================
# 模型保存与加载模块
# =====================

def save_model(model, optimizer, epoch, path="model.pth"):
    """
    保存模型参数
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, path)
    print(f"模型已保存至 {path}")

def load_model(model, optimizer, path="model.pth"):
    """
    加载模型参数
    """
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"已加载模型参数，继续从第 {start_epoch} 轮训练")
        return model, optimizer, start_epoch
    else:
        print(f"未找到模型文件 {path}，从头开始训练")
        return model, optimizer, 0

# =====================
# 训练模块
# =====================

def train_model(text, embed_dim=128, hidden_dim=256, num_layers=16, batch_size=32, num_epochs=50, lr=1e-3, model_path="model.pth", vocab_path="vocab.json"):
    """
    训练生成式去噪模型
    """
    # 加载或构建词汇表
    vocab, reverse_vocab = load_vocab(vocab_path)
    if vocab is None or reverse_vocab is None:
        print("词汇表无效或不存在，将重新生成...")
        vocab, reverse_vocab = build_vocab(text)
        save_vocab(vocab, vocab_path)
    
    vocab_size = len(vocab)

    # 构建数据集和数据加载器
    dataset = TextDataset(text, vocab)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, vocab))

    # 初始化模型
    model = GenerativeDenoisingModel(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

    # 加载模型参数（如果存在）
    model, optimizer, start_epoch = load_model(model, optimizer, model_path)
    
    # 记录损失
    losses = []

    # 训练循环（当num_epochs > 0时执行）
    if num_epochs > 0:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0

            for input_seq, target_seq in data_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                logits = model(input_seq)  # [batch_size, seq_len, vocab_size]
                logits = logits.view(-1, logits.size(-1))  # 展平 logits
                target_seq = target_seq.view(-1)          # 展平 target_seq

                loss = criterion(logits, target_seq)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # 每轮训练后保存模型
            save_model(model, optimizer, epoch, model_path)

        # 绘制损失曲线
        plt.plot(range(start_epoch + 1, num_epochs + 1), losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("training_loss.png")  # 保存图像而不是显示
        print("训练损失曲线已保存至 training_loss.png")

    return model, vocab, reverse_vocab

# =====================
# 推理模块
# =====================

def generate_text(model, start_text, max_len, vocab, reverse_vocab, temperature=1.0):
    """
    使用模型生成文本
    """
    model.eval()
    input_ids = torch.tensor([vocab.get(word, vocab["<UNK>"]) for word in start_text.split()], dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(input_ids)
            probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            if next_token == vocab["<PAD>"]:
                break

            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    generated_text = " ".join([reverse_vocab[idx] for idx in input_ids.squeeze().tolist()])
    return generated_text

def get_word_embedding(model, word, vocab):
    """
    获取特定词的嵌入向量
    """
    if word in vocab:
        word_id = vocab[word]
        with torch.no_grad():
            embedding_vector = model.embedding(torch.tensor([word_id], device=device))
        return embedding_vector.squeeze().cpu().numpy()
    else:
        return None

# =====================
# 主程序
# =====================

if __name__ == "__main__":
    print(f"使用设备: {device}")
    
    # 从外部文件加载文本数据
    file_path = "medium.txt"  # 替换为您的文本文件路径
    text = load_text_from_file(file_path)

    # 模型训练
    model, vocab, reverse_vocab = train_model(
        text, 
        num_epochs=0,
        batch_size=32, 
        model_path="model.pth", 
        vocab_path="vocab.json"
    )

    # 文本生成示例
    start_text = "This is"
    generated_text = generate_text(model, start_text, max_len=100, vocab=vocab, reverse_vocab=reverse_vocab)
    print("\n生成的文本:")
    print(generated_text)
    
    # 词向量获取示例
    test_word = "I"
    embedding = get_word_embedding(model, test_word, vocab)
    if embedding is not None:
        print(f"\n词 '{test_word}' 的嵌入向量 (前10维):")
        print(embedding[:10])
    else:
        print(f"\n词汇表中未找到 '{test_word}'")