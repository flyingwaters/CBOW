# @ 2019 12.2
# @ 冯龙宇
# @ 深入理解CBOW
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--save_path', required=True)
arg = ap.parse_args()
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right 滑动窗口
raw_text = "我 们 是 共 产 主 义 接 班 人 ， 爱 祖 国 ，爱 人 民 ， 爱 社 会 主 义".split(' ')

vocab = set(raw_text)
# vocab
word_to_idx = {word: i for i, word in enumerate(vocab)}
# 生成字到数字的字典
data = []
# 真实应该是从第一个词作扫描 但是有一些特殊情况要处理不方便编码
for i in range(CONTEXT_SIZE, len(raw_text)-CONTEXT_SIZE):
    context = [raw_text[i-CONTEXT_SIZE], raw_text[i-CONTEXT_SIZE+1], raw_text[i+CONTEXT_SIZE-1], raw_text[i+CONTEXT_SIZE]]
    # 上下文
    target = raw_text[i]
    # 中心词
    data.append((context, target))
    # 输入输出对

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        # 随机矩阵 几个词 * n_dim
        # 从n_words字符串作为输入， 然后
        # 2*context_size 个单词* n_dim
        self.linear1 = nn.Linear(2*context_size*n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(1, -1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        # 省去反复申请存储空间的过程
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x


model = CBOW(len(word_to_idx), 100, CONTEXT_SIZE)
# 初始化一个类
if torch.cuda.is_available():
    model = model.cuda()
    # 一个类调用cuda()函数
criterion = nn.CrossEntropyLoss()
    # 一个CrossEntroptLoss() 类
optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # 一个SGD优化器的类
for epoch in range(10000):
    print('epoch {}'.format(epoch))
    print('*'*10)
    running_loss = 0
    for word in data:
        context, target = word
        context = Variable(torch.LongTensor([word_to_idx[i] for i in context]))
        target = Variable(torch.LongTensor([word_to_idx[target]]))

        if torch.cuda.is_available():
            context = context.cuda()
            target = target.cuda()
        # forward
        out = model(context)
        loss = criterion(out, target)
        running_loss += loss.detach().item()
        # 使用detach()[0]
        # backward
        optimizer.zero_grad()
        # 所有loss反向得梯度为0
        loss.backward()
        optimizer.step()
        # 更新所有参数
    print('loss: {:.6f}'.format(running_loss / len(data)))
    if running_loss / len(data) < 0.01:
        break

torch.save(model.embedding.state_dict(), arg.save_path)
