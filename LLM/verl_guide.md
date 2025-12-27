# verl 仓库管理指南

## 仓库结构

- **origin**: `https://github.com/lu-jun-yu/verl.git`
- **upstream**: `https://github.com/volcengine/verl.git`

## 常用操作

### 1. 拉取原仓库更新

```bash
cd LLM/verl
git fetch upstream
git merge upstream/main
# 或者使用 rebase
git rebase upstream/main
```

### 2. 推送你的修改到 fork

```bash
cd LLM/verl
git add .
git commit -m "你的提交信息"
git push origin main
```

### 3. 更新主仓库中的 submodule 引用

在 verl 中提交修改后，回到主仓库更新 submodule 引用：

```bash
cd ../..  # 回到 OPTS 根目录
git add LLM/verl
git commit -m "update verl code"
git push
```

### 4. 克隆主仓库后初始化 submodule

```bash
git clone https://github.com/你的用户名/OPTS.git
cd OPTS
git submodule init
git submodule update
```

或一步完成：

```bash
git clone --recurse-submodules https://github.com/你的用户名/OPTS.git
```
