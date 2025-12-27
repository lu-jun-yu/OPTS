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
git clone https://github.com/lu-jun-yu/OPTS.git
cd OPTS
git submodule init
git submodule update
```

或一步完成：

```bash
git clone --recurse-submodules https://github.com/lu-jun-yu/OPTS.git
```

### 5. 拉取主仓库时同步更新 submodule

```bash
# 方法1：拉取时自动更新 submodule
git pull --recurse-submodules

# 方法2：先拉取主仓库，再手动更新 submodule
git pull
git submodule update

# 方法3：设置为默认行为（推荐，只需设置一次）
git config --global submodule.recurse true
# 之后直接 git pull 就会自动更新 submodule
```

### 6. 更新 submodule 到最新 commit

```bash
# 进入 submodule 拉取最新代码
cd LLM/verl
git pull origin main

# 回到主仓库提交 submodule 引用更新
cd ../..
git add LLM/verl
git commit -m "update verl submodule"
git push
```
