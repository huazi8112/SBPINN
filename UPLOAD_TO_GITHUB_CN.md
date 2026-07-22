# 上传到现有 GitHub 仓库

论文中使用的仓库地址为：

```text
https://github.com/huazi8112/SBPINN
```

推荐使用 Git 命令更新现有仓库：

```powershell
# 1. 克隆现有仓库
git clone https://github.com/huazi8112/SBPINN.git
cd SBPINN

# 2. 保留 .git 文件夹，删除仓库中的旧代码文件
#    然后将本压缩包中 SBPINN 文件夹内的全部内容复制到这里

# 3. 检查并提交
git status
git add .
git commit -m "Update code repository to match the revised manuscript"
git push origin main
```

注意：不要把压缩包本身、`.idea`、`__pycache__`、模型检查点、`.npz` 稠密结果或本地运行产生的 `results_*` 文件夹提交到 GitHub。仓库中的 `.gitignore` 已配置为忽略这些内容。
