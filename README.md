# Analyze_Star 🪐
学生作业练习：基于随机森林的天体分类项目

## 版本说明
- `cpu` 分支：纯 CPU 运行版本（无 GPU 环境可用）
- `gpu` 分支：GPU 加速版本（依赖 RAPIDS 库，训练更快）
- `main` 分支：项目说明与基础配置

---
## 运行步骤
1.  切换到对应版本分支
2.  配置环境
    conda env create -f environment.yml
    conda activate <你的环境名>
3.  运行代码
    # CPU 版本
    python Project_CPU.py
    # GPU 版本
    python Project_GPU.py
###瞎写的，别深究:)