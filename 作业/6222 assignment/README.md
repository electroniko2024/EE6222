# Assignment 1：降维与分类

署名：MENG XIANGYU；学号：G2607496E。

最终提交文件：`output/pdf/MENGXIANGYU-G2607496E.pdf`。英文正文为 5 页，随后附完整实验程序。

## 运行实验

使用 Python 3.12，在本目录运行：

```powershell
python -m pip install -r requirements.txt
python experiment.py
```

数据包含在 scikit-learn 的 `load_digits()` 中，运行时无需另行下载。数据为公开 UCI 手写数字数据的 1797 张图像子集，每张图像为 64 维。主实验比较原始特征、PCA、LDA，分别搭配 kNN 和 Ridge 线性分类器。

训练开发集占 80%，内部划分为约 60% 拟合、20% 验证；测试集占 20%。维度和分类器参数由验证集选择，随后在全部开发集重拟合。测试曲线不参与参数选择。

## 文件说明

- `experiment.py`：完整实验程序，也是 PDF 附录中的程序。
- `requirements.txt`：复现依赖版本。
- `report.md`：自动导出的英文正文预览，便于阅读与复制；重建 PDF 时会覆盖此预览。
- `build_report.py`：报告的正文源文件及 PDF 生成程序；需要修改正文时编辑这里的段落。
- `report_zh.md`：可编辑的中文说明正文，包含方法解释、结果解读及程序阅读指南。
- `build_chinese_report.py`：从 `report_zh.md` 生成中文说明 PDF，使用 Windows 的微软雅黑字体。
- `output/pdf/MENGXIANGYU-G2607496E-中文说明报告.pdf`：五页中文说明材料。
- `results/validation_grid.csv`：全部 220 个验证集配置的结果。
- `results/selections.json`：测试评价前保存的六组参数选择。
- `results/test_curves.csv`：全部预设维度的测试准确率和错误率。
- `results/results.json`：划分、软件版本、所选配置、置信区间和实现检查。
- `results/split_indices.npz`：可核查的训练、验证、测试索引。
- `results/digits_data.npz`：此次实验所用数据副本。
- `results/test_predictions.npz`：所选配置逐样本预测。
- `results/*_confusion.csv`：所选配置的混淆矩阵（行是真实类别，列是预测类别，0–9）。
- `results/accuracy_curves.png`、`results/pca_variance.png`：报告图表。

重建 PDF：

```powershell
python build_report.py
```

重建中文说明 PDF（Windows）：

```powershell
python build_chinese_report.py
```

## 主要结果

原始 1-NN 测试准确率为 99.17%；验证选择的 PCA 24 维 + 5-NN 为 98.61%，维度减少 62.5%；LDA 9 维 + 1-NN 为 95.56%。本实验观察到压缩价值，但没有观察到所选降维方案超过对应原始基线。报告没有根据测试集峰值重新选择模型，也不宣称差异显著。

## 提交

题目要求通过 NTULearn 提交 PDF，截止日期为 2026 年 10 月 25 日。文件已按姓名和学号命名；此工作仅生成本地材料，未向 NTULearn 提交。
