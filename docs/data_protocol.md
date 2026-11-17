# D01：CWRU 数据清单、划分与首轮处理约定

版本：v0.2｜来源核查：2026-10-21｜验收修订：2026-10-22（Asia/Hong_Kong）｜范围：D01

本轮确定 16 个候选记录，均使用驱动端加速度通道。三种故障统一取 0.007 inch 缺陷，外圈统一取 6 点位置。负载 0、1 HP 用于训练，2 HP 用于验证，3 HP 留作测试。首轮预处理参数已作合成信号检查，仍需 D02 用训练记录试跑。尚未生成样本库，也未训练分类模型。

机器可读清单见 [`data/records.csv`](../data/records.csv)，配置见 [`configs/data.yaml`](../configs/data.yaml)。16 个官方 MAT 文件已为本次元数据核查下载到 `data/raw/cwru/`，合计 70,056,712 bytes；D02 可检查校验值后复用。`records.csv` 是程序输入清单，保持 UTF-8 CSV，不另造 Excel 副本。

## 1. 官方来源与取值原则

| 编号 | 来源 | 本项目用途 |
| --- | --- | --- |
| S1 | [CWRU 下载说明](https://engineering.case.edu/bearingdatacenter/download-data-file) | DE、FE、BA、time、RPM 的变量含义。 |
| S2 | [Normal Baseline Data](https://engineering.case.edu/bearingdatacenter/normal-baseline-data) | 四条正常记录、下载链接、负载和近似转速。 |
| S3 | [12k Drive End Bearing Fault Data](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) | 三类故障的 12 kHz 系列、0.007 inch、负载及外圈钟点位置。 |
| S4 | [Apparatus & Procedures](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures) | 人工单点缺陷、试验台和传感器布置。 |
| S5 | [Fault Specifications](https://engineering.case.edu/bearingdatacenter/fault-specifications) | 选定故障直径 0.007 inch、深度 0.011 inch、SKF。 |
| S6 | [Bearing Information](https://engineering.case.edu/bearingdatacenter/bearing-information) | 驱动端轴承型号 6205-2RS JEM SKF。 |
| S7 | [Radtke & Bock, 2022](https://doi.org/10.36001/phme.2022.v7i1.3329)，[PDF](https://papers.phmsociety.org/index.php/phme/article/download/3329/1947) | §4.1.1、§4.2，PDF 第 6–7 页／论文集 426–427 页：其研究把正常基线按 48 kHz、故障按 12 kHz 处理，并降采样正常基线。仅用来补充采样率依据。 |
| S8 | [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html) | 多相 FIR 抗混叠降采样。 |
| S9 | [PyWavelets CWT](https://pywavelets.readthedocs.io/en/latest/ref/cwt.html) | Morlet 小波、尺度与频率换算；小尺度可能出现混叠。实际调用还核对了本机函数签名。 |

S1–S6 的网页快照及 SHA-256 见 `runs/d01/sources/` 和 `runs/d01/source_audit.json`。每条文件的来源 URL、变量、长度、校验值和原始标量均可在审计 JSON 中复查。网上的镜像命名或其他项目的标签不覆盖官方表及本次 MAT 检查。

**正常采样率的证据边界：** S2 没有逐条注明采样率，本次 MAT 变量中也没有明确的采样率或时间戳字段；`*_time` 是采样值数组。采用 48,000 Hz 是依据 S7 作出的明确项目选择，`fs_status` 标记为 `literature_supported_not_official_per_file`，不能改写成“官方逐文件确认 48 kHz”。样本数量本身不能证明采样率。D02 沿用此值并在运行记录保留依据；若取得相反的直接时间基准证据，先修订版本、再重新生成全部派生产物，不按分类得分选择采样率。故障的 12,000 Hz 依据 S3 的数据系列标题。

**外圈位置冲突：** S3 表头把 6 点位置列为载荷区中心，S4 正文却把 3 点位置描述为直接受载。清单保留无歧义的 `6:00` 与链接名 `OR007@6`，另记文字冲突；不要据此进一步解释受力方向或优劣。只选一种外圈位置可减少首轮条件差异。

## 2. 四类记录及工况

下列编号就是下载文件名，通道必须逐字匹配。正常、内圈、外圈、滚动体分别映射为 `normal`、`inner_race`、`outer_race`、`rolling_element`。缺陷所在轴承和加速度计所在测点是不同概念：本轮只取驱动端故障及 DE 测点，正常记录也取 DE。

| 类别 | 官方记录名 | 文件 | 选定通道 | HP | 表中近似 RPM | MAT RPM | 原始采样率 Hz | 原始点数 | 划分 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| normal | Normal_0 | 97.mat | X097_DE_time | 0 | 1797 | 1796 | 48000¹ | 243938 | train |
| normal | Normal_1 | 98.mat | X098_DE_time | 1 | 1772 | 缺失 | 48000¹ | 483903 | train |
| normal | Normal_2 | 99.mat | X099_DE_time | 2 | 1750 | 缺失 | 48000¹ | 485063 | validation |
| normal | Normal_3 | 100.mat | X100_DE_time | 3 | 1730 | 1725 | 48000¹ | 485643 | test |
| inner_race | IR007_0 | 105.mat | X105_DE_time | 0 | 1797 | 1797 | 12000 | 121265 | train |
| inner_race | IR007_1 | 106.mat | X106_DE_time | 1 | 1772 | 1772 | 12000 | 121991 | train |
| inner_race | IR007_2 | 107.mat | X107_DE_time | 2 | 1750 | 1748 | 12000 | 122136 | validation |
| inner_race | IR007_3 | 108.mat | X108_DE_time | 3 | 1730 | 1721 | 12000 | 122917 | test |
| outer_race | OR007@6_0 | 130.mat | X130_DE_time | 0 | 1797 | 1796 | 12000 | 121991 | train |
| outer_race | OR007@6_1 | 131.mat | X131_DE_time | 1 | 1772 | 1773 | 12000 | 122426 | train |
| outer_race | OR007@6_2 | 132.mat | X132_DE_time | 2 | 1750 | 1750 | 12000 | 121410 | validation |
| outer_race | OR007@6_3 | 133.mat | X133_DE_time | 3 | 1730 | 1725 | 12000 | 122571 | test |
| rolling_element | B007_0 | 118.mat | X118_DE_time | 0 | 1797 | 1796 | 12000 | 122571 | train |
| rolling_element | B007_1 | 119.mat | X119_DE_time | 1 | 1772 | 1772 | 12000 | 121410 | train |
| rolling_element | B007_2 | 120.mat | X120_DE_time | 2 | 1750 | 1748 | 12000 | 121556 | validation |
| rolling_element | B007_3 | 121.mat | X121_DE_time | 3 | 1730 | 1722 | 12000 | 121556 | test |

¹ 正常采样率依据及限制见上节。所有记录取表中列出的单一通道，其余信号不自动加入样本。

故障直径 0.007 inch = 0.1778 mm，深度 0.011 inch = 0.2794 mm。正常的 `fault_size=0` 是“无人工缺陷”的编码；正常的缺陷深度留空表示不适用。`rpm` 优先取匹配编号的 MAT RPM，缺失时使用官方表的近似值并标记 `rpm_basis`；`rpm_mat` 留空，保留缺失原因。`99.mat` 的 `ans` 没有被解释为 RPM。

**99.mat 的特殊处理已核实：** 文件同时带有 X098、X099 的 DE/FE 数组。其中 X098 数组与 `98.mat` 完全相同。读入时只允许 `X099_DE_time`；不得遍历全部 `_DE_time` 当作额外样本，也不得按排序取第一个。两个重复数组的比较结果及哈希保存在审计文件中。16 个选定 DE 数组哈希互不相同，这只排除了整段完全相同，不是统计独立或试件独立的证明。

不纳入首轮：其他故障尺寸、外圈 3／12 点、FE 故障系列、48 kHz 故障系列、FE／BA 观测通道。它们属于后续扩展候选，不能为增加图片数量随意混入本轮数据。

## 3. 来源组与训练／验证／测试

| 数据集 | 负载 | 记录数 | 每类记录数 | 用途 |
| --- | --- | ---: | ---: | --- |
| train | 0、1 HP | 8 | 2 | 试跑、图像颜色范围拟合、特征 Scaler 拟合和 SVM 训练。 |
| validation | 2 HP | 4 | 1 | 首轮训练流程固定后调参，保留每次选择记录。 |
| test | 3 HP | 4 | 1 | D03 固定参数、指标和模型版本后最终评估。 |

先固定清单中的 `group_id` 和 `split`，再切片。`group_id=cwru_文件编号` 表示该记录的选定原始信号；其所有片段、复核图、不同处理版本和未来同源通道均继承该分组。禁止片段随机打散后按百分比分割；禁止相同信号以副本文件名跨组。由于 `99.mat` 是混合容器，来源识别以明确选定的 X099 信号为准，X098 副本始终排除。

`source_family_id` 把相同部位、尺寸和位置的不同负载记录关联起来。公开页面没有给出可核实的逐记录物理试件编号，所以 `physical_specimen_id` 留空、`specimen_status` 说明原因。同一来源族可能来自同一轴承，在三个数据集中均存在；本方案不宣称试件隔离。若今后得到试件映射，需检查能否进行试件级划分；若四类无法同时覆盖三个集合，就明确报告无法开展该评价，不能给现有分组改名为试件级划分。

这个小范围方案用于**所选实验来源中从 0／1 HP 到保留 2／3 HP 工况的迁移**，包含新记录和负载／转速变化，不能单独分离负载效应与记录效应。每类测试仅一条原始记录，片段再多也不能当作多个独立轴承；不据此推广到其他故障尺寸、位置或航空轴承。

划分在看到分类得分之前确定；D01 只读全体记录的键、长度、有限性、标量元数据与哈希，没有使用验证／测试的图样或预测结果调参。D02 的视觉试跑仅用训练记录。主方案不把验证重新并入训练；以后如需重新拟合，先登记新版本并保持测试封存。

后续分类报告至少保留混淆矩阵、各类召回率、macro-F1、balanced accuracy 和逐记录结果，指标以 D03 冻结稿为准。不得把相关片段当成独立试件计算置信区间，也不靠调换测试负载来挑好看的分数。

## 4. 首轮预处理配置

配置状态为 `candidate_for_D02_training_only_pilot`。下面的参数是本项目首轮选择，不是 CWRU 官方推荐，也不代表已验证分类效果。

1. **校验与取信号：** 按文件 SHA-256 和完整变量名读入，转为一维 float64；异常值、空数组、常量窗口报错并登记。保留原始文件，不改写 MAT。
2. **统一采样率：** 正常 48k→12k，先对整条选定记录执行 `scipy.signal.resample_poly(up=1, down=4, window=('kaiser',5.0), padtype='line')`；故障维持 12k。该函数先低通再抽取，不能使用 `x[::4]` 代替抗混叠。不同原始采集链和重采样步骤仍可能留下类别相关特征，共同采样率不等于完全消除采集差异。
3. **窗口：** 每条处理后的信号两端各排除 128 点；窗口 2048 点，步长 2048 点，首轮无重叠、无增强，不足一窗的尾部舍弃。每窗约 0.1707 秒，相当于约 5 次转动。`start_sample`／`end_sample` 使用**完整重采样信号**坐标，零起点、左闭右开，第一窗 `[128,2176)`；不在去边后的数组上重新编号。原始对应区间由重采样比另算，FIR 影响范围不等同于此名义索引区间。
4. **幅值：** 每窗减均值，不逐窗做单位方差归一化或 min-max，保留幅值差异。真实信号的单位尚未由本次页面确认，CWT 颜色不标作物理损伤量或标定 PSD。
5. **CWT：** `cmor1.5-1.0`，FFT 方法，`precision=12`；100–3000 Hz，128 个对数等距频点。用 `pywt.frequency2scale(wavelet, frequencies/12000)` 换算，尺度为 120 到 4，避免最小尺度 1／2 的明显混叠问题。每窗反射填充左右各 1024 点，变换后裁回 2048 点；填充只来自本窗，不能从其他数据集借上下文。这减轻边界影响，不保证边缘系数完全无偏。频段是试跑起点，可能舍去较高频共振信息；改动只能根据训练／验证依据登记新版本。
6. **颜色与图像：** 计算 `10*log10(max(abs(coef)**2,1e-12))`，不额外乘尺度系数。训练平衡后全部 CWT 矩阵像素的 1%／99% 分位数作为公共颜色上下限，分位数计算用 `method='linear'`，保存到 `configs/cwt_color_limits.json`。此文件由 D02 拟合后生成，当前不存在。验证／测试完全复用，不逐图自动拉伸。先将标量矩阵用 Pillow F 模式 BILINEAR 缩到 224×224，再裁剪到公共范围并应用 viridis、输出 RGB PNG；最高频率在上，时间向右。模型图没有坐标、类别、文件名和边框。复核图使用相同数值及颜色范围，增加时间、频率轴和色标。
7. **训练记录平衡：** 所有训练记录各取相同窗数，数量为训练记录可用窗数的最小值；用 `floor(linspace(0,n-1,m))` 均匀选取，保留选择索引。正常 0 HP 记录较短，不用重复窗口补齐。验证／测试保留全部有效完整窗口。该规则不从验证／测试的数量或幅值学习参数。
8. **后续输入接口：** ResNet-18 预训练权重候选标识固定为 `IMAGENET1K_V1`，224×224 图直接转 `[0,1]` 后按 ImageNet mean/std 标准化，不再调用会裁剪图片的默认变换。特征 Scaler 只在训练集拟合；SVM 参数只在验证集选择。权重尚未下载，D01 没有运行这一分类流程。

按当前长度和配置计算，训练平衡后预计每条训练记录 29 窗。详细预计窗数见 `runs/d01/validation.json`；这些是长度计算，不是已生成的图片数量。增加重叠率不能解决只有 16 条记录或试件身份未知的问题。

## 5. 字段、证据与验收

`records.csv` 中的 `source_url` 是 MAT 直链，`metadata_url` 是官网表；`sha256` 对原文件字节计算。`signal_sha256_float64` 对选定信号按 float64 little-endian 展平后的字节计算，用于识别等值副本。`channel_status` 表示已实际读入，不是猜测键名。

`fault_size` 的单位由 `fault_size_unit=inch_diameter` 指定；外圈钟点和载荷区关系分列保存。`ignored_time_keys` 列出不参与建样本的其他通道／混入数组。空字符串表示未知或不适用，解释分别见 `rpm_missing_reason`、`specimen_status` 及本文，不能把缺失物理试件编号填成独立文件编号。

验证与复查：

```bash
conda activate aae5103
python -m pip check
python scripts/validate_d01.py
python -m unittest discover -s tests -v
```

`validate_d01.py` 先对照 `runs/d01_fix/deliverables.sha256` 检查冻结文件，再逐记录核对来源身份、标签、负载、尺寸、位置、采样率和精确通道，检查信号、分组与重复数组。采样率、重采样比例、CWT 采样间隔、边界及窗口索引必须一致。合成信号检查使用 `scripts/data_contract.py` 的同一配置解析与重采样函数；D02 实际处理应复用该函数。

每次结果写入独立的 `runs/d01_checks/<时间戳与标识>/validation.json`；可用 `--output` 指定尚不存在的文件，已有文件及 `runs/d01/` 目录禁止写入。快照检查与内容检查分别报告；快照不符时停止内容检查并返回非零退出码。不生成正式样本、不训练、不计算测试分数。

原 `runs/d01/deliverables.sha256`、`validation.json` 和 `validation.log` 保留。修复前 34 项校验全部通过；有意变化及旧／新哈希见 `runs/d01_fix/changes.json`。当前修复验收见 `runs/d01_fix/validation.json`，回归检查见 `runs/d01_fix/regression.log`。后续有意修改已冻结文件时，先保留旧版、记录原因及差异，再建立新阶段清单，通过 `--baseline <新清单>` 验收；不能为消除失败直接覆盖旧基线。详见 [D01_修复说明.md](D01_修复说明.md)。

`audit_d01_sources.py` 是本轮官方网页与 MAT 元数据收集脚本，`build_d01_manifest.py` 从审计快照生成当前清单。二者用于 D01 溯源；进入 D02 后不要无意重建并覆盖已补充的清单。审计脚本复用现有下载时会显式记录缓存来源；如需重新核查远端，应另建快照再比较，避免覆盖已冻结校验值。

本机复用了 `aae5103`／Python 3.12.15，没有安装或改动依赖。依赖检查通过。受限执行上下文中 `torch.backends.mps.is_available()` 为 false，联网审计所用的非受限上下文为 true，说明本次可见性依执行上下文变化；不能据此认定硬件不支持 MPS。D01 数值检查均在 CPU 上完成。完整版本见 `source_audit.json`，离线验收上下文见 `validation.json`。

## 6. D02 的明确起点

1. 修改前先运行离线验收及回归检查，确认与修订 003 的 `runs/d01_fix/deliverables.sha256` 相符；复用原始文件。后续改动按第 5 节登记新版本，不覆盖 D01 证据。
2. 按固定记录分组实现读取、抗混叠、索引与切片。先只取训练中的 `97、105、130、118` 各一窗，生成可复核的时频图，检查频率轴、方向、颜色与确定性；公共颜色范围仍需按第 4 节从完整平衡训练部分拟合，不能只用这四窗冻结。
3. 保存 `samples.csv`、颜色范围、处理版本、实际窗数、剔除原因和复现命令；用同一片段重复生成比较数组／图片结果。真实数据的质量检查可能减少预计窗数，发生时记录原因再按训练规则更新，不动负载划分。
4. 确认训练处理流程后再批量生成固定配置的验证／测试产物。测试允许机械处理和完整性检查，但不人工选图调参数、不计算分类得分；正式测试留至 D03 冻结配置。

D01 的完成指清单、划分、候选配置与依据齐备；D02 仍待开始，未把元数据核查下载等同于样本库或分类实验完成。
