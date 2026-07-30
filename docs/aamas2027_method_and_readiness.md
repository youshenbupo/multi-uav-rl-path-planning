# AAMAS 2027 方法建模、实现映射与就绪度

**状态：** 研究记录，不是论文结果表。
**截至：** 2026-07-30。
**主线：** 通信陈旧度、时延和丢包 → 邻机状态预测不确定性 → 不确定性感知预测交互图与协同行动 → 不确定性自适应 CBF 安全裕度 → 动态障碍环境中的安全执行。

本文以当前实现为准，区分已核验的代码/原始产物与尚未证明的论文结论。

## 1. 任务、动力学与信息边界

对 (N) 架 UAV，令位置、速度、目标和归一化策略动作分别为

$$
p_i^t\in\mathbb R^3,\quad v_i^t\in\mathbb R^3,\quad g_i\in\mathbb R^3,\quad a_i^t\in[-1,1]^3.
$$

环境将动作映射为请求速度 (u_{i,\mathrm{RL}}^t)。执行侧实际采用 CBF 输出 (u_{i,\mathrm{safe}}^t)，并以三维单积分器更新：

$$
p_i^{t+1}=\operatorname{clip}_{\mathcal W}\!\left(p_i^t+\Delta t\,u_{i,\mathrm{safe}}^t\right),\qquad
v_i^{t+1}=\frac{p_i^{t+1}-p_i^t}{\Delta t}.
$$

Actor 的信息集严格受限：

$$
\mathcal I_i^t=\{p_i^t,v_i^t,g_i,\text{局部地形/威胁/动态障碍},\mathcal K_i^t\},
$$

其中 (​\mathcal K_i^t) 仅含已经送达的邻机包。Actor 不得读 (p_j^t,v_j^t;(j\ne i)) 的当前真值。集中式 Critic 可以使用联合真值状态

$$
s^t=\{p_{1:N}^t,v_{1:N}^t,g_{1:N},\text{active mask},\text{terrain},\text{threats},\text{dynamic obstacles}\}.
$$

**例子。** 三机系统中，UAV 1 在 (t=8) 只能看自己的真值和 UAV 2 在 (t=6) 发出、后来送达的包；仿真器即使知道 (p_2^8)，也不能把它给 Actor 1。Critic 可以用 (p_2^8) 训练 (V(s^8))。

**实现：** `multiuav/envs/communication.py`、`observations.py`、`learning/conflict_graph.py`、`graph_networks.py` 和 `graph_runner.py`。

## 2. 延迟、丢包、陈旧度与状态预测

UAV (j) 在源时刻 (s) 发送

$$
z_{j\to i}^{s}=(p_j^s,v_j^s,s).
$$

通信在范围内且双方活跃时，经过固定整数延迟 (d) 送达；每个包独立以概率 (q) 丢失：

$$
B_{j\to i}^{s}\sim\operatorname{Bernoulli}(1-q),\qquad t_{\mathrm{deliver}}=s+d.
$$

接收方只保留最新已送达包。当前年龄为

$$
\tau_{ij}^t=t-s,
$$

且只有 (0\le\tau_{ij}^t\le\tau_{\max}) 时该包有效；过期时是不可见，而非真值回填。当前动态协议设置 (d=1)、(q=0.1)、(​\tau_{\max}=3)。

对有效包，Actor 用恒速外推：

$$
\hat p_{j\mid i}^t=p_j^s+\tau_{ij}^t\Delta t\,v_j^s,
\qquad
\sigma_{j\mid i}^t=\eta\tau_{ij}^t.
$$

这里 (​\eta=​\texttt{communication\_uncertainty\_growth\_per\_step})，当前为 1.0。

**例子。** (p_2^6=(10,0,20))、(v_2^6=(2,0,0))、(​\Delta t=1)。在 (t=8)，UAV 1 得到

$$
\hat p_{2\mid1}^{8}=(14,0,20),\qquad \sigma_{2\mid1}^{8}=2.
$$

这只是预测；不是 UAV 2 的当前真值。若到 (t=10) 仍无新包，年龄 4 超出窗口，边必须消失。

## 3. 不确定性感知 CPA 预测交互图

对接收方 (i) 与其可见邻机 (j)，定义

$$
r_{ij}^t=\hat p_{j\mid i}^t-p_i^t,\qquad w_{ij}^t=v_j^s-v_i^t.
$$

在预测时域 (H) 内，closest point of approach (CPA) 为

$$
t_{ij}^{\mathrm{CPA}}=\operatorname{clip}\left(-\frac{r_{ij}^{t\top}w_{ij}^t}{\lVert w_{ij}^t\rVert_2^2+\varepsilon},0,H\right),
$$

$$
d_{ij}^{\mathrm{CPA}}=\left\lVert r_{ij}^t+t_{ij}^{\mathrm{CPA}}w_{ij}^t\right\rVert_2.
$$

当前 (H=5)、风险距离 (d_{\mathrm{risk}}=12)。完整不确定性图使用保守 CPA 距离和风险短缺：

$$
\tilde d_{ij}^{\mathrm{CPA}}=\max\left(0,d_{ij}^{\mathrm{CPA}}-\rho_g\sigma_{j\mid i}^t\right),
\qquad
c_{ij}=\frac{\max(0,d_{\mathrm{risk}}-\tilde d_{ij}^{\mathrm{CPA}})}{d_{\mathrm{risk}}}.
$$

当前 (​\rho_g=​\texttt{graph\_uncertainty\_risk\_gain}=1.0)。边特征还含归一化相对位置/速度、CPA 时间/距离、年龄、valid 标记和不确定性；邻接由通信半径、预测风险和 top-​(k) 近邻共同决定。

**例子。** 若名义 (d^{\mathrm{CPA}}=13)、(​\sigma=2)、(​\rho_g=1)，则 (​\tilde d=11<12)，并有

$$
c=\frac{12-11}{12}\approx0.083.
$$

陈旧包因而会提升而非掩盖交互风险。

### 三个 GraphMAPPO 臂

| 臂 | `graph_mode` | 已送达包预测 | 不确定性特征 |
| --- | --- | --- | --- |
| Raw graph | `mappo` | 否 | 否 |
| Predictive graph | `predictive_graph` | 是 | 否 |
| 完整主方法 | `uncertainty_predictive_graph` | 是 | 是 |

所有臂必须匹配环境、通信、动态障碍、CBF 和独立五种子预算；不得用替代控制器伪造基线。

## 4. 去中心化 Actor、集中式 Critic 与 MAPPO

Graph Actor 的邻居消息由可见边特征决定。post-isolation 结构禁止把邻机私有局部节点隐藏向量作为 Actor 的 key/value。可抽象为

$$
\alpha_{ij}^{(h)}=\operatorname{softmax}_{j\in\mathcal N_i}(W_{e,h}e_{ij}),
\qquad
m_i=\sum_{j\in\mathcal N_i}\alpha_{ij}^{(h)}V_e e_{ij},
$$

$$
h_i^{\ell+1}=\operatorname{Norm}\bigl(h_i^\ell+\phi(m_i)\bigr),
\qquad
a_i=\tanh(\tilde a_i),\quad
\tilde a_i\sim\mathcal N(\mu_\theta(h_i),\operatorname{diag}(\sigma_\theta^2)).
$$

Critic 对联合真值状态/图作掩码聚合，输出 (V_\phi(s^t))，但不向 Actor 回传邻机真值。

**例子。** 改变 UAV 2 当前私有观测而保持其已送达包不变时，UAV 1 的 Actor 输出应该不变；对应不泄漏测试已加入。UAV 1 只能因“包年龄、预测位置和不确定性”变化而改变动作。

GAE 与 clipped PPO 为

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),\qquad
A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},
$$

$$
L_{\mathrm{PPO}}=\mathbb E_t\!\left[\min\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right],
$$

其中 (r_t=\pi_\theta(a_t\mid\mathcal I_t)/\pi_{\theta_{\mathrm{old}}}(a_t\mid\mathcal I_t))。图训练还含未来冲突和最小距离辅助头，它们不改变 Actor 信息边界。

**例子。** 若 (r_t=1.30)、(​\epsilon=0.2)、优势为正，PPO 目标使用 1.20 而不是 1.30 的增益，限制过大单步策略更新。

## 5. 不确定性自适应 CBF-QP

CBF 是执行侧联合安全层，运行在 CPU OSQP，不通过 PPO 反向传播。对 (u_{\mathrm{RL}}\in\mathbb R^{3N})，求解

$$
\begin{aligned}
\min_{u,\xi}\quad & \frac12\lVert u-u_{\mathrm{RL}}\rVert_2^2+\frac{\lambda_\xi}{2}\lVert\xi\rVert_2^2\\
\text{s.t.}\quad & \nabla h_k(p)^\top u+\xi_k\ge-\alpha h_k(p),\\
&\xi_k\ge0,\qquad u\in\mathcal U_{\mathrm{speed}}.
\end{aligned}
$$

当前值为 (​\lambda_\xi=100)、最大 20,000 iterations、0.1 s solve limit；水平速度球采用 16 边凸多边形近似。

### UAV 隔离裕度

$$
h_{ij}=\lVert p_i-p_j\rVert_2^2-\left(d_{\mathrm{safe}}+m_{ij}\right)^2,
$$

$$
m_{ij}=\min\left(m_{\max},\kappa\max(\sigma_{j\mid i},\sigma_{i\mid j})\right).
$$

当前 (​\kappa=0.5)、(m_{\max}=5.0)。只用双方已知的不确定性；没有有效包时不制造额外观测。

**例子。** 基础隔离 10 m，最大可用不确定性为 2 m，则 (m=1) m，有效隔离变为 11 m。这个裕度收紧仅作用于执行安全层。

### 其他 barrier

地形、静态圆柱威胁、动态圆柱和世界边界均形成线性 CBF 行。对动态障碍中心 (c_o^t=c_o^0+t\Delta t v_o)，令 (r_{io}=p_{i,xy}-c_{o,xy}^t)：

$$
h_{io}=\lVert r_{io}\rVert_2^2-(r_o+m_{\mathrm{threat}})^2,
$$

$$
2r_{io}^\top u_{i,xy}\ge-\alpha h_{io}+2r_{io}^\top v_{o,xy}.
$$

**例子。** 移动圆柱向 (+y) 穿过飞行区时，朝向它的请求速度会违反上式；QP 修正该速度。若 OSQP 非 `solved`、超时、非有限或异常，系统执行有界 hover/repulsion emergency policy，而不是放行 RL 动作。

## 6. 一步端到端例子

1. UAV 1 在 (t=8) 拥有自身真值和 UAV 2 的 (t=6) 包；UAV 3 包过期。
2. UAV 1 得到 (​\hat p_{2\mid1}^8)、年龄 2、不确定性 2；UAV 3 边无效。
3. CPA 图若因保守距离小于 12 m 则保留 UAV 1–2 风险边。
4. Actor 从自身节点和已送达边生成 (u_{1,\mathrm{RL}})，不读 UAV 2 当前真值。
5. 联合 CBF-QP 用真值物理几何执行安全约束，并将通信不确定性转为 1 m 隔离增量。
6. `solved` 时执行 QP 解；否则记录完整 context 并用 emergency policy。

该例说明数据流和约束语义，不证明成功率或优越性。

## 7. 已核验工作与结果边界

### 已核验

- Actor 信息边界修复完成：无通信不以邻机真值填充；已送达包不会被发送方当前活动状态擦除；Graph Actor 对邻机私有特征/当前活动状态不敏感。
- targeted pytest 曾通过 39 项，Ruff/mypy 已通过；完整 pytest 保留一个独立 MATLAB legacy-inventory 失败，见 `docs/known_issues.md`。
- post-isolation MLP：五个有效 3-UAV CUDA 训练，五个最终检查点的六场景 × 20 episode 评估，共 30 格、600 条原始 JSONL；全源 CBF 回放保留 27 事件与一个 OOD 动态障碍不可重放错误。
- raw GraphMAPPO (`mappo`)：四个有效种子，20260719 的 `launcherretry1`、20260720、20260721、20260722；均为 100,032 transitions / 1,042 updates。初次 20260719 根在 10,848 transitions 中止，带 `ABORTED.json`，已排除。
- SSH 已恢复，远程为 `origin`；分支已推送至 `ccdd646`，此后文档提交仍须继续推送。

### 绝不能宣称

- 单种子、短评估、训练遥测或 CBF 回放不证明性能、安全性、回退率、显著性或基线优越性。
- 所有 actor-boundary 修复前的学习检查点、评估、汇总和 CBF 回放均协议不合格，必须保留但排除。
- 目前尚未投稿，也未获得向外部投稿系统提交的授权。

## 8. 论文准备

现在可以准备 methods 图、问题陈述、实验协议表、原始证据索引、无效尝试/回放排除表和已全文核验文献综述。投稿前仍需要：

1. raw graph seed 20260723 和该臂五种子六场景评估；
2. `predictive_graph` 与 `uncertainty_predictive_graph` 各五种子训练、统一评估和全源 CBF replay；
3. 5/8 UAV 独立训练和关键消融；
4. 匹配种子统计、置信区间、排除规则和多重比较处理；
5. 全文核验一手论文，更新 `docs/related_work_matrix.md`；
6. AAMAS 格式、复现材料、限制与伦理/数据/AI 使用声明。

当前证据支持严谨的方法与实验计划，不足以支持结果性论文主张或投稿就绪结论。

## 9. 实现索引

| 内容 | 文件 |
| --- | --- |
| 包、年龄、预测与不确定性 | `multiuav/envs/communication.py` |
| 局部 Actor / 集中 Critic 状态 | `multiuav/envs/observations.py` |
| 环境、动态障碍和通信调度 | `multiuav/envs/multi_uav_env.py`, `dynamic_world.py` |
| CPA 图 | `multiuav/learning/conflict_graph.py` |
| Graph Actor/Critic | `multiuav/learning/graph_networks.py` |
| GraphMAPPO 编排 | `multiuav/learning/graph_runner.py` |
| PPO/GAE | `multiuav/learning/mappo.py`, `graph_mappo.py` |
| barrier、QP 与回退 | `multiuav/safety/cbf_constraints.py`, `qp_filter.py`, `emergency_policy.py` |
