"""
Active sampling configuration definitions.

This module centralises hyper-parameters used by the validation-active
sampling pipeline so that examples and scripts share a single source of truth.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ActiveSamplingConfig:
    """Configuration for the validation-active sampling module."""

    # 论文默认使用冻结的 GDDN 生成先验。
    estimator_mode: str = "gddn"

    # Candidate generation
    candidate_pool_size: int = 64
    coarse_candidate_count: int = 32
    fine_candidate_count: int = 16
    fibonacci_start_index: int = 0
    sphere_radius: float = 3.0
    elevation_range: Tuple[float, float] = (20.0, 160.0)
    azimuth_range: Tuple[float, float] = (0.0, 360.0)
    min_view_cosine_distance: float = 0.01  # 排除角≈8°，与min_selection_angle_deg一致
    min_selection_angle_deg: float = 8.0
    # 覆盖锥约束 (Coverage-Aware Candidate Generation, CACG)
    # 限制候选视角与最近真实/已选视角的最大角度偏差（度）
    # 90°=排除纯背面, 60°=仅侧面扩展, 45°=紧凑间隙填充
    max_view_deviation_angle: float = 90.0

    # Active selection
    num_views_per_round: int = 4
    diversity_cosine_margin: float = 0.15
    ablation_preset: str = "full"
    ablation_random_selection: bool = False
    ablation_random_selection_seed: int = 0
    quality_variation_threshold: float = 0.03  # 收紧质量门控，过滤稳定但退化的候选
    quality_min_brightness: float = 0.03
    quality_min_contrast: float = 0.06
    selection_min_q_gen: float = 0.88
    selection_use_q_gen_gate: bool = False     # q_gen 当前未校准，默认不作为 Phase 2 质量硬门控
    selection_min_q_gain: float = 0.50
    selection_use_q_gain_gate: bool = False    # q_gain 为残差净收益预测器，默认先保留为可选门控
    selection_sort_by_verified_utility: bool = True
    selection_use_verified_utility_gate: bool = True
    selection_min_verified_utility: float = 0.10
    selection_min_probe_verified_coverage: float = 0.05

    # MC-Dropout uncertainty
    mc_dropout_samples_coarse: int = 2
    mc_dropout_samples_fine: int = 6
    dropout_rate: float = 0.15
    score_type: str = "variance"
    use_mask_for_scores: bool = True
    mask_score_min_coverage: float = 0.05
    score_fallback_variance_eps: float = 1e-8
    aleatoric_variance_floor: float = 1e-4
    use_mixed_precision: bool = False  # 在线阶段默认全局使用FP32，禁用autocast
    batch_size: int = 4  # 从8降至4（24G显存优化）
    estimator_force_single_candidate_generation: bool = True  # 强制在线估计走单候选链路，避免 batch>1 退回 fallback 破坏排序一致性
    primitive_dropout_p: float = 0.2
    opacity_scale_jitter: float = 0.15
    mc_dropout2d_p: float = 0.2
    mc_token_dropout_p: float = 0.08
    mc_cond_noise_sigma: float = 0.02
    mc_condition_alpha: float = 0.5
    mc_latent_noise_std: float = 0.005
    mc_sampler_type: str = "ddim"

    # Layer 2 截断MC-Dropout (频域解耦不确定性估计)
    # 原理：仅在DDIM前k步启用Dropout（高噪声=低频几何），后T-k步确定性精化
    # 效果：滤除高频纹理方差，提高Var_trunc作为几何缺失代理的Precision
    mc_truncation_enable: bool = True       # 是否启用截断MC-Dropout
    mc_truncation_k: int = 15               # 截断步数，占总步数30-40%（建议15-20）
    mc_fix_initial_noise: bool = True       # 固定初始噪声z_T，确保方差仅来自Dropout
    # 两层MC采样（Law of Total Variance，可选增强）
    # 仅在使用随机采样器(DDPM/DDIM with η>0)时需要；DDIM确定性模式下算法1已足够
    mc_two_layer_enable: bool = False       # 是否启用两层MC采样
    mc_two_layer_outer: int = 4             # 外层MC-Dropout采样次数（不同Dropout mask）
    mc_two_layer_inner: int = 2             # 内层噪声种子采样次数（不同z_T）

    # Diffusion sampling schedule (gddn mode, 24G显存优化)
    # Two-stage settings: coarse for fast screening, fine for precise selection
    coarse_resolution: int = 256  # 保持256
    coarse_steps: int = 100  # 增加以改善生成质量（从25提升）
    fine_resolution: int = 384  # 从512降至384（关键！）
    fine_steps: int = 50  # 降低到50以加速主动采样验证阶段
    ensemble_sampler_type: str = "ddim"
    ensemble_noise_scale: float = 0.0

    # Validation
    pearson_threshold: float = 0.2  # 相关性过低时回退到更保守的选择
    ece_bins: int = 10
    validation_use_mask: bool = True
    # Small-ensemble/Bootstrap repeats when true ensemble models are unavailable
    validation_bootstrap_repeats: int = 10  # Bootstrap重复次数（10次提高方差估计稳定性和Pearson信号质量）
    validation_candidate_batch_size: int = 1  # Bootstrap单次只评估一个候选，避免batch>1走fallback路径污染可靠性
    # Health gating for GDDN outputs
    health_min_std: float = 0.12
    health_min_mean: float = 0.08
    health_max_mean: float = 0.92

    # Training integration
    guidance_interval: int = 100
    warmup_iterations: int = 5000
    lambda_guidance: float = 0.1
    pseudo_loss_weight: float = 0.1
    phase3_real_view_probability: float = 0.7
    phase3_real_only_training: bool = True   # Phase 3仅以真实相机为训练样本，伪视图只作verified guidance
    log_selected_images: bool = False
    save_pseudo_png: bool = False
    save_gsplat_render: bool = True      # 保存gsplat渲染结果（用于对比）
    save_uncertainty_maps: bool = True   # 分别保存epistemic/aleatoric不确定性图
    save_comparison_png: bool = True     # 保存GDDN vs gsplat对比拼接图
    
    # 两阶段训练 - 阶段B纯致密化
    post_densify_iterations: int = 30000  # 30000轮充分致密化（与标准3DGS一致）

    # Dynamic Lambda Scheduling (动态λ调整 - 模块四)
    lambda_dynamic_enable: bool = True       # 是否启用动态λ调整
    lambda_min: float = 0.05                 # 最小λ值（阶段2起始、阶段3结束）
    lambda_max: float = 0.2                  # 最大λ值（阶段2峰值）
    lambda_peak_iteration: int = 9000        # 阶段2结束/阶段3开始的迭代点，从25000降至9000
    lambda_end_iteration: int = 10000        # 训练总迭代数，从30000降至10000

    # Robust NLL Loss Parameters (BayesGS-Diff Module II)
    nll_enable: bool = True                  # 是否启用鲁棒异方差NLL损失
    nll_huber_delta: float = 0.1             # Huber Loss拐点
    nll_variance_epsilon: float = 1e-3       # 方差下限，用于clamp
    nll_epistemic_threshold: float = 2.5     # 认知不确定性阈值，超过即视为OOD（从1.5提高到2.5，解决valid_mask全0问题）

    # Soft Gating Parameters (软门控机制模块三)
    gate_enable: bool = True              # 是否启用软门控机制
    gate_epsilon: float = 1e-3            # 数值稳定项 ϵ，防止除零
    gate_gamma: float = 1.0               # 数据不确定性敏感度 γ ∈ [0.5, 2.0]
    gate_beta: float = 10.0               # 认知不确定性sigmoid陡峭度 β ∈ [5, 20]
    gate_tau: float = 0.5                 # 认知不确定性阈值 τ，需根据验证集调整
    gate_strategy: str = "multiply"       # 组合策略: "multiply"(相乘), "add"(相加归一化), "geometric"(加权几何平均)
    gate_alpha: float = 0.5               # 几何平均权重参数 α ∈ [0, 1]，仅用于geometric策略
    gate_tau_auto: bool = False           # 是否自动估计τ
    gate_tau_percentile: float = 70.0     # 自动估计时使用的不确定性分位点
    gate_tau_ema_alpha: float = 0.1       # τ的EMA平滑系数
    gate_variance_floor: float = 3e-4     # 软门控的最小方差钳位，确保权重不至于完全失效

    # Adaptive Gating Parameters (自适应门控，模块三增强)
    adaptive_gating_enable: bool = True           # 是否启用自适应调整
    tau_range: Tuple[float, float] = (0.4, 0.6)   # τ动态调整范围
    beta_range: Tuple[float, float] = (5.0, 15.0) # β动态调整范围
    adjustment_cooldown_steps: int = 500          # 参数调整冷却期（步）

    # Three-Stage Training Parameters (三阶段训练，模块四，缩短以快速验证)
    scaffold_iterations: int = 2000       # 阶段1（骨架构建）结束迭代，从5000降至2000
    guided_iterations: int = 9000         # 阶段2（引导求精）结束迭代，从25000降至9000
    refinement_iterations: int = 10000    # 阶段3（精细化）结束迭代，从30000降至10000

    # Dynamic Guidance Weight (动态引导权重)
    lambda_guide_min: float = 0.05        # λguide最小值
    lambda_guide_max: float = 0.2         # λguide最大值

    # Phase Transition Criteria (阶段切换判据)
    phase_transition_psnr_threshold: float = 18.0     # 骨架→引导的PSNR阈值
    early_refinement_loss_threshold: float = 0.001    # 提前进入精细化的损失收敛阈值

    # Depth Anything / DACD Settings (BayesGS-Diff Module III)
    depth_anything_enable: bool = True
    depth_anything_encoder: str = "vitg"
    depth_anything_checkpoint: Optional[str] = "models/depth_anything_v2/depth_anything_v2_vitg.pth"
    depth_anything_device: str = "cuda"
    dacd_enable: bool = True
    dacd_interval: int = 300                 # 从200增至300 - 延迟DACD直到渲染深度更可靠
    dacd_anchor_opacity_threshold: float = 0.8
    dacd_min_anchor_ratio: float = 0.03      # 从0.05降至0.03 - 降低有效像素比例要求
    dacd_min_r2: float = 0.5
    dacd_void_opacity_threshold: float = 0.05
    dacd_safe_epistemic_threshold: float = 0.5
    dacd_far_plane: float = 10.0
    dacd_spawn_opacity: float = 0.1
    dacd_max_spawn_per_view: int = 512

    # ============================================================
    # 两阶段训练管线 (Phase 1→2→3)
    # ============================================================
    # Phase 1: 纯3DGS预训练 + 收敛检测
    phase1_max_iterations: int = 15000         # Phase 1最大迭代数（需足够步数产生3000+高斯体）
    phase1_convergence_window: int = 100       # 收敛检测滑动窗口大小
    phase1_convergence_threshold: float = 0.01 # loss变化率阈值(1%)
    phase1_convergence_psnr: float = 20.0      # PSNR收敛阈值
    phase1_convergence_min_iter: int = 2000    # 最少训练步数后才开始检测收敛

    # Phase 2: MC-Dropout主动采样
    phase2_rounds: int = 5                     # MC-Dropout主动采样轮数
    phase2_probe_rerank_enable: bool = True    # 是否对fine阶段Top-K候选执行单视图recoverability重排
    phase2_probe_topk: int = 8                 # 执行单视图probe重排的Top-K候选数量
    phase2_probe_rerank_strength: float = 1.0  # 0=仅记录probe日志，1=完全按recoverability乘性重排
    phase2_frontier_band_radius: int = 3       # frontier边界带半径（像素，基于mask邻域）
    phase2_generation_retry_budget: int = 12   # Phase 2生成阶段最多尝试多少个候选，允许拒绝后继续尝试后备候选

    # Phase 3: 伪视图增强训练
    phase3_iterations: int = 29000             # Phase 3训练迭代数
    pseudo_mode: str = "guidance_only"         # 伪视图模式: direct/weighted/guidance_only；默认仅作为辅助引导而非观测真值
    pseudo_min_q_gen: float = 0.90
    pseudo_use_q_gen_gate: bool = False        # q_gen 当前未校准，默认不作为伪视图接纳硬门控
    phase3_use_q_gen_weight: bool = False      # q_gen 当前未校准，默认不参与 Phase 3 整图loss缩放
    pseudo_min_q_gain: float = 0.50
    pseudo_use_q_gain_gate: bool = False       # q_gain 可作为 Phase 2 伪视图接纳的残差净收益门控
    phase3_use_q_gain_weight: bool = False     # q_gain 可作为 Phase 3 的整图净收益权重
    q_gain_calibrator_path: str = ""           # 由 withheld 诊断拟合得到的 q_gain 校准器 JSON
    pseudo_exist_region_min_coverage: float = 0.05
    pseudo_exist_region_l1_max: float = 0.20
    pseudo_degraded_accept_enable: bool = True
    pseudo_degraded_accept_l1_max: float = 0.24
    pseudo_degraded_accept_weight_scale: float = 0.15
    pseudo_degraded_accept_max_unsupported_l1: float = 0.25
    pseudo_degraded_accept_min_supported_coverage: float = 0.15
    pseudo_require_verified_evidence: bool = True   # 伪视图必须先形成已验证证据区域，否则直接拒绝
    pseudo_verified_min_coverage: float = 0.12      # 已验证证据区域最小覆盖率，低于该值说明视角“新但不可稳定修复”
    pseudo_verified_allow_supported_fallback: bool = False  # 禁止把supported_exist_mask冒充verified evidence
    pseudo_verified_pixel_l1_max: float = 0.18      # pseudo与当前render逐像素最大允许偏差
    pseudo_verified_warp_l1_max: float = 0.18       # pseudo与几何warp逐像素最大允许偏差
    pseudo_verified_warp_render_l1_max: float = 0.12  # warp与当前render在支持区的最大允许偏差
    pseudo_verified_min_warp_valid_ratio: float = 0.10
    pseudo_verified_min_support_projection_ratio: float = 0.05
    pseudo_verified_neighborhood_kernel: int = 3    # verified evidence需通过局部连通性过滤，抑制孤立噪点
    pseudo_verified_neighborhood_ratio: float = 0.55
    pseudo_verified_patch_min_pixels: int = 64      # verified evidence连通区域小于该像素数时不作为proposal patch
    pseudo_verified_patch_padding: int = 4          # proposal patch外扩边界，给局部纹理与上下文留缓冲
    pseudo_verified_max_patches_per_view: int = 8   # 每个视角最多保留多少个verified proposal patch
    pseudo_verified_patch_min_score: float = 0.001  # patch分数过低则丢弃，避免极小伪证据干扰训练
    pseudo_verified_patch_frontier_bonus: float = 0.5  # frontier-rich patch额外加分，优先吸收边界增量证据
    phase3_use_patch_proposals_only: bool = True    # Phase 3只消费verified patch proposals，不再消费整图伪监督
    plan6_four_stage_enable: bool = False           # 启用PLAN6四阶段协议输出；默认关闭，不改变现有训练
    plan6_gate0_status: str = ""                    # go/conditional_go才允许进入校准后的Phase3权重
    plan6_trust_is_calibrated: bool = False         # WhenTrust未校准时只输出proxy，不进入Phase3
    plan6_rgb_trust_is_calibrated: Optional[bool] = None
    plan6_depth_trust_is_calibrated: Optional[bool] = None
    plan6_alpha_trust_is_calibrated: Optional[bool] = None
    plan6_wrong_threshold: float = 0.35
    plan6_repairable_threshold: float = 0.25
    plan6_trust_threshold: float = 0.50
    plan6_min_trust_coverage: float = 0.01
    plan6_fusion_mode: str = "max"
    plan6_cache_valid_iters: int = 0                # >0时为Phase2 cache设置有效迭代窗口
    phase3_use_plan6_trust_weights: bool = False    # Phase3正式消费PLAN6 trust-gated权重
    phase3_use_plan6_depth_alpha_losses: bool = False  # 启用PLAN6 depth/alpha几何与opacity监督
    phase3_plan6_depth_loss_weight: float = 0.05
    phase3_plan6_alpha_loss_weight: float = 0.02
    phase2_probe_min_verified_coverage: float = 0.10  # probe阶段预计verified coverage过低时强惩罚
    phase3_region_weight_enable: bool = True   # 是否启用supported/frontier/unsupported区域化伪监督
    phase3_supported_exist_scale: float = 1.0
    phase3_frontier_weight_scale: float = 1.5
    phase3_unsupported_novel_weight_scale: float = 0.25
    phase3_use_verified_evidence_only: bool = True  # Phase 3伪监督只允许作用于已验证证据区域，禁止整图直接监督

    # DDUD ranking
    ddud_enable: bool = True
    ddud_geo_weight: float = 0.60
    ddud_hole_weight: float = 0.35
    ddud_frontier_weight: float = 0.5
    ddud_unsupported_penalty: float = 0.3

    # ============================================================
    # Clone/Split/Prune Densification 参数（显式化）
    # ============================================================
    refine_start_iter: int = 100               # 开始densification的迭代
    refine_every: int = 200                    # densification间隔，降低伪监督噪声被放大的机会
    refine_stop_iter: int = 10000              # 更早停止densification，减少错误高斯持续增殖
    grow_grad2d: float = 0.0006                # 提高2D梯度阈值，仅让更稳定的真实梯度触发致密化
    grow_scale3d: float = 0.01                 # 3D尺度阈值
    prune_opa: float = 0.01                    # 更积极剪枝，避免低透明度错误高斯长期滞留
    allow_pseudo_densification_stats: bool = False
