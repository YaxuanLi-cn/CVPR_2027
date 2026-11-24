# Method 逐句结构来源对照表

对应文件：`method.tex`（同目录）。共 **104 句**（含图注 4 句），每一句的句式骨架都取自下表 14 篇论文中的一句原文；内容（方法细节）来自 `SUPERGLUE_EXPERIMENTS.md`、`STAGE2_IMAGINATION.md` 及对应代码。

## 核验方式

- **原句逐字核验**：每条“原”都由脚本在该论文的 arXiv LaTeX 源码（Nature 论文为 nature.com 正文，LightGBM 为 NeurIPS 官方 PDF）中做了逐字匹配，全部命中。下面显示时只做了排版清理：引用写成 `[cite]`、交叉引用写成 `[ref]`、去掉 `\emph` 等格式命令，数学公式保持 LaTeX 原样。
- **会议正式版交叉核对**：又把每条原句的文字部分与会议官方 PDF（CVF Open Access / NeurIPS Proceedings）做了自动比对，103/104 句完全一致。唯一的例外是 **F6**（DUSt3R）：CVPR 正式版写的是 “Given camera intrinsics …”（没有 “the”，公式写法也略有不同），表中 F6 已直接采用正式版措辞。
- **句子拆分口径**：以句号为界；`where …` 这类紧跟公式的从句单独算一句；“Metric Scale Recovery” 里的三项编号列表每项算一句。段落小标题和公式本身不计。

## 来源论文

| 简称 | 论文 | 发表 | 取用句数 |
|---|---|---|---:|
| OrienterNet | Sarlin et al., *OrienterNet: Visual Localization in 2D Public Maps with Neural Matching* | CVPR 2023 | 25 |
| PixLoc | Sarlin et al., *Back to the Feature: Learning Robust Camera Localization from Pixels to Pose* | CVPR 2021 | 17 |
| SuperGlue | Sarlin et al., *SuperGlue: Learning Feature Matching with Graph Neural Networks* | CVPR 2020 (Oral) | 14 |
| LoFTR | Sun et al., *LoFTR: Detector-Free Local Feature Matching with Transformers* | CVPR 2021 | 13 |
| Pixel-Perfect SfM | Lindenberger et al., *Pixel-Perfect Structure-from-Motion with Featuremetric Refinement* | ICCV 2021 (Oral) | 8 |
| SfMLearner | Zhou et al., *Unsupervised Learning of Depth and Ego-Motion from Video* | CVPR 2017 (Oral) | 7 |
| Kaufmann et al. (Swift) | Kaufmann et al., *Champion-level drone racing using deep reinforcement learning* | Nature 620, 2023 | 6 |
| DUSt3R | Wang et al., *DUSt3R: Geometric 3D Vision Made Easy* | CVPR 2024 | 4 |
| Navigation World Models | Bar et al., *Navigation World Models* | CVPR 2025 | 4 |
| Kendall & Cipolla | Kendall and Cipolla, *Geometric Loss Functions for Camera Pose Regression with Deep Learning* | CVPR 2017 | 2 |
| Yi et al. | Yi et al., *Learning to Find Good Correspondences* | CVPR 2018 (Oral) | 1 |
| LightGlue | Lindenberger et al., *LightGlue: Local Feature Matching at Light Speed* | ICCV 2023 | 1 |
| LightGBM | Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* | NIPS 2017 | 1 |
| Kendall & Gal | Kendall and Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* | NIPS 2017 | 1 |

没有使用列表之外的来源；ECCV、TPAMI、arXiv-only 等不在你指定范围内的论文一律未用作句式来源（MASt3R 是 ECCV 2024，只作为被引用的方法出现，不是句式来源）。

## 逐句对照

每条三行：**新** = `method.tex` 中的句子；**原** = 来源原句；**构** = 两句共用的句式骨架。

### Figure caption

**C1** · SuperGlue（CVPR 2020 (Oral)）
```text
新: \textbf{The DreamNav architecture.}
原: The SuperGlue architecture.
构: The X architecture.
```
**C2** · SuperGlue（CVPR 2020 (Oral)）
```text
新: DreamNav is made up of two major components: the \emph{coarse-estimation stage} (Section~\ref{Coarse-estimation}), and the \emph{imagination-refined stage} (Section~\ref{sec:refine}).
原: SuperGlue is made up of two major components: the attentional graph neural network (Section [ref]), and the optimal matching layer (Section [ref]).
构: X is made up of two major components: the A (Section), and the B (Section).
```
**C3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: The first component uses \emph{rotation-aware matching} to map the source and target images into a calibrated two-view geometry, and then uses learned priors, metric scale cues and a heading network to create a metric pose hypothesis $\mathbf{p}_{\text{coarse}}$.
原: The first component uses a keypoint encoder to map keypoint positions $\mathbf{p}$ and their visual descriptors $d$ into a single vector, and then uses alternating self- and cross-attention layers (repeated $L$ times) to create more powerful representations $f$.
构: The first component uses A to map B into C, and then uses D to create E.
```
**C4** · SuperGlue（CVPR 2020 (Oral)）
```text
新: The imagination-refined stage renders the view implied by this hypothesis with a ground-plane homography, re-matches it against the real observation tile by tile, then finds the refined pose $\mathbf{p}^\star$ using reliability-weighted fusion.
原: The optimal matching layer creates an $M$ by $N$ score matrix, augments it with dustbins, then finds the optimal partial assignment using the Sinkhorn algorithm (for $T$ iterations).
构: The B [verb] A, [verb] it with C, then finds the D using E.
```

### Overview

**O1** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: DreamNav uses a combination of learned priors and calibrated two-view geometry to map a source--target image pair to a metric pose change.
原: Swift uses a combination of learning-based and traditional algorithms to map onboard sensory readings to control commands.
构: X uses a combination of A and B to map C to D.
```
**O2** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: This mapping comprises two stages (Figure~\ref{fig:framework}): (1) a coarse-estimation stage, which distils sparse correspondences and learned priors into an initial pose hypothesis, and (2) an imagination-refined stage that renders the view implied by this hypothesis and corrects the pose from its discrepancy with the real observation.
原: This mapping comprises two parts: (1) an observation policy, which distils high-dimensional visual and inertial information into a task-specific low-dimensional encoding, and (2) a control policy that transforms the encoding into commands for the drone.
构: This mapping comprises two parts: (1) an A, which distils B into C, and (2) a D that transforms ...
```
**O3** · DUSt3R（CVPR 2024）
```text
新: Note that the imagined view is rendered by a closed-form geometric warp, which radically differs from generative imagination but keeps the geometry exact on the ground plane (see Section~\ref{sec:refine}).
原: Note that both pointmaps are expressed in the same coordinate frame of $I^1$, which radically differs from existing approaches but offers key advantages (see [ref]).
构: Note that A is [verb-ed] ..., which radically differs from B but offers C (see ...).
```

### Problem formulation

**F1** · PixLoc（CVPR 2021）
```text
新: Our goal is to estimate the pose change $\mathbf{p}=(\Delta x,\Delta y,\Delta z,\Delta\phi,\Delta\theta)$ from a source image $S$ to a target image $G$, where $(\Delta x,\Delta y,\Delta z)$ is the metric displacement in the East--North--Up frame and $\Delta\phi$ and $\Delta\theta$ are the changes in camera tilt and heading.
原: Our goal is to estimate the 6-DoF pose $(R, t) \in \mathbf{SE}(3)$ of a query image $I_q$, where $R$ is a rotation matrix and $t$ is a translation vector in the camera frame.
构: Our goal is to estimate the X of Y, where A is ... and B is ... in the Z frame.
```
**F2** · OrienterNet（CVPR 2023）
```text
新: We consider image pairs with zero camera roll and a shared pinhole calibration $\mathbf{K}$, which is estimated once from training pairs.
原: We consider an image $I$ with known pinhole camera calibration.
构: We consider [input] with [known] pinhole camera calibration.
```
**F3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Considering that the source and target cameras are tilted by $\phi_S$ and $\phi_G$ from nadir, with $\Delta\phi=\phi_G-\phi_S$, we can write the relative rotation $\mathbf{R}$ of the rigid motion $\mathbf{X}_G=\mathbf{R}\mathbf{X}_S+\mathbf{t}$ between the two camera frames as:
原: Considering that query keypoint $i$ is in the image $Q$ and all source keypoints are in image $S$, $(Q, S) \in \{A, B\}^2$, we can write:
构: Considering that A is ... and B is ..., [side condition], we can write: [equation]
```
**F4** · PixLoc（CVPR 2021）
```text
新: where $\mathbf{A}(\phi)$ is the rotation from a camera tilted by $\phi$ to its gravity-aligned frame and $\mathbf{R}_z(\cdot)$ is a rotation about the vertical axis.
原: where $\rho$ is a robust cost function [cite] with derivative $\rho'$ and $w_k^i$ is a per-residual weight.
构: where A is a ... with/from ... and B is a ...
```
**F5** · OrienterNet（CVPR 2023）
```text
新: Under this factorization, we reduce the problem to estimating a relative pose $(\Delta\theta,\phi_S,\phi_G,\mathbf{u},d)$ consisting of three angles, a unit translation direction $\mathbf{u}\in\mathbb{S}^2$ expressed in the gravity-aligned frame of the source camera, and a metric distance $d>0$.
原: Under realistic assumptions, we reduce this problem to estimating a 3-DoF pose $\xi = (x,y,\theta)$ consisting of a location $(x,y) \in \mathbb{R}^2$ and heading angle $\theta\in(-\pi,\pi]$.
构: Under X, we reduce this problem to estimating a pose (...) consisting of A in R^n and B in [...].
```
**F6** · DUSt3R（CVPR 2024）
```text
新: Given the absolute heading $\psi_S$ of the source camera, the displacement in the world frame can be straightforwardly obtained from the levelled displacement $(\ell_x,\ell_y,\Delta z)^\top=d\,\mathbf{u}$ as
原: Given camera intrinsics K ∈ R^{3×3}, the pointmap X of the observed scene can be straightforwardly obtained from the ground-truth depthmap D ∈ R^{W×H} as X_{i,j} = K^{-1} D_{i,j} [i, j, 1]^⊤.
构: Given the A, the B can be straightforwardly obtained from the C as [equation].
```
**F7** · Yi et al.（CVPR 2018 (Oral)）
```text
新: Given two calibrated images, it is possible to estimate the three angles and $\mathbf{u}$ exactly---in theory---and the displacement only up to its scale $d$~\citep{hartley2003multiple} and the absolute heading $\psi_S$.
原: Given two images, it is possible to estimate rotation exactly---in theory---and translation only up to a scale factor [cite].
构: Given two images, it is possible to estimate A exactly---in theory---and B only up to C [cite].
```
**F8** · PixLoc（CVPR 2021）
```text
新: We should thus focus learning on these two quantities and on the reliability of each estimate, making the remaining pose parameters tightly constrained by geometry.
原: We should thus focus on learning robust and generic features, making the pose estimation scene-agnostic and tightly constrained by geometry.
构: We should thus focus on [learning] A, making B ... and tightly constrained by geometry.
```

### Coarse stage: intro

**S0** · LoFTR（CVPR 2021）
```text
新: Given the image pair $S$ and $G$, the coarse stage uses sparse correspondences and learned priors to produce an initial estimate $\mathbf{p}_{\text{coarse}}$.
原: Given the image pair $I^A$ and $I^B$, the existing local feature matching methods use a feature detector to extract interest points.
构: Given the image pair A and B, the X use(s) Y to [verb] Z.
```

### Rotation-aware multi-scale matching

**M1** · LoFTR（CVPR 2021）
```text
新: Directly applying a pretrained SuperPoint--SuperGlue matcher~\citep{detone2018superpoint,sarlin2020superglue} in the context of UAV image pairs is impractical, because neither network is invariant to in-plane rotation while the heading change spans the full circle.
原: Directly applying the vanilla version of Transformer in the context of local feature matching is impractical even when the input length is reduced by the local feature CNN.
构: Directly applying A in the context of B is impractical [subordinate clause].
```
**M2** · LoFTR（CVPR 2021）
```text
新: To remedy this problem, we propose to use a rotation-aware, coarse-to-fine inference scheme around the frozen matcher.
原: To remedy this problem, we propose to use an efficient variant of the vanilla attention layer in Transformer.
构: To remedy this problem, we propose to use a(n) [adj.] variant/scheme of/around X.
```
**M3** · PixLoc（CVPR 2021）
```text
新: To find the rotation, we apply the matcher to a low-resolution copy of the pair, rotating $G$ by $k\cdot90^\circ$ for $k\in\{0,1,2,3\}$, yielding four match sets $\mathcal{M}^{(k)}=\{(\mathbf{x}_i,\mathbf{x}'_i,w_i)\}$ with confidences $w_i$.
原: To further increase it, we apply PixLoc to image pyramids, starting at the lowest resolution, yielding coarsest feature maps of size $W{=}16$.
构: To [purpose], we apply X to Y, [participle clause], yielding Z of/with ...
```
**M4** · OrienterNet（CVPR 2023）
```text
新: We select a single rotation by maximum total confidence: $k^\star=\operatorname*{argmax}_{k}\sum_{i\in\mathcal{M}^{(k)}}w_i$.
原: We estimate a single pose by maximum likelihood: $\xi^* = \operatorname*{argmax}_{\xi} P(\xi|I,\text{map},\xi_\text{prior})$.
构: We [estimate/select] a single X by maximum Y: x* = argmax ...
```
**M5** · LoFTR（CVPR 2021）
```text
新: After selecting $k^\star$, the pair is matched again at a higher resolution with a larger keypoint budget, and all keypoints are mapped back to the unrotated image at the original resolution.
原: After establishing coarse matches, these matches are refined to the original image resolution with the coarse-to-fine module.
构: After [gerund] A, [subject] is/are [refined] ... the original (image) resolution with B.
```
**M6** · PixLoc（CVPR 2021）
```text
新: Coarse matches are thus responsible for the robustness of the pose estimate while fine matches enhance its accuracy.
原: Low-resolution feature maps are thus responsible for the robustness of the pose prediction while finer features enhance its accuracy.
构: A are thus responsible for the robustness of the pose [estimate] while B enhance its accuracy.
```
**M7** · PixLoc（CVPR 2021）
```text
新: To prevent imprecise coarse keypoints from degrading well-matched pairs, we add the coarse matches to the pose estimation only if the fine matches alone yield too few inliers.
原: To prevent hard examples from smoothing the fine features, we apply the loss at a given level only if the previous one succeeded in bringing the pose sufficiently close to the ground truth.
构: To prevent A from [gerund] B, we [apply] C ... only if D ...
```

### Relative pose from two-view geometry

**G1** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: Once the matches are established, the geometric estimation proceeds, using robust essential-matrix fitting with MAGSAC++~\citep{barath2020magsacpp} followed by a nonlinear refinement of the Sampson error~\citep{hartley2003multiple}.
原: Once all tracks are refined, the geometric estimation proceeds, typically using two-view epipolar geometric verification followed by incremental or global SfM.
构: Once A are [done], the geometric estimation proceeds, using B followed by C.
```
**G2** · SuperGlue（CVPR 2020 (Oral)）
```text
新: In resolving the four-fold decomposition of the essential matrix, some regularities of UAV imagery can be leveraged: triangulated points lie in front of both cameras, the camera roll is zero, and the tilt stays within a limited range.
原: In the image matching problem, some regularities of the world could be leveraged: the 3D world is largely smooth and sometimes planar, all correspondences for a given image pair derive from a single epipolar transform if the scene is static, and some poses are more likely than others.
构: In [problem], some regularities of X could be leveraged: A, B, and C.
```
**G3** · OrienterNet（CVPR 2023）
```text
新: Since the roll is zero, each estimated $\mathbf{R}$ corresponds to a triplet $(\Delta\theta,\phi_S,\phi_G)$ obtained in closed form from the factorization in Eq.~\eqref{eq:rot}.
原: Since the image is gravity-aligned, each of the $U$ columns in $X$ corresponds to a vertical plane in the 3D space.
构: Since [premise], each A corresponds to a B in/from C.
```
**G4** · SuperGlue（CVPR 2020 (Oral)）
```text
新: This also lets us express the translation direction in the gravity-aligned frame of the source camera, $\mathbf{u}=-\mathbf{A}(\phi_S)\mathbf{R}^\top\mathbf{t}/\lVert\mathbf{t}\rVert_2$, whose vertical component gives the direction of $\Delta z$.
原: This also lets us label some keypoints $\mathcal I \subseteq \mathcal{A}$ and $\mathcal J \subseteq \mathcal{B}$ as unmatched if they do not have any reprojection in their vicinity.
构: This also lets us [verb] A [math] as/in B [trailing clause].
```

### MAP pose with learned priors

**P1** · OrienterNet（CVPR 2023）
```text
新: Two-view pose estimation is ambiguous in pairs that exhibit a near-planar scene and a short baseline, as a pitch rotation then resembles a forward translation.
原: Single-image localization is ambiguous in locations that exhibit few distinctive semantic elements or repeated patterns.
构: X is ambiguous in [cases] that exhibit A or/and B.
```
**P2** · OrienterNet（CVPR 2023）
```text
新: Such ambiguity can be resolved by injecting learned priors on the heading change and on the tilt of each camera when the matches are uninformative.
原: Such challenge can be disambiguated by accumulating additional cues over multiple views when their relative poses are known.
构: Such X can be [resolved] by [gerund] A over/on B when C.
```
**P3** · PixLoc（CVPR 2021）
```text
新: To this end, an MLP over pooled statistics of a frozen MASt3R model~\citep{leroy2024mast3r} predicts the priors $(\overline{\Delta\theta},\bar\phi_S,\bar\phi_G)$ along with a direct regression of every pose parameter.
原: To this end, the CNN predicts an uncertainty map $U_k^l \in \mathbb{R}_{>0}^{W_l\times H_l}$ along with each feature map.
构: To this end, the [network] predicts A [math] along with B.
```
**P4** · PixLoc（CVPR 2021）
```text
新: The total error over the $N$ inlier matches is
原: The total error over $N$ observations is
构: The total error over N [observations] is [equation]
```
**P5** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: where $e_i$ is the Sampson error of match $i$ in pixels, $\ominus$ denotes the wrapped angular difference, and $\rho$ is the pseudo-Huber cost.
原: where $\mathcal{T}(j)$ is the set the images and keypoints in track $j$, $\Pi(\cdot)$ projects to the image plane, and $\lVert\cdot\rVert_\gamma$ is a robust norm [cite].
构: where A is the ... in ..., B [verb] ..., and C is a robust [norm/cost].
```
**P6** · PixLoc（CVPR 2021）
```text
新: This nonlinear least-squares cost is iteratively minimized from both the essential-matrix solution and the prior pose using a trust-region solver, and the better-supported solution is kept.
原: This nonlinear least-squares cost is iteratively minimized from an initial estimate $(R_0, t_0)$ using the Levenberg-Marquardt (LM) algorithm [cite].
构: This nonlinear least-squares cost is iteratively minimized from [initial estimate] using the X algorithm/solver.
```
**P7** · PixLoc（CVPR 2021）
```text
新: The data term dominates if the matches tightly constrain the pose.
原: The weight is $1$ if the 3D point projects into a location with low uncertainty in both the query and the reference images.
构: The A is/does X if B ...
```
**P8** · PixLoc（CVPR 2021）
```text
新: The priors take over as the geometry degenerates, which pins the tilts and thereby stabilizes the translation direction.
原: It tends to $0$ as either of the location is uncertain.
构: It [verb] ... as [condition degrades].
```

### Pose-guided zoom re-matching

**Z1** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: Matching entire images is necessary to establish correspondences when nothing is known about the relative pose.
原: Summarizing dense image information into sparse points is necessary to perform global data association and optimization at scale.
构: [Gerund phrase] is necessary to [perform] A ...
```
**Z2** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: However, refining the pose is an inherently local operation, which, we show, can substantially benefit from concentrating keypoints and resolution on the co-visible region.
原: However, refining geometry is an inherently local operation, which, we show, can efficiently benefit from locally-dense pixels.
构: However, refining X is an inherently local operation, which, we show, can [adv.] benefit from Y.
```
**Z3** · LightGlue（ICCV 2023）
```text
新: Given the rotation-only homography $\mathbf{K}\mathbf{R}^\top\mathbf{K}^{-1}$ of the MAP pose, we warp $G$ into the orientation of $S$ and predict the window in which the two views overlap.
原: Given a homography or pixel-wise depth and a relative pose, we wrap points from $A$ to $B$ and conversely.
构: Given a homography ... and a (relative) pose, we warp A [from/into] B and ...
```
**Z4** · LoFTR（CVPR 2021）
```text
新: For every pair, we first locate this window in both images, and then crop and resample it from the original resolution to the input size of the matcher before matching again.
原: For every coarse match $(\tilde{i}, \tilde{j})$, we first locate its position $(\hat{i}, \hat{j})$ at fine-level feature maps $\hat F^A$ and $\hat F^B$, and then crop two sets of local windows of size $w\times w$.
构: For every X, we first locate its/this A at/in B, and then crop ...
```
**Z5** · OrienterNet（CVPR 2023）
```text
新: The zoomed matches are not pooled with those of the first pass as we found the pose solved from the zoomed matches alone to be more accurate.
原: $F$ is not normalized as we let $\Phi_\text{map}$ modulate its norm as importance weight in the matching.
构: A is/are not [verb-ed] as we [let/found] B ...
```
**Z6** · LoFTR（CVPR 2021）
```text
新: We denote the two poses re-estimated from them as the zoom-pass solutions.
原: We denote the transformed features as $\tilde{F}^A_{tr}$ and $\tilde{F}^B_{tr}$.
构: We denote the [participle] X as Y.
```

### Metric scale recovery

**K1** · DUSt3R（CVPR 2024）
```text
新: The translation $\mathbf{t}$ is recovered up to an unknown scale factor.
原: The output pointmaps $X^{1,1}$ and $X^{2,1}$ are regressed up to an unknown scale factor.
构: The [output] X is/are [regressed] up to an unknown scale factor.
```
**K2** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: The metric distance $d$ is impossible to recover with a purely geometric approach.
原: The aerodynamic forces and torques are difficult to model with a first-principles approach.
构: The X is/are difficult to [model] with a [first-principles] approach.
```
**K3** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: We thus use three complementary metric cues:
原: We thus use a data-driven model [cite].
构: We thus use [a data-driven] X. (list of three numbered items follows, as in the OrienterNet overview)
```
**K5** · OrienterNet（CVPR 2023）
```text
新: 1)~The prior MLP regresses the distance from the MASt3R statistics and thus provides an estimate $d_{\text{reg}}$ without relying on any match.
原: 1) The image-CNN extracts semantic features from the image and lifts them to an orthographic Bird's-Eye View (BEV) representation $T$ by inferring the 3D structure of the scene.
构: 1) The [network] [verb] A from B and [verb] ... C by/without [gerund] D.
```
**K6** · OrienterNet（CVPR 2023）
```text
新: 2)~The baseline-to-depth ratio $s$ of the triangulated inliers is combined with the metric scene depth $D$ of MASt3R into an estimate $d_{\text{depth}}=c\,s\,D$ that makes no assumption on the flight altitude.
原: 2) The OSM map is encoded by the map-CNN into a neural map $F$ that embeds semantic and geometric information.
构: 2) The A is [verb-ed] by/with B into a C [math] that [verb] ...
```
**K7** · OrienterNet（CVPR 2023）
```text
新: 3)~We estimate $d_{\text{alt}}=H_0/\tilde h$ by comparing the height $\tilde h$ of the camera above the triangulated ground, in baseline units, against the typical flight altitude $H_0$.
原: 3) We estimate a probability distribution over camera poses $\xi$ by exhaustively matching the BEV against the map.
构: 3) We estimate A by [gerund] B against C.
```
**K8** · OrienterNet（CVPR 2023）
```text
新: The third cue therefore encodes the flight altitude of the training data in a single constant, requiring recalibration at other altitudes.
原: They therefore encode the focal length in the network weights, learning the mapping from object scale to depth.
构: They therefore encode the A in the B, [participle clause].
```

### Reliability-weighted candidate fusion

**U1** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Committing to a single estimator for all image pairs would be brittle.
原: Building a separate representation for all $M\times N$ potential matches would be prohibitive.
构: [Gerund] a [single/separate] A for all B would be [adj.].
```
**U2** · SuperGlue（CVPR 2020 (Oral)）
```text
新: We instead express each pose parameter as a reliability-weighted blend of candidate estimates followed by a learned residual:
原: We instead express the pairwise score as the similarity of matching descriptors:
构: We instead express the A as the B of C: [equation]
```
**U3** · SfMLearner（CVPR 2017 (Oral)）
```text
新: where $k$ indexes over the candidates $c_k$ of a pose parameter $y\in\{\Delta\theta,\Delta\phi,\log d,\Delta z,\ell_x,\ell_y\}$, $\hat e_k$ is the absolute error of $c_k$ predicted from pair-level features $\mathbf{f}$ such as inlier counts, residuals and the agreement between candidates, and $\bar y$ and $r$ are the blended estimate and the residual regressor, respectively.
原: where $l$ indexes over different image scales, $s$ indexes over source images, and $\lambda_s$ and $\lambda_e$ are the weighting for the depth smoothness loss and the explainability regularization, respectively.
构: where k indexes over A, ..., and X and Y are the P and the Q, respectively.
```
**U4** · OrienterNet（CVPR 2023）
```text
新: The candidates consist of the essential-matrix and MAP solutions of both matching passes and the direct regressions from the MASt3R statistics, each with a different failure mode and whose displacement components are given as $\hat d\,\mathbf{u}$ with the fused distance $\hat d$.
原: The data consists of a collection of polygons, lines, and points, each of a given semantic class and whose coordinates are given in the same local reference frame.
构: The X consist(s) of A, B, and C, each of/with a ... and whose ... are given in/as ...
```
**U5** · LightGBM（NIPS 2017）
```text
新: For each candidate, a GBDT~\citep{ke2017lightgbm} learns $\hat e_k$ by fitting the square root of its absolute error (a compromise between typical and tail errors).
原: In each iteration, GBDT learns the decision trees by fitting the negative gradients (also known as residual errors).
构: In/For each X, GBDT learns A by fitting B (parenthetical gloss).
```
**U6** · LoFTR（CVPR 2021）
```text
新: We use the $\ell_1$ loss for the residual regressor.
原: We use the $\ell_2$ loss for fine-level refinement.
构: We use the l_p loss for X.
```
**U7** · LoFTR（CVPR 2021）
```text
新: For the heading change, the blend $\bar y$ and the residual are computed on the unit circle.
原: For self-attention layers, the input features $f_i$ and $f_j$ (shown in Fig. [ref]) are the same (either $\tilde{F}^A$ or $\tilde{F}^B$).
构: For X, the A and B are [predicate].
```
**U8** · Kendall & Gal（NIPS 2017）
```text
新: This makes the fusion robust to failed estimators: candidates for which the model predicts a large error will have a smaller effect on the blend.
原: This makes the model more robust to noisy data: inputs for which the model learned to predict high uncertainty will have a smaller effect on the loss.
构: This makes the A (more) robust to B: C for which the model [predicts] high D will have a smaller effect on the E.
```
**U9** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: During this fusion, only the gradient-boosted regressors are trained, whereas the weights of the matcher and of MASt3R are kept frozen.
原: During this fine-tuning phase, only the weights of the control policy are updated, whereas the weights of the gate-detection network are kept constant.
构: During this X, only A are updated, whereas the weights of B are kept constant.
```
**U10** · PixLoc（CVPR 2021）
```text
新: As each regressor never sees the scenes it predicts (five-fold cross-fitting grouped by scene), its outputs on training pairs follow the same error distribution as at test time.
原: As the CNN never sees 3D points, PixLoc can generalize to any 3D structure available.
构: As the A never sees B, [consequence].
```

### Absolute heading

**H1** · OrienterNet（CVPR 2023）
```text
新: Converting the levelled displacement to the world frame requires extracting the absolute heading $\psi_S$ from single-image cues such as the direction of cast shadows.
原: Cross-modal matching between the image and the map requires extracting semantic information from visual cues.
构: [Gerund phrase] requires extracting A from [visual] cues.
```
**H2** · Kendall & Cipolla（CVPR 2017）
```text
新: Our heading network, a ConvNeXt~\citep{liu2022convnet}, outputs a von Mises distribution over the heading of each image, given by a mean direction $\mu$ and a concentration $\kappa$.
原: Our network outputs an estimate, $\mathbf{\hat{p}}$, for pose, $\mathbf{p}$, given by a 3-D camera position $\mathbf{\hat{x}}$ and orientation $\mathbf{\hat{q}}$.
构: Our network outputs a(n) A for/over B, given by a C and (a) D.
```
**H3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Given the heading labels of the training images, we minimize the negative log-likelihood of this distribution:
原: Given these labels, we minimize the negative log-likelihood of the assignment $\bar{\mathbf P}$:
构: Given these labels, we minimize the negative log-likelihood of X: [equation]
```
**H4** · SuperGlue（CVPR 2020 (Oral)）
```text
新: where $I_0$ is the modified Bessel function of the first kind of order zero.
原: where $<\cdot,\cdot>$ is the inner product.
构: where X is the Y.
```
**H5** · OrienterNet（CVPR 2023）
```text
新: For the source camera, we express the joint estimate over the two single-image predictions as
原: For an arbitrary reference view $i$, we express the joint likelihood over all single-view predictions as
构: For [a reference view], we express the joint X over all single-view predictions as [equation]
```
**H6** · OrienterNet（CVPR 2023）
```text
新: where $\Delta\hat\theta$ denotes the fused heading change.
原: where $\oplus$ denotes the pose composition operator.
构: where X denotes the Y.
```
**H7** · OrienterNet（CVPR 2023）
```text
新: This is simply computed by rotating the prediction of the target image into the source camera and summing the natural parameters of the two distributions.
原: This is efficiently computed by warping each probability volume $P_j$ to the reference frame $i$.
构: This is [adv.] computed by [warping] A to/into the reference B.
```
**H8** · SuperGlue（CVPR 2020 (Oral)）
```text
新: After fusion, we rotate $(\hat\ell_x,\hat\ell_y)$ by $\hat\psi_S$ with Eq.~\eqref{eq:world} and recover $\mathbf{p}_{\text{coarse}}=(\Delta\hat x,\Delta\hat y,\Delta\hat z,\Delta\hat\phi,\Delta\hat\theta)$.
原: After $T$ iterations, we drop the dustbins and recover $\mathbf{P} = \bar{\mathbf{P}}_{1:M,1:N}$.
构: After X, we [verb] A and recover B = ...
```

### Refined stage: motivation

**R1** · Navigation World Models（CVPR 2025）
```text
新: Intuitively, if a pose hypothesis is correct, we can use it to imagine one view from the other, and the imagined view will coincide with the real observation.
原: Intuitively, if our world model is familiar with an environment, we can use it to simulate navigation trajectories, and choose the ones which reach the goal.
构: Intuitively, if A is B, we can use it to [simulate] C, and ...
```
**R2** · SfMLearner（CVPR 2017 (Oral)）
```text
新: The key refinement signal of this stage comes from the task of \emph{novel view synthesis}: given one view of a scene and a pose hypothesis, synthesize the image of the scene seen from the other camera.
原: The key supervision signal for our depth and pose prediction CNNs comes from the task of novel view synthesis: given one input view of a scene, synthesize a new image of the scene seen from a different camera pose.
构: The key X signal for/of Y comes from the task of novel view synthesis: given one view of a scene, synthesize ... seen from ...
```
**R3** · Kendall & Cipolla（CVPR 2017）
```text
新: This is a particularly demanding task in our setting because a displacement of one metre shifts the ground by only about 16 pixels at the native resolution---roughly two pixels at the 512-pixel resolution of latent diffusion models---which leaves little margin for the spatial inaccuracy of generated images.
原: This is a particularly challenging objective because it involves learning two distinct quantities - rotation and translation - with different units and scales.
构: This is a particularly challenging X because [clause] - [parenthetical] - [trailing phrase].
```
**R4** · LoFTR（CVPR 2021）
```text
新: We propose to tackle the limited fidelity of generative imagination with a generator-free design.
原: We propose to tackle the repeatability issue of feature detectors with a detector-free design.
构: We propose to tackle the A issue of B with a B-free design.
```

### Pose hypotheses

**Y1** · PixLoc（CVPR 2021）
```text
新: We are given the pose estimates of the coarse stage, namely the fused estimate $\mathbf{p}_{\text{coarse}}$ and the essential-matrix and MAP solutions of both matching passes, collectively called the hypothesis set $\mathcal{P}$.
原: We are given a 3D representation of the environment, such as a sparse or dense 3D point cloud $\{P_i\}$ and posed reference images $\{I_k\}$, collectively called the reference data.
构: We are given A, such as/namely B and C, collectively called the D.
```
**Y2** · PixLoc（CVPR 2021）
```text
新: To keep the pipeline simple, we select the guiding hypothesis as the fused estimate $\mathbf{p}_{\text{coarse}}$ and imagine a single view per pair.
原: To keep the pipeline simple, we select the initial pose as the pose of the first reference image returned by image retrieval.
构: To keep the pipeline simple, we select the A as the B ...
```

### Geometric imagination

**I1** · OrienterNet（CVPR 2023）
```text
新: We observe that the scene seen from flight altitude is mostly planar and that the parallax of elevated structures such as rooftops is small relative to the image motion induced by the pose.
原: We also observe that our world is mostly planar and that the motion of people and objects in outdoor spaces is mostly restricted to 2D surface.
构: We also observe that A is mostly planar and that the B of C is [mostly restricted/small] ...
```
**I2** · SfMLearner（CVPR 2017 (Oral)）
```text
新: Let $\tilde{\mathbf{x}}$ denote the homogeneous coordinates of a pixel in the source view, and $\mathbf{n}=(0,\sin\phi_S,\cos\phi_S)^\top$ denote the downward unit normal of the ground plane in the source camera frame.
原: Let $p_t$ denote the homogeneous coordinates of a pixel in the target view, and $K$ denote the camera intrinsics matrix.
构: Let x denote the homogeneous coordinates of a pixel in the A view, and B denote the C.
```
**I3** · SfMLearner（CVPR 2017 (Oral)）
```text
新: We can obtain its projected coordinates onto the target view $\tilde{\mathbf{x}}'$ by
原: We can obtain $p_t$'s projected coordinates onto the source view $p_s$ by
构: We can obtain its projected coordinates onto the A view x' by [equation]
```
**I4** · PixLoc（CVPR 2021）
```text
新: where $\mathbf{t}=-\mathbf{R}\mathbf{A}(\phi_S)^\top(\hat\ell_x,\hat\ell_y,\Delta\hat z)^\top$ is the metric translation given the guiding hypothesis and $h$ is the height of the source camera above the ground.
原: where $p_q^i = \Pi\left(RP_i+t\right)$ is the projection of $i$ in the query given its current pose estimate and $\left[\cdot\right]$ is a lookup with sub-pixel interpolation.
构: where A = ... is the B given the current pose estimate and C is a D.
```
**I5** · OrienterNet（CVPR 2023）
```text
新: The height of the camera can be estimated as the distance to the triangulated ground in the two-view reconstruction of the coarse stage, scaled to metres by the fused distance $\hat d$.
原: The precise height of the camera can always be estimated as the distance to the ground in a local SLAM reconstruction.
构: The height of the camera can be estimated as the distance to the ground in a(n) X reconstruction.
```
**I6** · PixLoc（CVPR 2021）
```text
新: When this reconstruction is not sufficiently reliable, as in pairs with few inliers, we replace $h$ by the median camera height $H_0$ of the training set.
原: When retrieval is not sufficiently robust and returns an incorrect location, as in the most challenging conditions, one could improve the performance by reranking using covisiblity clustering [cite] or pose verification with sparse [cite] or dense matching [cite].
构: When X is not sufficiently robust ..., as in [hard cases], one could/we [remedy] by ...
```
**I7** · SfMLearner（CVPR 2017 (Oral)）
```text
新: As indicated in Eq.~\eqref{eq:homography}, a key component of our refinement stage is a plane-induced renderer that imagines the source view $\widehat{S}$ by sampling pixels from the target view $G$ based on the homography $\mathbf{H}$, i.e., $\widehat{S}(\mathbf{x})=G(\mathbf{H}\mathbf{x})$.
原: As indicated in Eq. [ref], a key component of our learning framework is a differentiable depth image-based renderer that reconstructs the target view $I_t$ by sampling pixels from a source view $I_s$ based on the predicted depth map $\hat{D}_t$ and the relative pose $\hat{T}_{t \rightarrow s}$.
构: As indicated in Eq., a key component of our X is a Y renderer that reconstructs the A view by sampling pixels from a B view based on C.
```
**I8** · SfMLearner（CVPR 2017 (Oral)）
```text
新: Note that the above formulation implicitly assumes 1) the ground is planar; 2) the cameras have zero roll and known intrinsics; 3) the guiding hypothesis is accurate enough for the two views to overlap after warping.
原: Note that when applied to monocular videos the above view synthesis formulation implicitly assumes 1) the scene is static without moving objects; 2) there is no occlusion/disocclusion between the target view and the source views; 3) the surface is Lambertian so that the photo-consistency error is meaningful.
构: Note that ... the above formulation implicitly assumes 1) A; 2) B; 3) C.
```
**I9** · SfMLearner（CVPR 2017 (Oral)）
```text
新: If the first assumption is violated by elevated structures, the imagined view is locally displaced by a residual parallax, which the subsequent matching step tolerates.
原: If any of these assumptions are violated in a training sequence, the gradients could be corrupted and potentially inhibit training.
构: If [any of] these assumption(s) is/are violated in/by X, the Y could be/is Z ...
```

### Tile-wise re-matching at native resolution

**T1** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: Because $\widehat{S}$ and $S$ are aligned up to a small residual, corresponding content falls into the same image window.
原: Because a 3D point has a single projection on a given image plane, valid tracks cannot contain multiple keypoints detected in the same image.
构: Because A ..., B ... in(to) the same image (window).
```
**T2** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: We can leverage this property to match the overlap tile by tile at the native resolution, which is not possible before alignment.
原: We can leverage this property to efficiently prune out most incorrect matches using the track separation algorithm introduced in [cite].
构: We can leverage this property to [verb] A ...
```
**T3** · LoFTR（CVPR 2021）
```text
新: The frozen matcher then processes each of up to $3{\times}3$ co-located tiles with its full keypoint budget, yielding correspondences $(\mathbf{x}_i,\hat{\mathbf{x}}_i)$ between $S$ and $\widehat{S}$.
原: A smaller LoFTR module then transforms the cropped features within each window by $N_f$ times, yielding two transformed local feature maps $\hat{F}^A_{tr}(\hat{i})$ and ${\hat{F}^B_{tr}(\hat{j})}$ centered at $\hat{i}$ and $\hat{j}$, respectively.
构: A [module] then [transforms] the A within each window ..., yielding B ...
```
**T4** · LoFTR（CVPR 2021）
```text
新: Mapping every $\hat{\mathbf{x}}_i$ back to $G$ through $\mathbf{H}$ and gathering all tiles produces the imagined match set $\mathcal{M}_{\text{im}}=\{(\mathbf{x}_i,\mathbf{H}\hat{\mathbf{x}}_i)\}$.
原: Gathering all the matches $\{(\hat{i}, \hat{j}^\prime)\}$ produces the final fine-level matches $\mathcal{M}_f$.
构: Gathering all the A produces the final B M.
```
**T5** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: The relative pose can then be re-estimated from $\mathcal{M}_{\text{im}}$ with the same essential-matrix and MAP solvers.
原: The estimated structure and motion can then be refined with a similar featuremetric cost.
构: The [estimated] A can then be refined with a similar/the same B.
```
**T6** · OrienterNet（CVPR 2023）
```text
新: The residual displacement $\delta_i=\lVert\hat{\mathbf{x}}_i-\mathbf{x}_i\rVert_2$ reflects how far the imagined view is from the real observation at each match.
原: This score reflects how likely an image is to be taken at each location.
构: This X reflects how [adj.] A is ... at each B.
```

### Hypothesis verification

**V1** · Navigation World Models（CVPR 2025）
```text
新: Assuming we have the hypothesis set $\mathcal{P}$ of the coarse stage, we can use $\mathcal{M}_{\text{im}}$ to verify each of its members.
原: Assuming we have an existing navigation policy $\Pi(\mathbf{a}|s_0, s^*)$, we can use NWMs to rank sampled trajectories.
构: Assuming we have an existing A, we can use B to rank/verify C.
```
**V2** · OrienterNet（CVPR 2023）
```text
新: Each score is computed by testing $\mathcal{M}_{\text{im}}$ against the epipolar geometry of the corresponding hypothesis as
原: Each element is computed by correlating $F$ with $T$ transformed by the corresponding pose as
构: Each X is computed by [gerund] A with/against B ... the corresponding pose as [equation]
```
**V3** · DUSt3R（CVPR 2024）
```text
新: where $e_i(\mathbf{p}_k)$ is the Sampson error of match $i$ under hypothesis $\mathbf{p}_k$, and $\tau$ is a threshold controlling the strictness of the test.
原: where $C^{v,1}_i$ is the confidence score for pixel $i$, and $\alpha$ is a hyper-parameter controlling the regularization term [cite].
构: where A is the B for/of C i, and D is a hyper-parameter controlling the E.
```
**V4** · OrienterNet（CVPR 2023）
```text
新: Instead of directly selecting the hypothesis with the best score, we pass the scores $v_\tau$ and the statistics of $\delta_i$ to the fusion as reliability features that complement the candidates themselves.
原: Instead of directly regressing the distribution $\alpha$ over depths, we regress a distribution $S$ over scales that are independent from the camera calibration parameters.
构: Instead of directly [gerund] A, we [verb] B ... that are/complement C.
```

### Refined fusion

**W1** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: We use the same reliability-weighted fusion for the coarse and refined stages, requiring a single additional set of regressors.
原: We use the same representations for keypoint and bundle adjustments, requiring a single forward pass per image.
构: We use the same A for B and C, requiring a single D ...
```
**W2** · OrienterNet（CVPR 2023）
```text
新: It is the combination of the coarse estimate $\hat y$ and the two candidates re-estimated from $\mathcal{M}_{\text{im}}$:
原: It is the combination of an image-map matching term $M$ and the location prior $\Omega$:
构: It is the combination of a(n) A and the B: [equation]
```
**W3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: where $\operatorname{blend}(\cdot)$ denotes the reliability-weighted average of Eq.~\eqref{eq:blend}.
原: where $[\cdot\,||\,\cdot]$ denotes concatenation.
构: where [operator] denotes X.
```
**W4** · LoFTR（CVPR 2021）
```text
新: For the distance, we use the same scale cues as in the coarse stage, recomputed from $\mathcal{M}_{\text{im}}$.
原: With the optimal transport layer, we use the same loss formulation as in [cite].
构: With/For X, we use the same A as in B.
```
**W5** · PixLoc（CVPR 2021）
```text
新: In training, we leverage the held-out predictions of the coarse stage to match the test and training conditions and prevent the refinement from over-trusting its initialization.
原: In contrast, we leverage the power of differentiable programming to match the test and training conditions and learn additional strong priors from noisy data.
构: [Adverbial], we leverage A to match the test and training conditions and [verb] B.
```

### Canonical-view heading verification

**B1** · Navigation World Models（CVPR 2025）
```text
新: The imagination in Eq.~\eqref{eq:homography} refines the relative pose but does not constrain the absolute heading $\psi_S$.
原: The formulation in Equation [ref] models action but does not allow control over the temporal dynamics.
构: The X in Equation models A but does not [allow/constrain] B.
```
**B2** · Navigation World Models（CVPR 2025）
```text
新: We extend this paradigm with a second imagined view, rectifying each image to nadir, thus now a change of camera heading amounts to an in-plane rotation of the image.
原: We extend this formulation with a time shift input $k\in[{T_\text{min}}, {T_\text{max}}]$, setting $a_\tau = (u,\phi, k)$, thus now $a_\tau$ specifies the time change $k$, used to determine how many steps should the model move into the future (or past).
构: We extend this formulation with a(n) A, [participle clause], thus now B [verb] C ...
```
**B3** · OrienterNet（CVPR 2023）
```text
新: Each image is rectified via a rotation-only homography computed from its estimated tilt such that its tilt is zero -- its principal axis is then vertical.
原: The image is rectified via a homography computed from the known gravity such that its roll and tilt are zero -- its principal axis is then horizontal.
构: The image is rectified via a homography computed from the A such that its ... tilt is/are zero -- its principal axis is then B.
```
**B4** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Unlike the plane-induced homography of Eq.~\eqref{eq:homography}, it is also exact for any scene depth, which removes the tilt as a nuisance factor and provides an exact rotation augmentation.
原: Unlike other handcrafted or learned approaches, it is also equivariant to permutation of the images, which better reflects the symmetry of the problem and provides a beneficial inductive bias.
构: Unlike A, it is also [property] ..., which [verb] B and provides a(n) C.
```
**B5** · OrienterNet（CVPR 2023）
```text
新: The heading is discretized into $K$ rotations sampled at regular intervals and their mirrored copies.
原: It is discretized into each map location and $K$ rotations sampled at regular intervals.
构: It is discretized into A and K rotations sampled at regular intervals.
```
**B6** · OrienterNet（CVPR 2023）
```text
新: This verification benefits from a simple implementation by rotating the nadir view $K$ times and summing, after undoing each rotation, the von Mises natural parameters predicted by a second heading network.
原: This formulation benefits from an efficient implementation by rotating $T$ $K$ times and performing a single convolution as a batched multiplication in the Fourier domain [cite].
构: This X benefits from a(n) [adj.] implementation by rotating A K times and [gerund] B ...
```
**B7** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: The estimate of the absolute heading obtained from the nadir views is then combined with the estimate from the coarse stage by summing their natural parameters, resulting in a more accurate estimate $\psi_S^\star$ of the source heading.
原: The estimate of the global pose obtained from the gate detector is then combined with the estimate from the visual–inertial estimator by means of a Kalman filter, resulting in a more accurate representation of the robot’s state.
构: The estimate of the A obtained from the B is then combined with the estimate from the C by D, resulting in a more accurate E of the F.
```
**B8** · LoFTR（CVPR 2021）
```text
新: The final prediction consists of the refined relative pose and the displacement rotated to the world frame with Eq.~\eqref{eq:world}: $\mathbf{p}^\star=(\Delta x^\star,\Delta y^\star,\Delta z^\star,\Delta\phi^\star,\Delta\theta^\star)$.
原: The final loss consists of the losses for the coarse-level and the fine-level: $\mathcal{L} = \mathcal{L}_{c} + \mathcal{L}_{f}$.
构: The final X consists of the A for/of the B and the C: [equation].
```

### Short-range mode for closed-loop flight（2026-11-17 补充）

这一段是为闭环实验补写的，共 10 句（`where …` 从句单独算一句），加在 `method.tex` 末尾；上文的句数统计（104 句）不含这 10 句。原句同样在 arXiv LaTeX 源码中逐字匹配，并与 CVF 正式版 PDF 比对一致。内容依据 `online/plane_servo.py` 和 `online/sim_nav.py`。

**X1** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Estimating the translation near the goal is very challenging due to the lack of parallax, the degeneracy of the essential matrix, the bias of the fused distance at such baselines, and the scarcity of short baselines in the training pairs.
原: Indoor image matching is very challenging due to the lack of texture, the abundance of self-similarities, the complex 3D geometry of scenes, and large viewpoint changes.
构: A is very challenging due to the lack of B, the C of D, the E ..., and F.
```
**X2** · SuperGlue（CVPR 2020 (Oral)）
```text
新: As we show in the following, the imagined matches can effectively constrain the translation to overcome these challenges.
原: As we show in the following, SuperGlue can effectively learn priors to overcome these challenges.
构: As we show in the following, X can effectively [verb] A to overcome these challenges.
```
**X3** · OrienterNet（CVPR 2023）
```text
新: We then obtain the constraint on the scaled translation $\mathbf{w}=\mathbf{t}/h$ for each imagined match $i$ as
原: We then obtain the distribution $\alpha_{u,d}$ for each depth bin $d$ as
构: We then obtain the A for each B as [equation]
```
**X4** · OrienterNet（CVPR 2023）
```text
新: where $\bar{\mathbf{x}}=\mathbf{K}^{-1}\tilde{\mathbf{x}}$ denotes the normalized coordinates, with the rotation $\mathbf{R}$ fixed to its refined estimate.
原: where $[\cdot]$ denotes the linear interpolation.
构: where A denotes the B.
```
**X5** · SuperGlue（CVPR 2020 (Oral)）
```text
新: This is equivalent to solving a linear least-squares problem in $\mathbf{w}$, which remains well conditioned at zero baseline.
原: This is equivalent to solving a linear assignment problem.
构: This is equivalent to solving a linear A problem.
```
**X6** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Translation estimation is performed with the Direct Linear Transformation~\citep{hartley2003multiple} (DLT), which has a direct least-squares solution.
原: Homography estimation is performed with both RANSAC and the Direct Linear Transformation [cite] (DLT), which has a direct least-squares solution.
构: A estimation is performed with ... the Direct Linear Transformation (DLT), which has a direct least-squares solution.
```
**X7** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: The robust solution of Eq.~\eqref{eq:servo} is computed with iteratively reweighted least squares~\citep{holland1977robust}.
原: The robust mean in Eq. [ref] is computed with iteratively reweighted least squares [cite].
构: The robust A in/of Eq. is computed with iteratively reweighted least squares.
```
**X8** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: This ensures robustness to mismatches and accounts for the residual parallax of elevated structures.
原: This ensures robustness to outlier observations and accounts for the unknown topology of the feature space.
构: This ensures robustness to A and accounts for the B of the C.
```
**X9** · PixLoc（CVPR 2021）
```text
新: To prevent this estimate from replacing the fusion where the latter is reliable, we apply it at a given step only if the refined distance is below 5\,m and at least 200 matches agree with the plane.
原: To prevent hard examples from smoothing the fine features, we apply the loss at a given level only if the previous one succeeded in bringing the pose sufficiently close to the ground truth.
构: To prevent A from [gerund] B, we apply the C at a given D only if E.
```
**X10** · OrienterNet（CVPR 2023）
```text
新: The short-range mode does not require any additional model but only the camera height $h$ of Eq.~\eqref{eq:homography}, which converts $\mathbf{w}$ into a metric translation.
原: OrienterNet does not require such initialization but only a coarse position-only prior.
构: X does not require A but only a B.
```
