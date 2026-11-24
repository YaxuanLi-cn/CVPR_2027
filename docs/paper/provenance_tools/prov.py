# -*- coding: utf-8 -*-
"""Sentence-level provenance of experiment.tex: every sentence ('mine') with the published sentence ('src') whose structure it follows."""
PAPERS = {
    'superglue':   ('SuperGlue: Learning Feature Matching with Graph Neural Networks', 'Sarlin et al.', 'CVPR 2020 (Oral)', 'arXiv:1911.11763'),
    'loftr':       ('LoFTR: Detector-Free Local Feature Matching with Transformers', 'Sun et al.', 'CVPR 2021', 'arXiv:2104.00680'),
    'pixloc':      ('Back to the Feature: Learning Robust Camera Localization from Pixels to Pose', 'Sarlin et al.', 'CVPR 2021', 'arXiv:2103.09213'),
    'orienternet': ('OrienterNet: Visual Localization in 2D Public Maps with Neural Matching', 'Sarlin et al.', 'CVPR 2023', 'arXiv:2304.02009'),
    'pixsfm':      ('Pixel-Perfect Structure-from-Motion with Featuremetric Refinement', 'Lindenberger et al.', 'ICCV 2021 (Oral)', 'arXiv:2108.08291'),
    'dust3r':      ('DUSt3R: Geometric 3D Vision Made Easy', 'Wang et al.', 'CVPR 2024', 'arXiv:2312.14132'),
    'nwm':         ('Navigation World Models', 'Bar et al.', 'CVPR 2025', 'arXiv:2412.03572'),
    'reloc3r':     ('Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization', 'Dong et al.', 'CVPR 2025', 'arXiv:2412.08376'),
    'lightglue':   ('LightGlue: Local Feature Matching at Light Speed', 'Lindenberger et al.', 'ICCV 2023', 'arXiv:2306.13643'),
    'kendall_unc': ('What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?', 'Kendall and Gal', 'NIPS 2017', 'arXiv:1703.04977'),
    'nature_drone': ('Champion-level drone racing using deep reinforcement learning', 'Kaufmann et al.', 'Nature 620, 2023', 'nature.com/articles/s41586-023-06419-4'),
}
E = []
def add(i, para, mine, key, src, skel, pending=False): E.append(dict(id=i, para=para, mine=mine, key=key, src=src, skel=skel, pending=pending))

# ================================================================ opening
P = 'Opening'
add('E1', P, r'''We evaluate our single model for image-goal navigation in the context of both offline pose estimation and closed-loop flight.''', 'orienternet',
    r'''We evaluate our single model for localization in the context of both driving and AR.''',
    'We evaluate our single model for X in the context of both A and B.')
add('E2', P, r'''Figure~\ref{fig:qualitative} shows qualitative examples, while Figure~\ref{fig:breakdown} illustrates the effectiveness of the imagination-refined stage.''', 'orienternet',
    r'''\Cref{fig:qualitative-main} shows qualitative examples, while \cref{fig:qualitative-sequence} illustrates the effectiveness of multi-frame fusion.''',
    'Fig. A shows qualitative examples, while Fig. B illustrates the effectiveness of X.')
add('E3', P, r'''Our experiments show that: 1) DreamNav is more accurate than existing navigation policies, world models and visual representations for pose estimation between aerial views; 2) Geometric imagination helps estimate every pose parameter more accurately than the coarse stage alone; 3) DreamNav is significantly more accurate than a regression baseline on real drone photographs when considering baselines within the training range.''', 'orienternet',
    r'''Our experiments show that:
1) OrienterNet is more effective than existing deep networks for localization with 2D maps;
2) Planimetric maps help localize more accurately than overhead satellite imagery;
3) OrienterNet is significantly more accurate than an embedded consumer-grade GPS sensor when considering multiple views.''',
    'Our experiments show that: 1) X is more [adj] than existing A for B; 2) C help [verb] more accurately than D; 3) X is significantly more accurate than E when considering F.')

# ================================================================ setup
P = 'Experimental setup: dataset'
add('D1', P, r'''We use PairUAV (Section~\ref{PairUAV}), a large-scale aerial dataset composed of 1,412 sites with ground truth poses for every image, and well-defined training and test splits corresponding to different sites.''', 'superglue',
    r'''We use ScanNet~\cite{dai2017scannet}, a large-scale indoor dataset composed of monocular sequences with ground truth poses and depth images, and well-defined training, validation, and test splits corresponding to different scenes.''',
    'We use X, a large-scale [adj] dataset composed of A with ground truth B, and well-defined training ... and test splits corresponding to different scenes.')
add('D2', P, r'''Each site contains 150 images with headings over the full circle and tilts of up to $45^\circ$ from nadir.''', 'loftr',
    r'''ScanNet contains 1613 monocular sequences with ground truth poses and depth maps.''',
    'X contains N [items] with A and B.')
add('D3', P, r'''Holding out 290 sites for testing, we select 24,909 training and 6,364 test pairs.''', 'superglue',
    r'''Discarding pairs with too small or too large overlap, we select 230M training and 1500 test pairs.''',
    '[Gerund] A, we select N training and M test pairs.')
add('D4', P, r'''This dataset contains image pairs with baselines of up to 38\,m and arbitrary heading changes.''', 'loftr',
    r'''This dataset contains image pairs with wide baselines and extensive texture-less regions.''',
    'This dataset contains image pairs with A and B.')

P = 'Experimental setup: metrics'
add('M1', P, r'''We report for each pose parameter the mean absolute error (MAE), in metres or degrees, as well as the success rate (SR) at 1\,m for the distance.''', 'pixloc',
    r'''We report for each scene the median translation (cm) and rotation (\degree) errors~\cite{kendall2015posenet}, as well as the average localization recall at (5cm, 5\degree) for 7Scenes~\cite{shotton2013scene}.''',
    'We report for each A the [statistic] B errors, as well as the C at [thresholds] for D.')
add('M2', P, r'''The position error measures the distance between the predicted and the true goal position when executing the estimated displacement from the source pose.''', 'nature_drone',
    r'''The gate margin metric measures the distance between the drone and the closest point on the gate when crossing the gate plane.''',
    'The X measures the distance between A and B when [gerund] C.')
add('M3', P, r'''Distance errors are obtained from the norm of the estimated displacement.''', 'superglue',
    r'''Relative poses are obtained from essential matrix estimation with RANSAC.''',
    'A are obtained from B.')
add('M3b', P, r'''We also report the 3-D position error and the SR, where a pair is deemed successful based on its distance error.''', 'superglue',
    r'''We also report the match precision and the matching score~\cite{superpoint, yi2016lift}, where a match is deemed correct based on its epipolar distance.''',
    'We also report the A and the B, where a C is deemed correct based on its D.')
add('M4', P, r'''As in previous work~\citep{sarlin2020superglue,sun2021loftr}, we report the AUC of the pose error at the thresholds ($5^\circ$, $10^\circ$, $20^\circ$), where the pose error is the maximum of the angular errors in heading and tilt.''', 'superglue',
    r'''As in previous work~\cite{moo2018learning, zhang2019learning, brachmann2019neural}, we report the AUC of the pose error at the thresholds (5\degree$,\, $10\degree$,\, $20\degree), where the pose error is the maximum of the angular errors in rotation and translation.''',
    'As in previous work, we report the AUC of the pose error at the thresholds (...), where the pose error is the maximum of the angular errors in A and B.')

P = 'Experimental setup: baselines'
add('B1', P, r'''We compare DreamNav with four categories of methods: 1) visuomotor policies including AI2-THOR~\citep{7989381}, $\pi_{0.5}$~\citep{intelligence2025pi_}, $\pi_0$-FAST~\citep{pertsch2025fast}, ViNT~\citep{shah2023vint}, and NoMaD~\citep{sridhar2024nomad}, 2) world models that plan by imagination, i.e., DINO-WM~\citep{zhou2025dinowm} and Visual Foresight~\citep{ebert2018visual}, 3) visual representations including DINOv3~\citep{simeoni2025dinov3} and Sample4Geo~\citep{deuser2023sample4geo}, and 4) a regression on pooled MASt3R statistics, i.e., RASR~\citep{liang2026rasr}.''', 'loftr',
    r'''We compare LoFTR with three categories of methods:
1) detector-based local features including R2D2~\cite{revaudR2D2RepeatableReliable}, D2Net~\cite{dusmanuD2NetTrainableCNN2019},
and DISK~\cite{tyszkiewiczDISKLearningLocal2020},
2) a detector-based local feature matcher, i.e., SuperGlue~\cite{sarlinSuperGlueLearningFeature2019} on top of SuperPoint~\cite{detoneSuperPointSelfSupervisedInterest2018} features, and
3) detector-free matchers including Sparse-NCNet~\cite{roccoEfficientNeighbourhoodConsensus2020} and DRC-Net~\cite{liDualResolutionCorrespondenceNetworks2020}.''',
    'We compare X with N categories of methods: 1) A including a1, a2, and a3, 2) a B, i.e., b1 ..., and 3) C including c1 and c2.')
add('B2', P, r'''All baselines, including RASR, are trained on the training sites of the same split, supervised by the five pose parameters of each pair.''', 'superglue',
    r'''All learned methods, including SuperGlue, are trained on ground truth correspondences, found by projecting keypoints from one image to the other.''',
    'All A, including X, are trained on B, [participle clause].')
add('B3', P, r'''Other predictors are evaluated as reference: a constant output of zero motion, and RASR with its heading regressed as a sine--cosine pair using the same features.''', 'superglue',
    r'''Other local features are evaluated as reference: ORB~\cite{rublee2011orb} with GMS, D2-Net~\cite{dusmanu2019d2}, and ContextDesc~\cite{luo2019contextdesc} using the publicly available trained models.''',
    'Other A are evaluated as reference: a1 ..., and a2 using the B.')

P = 'Experimental setup: implementation'
add('I0', P, r'''We initialize the baselines with their official pre-trained weights, when available, before training on PairUAV.''', 'reloc3r',
    r'''We initialize Reloc3r with DUSt3R's pre-trained 512-DPT weights.''',
    "We initialize X with Y's pre-trained weights.")
add('I1', P, r'''All experiments are conducted on a single NVIDIA RTX 5090 GPU.''', 'reloc3r',
    r'''All evaluations are conducted on a 24GB NVIDIA GeForce RTX 4090 GPU and we use mixed precision fp16 / fp32 to improve speed and memory efficiency without sacrificing accuracy.''',
    'All [evaluations] are conducted on a [N] NVIDIA X GPU (first clause of the source).')
add('I3', P, r'''More details are in Appendix~I.''', 'pixloc',
    r'''More details are in the \supp.''',
    'More details are in the X.')

# ================================================================ main results
P = 'Comparison with baselines'
add('R1', P, r'''As shown in Table~\ref{tab:main_result}, DreamNav outperforms all baselines, on all pose parameters, when using either the coarse stage alone or both stages.''', 'superglue',
    r'''As shown in Table~\ref{tab:phototourism}, SuperGlue outperforms all baselines, at all relative pose thresholds, when applied to both SuperPoint and SIFT.''',
    'As shown in Table, X outperforms all baselines, at all A, when [participle] both B and C.')
add('R2', P, r'''Most notably, the success rate of the resulting navigation is very high (83.3\%), reinforcing the view that the last meter is within reach of a monocular camera.''', 'superglue',
    r'''Most notably, the precision of the resulting matching is very high (84.9\%), reinforcing the analogy that SuperGlue ``glues'' together local features.''',
    'Most notably, the A of the resulting B is (very) high (N%), reinforcing the C that D.')
add('R3', P, r'''Regression-based methods like $\pi_{0.5}$ and DINOv3 cannot predict a substantially more accurate horizontal displacement than the constant predictor itself, lacking any cue on the absolute heading (see Table~\ref{tab:main_result} and Figure~\ref{fig:cdf}).''', 'superglue',
    r'''Outlier rejection methods like PointCN and OANet cannot predict more correct matches than the NN matcher itself, overly relying on the initial descriptors (see Figure~\ref{fig:qual-all} and \supp~\ref{sec:results-supp}).''',
    'A methods like a1 and a2 cannot predict more [adj] B than the C itself, [participle clause] (see Fig. and ...).')
add('R4', P, r'''Direct regression disregards any knowledge of projective geometry and performs mere interpolation between training pairs without any geometric constraint.''', 'orienternet',
    r'''a)~Map retrieval disregards any knowledge of projective geometry and performs mere recognition without any geometric constraint.''',
    'X disregards any knowledge of projective geometry and performs mere Y without any geometric constraint.')
add('R5', P, r'''DreamNav outperforms RASR by a large margin (83.3\% against 11.7\% at SR@1\,m), which demonstrates the effectiveness of the correspondence-based design.''', 'loftr',
    r'''LoFTR outperforms SuperGlue by a large margin (13\% at AUC@10\degree), which demonstrates the effectiveness of the detector-free design.''',
    'X outperforms Y by a large margin (N% at METRIC), which demonstrates the effectiveness of the Z design.')
add('R6', P, r'''The imagination-refined stage brings consistent improvement on PairUAV, especially in the tilt change, with up to $-$54.6\% error.''', 'pixloc',
    r'''PixLoc brings consistent improvement on CMU, especially in the fine threshold, with up to +2.4\% recall.''',
    'X brings consistent improvement on D, especially in the A, with up to N% B.')
add('R7', P, r'''At all error thresholds up to the 99th percentile, the refined estimate is consistently more accurate than its coarse counterpart.''', 'pixsfm',
    r'''At all error thresholds, featuremetric optimization is consistently more accurate than its geometric counterparts.''',
    'At all error thresholds, A is consistently more accurate than its B counterpart(s).')

# ================================================================ ablation
P = 'Ablation study'
add('A1', P, r'''To evaluate our design decisions, we repeat the experiments on the same test pairs, but this time focusing on different DreamNav variants.''', 'superglue',
    r'''To evaluate our design decisions, we repeat the indoor experiments with SuperPoint features, but this time focusing on different SuperGlue variants.''',
    'To evaluate our design decisions, we repeat the A experiments with/on B, but this time focusing on different X variants.')
add('A2', P, r'''This ablation study, presented in Table~\ref{tab:ablation}, shows that all DreamNav components are useful and bring performance gains.''', 'superglue',
    r'''This ablation study, presented in Table~\ref{tab:ablation}, shows that all SuperGlue blocks are useful and bring substantial performance gains.''',
    'This ablation study, presented in Table, shows that all X blocks are useful and bring (substantial) performance gains.')
add('A3', P, r'''When we regress the sine and cosine of the heading change with an MLP while keeping the features of RASR, we observe an improvement in $\mathrm{MAE}_\theta$ from 5.70$^\circ$ to 1.82$^\circ$.''', 'superglue',
    r'''When we additionally backpropagate through the SuperPoint descriptor network while training SuperGlue, we observe an improvement in AUC@20\degree from 51.84 to 53.38.''',
    'When we [verb] A while [gerund] B, we observe an improvement in METRIC from N1 to N2.')
add('A4', P, r'''We find that running the original matcher leads to a valid relative pose for only 47\% of the pairs, while searching over four rotations leads to a valid pose for 99.3\% of them.''', 'nwm',
    r'''We find that running the model with time only leads to poor performance, while not conditioning on time leads to small drop in performance as well.''',
    'We find that running A leads to B, while [gerund] C leads to D.')
add('A5', P, r'''Without the rotation search, the matcher loses its ability to match pairs whose heading change exceeds $60^\circ$, as shown in Appendix~I.''', 'lightglue',
    r'''Without the matchability classifier, the network loses its ability to discriminate between good and bad matches, as shown in Figure~\ref{fig:matchability}.''',
    'Without the A, the B loses its ability to [verb] C, as shown in Fig.')
add('A6', P, r'''World-frame displacements stay at the level of the constant predictor when the absolute heading is unknown.''', 'orienternet',
    r'''b)~Featuremetric refinement converges to incorrect locations when the initial pose is inaccurate.''',
    'A [verb] to/at B when the C is [adj].')
add('A7', P, r'''Zoom re-matching is particularly effective in weakly matched pairs, as full-image matching cannot fully exploit the co-visible region due to the fixed budget of keypoints.''', 'pixsfm',
    r'''Featuremetric optimization is particularly effective in this situation, as geometric optimization cannot fully suppress the detection noise due to the small number of observations.''',
    'A is particularly effective in B, as C cannot fully [verb] D due to the E.')
add('A8', P, r'''Verifying the heading in the canonical view only marginally improves the performance, confirming that the remaining heading error can be sufficiently systematic within a site for a more exact input representation to leave it unchanged.''', 'pixloc',
    r'''Localizing with the oracle prior only marginally improves the performance, confirming that image retrieval can be sufficiently accurate for the pose optimization to converge to the correct minimum.''',
    '[Gerund] A only marginally improves the performance, confirming that B can be sufficiently [adj] for C to [verb] D.')
add('A9', P, r'''Using the true absolute heading halves the horizontal errors, and makes DreamNav accurate to half a metre in both $\Delta x$ and $\Delta y$.''', 'pixloc',
    r'''Using the oracle prior partially bridges the gap, and makes PixLoc competitive on driving datasets like CMU and RobotCar.''',
    'Using the oracle A [verb] the B, and makes X [predicate].')

P = 'Ablation of the imagination-refined stage'
add('G1', P, r'''We render imagined views while varying the number of tiles per pair from $1$ to $3{\times}3$ (see Table~\ref{tab:imag}a).''', 'nwm',
    r'''We train models while varying the number of conditioning frames from $1$ to $4$ (see Table~\ref{tab:ablations}).''',
    'We [verb] A while varying the number of B from N1 to N2 (see Table).')
add('G2', P, r'''Unsurprisingly, more tiles help, and with a single tile the matcher often ``runs out'' of keypoints, leading to five times fewer inliers.''', 'nwm',
    r'''Unsurprisingly, more context helps, and with short context the model often ``lose track'', leading to poor predictions.''',
    'Unsurprisingly, more A helps, and with [little] A the B often ``...\'\', leading to C.')
add('G3', P, r'''Re-aligning with the coarse pose, replacing the rotation-only warp by the plane-induced homography, and matching tiles at the native resolution all contribute to the performance of the refinement over the zoom pass of the coarse stage.''', 'pixloc',
    r'''Unrolling the optimizer and learning features, damping factor, and confidences all contribute to the performance of PixLoc over classical photometric alignment.''',
    '[Gerund] A, B, and C all contribute to the performance of X over Y.')
add('G4', P, r'''This improvement is not brought by the additional layer of regressors.''', 'pixloc',
    r'''This improvement is not brought by the significantly less accurate image retrieval.''',
    'This improvement is not brought by the A.')
add('G5', P, r'''The findings of our control reveal that the same regressors without the imagined matches result in no gain over the coarse stage and a slight increase in error (Table~\ref{tab:imag}b).''', 'nature_drone',
    r'''The findings of our analysis reveal that subsequent fine-tuning operations result in negligible enhancements in performance and alterations in behaviour (Extended Data Fig. 2 ).''',
    'The findings of our A reveal that B result in [negligible] C and D (Fig./Table).')
add('G6', P, r'''Results (Table~\ref{tab:imag}b) show DreamNav refines effectively without the verification scores, with only minor performance drops (0.455\,m to 0.468\,m in $\ell_x$).''', 'nwm',
    r'''Results (Table~\ref{tab:plan_constraints}) show NWM plans effectively under constraints, with only minor performance drops (see examples in Figure~\ref{fig:plan_constraints}).''',
    'Results (Table) show X [verb] effectively under/without A, with only minor performance drops (...).')
add('G7', P, r'''We investigate the extent of variations in accuracy across iterations.''', 'nature_drone',
    r'''We investigate the extent of variations in behaviour across iterations.''',
    'We investigate the extent of variations in A across iterations.')
add('G8', P, r'''We observe that the displacement between imagined and real views, which is a label-free measure of the accuracy of the guiding pose, is 7.6 pixels after the coarse stage and 5.0 pixels after one refinement.''', 'nature_drone',
    r'''We observe that the difference in distance from gate centres, which is a metric for the safety of the policy, is 0.09 ± 0.08 m.''',
    'We observe that the A, which is a metric for the B of the C, is N.')
add('G9', P, r'''Note that the gain of a second round in the median translation-direction error (0.03$^\circ$) is substantially smaller than the difference between the errors of the zoom pass and the first round (0.75$^\circ$).''', 'nature_drone',
    r'''Note that this lap-time difference is substantially smaller than the difference between the single-lap completion times of Swift and the human pilots (0.16 s).''',
    'Note that this A is substantially smaller than the difference between the B of C and D (N).')
add('G10', P, r'''Diffusion retrieval replaces the plane-induced warp and matching by a comparison of the target view and of the views that a diffusion model synthesizes for candidate poses around the coarse estimate.''', 'orienternet',
    r'''a)~\underline{Map retrieval}~\cite{xia2022visual} replaces the BEV inference and matching by a correlation of the neural map and with a global image embedding.''',
    'X retrieval replaces the A and matching by a [comparison] of the B and (with) C.')
add('G11', P, r'''Replacing the geometric imagination by diffusion retrieval on top of the same coarse stage results in a significant increase in error for every pose parameter, up to 0.43$^\circ$ from 0.12$^\circ$ in $\mathrm{MAE}_\theta$ (Table~\ref{tab:imag}c).''', 'loftr',
    r'''1) Replacing the LoFTR module by convolution with a comparable number of parameters results in a significant drop in AUC as expected.''',
    'Replacing the A by B with C results in a significant [drop/increase] in D.')
add('G12', P, r'''We attribute the degradation to the use of synthesized views and the retrieval among candidate poses, giving up pixel-level geometry and tying the estimate to the sampled candidates.''', 'loftr',
    r'''We attribute the prominence to the use of the Transformer and the optimal transport layer, taking advantage of global information and jointly bringing global consensus into the final matches.''',
    'We attribute the A to the use of the B and the C, [gerund] D and [gerund] E.')

# ================================================================ analysis
P = 'Analysis: where the refinement helps'
add('N1', P, r'''Figure~\ref{fig:breakdown} studies the effectiveness of the two stages -- coarse and refined -- on the test pairs for different ranges of inlier count, rotation and distance.''', 'lightglue',
    r'''Table~\ref{tab:adaptive} studies the effectiveness of the two pruning mechanisms -- adaptive depth and width -- on MegaDepth image pairs for different ranges of visual overlap.''',
    'Table/Fig. studies the effectiveness of the two A -- a1 and a2 -- on B pairs for different ranges of C.')
add('N2', P, r'''The improvements are consistent across all bins, both in rotation and in distance.''', 'pixsfm',
    r'''The improvements are consistent across all local features, both indoors and outdoors.''',
    'The improvements are consistent across all A, both B and C.')
add('N3', P, r'''For moderately difficult samples, such as the pairs with 100 to 200 coarse inliers, the refinement recovers many more correspondences and nearly halves the position error, resulting in a drop from 3.98\,m to 2.21\,m.''', 'lightglue',
    r'''For easy samples, such as the successive frames of a video,
the network quickly converges and exits after a few layers, resulting in a 1.86$\times$ speedup.''',
    'For [adj] samples, such as the A, the B [verb] and [verb] ..., resulting in a N [change].')
add('N4', P, r'''In cases of low visual overlap, e.g. fewer than 100 inliers, the refinement requires more correspondences to succeed.''', 'lightglue',
    r'''In cases of low visual overlap, e.g. loop closure, the network requires more layers to converge.''',
    'In cases of low visual overlap, e.g. A, the B requires more C to [verb].')
add('N5', P, r'''After evaluation, we resample the 290 test sites 1,000 times to report 95\% confidence intervals.''', 'nwm',
    r'''After training, we sample $5$ times from each model to report mean and std results.''',
    'After A, we sample N times from B to report C.')
add('N6', P, r'''The refined stage more consistently reaches the goal, exhibiting a higher success rate by 10.2 points (interval of 9.1 to 11.3) and a lower mean position error by 0.68\,m (0.61 to 0.74).''', 'nature_drone',
    r'''The autonomous drone more consistently pushes for fast lap times, exhibiting lower mean and variance.''',
    'The A more consistently [verb] B, exhibiting higher/lower C and D.')

P = 'Analysis: reliability of the predicted error'
add('V0', P, r'''In this section we wish to study the effectiveness of predicting the error of every candidate estimate.''', 'kendall_unc',
    r'''In this section we wish to study the effectiveness of modeling aleatoric and epistemic uncertainty.''',
    'In this section we wish to study the effectiveness of [gerund] A.')
add('V1', P, r'''In particular, we wish to quantify the performance of the error predictions that weight our fusion and analyze what they capture.''', 'kendall_unc',
    r'''In particular, we wish to quantify the performance of these uncertainty measurements and analyze what they capture.''',
    'In particular, we wish to quantify the performance of these A and analyze what they capture.')
add('V2', P, r'''Firstly, in Figure~\ref{fig:reliability} (left) we show sparsification curves for four pose parameters.''', 'kendall_unc',
    r'''Firstly, in Figure \ref{fig:prec_recall} we show precision-recall curves for regression and classification models.''',
    'Firstly, in Figure we show A curves for B.')
add('V3', P, r'''They show how our model performance improves by removing pairs with predicted error larger than various percentile thresholds.''', 'kendall_unc',
    r'''They show how our model performance improves by removing pixels with uncertainty larger than various percentile thresholds.''',
    'They show how our model performance improves by removing A with B larger than various percentile thresholds.')
add('V4', P, r'''Keeping the most confident half reduces the mean error of the tilt change from 0.20$^\circ$ to 0.02$^\circ$ for the test pairs.''', 'pixsfm',
    r'''Precomputing distance maps reduces the peak memory requirement of the bundle adjustment from 80 GB to less than 10GB for 1000 images.''',
    '[Gerund] A reduces the B of the C from N1 to N2 for D.')
add('V5', P, r'''Secondly, in Figure~\ref{fig:reliability} (right) we analyze the quality of our error prediction using calibration plots from our model on the test set.''', 'kendall_unc',
    r'''Secondly, in Figure \ref{fig:calibrationplot} we analyze the quality of our uncertainty measurement using calibration plots from our model on the test set.''',
    'Secondly, in Figure we analyze the quality of our A using calibration plots from our model on the test set.')
add('V6', P, r'''Better performing error predictions should correlate more accurately with the line $y=x$ in the calibration plots.''', 'kendall_unc',
    r'''Better performing uncertainty estimates should correlate more accurately with the line $y=x$ in the calibration plots.''',
    'Better performing A should correlate more accurately with the line y=x in the calibration plots.')
add('V7', P, r'''The predicted error is however lower than the actual one by a nearly constant factor of 1.5 to 2, which is mostly due to the square-root regression target, as our regressors cannot recover the mean of a skewed distribution if the fit is performed after such a concave transform.''', 'pixloc',
    r'''PixLoc is however less robust than the feature matching pipelines, which is mostly due to the naive pose prior, as our algorithm cannot converge if the retrieval returns the incorrect location.''',
    'X is however less/lower [adj] than Y, which is mostly due to the A, as our B cannot [verb] if C.')

P = 'Analysis: limitations'
add('L1', P, r'''Estimating the pose of a pair is challenging when the two views barely overlap or when the scene lacks cues on the absolute heading.''', 'orienternet',
    r'''Localizing an image or a sequence is challenging when the environments lacks distinctive elements or when they are not registered in the map.''',
    '[Gerund] A is challenging when B or when C.')
add('L2', P, r'''Out of the 2,422 position errors above 1\,m recorded for DreamNav, 44\% were because of an error in the absolute heading alone, 17\% because of fewer than 200 coarse inliers and 39\% because of the remaining errors in scale and translation direction.''', 'nature_drone',
    r'''Out of the 10 losses recorded for Swift, 40% were because of a collision with the opponent, 40% because of collision with a gate and 20% because of the drone being slower than the human pilot.''',
    'Out of the N losses recorded for X, a% were because of A, b% because of B and c% because of C.')
add('L3', P, r'''DreamNav can also fail for sites in the southern hemisphere due to their reversed shadows and is more sensitive to changes of the flight altitude.''', 'pixloc',
    r'''PixLoc can also fail for large outliers ratios due prominent occluders and is more sensitive to camera miscalibration.''',
    'X can also fail for A due (to) B and is more sensitive to C.')

# ================================================================ real data
P = 'Zero-shot transfer to real drone photographs'
add('T1', P, r'''While the previous experiments precisely quantify the accuracy of DreamNav, they do not contain any variations of sensor, altitude or image formation.''', 'pixsfm',
    r'''While the previous experiments precisely quantify the accuracy of the refinement, they do not contain any variations of appearance or camera models.''',
    'While the previous experiments precisely quantify the accuracy of X, they do not contain any variations of A or B.')
add('T2', P, r'''We thus turn to real drone photographs and evaluate the benefits of our geometric imagination in a zero-shot setting.''', 'pixsfm',
    r'''We thus turn to crowd-sourced imagery and evaluate the benefits of our featuremetric optimization in an end-to-end reconstruction pipeline.''',
    'We thus turn to A and evaluate the benefits of our B in a(n) C.')
add('T3', P, r'''We consider the pose estimation in real flights with the M\'ezi\`eres dataset~\citep{heigvd_mezieres}, following the metrics of Section~\ref{sec:setup}.''', 'orienternet',
    r'''We consider the localization in driving scenarios with the KITTI dataset~\cite{geiger2013vision},
following the closest existing setup~\cite{shi2022beyond}.''',
    'We consider the A in B scenarios with the X dataset, following the C.')
add('T4', P, r'''Images are captured by the camera of a DJI Phantom 4 RTK flying 60\,m above a village and have GT poses from RTK and bundle adjustment.''', 'orienternet',
    r'''Images are captured by cameras mounted on a car driving in urban and residential areas and have GT poses from RTK.''',
    'Images are captured by cameras [of/on] a A [gerund] in/above B and have GT poses from RTK.')
add('T5', P, r'''We only use the pairs of photographs whose cameras are at most 30\,m apart, from which we sample 600 pairs for evaluation.''', 'loftr',
    r'''We follow DISK~\cite{tyszkiewiczDISKLearningLocal2020} to only use the scenes of ``Sacre Coeur'' and ``St. Peter's Square'' for validation, from which we sample 1500 pairs for a fair comparison.''',
    'We ... only use the A of B ..., from which we sample N pairs for C.')
add('T6', P, r'''The pairs were sampled in a way that the difficulty is balanced based on the baseline-to-height ratio.''', 'lightglue',
    r'''The data was collected in a way that the difficulty is balanced based on visual overlap.''',
    'The A was collected/sampled in a way that the difficulty is balanced based on B.')
add('T7', P, r'''We undistort the photographs and resample them such that their intrinsics match the calibration $\mathbf{K}$ of PairUAV.''', 'orienternet',
    r'''We render $\SI{64}{\meter}{\times}\SI{64}{\meter}$ map tiles and resize the images such that their focal length matches the median of MGL.''',
    'We [verb] A and resize/resample the images such that their B matches the C of D.')
add('T8', P, r'''We emphasize that all results are obtained with the \emph{same} DreamNav model (the only site-specific input is the flight height reported by the aircraft, by which the estimated displacement is rescaled), \ie we never finetune our model on real photographs.''', 'dust3r',
    r'''We emphasize that all results are obtained with the \emph{same} \duster{} model (our default model is denoted as `\duster{} 512', other \duster{} models serves for the ablations in Section~\cref{ssec:ablations}), \ie we never finetune our model on a particular downstream task.''',
    'We emphasize that all results are obtained with the same X model (...), i.e. we never finetune our model on A.')
add('T9', P, r'''We compute the position error in the levelled frame of the source camera since the absolute heading is not transferable from rendered to real shadows.''', 'orienternet',
    r'''We compute the position error along directions perpendicular (lateral) and parallel (longitudinal) to the viewing axis~\cite{shi2022beyond} since the pose is generally less constrained along the road.''',
    'We compute the position error along/in A since the B is [adj] C.')
add('T10', P, r'''Table~\ref{tab:real} shows that DreamNav outperforms all regression baselines in both ranges of baseline-to-height ratio, in all metrics.''', 'orienternet',
    r'''\Cref{tab:eval-kitti} (a-b) shows that OrienterNet outperforms all existing approaches based on both satellite and OSM maps, in all metrics.''',
    'Table shows that X outperforms all A ... both B and C, in all metrics.')
add('T10b', P, r'''However, no improvement can be observed on the tilt change, while the constant predictor is more accurate there since the gimbal barely moves between these photographs.''', 'pixloc',
    r'''However, no improvement can be observed on RobotCar Day, while the refinement is detrimental on Aachen at 0.25m.''',
    'However, no improvement can be observed on A, while the B is [adj] on/there C.')
add('T11', P, r'''It is accurate within the training range, despite being trained on rendered images only.''', 'pixloc',
    r'''It is competitive for indoor scenes, despite being trained on outdoor Internet data only.''',
    'It is [adj] for/within A, despite being trained on B only.')
add('T12', P, r'''The refinement also increases the success rate at all thresholds on real photographs, which exhibit stronger parallax, a difficult condition for a plane-induced warp.''', 'pixloc',
    r'''It also increases the pose accuracy at all thresholds on RobotCar Night, which exhibits significant motion blur, a difficult condition for sparse keypoint detection.''',
    'It also increases the A at all thresholds on B, which exhibits C, a difficult condition for D.')
add('T13', P, r'''This confirms that a plane-induced warp is all we need for the imagination to pay off and that it generalizes well despite a fusion fitted on rendered pairs.''', 'pixloc',
    r'''This confirms that deep features are all we need for accurate localization and that they generalize well despite end-to-end training.''',
    'This confirms that A are all we need for B and that they generalize well despite C.')
add('T14', P, r'''The performance is however significantly lower beyond the training range (5.0\% SR), which highlights the difficulty of wide baselines at low altitude and the need for further research.''', 'orienternet',
    r'''The performance is however significantly lower than with driving data (\cref{sec:driving}), which highlights the difficulty of AR-like conditions and the need for further research.''',
    'The performance is however significantly lower than/beyond A (...), which highlights the difficulty of B and the need for further research.')
add('T15', P, r'''By contrast with its relative geometry, DreamNav's scale recovery assumes that the flight altitude is consistent with what was observed during training.''', 'nature_drone',
    r'''By contrast, Swift’s perception system assumes that the appearance of the environment is consistent with what was observed during training.''',
    'By contrast, X’s A assumes that the B is consistent with what was observed during training.')
add('T16', P, r'''If this assumption fails, the system can fail: without the reported flight height, the SR drops from 63.3\% to 2.0\%.''', 'nature_drone',
    r'''If this assumption fails, the system can fail.''',
    'If this assumption fails, the system can fail.')

# ================================================================ closed loop
P = 'Closed-loop navigation in simulation'
add('S1', P, r'''Next, we turn to describe experiments that measure how well we can navigate using DreamNav in closed loop.''', 'nwm',
    r'''Next, we turn to describe experiments that measure how well can we navigate using a NWM.''',
    'Next, we turn to describe experiments that measure how well can we navigate using X.')
add('S2', P, r'''To this end, we evaluate performance in an image-goal navigation task when deployed in eight held-out test sites.''', 'nature_drone',
    r'''To this end, we evaluate performance in a racing task when deployed in four different settings.''',
    'To this end, we evaluate performance in a(n) A task when deployed in N different B.')
add('S3', P, r'''We first reconstruct a 3D Gaussian splatting model~\citep{kerbl20233d} from 135 of the 150 images of each site, with known poses and calibration, using gsplat~\citep{ye2025gsplat}.''', 'lightglue',
    r'''We first triangulate a sparse 3D point cloud from the 4328 daytime reference images, with known poses and calibration, using COLMAP~\cite{schoenberger2016sfm}.''',
    'We first [verb] a 3D A from the N images, with known poses and calibration, using TOOL.')
add('S4', P, r'''For each of 10 held-out goal images per site, we sample 2 start poses among the other cameras, render the view of the drone, and execute the estimated pose change with actuation noise.''', 'lightglue',
    r'''For each of the 824 daytime and 98 nighttime queries, we retrieve 50 images with NetVLAD~\cite{arandjelovic2016netvlad}, match each of them, and estimate a camera pose with RANSAC and a Perspective-n-Point solver.''',
    'For each of the N A, we [verb] M B ..., [verb] C, and [verb] a D with E.')
add('S5', P, r'''These observations are rendered by sampling the splat model at the native resolution and are independent of the real image that serves as the goal.''', 'nature_drone',
    r'''These noisy observations are generated by sampling one realization from the residual observation model and are independent of the perception awareness of the deployed controller.''',
    'These A observations are generated by sampling B from the C model and are independent of the D.')
add('S6', P, r'''Each command is randomly perturbed by 3\% of its length and 5\,cm per axis, $0.2^\circ$ in heading and $0.1^\circ$ in tilt.''', 'nwm',
    r'''Each goal is randomly chosen between $\pm16$ seconds window around the current state.''',
    'Each A is randomly [participle] [by/between] B ...')
add('S7', P, r'''We assess the role of imagination by comparing two variants of DreamNav, with and without the imagination-refined stage.''', 'pixloc',
    r'''We justify our design decisions by comparing different variants of PixLoc.''',
    'We [verb] our A by comparing different variants of X.')
add('S7b', P, r'''The refined stage is run with its short-range mode (Section~\ref{sec:servo}) when evaluated in closed loop (and without it in the experiments above).''', 'pixloc',
    r'''PixLoc is trained on MegaDepth (CMU) when evaluated on Aachen (RobotCar and CMU).''',
    'X is [participle] on/with A (...) when evaluated on/in B (...).')
add('S8', P, r'''Policies are allowed to continue flying for up to 5 steps, provided their estimated distance to the goal is still above a threshold $\tau$.''', 'nature_drone',
    r'''Pilots are allowed to continue racing even after a crash, provided their vehicle is still able to fly.''',
    'A are allowed to continue [gerund] ..., provided their B is still C.')
add('S8b', P, r'''Choosing a smaller threshold can bring the drone closer to the goal but can also increase the risk of acting on baselines shorter than those of training.''', 'nature_drone',
    r'''Leaving a smaller gate margin can increase speed but can also increase the risk of collision or missing the gate.''',
    '[Gerund] a smaller A can [verb] B but can also increase the risk of C.')
add('S9', P, r'''Performance is assessed by evaluating the position error at the stop, the number of steps flown and the percentage of episodes ending within 1\,m of the goal.''', 'nature_drone',
    r'''Performance is assessed by evaluating the fastest lap time, the average and minimum observed gate margin of successfully passed gates and the percentage of track successfully completed.''',
    'Performance is assessed by evaluating the A, the B and the percentage of C.')
add('S10', P, r'''The results are summarized in Table~\ref{tab:nav}.''', 'nature_drone',
    r'''The results are summarized in Extended Data Table 1c .''',
    'The results are summarized in Table.')

add('S10b', P, r'''DreamNav however lags behind on rendered views, where the textures are significantly blurrier and the estimates are therefore coarser (0.48\,m against 0.36\,m of median $\ell_x$ error on the same pairs).''', 'pixloc',
    r'''It however lags behind on Aachen, where the reference images are significantly sparser and the initial priors are therefore much coarser.''',
    'X however lags behind on A, where the B are significantly [adj] and the C are therefore (much) coarser.')
add('S11', P, r'''Both variants manage to cover most of the initial distance when deployed for a single step, with the refined estimate yielding the lowest median error (1.32\,m from a start at 18.9\,m).''', 'nature_drone',
    r'''All approaches manage to successfully complete the task when deployed in idealized dynamics and ground-truth observations, with the time-optimal baseline yielding the lowest lap time.''',
    'All A manage to [verb] B when deployed in/for C, with the D yielding the lowest E.')
add('S12', P, r'''The SR of the full model is raised from 38.8\% to over 81\% by closing the loop for only 2.2 steps on average.''', 'orienternet',
    r'''The recall at 3m/3\degree is saturated to over 95\% by fusing the predictions from sequences of only 20 seconds.''',
    'The A at/of B is saturated/raised to over N% by [gerund] C ... of only D.')
add('S13', P, r'''The full model outperforms the coarse stage alone by 25 points of SR at $\tau = 1$\,m, demonstrating the effectiveness of the short-range mode.''', 'loftr',
    r'''LoFTR outperforms the detector-free method DRC-Net by 61\% at AUC@10\degree, demonstrating the effectiveness of the Transformer.''',
    'X outperforms the A by N% at METRIC, demonstrating the effectiveness of the B.')
add('S14', P, r'''Although the full model is globally more successful than the coarse stage, it is not more successful at all individual thresholds of the sweep (Figure~\ref{fig:nav}).''', 'nature_drone',
    r'''Although Swift is globally faster than all human pilots, it is not faster on all individual segments of the track (Extended Data Table 1 ).''',
    'Although X is globally more A than B, it is not more A on all individual C of the D (Table/Fig.).')
add('S14b', P, r'''The coarse stage is as successful for larger thresholds as the full model then stops after its first step in more than half of the episodes, before the short-range mode can act, while smaller thresholds leave it the steps to do so.''', 'pixsfm',
    r'''Patch Flow is more complete for larger thresholds as it partly solves a different problem by increasing the keypoint repeatability with its large receptive field, while we focus on their localization.''',
    'A is more/as [adj] for larger thresholds as it/B [verb] ..., while C ...')
add('S15', P, r'''One hypothesis is that the fusion extrapolates poorly to baselines shorter than 5\,m, which make up fewer than 1\% of its training pairs.''', 'nature_drone',
    r'''One hypothesis is that Swift optimizes trajectories on a longer timescale than human pilots.''',
    'One hypothesis is that X [verb] A ... than B.')
add('S16', P, r'''This is apparent, for example between 0.5\,m and 1\,m from the goal, for which the remaining distance is overestimated to 1.34\,m in median by the coarse stage, but estimated at 0.78\,m by the short-range mode.''', 'nature_drone',
    r'''This is apparent, for example in the split S (Fig. 4b,d ), for which human pilots are faster in the beginning and at the end of the manoeuvre, but slower overall (Extended Data Table 1d , segment 3).''',
    'This is apparent, for example in A, for which B are/is C ..., but D ...')
add('S17', P, r'''We report four thresholds (Table~\ref{tab:nav}) and find that the short-range mode improves navigation performance, with thresholds of at most 1.25\,m yielding better results.''', 'nwm',
    r'''We report ATE and RPE on all in-domain datasets (Table~\ref{tab:traj-follow}) and find that NWM-based trajectory ranking improves navigation performance, with more samples yielding better results.''',
    'We report A (Table) and find that B improves navigation performance, with C yielding better results.')
add('S18', P, r'''The full model is consistently faster at meeting the stopping condition, with 2.2 steps against 3.8 at $\tau = 1$\,m.''', 'nature_drone',
    r'''Swift is consistently faster at the start and in tight turns such as the split S.''',
    'X is consistently faster at A ...')

# ================================================================ qualitative
P = 'Qualitative results'
add('Q1', P, r'''The plane-induced warp yields realistic imagined views even in challenging situations.''', 'dust3r',
    r'''\duster{} yields high-quality dense 3D reconstructions even in challenging situations.''',
    'X yields [adj] A even in challenging situations.')
add('Q2', P, r'''The imagined views recover both the layout and the fine details of the real target, successfully coping with elevated structures and large heading and altitude changes.''', 'loftr',
    r'''LoFTR obtains more correct matches and fewer mismatches, successfully coping with low-texture regions and large viewpoint and illumination changes.''',
    'X obtains/recovers A and B, successfully coping with C and large D and E changes.')

# ================================================================ captions
P = 'Captions'
add('C1', P, r'''\textbf{Image-goal pose estimation on PairUAV.} We report the MAE of each pose parameter, the success rate (SR) and the AUC of the pose error, the latter two in percents \%.''', 'superglue',
    r'''\textbf{Wide-baseline indoor pose estimation.} We report the AUC of the pose error, the matching score (MS) and precision (P), all in percents \%.''',
    'Title. We report the A, the B (b) and C (c), all in percents %.')
add('C2', P, r'''DreamNav outperforms all policies, world models and representations when using either one or both stages.''', 'superglue',
    r'''SuperGlue outperforms all handcrafted and learned matchers when applied to both SIFT and SuperPoint.''',
    'X outperforms all A and B [baselines] when [participle] both C and D.')
add('C3', P, r'''\textbf{Cumulative errors.} We plot the cumulative position, heading and tilt errors on the test pairs of PairUAV.''', 'pixsfm',
    r'''\textbf{Camera pose estimation.}
We plot the cumulative translation error and report its AUC.''',
    'Title. We plot the cumulative A error(s) and ...')
add('C4', P, r'''Our refinement improves the accuracy of the coarse estimates for all pose parameters, even for the heading change, whose coarse estimates are already accurate to a fraction of a degree.''', 'pixsfm',
    r'''Our refinement improves the accuracy of the query camera poses for all local features, even when for SIFT, whose detections are already well-localized.''',
    'Our refinement improves the accuracy of the A for all B, even for C, whose D are already [adj].')
add('C5', P, r'''\textbf{Ablation of DreamNav.} While the sine--cosine regression alone improves over the baseline RASR regressor, the rotation-aware two-view geometry explains the majority of the gains brought by DreamNav.''', 'superglue',
    r'''\textbf{Ablation of SuperGlue.} While the optimal matching layer alone improves over the baseline Nearest Neighbor matcher, the Graph Neural Network explains the majority of the gains brought by SuperGlue.''',
    'Ablation of X. While the A alone improves over the baseline B, the C explains the majority of the gains brought by X.')
add('C6', P, r'''Both the absolute heading and the zoom re-matching are critical for accurate displacements, and the imagination-refined stage further improves the accuracy.''', 'superglue',
    r'''Both cross-attention and positional encoding are critical for strong gluing, and a deeper network further improves the precision.''',
    'Both A and B are critical for C, and D further improves the E.')
add('C7', P, r'''\textbf{Raw two-view geometry.} The learned priors improve the accuracy of poses estimated by epipolar geometry (E) or by the MAP estimator of Eq.~\eqref{eq:map} (MAP) with rotation-aware matches.''', 'pixsfm',
    r'''\textbf{End-to-end SfM.}
The proposed refinement improves the accuracy of poses estimated by epipolar geometry (stereo) or a complete SfM pipeline (multiview) with crowd-sourced imagery.''',
    'Title. The A improve(s) the accuracy of poses estimated by epipolar geometry (a) or B (b) with C.')
add('C8', P, r'''Improvements are substantial for both first-pass (pass 1) and pose-guided (zoom) matching configurations, especially when the rotation and the translation direction are weakly constrained.''', 'pixsfm',
    r'''Improvements are substantial for both standard (SIFT) and recent (SuperGlue) matching configurations, especially when few images $N$ observe the scene.''',
    'Improvements are substantial for both A (a) and B (b) matching configurations, especially when C.')
add('C9', P, r'''\textbf{Ablation of the imagination-refined stage.} a)~The plane-induced warp and the tile-wise matching both largely improve the number of inliers and the accuracy of the relative pose.''', 'pixsfm',
    r'''\textbf{Ablation study on ETH3D.}
i)~Featuremetric keypoint and bundle adjustments (KA and BA) both largely improve the \red{triangulation} and \blue{localization} accuracy.''',
    'Ablation ... i) A and B both largely improve the C and D accuracy.')
add('C10', P, r'''b)~Different refined fusions are better than the coarse stage when given the imagined matches but the full model (our default) works best.''', 'pixsfm',
    r'''iii)~Different image representations are better than the unrefined detections but S2DNet (our default) works best.''',
    'iii) Different A are better than the B but X (our default) works best.')
add('C10b', P, r'''c)~Diffusion retrieval produces a full imagined view because of its generative prior but is less accurate than the coarse stage.''', 'pixsfm',
    r'''Patch Flow produces a longer track length because of its larger receptive field but is less accurate.''',
    'X produces a [adj] A because of its B but is less accurate.')
add('C11', P, r'''\textbf{Where the refinement helps.} We show the position error of the two stages, in linear scale, for varying ranges of difficulty.''', 'pixsfm',
    r'''\textbf{Run-times.}
 We show the duration, in logarithmic scale, of the refinement for varying numbers of images.''',
    'Title. We show the A, in [linear/logarithmic] scale, of the B for varying C.')
add('C12', P, r'''Refinement helps most on moderately matched pairs, where the error drops by almost half.''', 'lightglue',
    r'''Early stopping helps most on smaller scenes, where the network stops after just half the layers.''',
    'A helps most on B, where the C [verb] ... half ...')
add('C13', P, r'''\textbf{Reliability of the predicted error.} Left: sparsification plots demonstrating that the predicted error of each parameter can effectively capture accuracy, as the error decreases with decreasing fraction of retained pairs.''', 'kendall_unc',
    r'''Precision Recall plots demonstrating both measures of uncertainty can effectively capture accuracy, as precision decreases with increasing uncertainty.''',
    'A plots demonstrating (that) B can effectively capture accuracy, as C decreases with increasing/decreasing D.')
add('C14', P, r'''Right: calibration plots. This plot shows how well the predicted error is calibrated, where perfect calibration corresponds to the line $y=x$, shown dashed.''', 'kendall_unc',
    r'''Uncertainty calibration plots. This plot shows how well uncertainty is calibrated, where perfect calibration corresponds to the line $y=x$, shown in black.''',
    'A calibration plots. This plot shows how well B is calibrated, where perfect calibration corresponds to the line y=x, shown in [style].')
add('C15', P, r'''\textbf{Zero-shot pose estimation on real drone photographs.} With data from a DJI Phantom 4 RTK, DreamNav outperforms the regression baseline and the constant predictor in both coarse and refined settings, in the range seen in training.''', 'orienternet',
    r'''\textbf{Localization of head-mounted devices for AR.}
With data from Aria glasses, OrienterNet outperforms the map retrieval baseline and the embedded GPS sensor in both single- and multi-frame settings, in both cities.''',
    'Title. With data from A, X outperforms the B baseline and the C in both a- and b- settings, in [both] D.')
add('C16', P, r'''Rescaling by the flight height does not close the gap of the regression baseline but strongly benefits our approach.''', 'orienternet',
    r'''Multi-frame fusion does not filter out the high noise of the GPS but strongly benefits our approach.''',
    'A does not [verb] the B of the C but strongly benefits our approach.')
add('C17', P, r'''\textbf{Comparing the cumulative position error} of DreamNav and RASR within and beyond the training range of baseline-to-height ratio, up to 60\,m of error on the 600 real pairs of the M\'ezi\`eres dataset.''', 'nwm',
    r'''\textbf{Comparing generation accuracy and quality} of NWM and DIAMOND at $1$ and $4$ FPS as function of time, up to $16$ seconds of generated video on the RECON dataset.''',
    'Comparing A of X and Y at/within B ..., up to N of C on the D dataset.')
add('C18', P, r'''\textbf{Geometric imagination.} We show the target views $\widehat{G}$ imagined from the source $S$ for 2 test pairs of PairUAV and 1 pair of real photographs of the M\'ezi\`eres dataset.''', 'pixsfm',
    r'''\textbf{Refined SfM tracks.}
 We show patches centered around reprojections of 3x 3D points observed in 4 images of the \emph{St. Peter's Square} scene.''',
    'Title. We show A ... of N B ... in M C of the D.')
add('C18c', P, r'''From top to bottom: source, imagined target, real target over the same region.''', 'dust3r',
    r'''From left to right: RGB, depth map, confidence map, reconstruction.''',
    'From X to Y: A, B, C.')
add('C19', P, r'''Although only based on a planar warp of the source pixels, the imagination works well thanks to the flight altitude, despite the coarse pose estimate and large viewpoint changes.''', 'pixloc',
    r'''Although only based on local gradients, direct alignment works well thanks to deep features, despite the coarse initial pose estimate and strong appearance changes.''',
    'Although only based on A, X works well thanks to B, despite the coarse (initial) pose estimate and strong/large C changes.')
add('C20', P, r'''\textbf{Closed-loop image-goal navigation.} We present results for navigating with DreamNav under four stopping thresholds $\tau$, reporting the mean number of steps, the median position error at the stop (Err., in m) and the SR at 1\,m (\%) next to the one-step result.''', 'nwm',
    r'''\textbf{Planning with Navigation Constraints.} We present results for planning with NWM under three action constraints, reporting the differences in final position ($\delta u$) and yaw ($\delta \phi$) relative to the no-constraints baseline.''',
    'Title. We present results for [gerund] with X under N A, reporting the B ... and C ...')

add('C1b', P, r'''$\ddagger$ indicates distances predicted by the official distance head of the model.''', 'loftr',
    r'''\dag indicates models trained on MegaDepth.''',
    '† indicates A [participle] on B.')
add('C7b', P, r'''The mean (median) of the angular errors in degrees is reported.''', 'loftr',
    r'''The AUC of the corner error in percentage is reported.''',
    'The A of the B error in C is reported.')
add('C16b', P, r'''The medians of both the position error and the translation-direction error and the MAE of the tilt change are reported, with the SR at 1\,m and 2\,m.''', 'loftr',
    r'''The evaluation results on both the local feature evaluation track and the full visual localization track are reported.''',
    'The A on/of both the B and the C are reported.')
add('C18b', P, r'''For a yellow region in the source (top), we show in the target camera (middle) the view rendered by the plane-induced warp of the coarse pose.''', 'pixloc',
    r'''For a \red{red} point in the reference image (left), we highlight in the query (right) the multilevel basin of attraction colored by the 2D gradient angle $\nicefrac{\partial\*F_q}{\partial\*p_q^i}^\top \*r_k^i$.''',
    'For a [colour] A in the B (position), we highlight/show in the C (position) the D [participle] by E.')
add('C21', P, r'''\textbf{Effect of the stopping threshold.} Comparison of the closed-loop behaviour of each variant, illustrated by the SR at 1\,m (left) and the mean number of steps flown (right) for varying thresholds $\tau$.''', 'nature_drone',
    r'''Comparison of the fastest race of each pilot, illustrated by the time behind Swift.''',
    'Comparison of the A of each B, illustrated by the C.')
