
Reading list for research in generation models.

We list the most popular methods for generation models, if we missed something, please submit a request.
(Note: We show the date the first edition of the paper was submitted to arxiv, but the link to the paper may be up to date.)

Backbone:
Date|Method|Conference|Title|Code/Project Page|abstract
-----|----|-----|-----|-----|-----
2024|VisionLLaMA|ECCV 2024|[VisionLLaMA: A Unified LLaMA Backbone for Vision Tasks](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08280.pdf)|[code](https://github.com/Meituan-AutoML/VisionLLaMA)|对视觉语言大模型的大一统backbone进行了探讨,主要提出了一个2d位置编码


Autoregressive generation model:
Date|Method|Conference|Title|Code/Project Page|abstract
-----|----|-----|-----|-----|-----
Jun 2024|LlamaGen|CVPR Jun 2024|[Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/abs/2406.06525)|[code](https://github.com/FoundationVision/LlamaGen)|采用与Llama相同的网络结构实现了图像的自回归生成，在语言视觉模型的范式统一上具有重要意义
28 Jul 2024|MAR|ARXIV 28 Jul 2024|[Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)|[code](https://github.com/LTH14/mar)|去除VQ过程的自回归生成方法
Jun 2024|VAR|PR Jun 2024|[Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)|[code](https://github.com/FoundationVision/VAR)|以不同scale图像作为自回归单位进行自回归生成
Oct 2024|MovieGen|Oct 2024|[Movie Gen: A Cast of Media Foundation Models](https://ai.meta.com/static-resource/movie-gen-research-paper/?utm_source=twitter&utm_medium=organic_social&utm_content=thread&utm_campaign=moviegen)|[poster](https://ai.meta.com/blog/movie-gen-media-foundation-models-generative-ai-video/)|继Sora后Meta推出的视频生成模型
Oct 2024|DART|Oct 2024|[DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation](https://arxiv.org/html/2410.08159v1)|[poster](https://arxiv.org/html/2410.08159v1)|提出了一种与LLM主流模型结构具有统一性的扩散生成模型（里面推导很多，暂时没看懂，先去看VDM了）
2 Oct 2024|ControlVAR|2 Oct 2024|[ControlVAR: Exploring Controllable Visual Autoregressive Modeling](https://arxiv.org/abs/2406.09750)|[code](https://github.com/lxa9867/ControlVAR)|coming soon
7 Oct 2024|CAR|Arxiv 7 Oct 2024|[CAR: Controllable Autoregressive Modeling for Visual Generation](https://arxiv.org/abs/2410.04671)|[code](https://github.com/MiracleDance/CAR)|参考controlNet的方式在自回归生成模型VAR上进行可控生成
14 Oct 2024|HART|Arxiv 14 Oct 2024|[HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://arxiv.org/abs/2410.10812)|[code](https://github.com/mit-han-lab/hart)|基于VAR生成模型构建的非离散的自回归图像生成
10 Oct 2024|Meissonic|Arxiv 10 Oct 2024|[Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis](https://arxiv.org/abs/2410.08261)|[code](https://huggingface.co/MeissonFlow/Meissonic)|一种高效的文生图模型，它将非自回归掩码图像建模 (MIM) 文本到图像提升到与 SDXL 等最先进的扩散模型相当的水平，并大大提高性能和效率
17 Oct 2024|Fluid|Arxiv 17 Oct 2024|[Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens](https://arxiv.org/abs/2410.13863)|[None]()|在自回归生成任务中对token的连续和离散，预测采用光栅扫描还是随机mask方式进行了深入探讨
27 Sep 2024|Emu3|Arxiv 27 Sep 2024|[Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869)|[poster](https://emu.baai.ac.cn/about)|首个采用Llama2架构实现文本、图像、视频大一统的模型(看起来确实很吊)
24 Oct 2024|FairQueue|Arxiv 24 Oct 2024|[FairQueue: Rethinking Prompt Learning for Fair Text-to-Image Generation](https://arxiv.org/abs/2410.18615)|[None]()|coming soon
30 Apr 2024|None|Arxiv 30 Apr 2024|[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)|[None]()|自回归加速推理（NLP）
29 Oct 2024|None|Arxiv 29 Oct 2024|[Towards Unifying Understanding and Generation in the Era of Vision Foundation Models: A Survey from the Autoregression Perspective](https://arxiv.org/abs/2410.22217)|[None]()|语言图像自回归大一统的探讨

Diffusion model：

Date|Method|Conference|Title|Code/Project Page/abstract
-----|----|-----|-----|-----
2015-xx-xx|Diffusion models|ICML 2015|[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](http://proceedings.mlr.press/v37/sohl-dickstein15.pdf)|None
2020-06-09|Denoised Diffusion models|NeurIPS 2020|[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)|[Diffusion Models](https://github.com/hojonathanho/diffusion)
2020-10-06|DDIM|ICLR 2021|[DENOISING DIFFUSION IMPLICIT MODELS](https://arxiv.org/pdf/2010.02502.pdf)|None
2020-11-26|SDE|ICLR 2021(Oral)|[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456.pdf)|None
2021-02-18|improved-diffusion|Arxiv 2021|[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672.pdf)|[improved-diffusion](https://github.com/openai/improved-diffusion)
2021-05-11|guided-diffusion|NeurIPS 2021|[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)|[guided-diffusion](https://github.com/openai/guided-diffusion)
2021-05-30|cascaded diffusion models|Arxiv 2021|[Cascaded Diffusion Models for High Fidelity Image Generation](https://arxiv.org/pdf/2106.15282.pdf)|None
2021-07-01|Variational Diffusion Models|NeurIPS 2021|[Variational Diffusion Models](https://arxiv.org/pdf/2107.00630.pdf)|[Variational Diffusion Models](https://github.com/google-research/vdm)
2021-09-28|Classifier-Free Diffusion|NeurIPS 2021 WorkShop|[Classifier-Free Diffusion Guidance](https://openreview.net/pdf?id=qw8AKxfYbI)|None
2021-10-06|DiffusionCLIP|Arxiv 2021|[DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation](https://arxiv.org/pdf/2112.10741.pdf)|[DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP)
2021-11-10|Palette|Arxiv 2021|[Palette: Image-to-Image Diffusion Models](https://arxiv.org/pdf/2111.05826.pdf)| None
2021-11-29|Blended diffusion|Arxiv 2021|[Blended Diffusion for Text-driven Editing of Natural Images](https://arxiv.org/pdf/2111.14818.pdf)|[Blended Diffusion](https://omriavrahami.com/blended-diffusion-page/)
2021-12-20|GLIDE|ICML 2022|[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/pdf/2112.10741.pdf)|[GLIDE](https://github.com/openai/glide-text2im)
2022-01-24|RePaint|CVPR 2022|[RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2201.09865.pdf)|[RePaint](https://www.git.io/RePaint)
2022-04-06|KNN-Diffusion|Arxiv 2022|[KNN-Diffusion: Image Generation via Large-Scale Retrieval](https://arxiv.org/pdf/2204.02849.pdf)|None
2022-04-13|DALL·E 2|Arxiv 2022|[Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/pdf/2204.06125.pdf)|None
2022-05-23|Imagen|Arxiv 2022|[Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/pdf/2205.11487.pdf)|None
2022-06-01|--|Arxiv 2022|[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364.pdf)|None
2022-06-03|Composable-Diffusion|Arxiv 2022|[Compositional Visual Generation with Composable Diffusion Models](https://arxiv.org/pdf/2206.01714.pdf)|[Composable-Diffusion](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch)
2022-06-22|videoDiffusion|NeurIPS 2022|[Video Diffusion Models](https://arxiv.org/pdf/2204.03458.pdf)|None
2022-08-03|PDDPM|Arxiv 2022|[Pyramidal Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2208.01864.pdf)|None
2022-08-25|DreamBooth|Arxiv 2022|[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/pdf/2208.12242.pdf)|None
2022-11-17|UViT|Arxiv 2022|[All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/pdf/2209.12152.pdf)|[UViT](https://github.com/baofff/U-ViT)
2022-11-17|RenderDiffusion|Arxiv 2022|[RenderDiffusion: Image Diffusion for 3D Reconstruction, Inpainting and Generation](https://arxiv.org/pdf/2211.09869.pdf)|[RenderDiffusion](https://github.com/Anciukevicius/RenderDiffusion)
2022-11-17|Null-text Inversion|Arxiv 2022|[Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://arxiv.org/pdf/2211.09794.pdf)|[Null-text Inversion](https://null-text-inversion.github.io/)
2022-11-25|3DDesigner|Arxiv 2022|[3DDesigner: Towards Photorealistic 3D Object Generation and Editing with Text-guided Diffusion Models](https://arxiv.org/pdf/2211.14108.pdf)|[3DDesigner](https://3ddesigner-diffusion.github.io/)
2024-x-x|DEADiff|CVPR 2024|[DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations](https://openaccess.thecvf.com/content/CVPR2024/papers/Qi_DEADiff_An_Efficient_Stylization_Diffusion_Model_with_Disentangled_Representations_CVPR_2024_paper.pdf)|[DEADiff](https://tianhao-qi.github.io/DEADiff/)
2024-8-14|TurboEdit|ARXIV 2024|[TurboEdit: Instant text-based image editing](https://arxiv.org/abs/2408.08332)|diffusion模型的目标解耦和图像编辑
2021-X-X|VDM|NIPS 2021|[Variational Diffusion Models](https://proceedings.neurips.cc/paper/2021/file/b578f2a52a0229873fefc2a4b06377fa-Paper.pdf)|哈哈，好像是换了个角度重新推了一遍DDPM，有点没看懂，公式推导好多啊
23 Oct 2024|None|Arxiv 23 Oct 2024|[Scalable Ranked Preference Optimization for Text-to-Image Generation](https://arxiv.org/abs/2410.18013)|diffusion model上的HFRL
23 Oct 2024|None|Arxiv 23 Oct 2024|[How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization?](https://arxiv.org/abs/2410.17594)|还没看
2025|None|CVPR 2025|[ACE: Anti-Editing Concept Erasure in Text-to-Image Models](https://arxiv.org/pdf/2501.01633)|unlearning相关
2025|None|CVPR 2025|[Silent Branding Attack: Trigger-free Data Poisoning Attack on Text-to-Image Diffusion Models](https://arxiv.org/pdf/2503.09669)|数据集攻击，安全相关

Segmentation：
Date|Method|Conference|Title|Code
-----|----|-----|-----|-----
2021-12-06|ddpm-segmentation|ICLR 2022|[Label-Efficient Semantic Segmentation with Diffusion Models](https://arxiv.org/pdf/2112.03126.pdf)|[ddpm-segmentation](https://github.com/yandex-research/ddpm-segmentation)

Other Discriminative Tasks：
Date|Method|Conference|Title|Code
-----|----|-----|-----|-----
2023-05-18| ~ |Arxiv 2023|[Discriminative Diffusion Models as Few-shot Vision and Language Learners](https://arxiv.org/pdf/2305.10722.pdf)| None

Survey:
Date|Conference|Title
-----|-----|-----
2022-09-02|Arxiv 2022|[Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/pdf/2209.00796.pdf)
2022-09-10|Arxiv 2022|[Diffusion Models in Vision: A Survey](https://arxiv.org/pdf/2209.04747.pdf)
2022-09-12|Arxiv 2022|[A Survey on Generative Diffusion Model](https://arxiv.org/pdf/2209.02646.pdf)
2023-04-02|Arxiv 2023|[Text-to-image Diffusion Models in Generative AI:A Survey](https://arxiv.org/pdf/2303.07909.pdf)
Sep 2024|AI 2024|[Multi-Modal Generative AI: Multi-modal LLM, Diffusion and Beyond](https://arxiv.org/abs/2409.14993)


Large Language Models：
Date|Method|Conference|Title|Code
-----|----|-----|-----|-----
Feb 2023|LLAMA|Arxiv Feb 2023|[Llama: Open and efficient foundation language models](https://arxiv.org/abs/2302.13971)|[code](https://github.com/meta-llama/llama)

Model Merging:
Date|Method|Conference|Title|Code/Project Page|abstract
-----|----|-----|-----|-----|-----
Sep 2024|None|ML Sep 2024|[REALISTIC EVALUATION OF MODEL MERGING FORCOMPOSITIONAL GENERALIZATION](https://arxiv.org/pdf/2409.18314)|None|None
Jun 2024|WATT|CVPR Jun 2024|[WATT: Weight Average Test-Time Adaptation of CLIP](https://arxiv.org/abs/2406.13875)|None|在test-time adaptation的问题中采用model merging的方法提高模型泛化能力
9 Oct 2024|DECOUPLE-THEN-MERGE|9 Oct 2024|[DECOUPLE-THEN-MERGE:TOWARDS BETTER TRAINING FOR DIFFUSION MODELS](https://arxiv.org/pdf/2410.06664)|None|分时间步分别训练diffusion模型再通过merging得到减少由于不同时间步导致的参数冲突问题
6 Jun 2024|None|6 Jun 2024|[B-ary Tree Push-Pull Method is Provably Efficient for Distributed Learning on Heterogeneous Data](https://arxiv.org/abs/2404.05454)|None|reading
27 Sept 2024|None|27 Sept 2024|[LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging](https://openreview.net/forum?id=J5sUOvlLbQ)|None|探讨观点：网络深层和浅层在finetune中的不同影响
8 Dec 2022|None|8 Dec 2022|[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)|None|task vector提出的论文
16 Oct 2023|None|16 Oct 2023|[Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards](https://arxiv.org/abs/2306.04488)|None|Rewarded Soup
24 Nov 2024|None|24 Nov 2024|[Less is More: Efficient Model Merging with Binary Task Switch](https://arxiv.org/abs/2412.00054)|None|动态merging方法，具体有点像剪枝

HFRL:
Date|Method|Conference|Title|Code/Project Page|abstract
-----|----|-----|-----|-----|-----
6 Oct 2024|TIS-DPO|arxiv 6 Oct 2024|[TIS-DPO: Token-level Importance Sampling for Direct Preference Optimization With Estimated Weights](https://arxiv.org/abs/2410.04350)|[None]|正在看，别催啦
17 Oct 2024|TIS-DPO|arxiv 17 Oct 2024|[Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://arxiv.org/abs/2410.13643)|[code](https://github.com/ChenyuWang-Monica/DRAKES)|提出了一种RL算法，结合蛋白质领域的离散扩散方法，直接通过奖励对模型进行优化解决不可导问题

Tutorial:

[CVPR 2022 Tutorial:Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://cvpr2022-tutorial-diffusion-models.github.io)





