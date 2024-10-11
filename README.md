
Reading list for research in generation models.

We list the most popular methods for generation models, if we missed something, please submit a request.
(Note: We show the date the first edition of the paper was submitted to arxiv, but the link to the paper may be up to date.)

Autoregressive generation model:
Date|Method|Conference|Title|Code/Project Page|abstract
-----|----|-----|-----|-----|-----
Jun 2024|LlamaGen|CVPR Jun 2024|[Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/abs/2406.06525)|[code](https://github.com/FoundationVision/LlamaGen)|采用与Llama相同的网络结构实现了图像的自回归生成，在语言视觉模型的范式统一上具有重要意义
Jun 2024|MAR|ARXIV Jun 2024|[Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)|[code](https://github.com/LTH14/mar)|去除VQ过程的自回归生成方法
Jun 2024|VAR|PR Jun 2024|[Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)|[code](https://github.com/FoundationVision/VAR)|以不同scale图像作为自回归单位进行自回归生成
Oct 2024|MovieGen|Oct 2024|[Movie Gen: A Cast of Media Foundation Models](https://ai.meta.com/static-resource/movie-gen-research-paper/?utm_source=twitter&utm_medium=organic_social&utm_content=thread&utm_campaign=moviegen)|[poster](https://ai.meta.com/blog/movie-gen-media-foundation-models-generative-ai-video/)|继Sora后Meta推出的视频生成模型
Oct 2024|DART|Oct 2024|[DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation](https://arxiv.org/html/2410.08159v1)|[poster](https://arxiv.org/html/2410.08159v1)|come soon

Diffusion model：

Date|Method|Conference|Title|Code/Project Page
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


Tutorial:

[CVPR 2022 Tutorial:Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://cvpr2022-tutorial-diffusion-models.github.io)





