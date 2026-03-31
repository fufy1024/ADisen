# A-Disen 
This repository contains the official implementation of the paper "Associating and Disentangling Foreground-Background for Text-based Image
Editing".

## 📖 Introduction
Text-based image editing, which aims to modify images with different complex rigid and non-rigid variations w.r.t the given text, has recently attracted extensive interest. Existing works are typically either global-entangled or hard-disentangled paradigms to implicitly or explicitly separate foreground and background, 
 but neglect the intrinsic associated interaction between foreground-background, leading to inaccurate disentanglement during the dynamic denoising process and a trade-off between foreground editability and background fidelity. In this paper, we propose a novel Associated-Disentangled attention framework (A-Disen), 
which explicitly models the associated interaction and step-wise dynamic disentanglement of foreground-background at each generation step, and thus achieves comprehensive improvements of editability and fidelity. Specifically, we design (1) dual branch negative-gating associated attention module, which innovatively transforms foreground negative attention into an inhibitory gating signal to regulate background attention, enabling more accurate and dynamic disentanglement in an associated-learning manner. 
 (2) Token-prior disentangled contrastive loss, which is designed based on the internal encoding pattern of textual cross-attention on foreground-background regions, to provide accurate token-level priors guidance for supervising foreground-background attention learning, and finally facilitates the associated-attention module. Comprehensive experiments demonstrate our superiority, exhibiting unprecedented editability on text guidance and fidelity on text-irrelevant image details.

 ![teaser](assets/fig-1-0105.jpg)

 ![teaser](assets/fig2-0914.jpg)
