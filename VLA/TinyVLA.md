# OpenVLA: An Open-Source Vision-Language-Action Model

## What is the paper doing?

1) existing VLAs are largely closed and inaccessible to the public
2) prior work fails to explore methods for efficiently fine-tuning VLAs for new tasks, a key component for adoption. Existing works do not provide best practices for deploying and adapting VLAs to new robots, environments, and tasks — especially on commodity hardware (e.g., consumer-grade GPUs).

OpenVLA, a 7B-parameter open-source VLA trained on a diverse collection of 970k real-world robot demonstrations.

with 7x fewer parameters compared to RT-2-X (55B) while outperforming 16.5% in absolute task success

strong generalization results

compute efficiency: OpenVLA can be fine-tuned on consumer GPUs via modern low-rank adaptation methods and served efficiently via quantization without a hit to downstream success rate

## What are the core insight and main conclusions of the paper?

What is the central claim or argument? Are the assumptions made (explicit or implicit) reasonable and well-justified? Do they hold under scrutiny in real-world or practical settings?

We argue that to develop a rich foundation for future research and development, robotics needs open-source, generalist VLAs that support effective fine-tuning and adaptation, akin to the existing ecosystem around open-source language models

work differs from RT-2-X in multiple important aspects:
1. by combining a strong open VLM backbone with a richer robot pretraining dataset, OpenVLA outperforms RT-2-X in our experiments while being an order of magnitude smaller
2. we thoroughly investigate fine-tuning of OpenVLA models to new target setups, while RT-2-X does not investigate the fine-tuning setting

We also demonstrated that OpenVLA can be easily adapted to new robot setups via parameter-efficient fine-tuning techniques.

## What are the key related works?

integrating pretrained language and vision-language models for robotic representation learning:

* S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta. R3m: A universal visual representation for robot manipulation. In CoRL, 2022.
* S. Karamcheti, S. Nair, A. S. Chen, T. Kollar, C. Finn, D. Sadigh, and P. Liang. Languagedriven representation learning for robotics. ArXiv, abs/2302.12766, 2023.
* M. Shridhar, L. Manuelli, and D. Fox. Cliport: What and where pathways for robotic manipulation. In Conference on robot learning, pages 894–906. PMLR, 2022.

pretrained language and vision-language models pretrained language and vision-language models:

* A. Stone, T. Xiao, Y. Lu, K. Gopalakrishnan, K.-H. Lee, Q. Vuong, P. Wohlhart, B. Zitkovich, F. Xia, C. Finn, et al. Open-world object manipulation using pre-trained vision-language models.
* D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. Palm-e: An embodied multimodal language model.

directly learning visionlanguage-action models (VLAs), which do not support efficient fine-tuning to new robot setups:

* Open X-Embodiment: Robotic learning datasets and RT-X models.
* Rt-2: Vision-language-action models transfer web knowledge to robotic control.
* A. S. et al. Introducing rfm-1: Giving robots human-like reasoning capabilities, 2024.
* Wayve. Lingo-2: Driving with natural language. 2024.

Existing works on VLAs either focus on training and evaluating in single robot or simulated setups:
* J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S.-C. Zhu, B. Jia, and S. Huang. An embodied generalist agent in 3d world. In Proceedings of the International Conference on Machine Learning (ICML), 2024.
* X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu, et al. Vision-language foundation models as effective robot imitators.
* H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan. 3d-vla: 3d visionlanguage-action generative world model.
* N. Dorka, C. Huang, T. Welschehold, and W. Burgard. What matters in employing vision language models for tokenizing actions in robot control? In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024.

**Visually-Conditioned Language Models**: patch features from pretrained visual transformers are treated as tokens, and are then projected into the input space of a language model. We employ these tools in our work to scale VLA training.

**Generalist Robot Policies** : Prior works like Octo typically compose pretrained components such as language embeddings or visual encoders with additional model components initialized from scratch, learning to “stitch” them together during the course of policy training. Unlike these works, OpenVLA adopts a more end-to-end approach, directly fine-tuning VLMs to generate robot actions by treating them as tokens in the language model vocabulary.

**Vision-Language-Action Models**: 
1) it performs alignment of pretrained vision and language components on a large, Internetscale vision-language dataset
2) the use of a generic architecture, not custom-made for robot control, allows us to leverage the scalable infrastructure underlying modern VLM training and scale to training billion-parameter policies with minimal code modifications
3) it provides a direct pathway for robotics to benefit from the rapid improvements in VLMs

## What is the key technical insight—the “one” idea—behind the solution?

**Preliminaries: Vision-Language Models**

build on the Prismatic-7B VLM：
* a 600M-parameter visual encoder
* a small 2-layer MLP projector
* a 7B-parameter Llama 2 language model backbone
* Prismatic uses a two-part visual encoder, consisting of pretrained SigLIP and DinoV2 models. Input image patches are passed separately through both encoders and the resulting feature vectors are concatenated channel-wise
* the addition of DinoV2 features has been shown to be helpful for improved spatial reasoning

**OpenVLA Training Procedure**

* We formulate the action prediction problem as a “vision-language” task, where an input observation image and a natural language task instruction are mapped to a string of predicted robot actions. (Rt-2)
* represent the actions in the output space of the LLM by mapping continuous robot actions to discrete tokens used by the language model’s tokenizer. Following Rt-2, we discretize each dimension of the robot actions separately into one of 256 bins. For each action dimension, we set the bin width to uniformly divide the interval between the 1st and 99th quantile of the actions in the training data
* Using quantiles instead of the min-max bounds Rt-2 used allows us to ignore outlier actions in the data that could otherwise drastically expand the discretization interval and reduce the effective granularity of our action discretization.
* overwriting the 256 least used tokens in the Llama tokenizer’s vocabulary (which corresponds to the last 256 tokens) with our action tokens.
* OpenVLA is trained with a standard next-token prediction objective, evaluating the cross-entropy loss on the predicted action tokens only.

**Training Data**

the Open X-Embodiment dataset [1] (OpenX) as a base to curate our training dataset
1. a coherent input and output space across all training datasets: restrict our training dataset to contain only manipulation datasets with at least one 3rd person camera and use single-arm end-effector control
2. a balanced mix of embodiments, tasks, and scenes in the final training mixture: we leverage the data mixture weights of Octo for all datasets that pass the first round of filtering. Octo heuristically down-weights or removes less diverse datasets and up-weights datasets with larger task and scene diversity; see Octo Model for details.

**OpenVLA Design Decisions**

Concretely, we trained and evaluated OpenVLA models on BridgeData V2 [6] for our initial experiments, instead of training on the full OpenX mixture, to increase iteration speed and reduce computational cost.

***VLM Backbone***

* LLaVA and IDEFICS-1 performed comparably on tasks with only one object in the scene
* LLaVA demonstrated stronger language grounding in tasks that involved multiple objects in the scene and required the policy to manipulate the correct object, i.e., the object specified in the language instruction.
* SigLIP-DinoV2 backbones: The fine-tuned Prismatic VLM policy achieved further improvements, outperforming the LLaVA policy by roughly 10% in absolute success rate across both simple single-object tasks and multi-object, language grounding tasks.
* Prismatic also provides a modular and easy-to-use codebase

***Image Resolution***
* many VLM benchmarks, increased resolution does improve performance [44, 86, 87], but we did not see this trend (yet) for VLAs

***Fine-Tuning Vision Encoder***

* Prior work on VLMs found that freezing vision encoders during VLM training typically leads to higher performance (from Prismatic VLM)
* However, we found fine-tuning the vision encoder during VLA training to be crucial for good VLA performance.
* We did not find learning rate warmup to provide benefits.

***Training Epochs and Learning Rate***

* we found it important for VLA training to iterate through the training dataset significantly more times. Our final training run completes 27 epochs through its training dataset.
* best results using a fixed learning rate of 2e-5 (the same learning rate used during VLM pretraining)

**Infrastructure for Training and Inference**

* a cluster of 64 A100 GPUs for 14 days, or a total of 21,500 A100-hours, using a batch size of 2048.
* During inference, OpenVLA requires 15GB of GPU memory when loaded in bfloat16 precision (i.e., without quantization) and runs at approximately 6Hz on one NVIDIA RTX 4090 GPU (without compilation, speculative decoding, or other inference speed-up tricks).

## Do the experiments support the paper’s central claims?

1. How does OpenVLA compare to prior generalist robot policies, when evaluating on multiple robots and various types of generalization?
2. Can OpenVLA be effectively fine-tuned on a new robot setup and task, and how does it compare to state-of-the-art data-efficient imitation learning approaches?
3. Can we use parameter-efficient fine-tuning and quantization to reduce the computational requirements for training and inference of OpenVLA models and make them more accessible? What are the performance-compute trade-offs?

**Datasets:** Open-X Embodiment dataset, a dataset that spans a wide range of robot embodiments, tasks, and scenes, consisting of more than 70 individual robot datasets, with more than 2M robot trajectories

**Baselines:**
* RT-2-X. OpenVLA outperforms the 55B-parameter RT-2-X model [1, 7], the prior state-of-the-art VLA, by 16.5% absolute success rate across 29 evaluation tasks on the WidowX and Google Robot embodiments.
* fine-tuned pretrained policies such as Octo
* from-scratch imitation learning with diffusion policies. OpenVLA shows substantial improvement on tasks involving grounding language to behavior in multi-task settings with multiple objects.

**Benchmarks:** across 29 evaluation tasks on the WidowX and Google Robot embodiments.

* visual (unseen backgrounds, distractor objects, colors/appearances of objects)
* motion (unseen object positions/orientations)
* physical (unseen object sizes/shapes)
* semantic (unseen target objects, instructions, and concepts from the Internet)

With LoRA, we can fine-tune OpenVLA on a new task within 10-15 hours on a single A100 GPU an 8x reduction in compute compared to full fine-tuning.

Notably, 4-bit quantization results in similar performance as bfloat16 half-precision inference despite requiring less than half the amount of GPU memory.

## Overall, what are the contributions of the paper?

open-source

the first to demonstrate the effectiveness of compute-efficient fine-tuning methods leveraging low-rank adaptation and model quantization to facilitate adapting OpenVLA models on consumer-grade GPUs instead of large server nodes without compromising performance.

## What questions remain open?

* We find that only fine-tuning the network’s last layer or freezing the vision encoder leads to poor performance, suggesting that further adaptation of the visual features to the target scene is crucial

* For narrower but highly dexterous tasks, Diffusion Policy still shows smoother and more precise trajectories; incorporating action chunking and temporal smoothing, as implemented in Diffusion Policy, may help OpenVLA attain the same level of dexterity and may be a promising direction for future work

1. it currently only supports single-image observations. In reality, real-world robot setups are heterogeneous, with a wide range of possible sensory inputs. Expanding OpenVLA to support multiple image and proprioceptive inputs as well as observation history is an important avenue for future work.
2. improving the inference throughput of OpenVLA is critical to enable VLA control for high-frequency control setups such as ALOHA, which runs at 50Hz.
3. While OpenVLA outperforms prior generalist policies, it does not yet offer very high reliability on the tested tasks, typically achieving <90% success rate.
4. What effect does the size of the base VLM have on VLA performance? 
5. Does co-training on robot action prediction data and Internet-scale vision-language data substantially improve VLA performance? 
6. What visual features are best-suited for VLA models?
