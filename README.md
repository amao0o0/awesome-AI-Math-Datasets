# AI Math Datasets

This repo contains recent **open-source** math datasets (mainly English) for training and evaluating Math Large Language Models (LLMs).

> [!NOTE]  
> This repo is currently under development and updated regularly.
> 
> 🗓️ Last updated: 2026-03-29

## Table of Contents
- [Pre-training](#pre-training)
- [Supervised Fine-Tuning](#supervised-fine-tuning)
- [Reinforcement Learning](#reinforcement-learning)
- [Benchmark](#benchmark)

---

## Pre-training

<!-- 📄 [Paper]() <br> 🔗 [Project]() <br> 🐙 [Repo]() <br> 🤗 [Dataset]() -->

### 📝 Text Only

| Dataset | Descriptions | References |
|---------|-------|-------|
|**Open-Web-Math**| An open dataset inspired by these works containing 14.7B tokens of mathematical webpages from Common Crawl. | 📄 [Paper](https://arxiv.org/pdf/2310.06786) <br> 🐙 [Repo](https://github.com/keirp/OpenWebMath) <br> 🤗 [Dataset](https://huggingface.co/datasets/open-web-math/open-web-math) | 
|**Open-Web-Math-Pro**|Refined from open-web-math using the ProX refining framework. It contains about 5B high-quality math-related tokens, ready for pre-training.| 📄 [Paper](https://arxiv.org/pdf/2409.17115) <br> 🐙 [Repo](https://github.com/GAIR-NLP/ProX) <br> 🤗 [Dataset](https://huggingface.co/datasets/gair-prox/open-web-math-pro)|
|**AMPS**| Auxiliary Mathematics Problems and Solutions.  A collection of mathematical problems and step-by-step solutions, comprising over 100,000 problems from Khan Academy and approximately 5 million problems generated using Mathematica scripts. | 🐙 [Repo](https://github.com/hendrycks/math?tab=readme-ov-file)|
|**NaturalProofs**| A dataset designed to study mathematical reasoning in natural language, comprising approximately 32,000 theorem statements and proofs, 14,000 definitions, and 2,000 additional pages sourced from diverse mathematical domains |📄 [Paper](https://arxiv.org/abs/2104.01112) <br> 🐙 [Repo](https://github.com/wellecks/naturalproofs)|
|**MathPile** | A math-centric corpus comprising about 9.5 billion tokens.| 📄 [Paper](https://huggingface.co/papers/2312.17120) <br> 🐙 [Repo](https://github.com/GAIR-NLP/MathPile/?tab=readme-ov-file) <br> 🤗 [Dataset](https://huggingface.co/datasets/GAIR/MathPile)|
|**AlgebraicStack** | A dataset of 11B tokens of code specifically related to mathematics.| 📄 [Paper](https://arxiv.org/abs/2310.10631) <br> 🐙 [Repo](https://github.com/EleutherAI/math-lm?tab=readme-ov-file) <br> 🤗 [Dataset](https://huggingface.co/datasets/EleutherAI/proof-pile-2/tree/main/algebraic-stack)|
|**MathCode-Pile** | Containing 19.2B tokens, with math-related data covering web pages, textbooks, model-synthesized text, and math-related code. | 📄 [Paper](https://arxiv.org/abs/2410.08196) <br> 🐙 [Repo](https://github.com/mathllm/MathCoder2) <br> 🤗 [Dataset](https://huggingface.co/datasets/MathGenie/MathCode-Pile)|
|**FineMath** |  Consisting of 34B tokens (FineMath-3+) and 54B tokens (FineMath-3+ with InfiMM-WebMath-3+) of mathematical educational content filtered from CommonCrawl. | 📄 [Paper](https://arxiv.org/abs/2502.02737v1) <br> 🤗 [Dataset](https://huggingface.co/datasets/HuggingFaceTB/finemath)|
|**Proof-Pile-2** | A 55 billion token dataset of mathematical and scientific documents from arxiv, open-web-math and algebraic-stack.| 📄 [Paper](https://arxiv.org/abs/2310.10631) <br> 🐙 [Repo](https://github.com/EleutherAI/math-lm?tab=readme-ov-file) <br> 🤗 [Dataset](https://huggingface.co/datasets/EleutherAI/proof-pile-2)|
|**AutoMathText** | A dataset encompassing around 200 GB of mathematical texts. It's a compilation sourced from a diverse range of platforms including various websites, arXiv, and GitHub (OpenWebMath, RedPajama, Algebraic Stack). |📄[Paper](https://arxiv.org/abs/2402.07625) <br> 🐙[Repo](https://github.com/yifanzhang-pro/AutoMathText) <br> 🤗[Dataset](https://huggingface.co/datasets/math-ai/AutoMathText)|
|**MegaMath** | An open math pretraining dataset curated from diverse, math-focused sources, with over 300B tokens. |📄[Paper](https://arxiv.org/abs/2504.02807) <br> 🐙[Repo](https://github.com/LLM360/MegaMath) <br> 🤗[Dataset](https://huggingface.co/datasets/LLM360/MegaMath)|
|**Nemotron-CC-Math**| A 133 billion token large-scale math corpus extracted from Common Crawl. |📄[Paper]() <br> 🐙[Repo](https://huggingface.co/blog/nvidia/nemotron-cc-math) <br> 🤗[Dataset](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1)|
| **SwallowMath** | A ~2.3B token math pre-training corpus refined from FineMath-4+ via LLM-driven rewriting (Llama-3.3-70B-Instruct) that removes boilerplate, restores missing context, and reformats solutions into concise step-by-step explanations. | 📄 [Paper](https://arxiv.org/abs/2505.02881) <br> 🐙 [Repo](https://github.com/rioyokotalab/swallow-code-math) <br> 🤗 [Dataset](https://huggingface.co/datasets/tokyotech-llm/swallow-math) |

### 🖼️ Vision-Text Modality

| Dataset | Descriptions | References |
|---------|-------|-------|
|**InfiMM-WebMath-40B**| A dataset of interleaved image-text documents. It comprises 24 million web pages, 85 million associated image URLs, and 40 billion text tokens, all meticulously extracted and filtered from CommonCrawl. | 📄 [Paper](https://arxiv.org/abs/2409.12568) <br> 🤗 [Dataset](https://huggingface.co/datasets/Infi-MM/InfiMM-WebMath-40B)|

---

## Supervised Fine-Tuning

<!-- 📄 [Paper]() <br> 🔗 [Project]() <br> 🐙 [Repo]() <br> 🤗 [Dataset]() -->

### 📝 Text Only

| Dataset | Descriptions | References |
|---------|-------|-------|
|**SVAMP**|A collection of 1,000 elementary-level math word problems.|📄 [Paper](https://arxiv.org/abs/2103.07191) <br> 🐙 [Repo](https://github.com/arkilpatel/SVAMP)|
|**GSM8K**|A dataset consists of 8.5K high-quality grade school math word problems. Each problem takes between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − × ÷) to reach the final answer. |📄 [Paper](https://arxiv.org/abs/2110.14168) <br> 🔗 [Project](https://openai.com/index/solving-math-word-problems/) <br> 🐙 [Repo](https://github.com/openai/grade-school-math?tab=readme-ov-file) |
|**MathQA**| A dataset of 37k English multiple-choice math word problems covering multiple math domain categories by modeling operation programs corresponding to word problems in the AQuA dataset | 🔗 [Project](https://math-qa.github.io/math-QA/) |
|**MATH**| A challenging dataset that extends beyond the high school level and covers diverse topics, including algebra, precalculus, and number theory. Each problem in MATH has a full step-by-step solution. | 🔗 [Project](https://github.com/hendrycks/math/)|
|**NuminaMath** | A comprehensive collection of 860,000 pairs ranging from high-school-level to advanced-competition-level. The dataset has both CoT and PoT rationales (NuminaMath-CoT and -TIR (tool integrated reasoning)) | 📄 [Paper](http://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf) <br> 🐙 [Repo](https://github.com/project-numina/aimo-progress-prize) <br> 🤗 [Dataset](https://huggingface.co/AI-MO) |
|**MetaMath**| A dataset with 395K samples created by bootstrapping questions from MATH and GSM8K. | 📄 [Paper](https://arxiv.org/abs/2309.12284) <br> 🔗 [Project](https://meta-math.github.io/) <br> 🐙 [Repo](https://github.com/meta-math/MetaMath) <br> 🤗 [Dataset](https://huggingface.co/datasets/meta-math/MetaMathQA) |
|**MathInstruct**| An instruction tuning dataset that combines data from 13 mathematical rationale datasets, uniquely focusing on the hybrid use of chain-of-thought (CoT) and program-of-thought (PoT) rationales. | 📄 [Paper](https://arxiv.org/pdf/2309.05653) <br> 🔗 [Project](https://tiger-ai-lab.github.io/MAmmoTH/) <br> 🐙 [Repo](https://github.com/TIGER-AI-Lab/MAmmoTH) <br> 🤗 [Dataset](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) |
|**CoinMath**| A dataset designed to enhance mathematical reasoning in large language models by incorporating diverse coding styles into code-based rationales. It includes math questions annotated with code-based solutions that feature concise comments, descriptive naming conventions, and hardcoded solutions | 📄 [Paper](https://arxiv.org/abs/2412.11699) <br> 🐙 [Repo](https://github.com/TIGER-AI-Lab/MAmmoTH) <br> 🤗 [Dataset](https://huggingface.co/datasets/amao0o0/CoinMath) |
|**OpenMathInstruct-2**| A math instruction tuning dataset with 14M problem-solution pairs generated using the Llama3.1-405B-Instruct model. | 📄 [Paper](https://arxiv.org/abs/2410.01560) <br> 🤗 [Dataset](https://huggingface.co/collections/nvidia/openmath-2-66fb142317d86400783d2c7b) |
|**CAMEL Math**| Containing 50K problem-solution pairs obtained using GPT-4. The dataset problem-solutions pairs were generated from 25 math topics, and 25 subtopics for each topic. | 📄 [Paper](https://arxiv.org/abs/2303.17760) <br> 🤗 [Dataset](https://huggingface.co/datasets/camel-ai/math) |
|**AoPS-Instruct**| A large-scale dataset of over 650,000 Olympiad-level math question-answer pairs mined from the AoPS forum, accompanied by LiveAoPSBench—a dynamic, contamination-resistant benchmark | 📄 [Paper](https://arxiv.org/pdf/2501.14275) <br> 🔗 [Project](https://livemathbench.github.io/) <br> 🤗 [Dataset](https://huggingface.co/datasets/jojo23333/LiveAoPSBench-2024) |
| **OpenMathReasoning** | A large-scale dataset of 306K unique math problems sourced from AoPS forums, with solutions generated by DeepSeek-R1 and QwQ-32B in both CoT and TIR (tool-integrated reasoning) formats. Formed the foundation of the AIMO-2 Kaggle competition winning solution. | 📄 [Paper](https://arxiv.org/abs/2504.16891) <br> 🤗 [Dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning) |


### 🖼️ Vision-Text Modality

| Dataset | Descriptions | References |
|---------|-------|-------|
|**GeoQA**| Containing 4,998 Chinese geometric multiple-choice questions with rich domain-specific program annotations. |📄 [Paper](https://arxiv.org/abs/2105.14517) <br> 🐙 [Repo](https://github.com/chen-judge/GeoQA)| 
|**UniGeo**| Containing 4,998 calculation problems and 9,543 proving problems. | 📄 [Paper](https://arxiv.org/abs/2212.02746) <br> 🐙 [Repo](https://github.com/chen-judge/UniGeo)|
|**Geo170K**| A synthesize dataset witch contains around 60,000 geometric image caption pairs and more than 110,000 question answer pairs. | 📄 [Paper](https://arxiv.org/abs/2312.11370) <br> 🐙 [Repo](https://github.com/pipilurj/G-LLaVA?tab=readme-ov-file) <br> 🤗 [Dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main)|
|**MAVIS**| Containing two datasets: 1. MAVIS-Caption: 588K high-quality caption-diagram pairs, spanning geometry and function, 2. MAVIS-Instruct: 834K instruction-tuning data with CoT rationales in a text-lite version. | 📄 [Paper](https://arxiv.org/html/2407.08739v1) <br> 🐙 [Repo](https://github.com/ZrrSkywalker/MAVIS?tab=readme-ov-file)|
|**Geometry3K**| Consisting of 3,002 geometry problems with dense annotation in formal language. | 📄 [Paper](https://arxiv.org/abs/2105.04165) <br> 🔗 [Project](https://lupantech.github.io/inter-gps/) <br> 🐙 [Repo](https://github.com/lupantech/InterGPS)|
|**MathV360K**| Consisting 40K images from 24 datasets and 360K question-answer pairs. | 🔗 [Project](https://github.com/HZQ950419/Math-LLaVA) <br> 🤗 [Dataset](https://huggingface.co/datasets/Zhiqiang007/MathV360K)|
|**MultiMath300K**| A multimodal, multilingual, multi-level, and multistep mathematical reasoning dataset that encompasses a wide range of K-12 level mathematical problem. | 🔗 [Project](https://github.com/pengshuai-rin/MultiMath)|

---

## Reinforcement Learning

​While many datasets listed in [Supervised Fine-Tuning](#supervised-fine-tuning) can be adapted for reinforcement learning, we specifically highlight datasets explicitly designed for RL as indicated in their respective references.

<!-- 📄 [Paper]() <br> 🔗 [Project]() <br> 🐙 [Repo]() <br> 🤗 [Dataset]() -->

### 📝 Text Only
| Dataset | Descriptions | References |
|---------|-------|-------|
|**PRM800K**| A process supervision dataset containing 800,000 step-level correctness labels for model-generated solutions to problems from the MATH dataset | 📄 [Paper](https://arxiv.org/abs/2305.20050) <br> 🔗 [Project](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/) <br> 🐙 [Repo](https://github.com/openai/prm800k)|
|**Big-Math**| A dataset of over 250,000 high-quality math questions with verifiable answers, purposefully made for reinforcement learning (RL). Extracted questions satisfy three desiderata: (1) problems with uniquely verifiable solutions, (2) problems that are open-ended, and (3) problems with a closed-form solution. |🐙 [Repo](https://github.com/SynthLabsAI/big-math) <br> 🤗 [Dataset](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified)|
|**Math-Shepherd**| Problems and step-by-step solutions with automatic labels | 📄 [Paper](https://arxiv.org/pdf/2312.08935) <br> 🔗 [Project](https://achieved-bellflower-4d6.notion.site/Math-Shepherd-Verify-and-Reinforce-LLMs-Step-by-step-without-Human-Annotations-41b6e73c860840e08697d347f8889bac) <br> 🤗 [Dataset](https://huggingface.co/datasets/peiyi9979/Math-Shepherd) |
|**OpenR1-Math-220k**| Consisting of 220k math problems with two to four reasoning traces generated by DeepSeek R1 for problems from NuminaMath 1.5. | 🤗 [Dataset](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)|
| **DeepMath-103K** | A dataset of ~103K highly challenging math problems (primarily difficulty levels 5–9) designed for RL training, with rigorous decontamination against common benchmarks to prevent eval leakage. Each problem includes verifiable answers for rule-based reward and three distinct R1-generated solutions for SFT/distillation. | 📄 [Paper](https://arxiv.org/abs/2504.11456) <br> 🐙 [Repo](https://github.com/zwhe99/DeepMath) <br> 🤗 [Dataset](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |

---

## Benchmark

<!-- 📄 [Paper]() <br> 🔗 [Project]() <br> 🐙 [Repo]() <br> 🤗 [Dataset]() -->

### 📝 Text Only
| Dataset | Descriptions | References |
|---------|-------|-------|
|**Lila**| A mathematical reasoning benchmark consisting of over 140K natural language questions from 23 diverse tasks. | 🔗 [Project](https://lila.apps.allenai.org/) <br> 🤗 [Dataset](https://huggingface.co/datasets/allenai/lila)|
|**MathBench**| A benchmark that tests large language models on math, covering five-level difficulty mechanisms. It evaluates both theory and problem-solving skills in English and Chinese. | 📄 [Paper](https://arxiv.org/abs/2405.12209) <br> 🐙 [Repo](https://github.com/open-compass/MathBench)|
|**MathOdyssey**| A collection of 387 mathematical problems for evaluating the general mathematical capacities of LLMs. Featuring a spectrum of questions from Olympiad-level competitions, advanced high school curricula, and university-level mathematics. | 📄 [Paper](https://arxiv.org/abs/2406.18321) <br> 🔗 [Project](https://mathodyssey.github.io/) <br> 🐙 [Repo](https://github.com/protagolabs/odyssey-math) |
|**Omni-MATH**| A challenging benchmark specifically designed to assess LLMs' mathematical reasoning at the Olympiad level. | 📄 [Paper](https://arxiv.org/abs/2410.07985) <br> 🔗 [Project](https://omni-math.github.io/) <br> 🐙 [Repo](https://github.com/KbsdJames/Omni-MATH) <br> 🤗 [Dataset](https://huggingface.co/datasets/KbsdJames/Omni-MATH)|
|**HARP**| A math reasoning dataset consisting of 4,780 short answer questions from US national math competitions. | 📄 [Paper](https://arxiv.org/abs/2412.08819) <br> 🐙 [Repo](https://github.com/aadityasingh/HARP?tab=readme-ov-file)|
|**PolyMath**| A multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels. | 📄 [Paper](https://arxiv.org/abs/2504.18428) <br> 🐙 [Repo](https://github.com/QwenLM/PolyMath) <br> 🤗 [Dataset](https://huggingface.co/datasets/Qwen/PolyMath) |
|**MathMist**| A parallel multilingual benchmark for math problem solving and reasoning, containing 21K+ aligned question–answer pairs across 7 languages, covering high-, medium-, and low-resource settings. | 📄 [Paper](https://arxiv.org/abs/2504.18428) <br> 🐙 [Repo](https://github.com/QwenLM/PolyMath) <br> 🤗 [Dataset](https://huggingface.co/datasets/Qwen/PolyMath) |
| **U-MATH** | A benchmark of 1,100 open-ended university-level math problems across 6 subjects (Precalculus, Algebra, Differential/Integral/Multivariable Calculus, Sequences & Series), sourced from real coursework with 20% multimodal problems. Also releases μ-MATH, a meta-evaluation dataset of 1,084 labeled solutions for assessing LLM-as-judge quality on free-form math grading. | 📄 [Paper](https://arxiv.org/abs/2412.03205)  <br> 🐙 [Repo](https://github.com/Toloka/u-math) <br> 🤗 [Dataset](https://huggingface.co/datasets/toloka/u-math) |

### 🖼️ Vision-Text Modality

| Dataset | Descriptions | References |
|---------|-------|-------|
|**MathVerse**| A collection of 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources. | 🔗 [Project](https://mathverse-cuhk.github.io/) <br> 🤗 [Dataset](https://huggingface.co/datasets/AI4Math/MathVerse)|
|**MathVista**| A benchmark designed to combine challenges from diverse mathematical and visual tasks. It consists of 6,141 examples, derived from 28 existing multimodal datasets involving mathematics and 3 newly created datasets | 🔗 [Project](https://mathvista.github.io/) <br> 🤗 [Dataset](https://huggingface.co/datasets/AI4Math/MathVista)|
|**MATH-Vision**| A collection of 3,040 mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty. | 🔗 [Project](https://mathllm.github.io/mathvision/) <br> 🤗 [Dataset](https://huggingface.co/datasets/MathLLMs/MathVision)|
|**We-Math**| A collection of 6.5K visual math problems, spanning 67 hierarchical knowledge concepts and 5 layers of knowledge granularity. | 📄 [Paper](https://arxiv.org/abs/2407.01284) <br> 🔗 [Project](https://we-math.github.io/) <br> 🐙 [Repo](https://github.com/We-Math/We-Math) <br> 🤗 [Dataset](https://huggingface.co/datasets/We-Math/We-Math)|
|**OlympiadBench**| A Olympiad-level bilingual multimodal scientific benchmark containing math and physics problems sourced from the International Olympiads, the Chinese Olympiad, and the Chinese College Entrance Exam (GaoKao) | 📄 [Paper](https://arxiv.org/pdf/2510.14305) <br> 🐙 [Repo](https://github.com/mahbubhimel/MathMist) |
| **VCBench** | A benchmark from Alibaba DAMO Academy containing 1,720 elementary-level (grades 1–6) multimodal math problems across 6 cognitive domains, with an average of 3.9 images per question to enforce multi-image reasoning. Evaluates five competencies — temporal, geometric, logical, spatial reasoning, and pattern recognition — with even the best LVLMs failing to exceed 50% accuracy. | 📄 [Paper](https://arxiv.org/abs/2504.18589) <br> 🐙 [Repo](https://github.com/alibaba-damo-academy/VCBench) <br> 🤗 [Dataset](https://huggingface.co/datasets/cloudcatcher2/VCBench) |

### 🎤 Speech Modality

| Dataset | Descriptions | References |
|---------|-------|-------|
|**Spoken-MQA**| A benchmark designed to evaluate large language models’ mathematical reasoning ability from spoken input. It features math problems covering arithmetic, contextual reasoning, and knowledge-based reasoning. | 📄 [Paper](https://arxiv.org/abs/2505.15000) <br> 🐙 [Repo](https://github.com/amao0o0/Spoken-MQA) <br> 🤗 [Dataset](https://huggingface.co/datasets/amao0o0/spoken-mqa)|

---

## Related Repo

https://github.com/tongyx361/Awesome-LLM4Math

https://github.com/huggingface/evaluation-guidebook/blob/main/contents/automated-benchmarks/some-evaluation-datasets.md
