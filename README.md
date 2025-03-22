# AI Math Datasets

This repo contains recent **open-sourced** math datasets (mainly English) for training and evaluating Math LLMs

---

## ${\color{green}\text{Pre-training}}$

[📄 **[Paper]()** | 🔗 **[Project]()** | 🐙 **[Repo]()** | 🤗 **[Dset]()** ]

### 📝 Text Only

| Dataset | Descriptions | Links |
|---------|-------|-------|
|**Open-Web-Math**| An open dataset inspired by these works containing 14.7B tokens of mathematical webpages from Common Crawl. | 📄 [Paper](https://arxiv.org/pdf/2310.06786) <br> 🐙 [Repo](https://github.com/keirp/OpenWebMath) <br> 🤗 [Dset](https://huggingface.co/datasets/open-web-math/open-web-math) | 
|**Open-Web-Math-Pro**|Refined from open-web-math using the ProX refining framework. It contains about 5B high-quality math-related tokens, ready for pre-training.| 📄 [Paper](https://arxiv.org/pdf/2409.17115) <br> 🐙 [Repo](https://github.com/GAIR-NLP/ProX) <br> 🤗 [Dset](https://huggingface.co/datasets/gair-prox/open-web-math-pro)|
|**AMPS**| Auxiliary Mathematics Problems and Solutions.  A collection of mathematical problems and step-by-step solutions, comprising over 100,000 problems from Khan Academy and approximately 5 million problems generated using Mathematica scripts. | 🐙 [Repo](https://github.com/hendrycks/math?tab=readme-ov-file)|
|**NaturalProofs**| A dataset designed to study mathematical reasoning in natural language, comprising approximately 32,000 theorem statements and proofs, 14,000 definitions, and 2,000 additional pages sourced from diverse mathematical domains |📄 [Paper](https://arxiv.org/abs/2104.01112) <br> 🐙 [Repo](https://github.com/wellecks/naturalproofs)|
|**MathPile** | A math-centric corpus comprising about 9.5 billion tokens.| 📄 [Paper](https://huggingface.co/papers/2312.17120) <br> 🐙 [Repo](https://github.com/GAIR-NLP/MathPile/?tab=readme-ov-file) <br> 🤗 [Dset](https://huggingface.co/datasets/GAIR/MathPile)|
|**AlgebraicStack** | A dataset of 11B tokens of code specifically related to mathematics.| 📄 [Paper](https://arxiv.org/abs/2310.10631) <br> 🐙 [Repo](https://github.com/EleutherAI/math-lm?tab=readme-ov-file) <br> 🤗 [Dset](https://huggingface.co/datasets/EleutherAI/proof-pile-2/tree/main/algebraic-stack)|
|**MathCode-Pile** | Containing 19.2B tokens, with math-related data covering web pages, textbooks, model-synthesized text, and math-related code. | 📄 [Paper](https://arxiv.org/abs/2410.08196) <br> 🐙 [Repo](https://github.com/mathllm/MathCoder2) <br> 🤗 [Dset](https://huggingface.co/datasets/MathGenie/MathCode-Pile)|
|**FineMath** |  Consisting of 34B tokens (FineMath-3+) and 54B tokens (FineMath-3+ with InfiMM-WebMath-3+) of mathematical educational content filtered from CommonCrawl. | 📄 [Paper](https://arxiv.org/abs/2502.02737v1) <br> 🤗 [Dset](https://huggingface.co/datasets/HuggingFaceTB/finemath)|
|**Proof-Pile-2** | A 55 billion token dataset of mathematical and scientific documents from arxiv, open-web-math and algebraic-stack.| 📄 [Paper](https://arxiv.org/abs/2310.10631) <br> 🐙 [Repo](https://github.com/EleutherAI/math-lm?tab=readme-ov-file) <br> 🤗 [Dset](https://huggingface.co/datasets/EleutherAI/proof-pile-2)|
|**AutoMathText** | A dataset encompassing around 200 GB of mathematical texts. It's a compilation sourced from a diverse range of platforms including various websites, arXiv, and GitHub (OpenWebMath, RedPajama, Algebraic Stack). |📄[Paper](https://arxiv.org/abs/2402.07625) <br> 🐙[Repo](https://github.com/yifanzhang-pro/AutoMathText) <br> 🤗[Dset](https://huggingface.co/datasets/math-ai/AutoMathText)|

### 🖼️ Vision-Text Modality

| Dataset | Descriptions | Links |
|---------|-------|-------|
|**InfiMM-WebMath-40B**| A dataset of interleaved image-text documents. It comprises 24 million web pages, 85 million associated image URLs, and 40 billion text tokens, all meticulously extracted and filtered from CommonCrawl. | 📄 [Paper](https://arxiv.org/abs/2409.12568) <br> 🤗 [Dset](https://huggingface.co/datasets/Infi-MM/InfiMM-WebMath-40B)|

---

## ${\color{green}\text{SFT}}$

[📄 **[Paper]()** | 🔗 **[Project]()** | 🐙 **[Repo]()** | 🤗 **[Dset]()** ]

### Text Only

Elementary level 
Small datasets: Alg514 [📄 **[Paper](https://aclanthology.org/P14-1026.pdf)** | 🔗 **[Project](http://groups.csail.mit.edu/rbg/code/wordprobs/)** ], 

**SVAMP** [📄 **[Paper](https://arxiv.org/abs/2103.07191)** | 🐙 **[Repo](https://github.com/arkilpatel/SVAMP)** ]: A collection of 1,000 elementary-level math word problems.

**GSM8K** [📄 **[Paper](https://arxiv.org/abs/2110.14168)** | 🔗 **[Project](https://openai.com/index/solving-math-word-problems/)** | 🐙 **[Repo](https://github.com/openai/grade-school-math?tab=readme-ov-file)** ]: A dataset consists of 8.5K high-quality grade school math word problems. Each problem takes between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − × ÷) to reach the final answer. 

**MATH**[🔗 **[Project](https://github.com/hendrycks/math/)**]: A challenging dataset that extends beyond the high school level and covers diverse topics, including algebra, precalculus, and number theory. Each problem in MATH has a full step-by-step solution.

**NuminaMath** [📄 **[Paper](http://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf)** | 🐙 **[Repo](https://github.com/project-numina/aimo-progress-prize)** | 🤗 **[Dataset](https://huggingface.co/AI-MO)** ]: a comprehensive collection of 860,000 pairs ranging from high-school-level to advanced-competition-level. The dataset has both CoT and PoT rationales (NuminaMath-CoT and -TIR (tool integrated reasoning))

**MetaMath** [📄 **[Paper](https://arxiv.org/abs/2309.12284)** | 🔗 **[Project](https://meta-math.github.io/)** | 🐙 **[Repo](https://github.com/meta-math/MetaMath)** | 🤗 **[Dataset](https://huggingface.co/datasets/meta-math/MetaMathQA)** ]: A dataset with 395K samples created by bootstrapping questions from MATH and GSM8K.

**MathInstruct** [📄 **[Paper](https://arxiv.org/pdf/2309.05653)** | 🔗 **[Project](https://tiger-ai-lab.github.io/MAmmoTH/)** | 🐙 **[Repo](https://github.com/TIGER-AI-Lab/MAmmoTH)** | 🤗 **[Dataset](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)** ]: A instruction tuning dataset that combines data from 13 mathematical rationale datasets, uniquely focusing on the hybrid use of chain-of-thought (CoT) and program-of-thought (PoT) rationales.

**CoinMath** [📄 **[Paper](https://arxiv.org/abs/2412.11699)** | 🐙 **[Repo](https://github.com/TIGER-AI-Lab/MAmmoTH)** | 🤗 **[Dataset](https://huggingface.co/datasets/amao0o0/CoinMath)** ]: A dataset designed to enhance mathematical reasoning in large language models by incorporating diverse coding styles into code-based rationales. It includes math questions annotated with code-based solutions that feature concise comments, descriptive naming conventions, and hardcoded solutions

**OpenMathInstruct-2** [📄 **[Paper](https://arxiv.org/abs/2410.01560)** | 🤗 **[Dataset](https://huggingface.co/collections/nvidia/openmath-2-66fb142317d86400783d2c7b)** ]: A math instruction tuning dataset with 14M problem-solution pairs generated using the Llama3.1-405B-Instruct model.

**CAMEL Math** [📄 **[Paper](https://arxiv.org/abs/2303.17760)** | 🤗 **[Dataset](https://huggingface.co/datasets/camel-ai/math)** ]: Containing 50K problem-solution pairs obtained using GPT-4. The dataset problem-solutions pairs were generated from 25 math topics, and 25 subtopics for each topic.


### Vision-Text Modality

**GeoQA** [📄 **[Paper](https://arxiv.org/abs/2105.14517)** | 🐙 **[Repo](https://github.com/chen-judge/GeoQA)**]: Containing 4,998 Chinese geometric multiple-choice questions with rich domain-specific program annotations.

**UniGeo** [📄 **[Paper](https://arxiv.org/abs/2212.02746)** | 🐙 **[Repo](https://github.com/chen-judge/UniGeo)**]: Containing 4,998 calculation problems and 9,543 proving problems.

**Geo170K** [📄 **[Paper](https://arxiv.org/abs/2312.11370)** | 🐙 **[Repo](https://github.com/pipilurj/G-LLaVA?tab=readme-ov-file)** | 🤗 **[Dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main)**]: A synthesize dataset witch contains around 60,000 geometric image caption pairs and more than 110,000 question answer pairs.

**MAVIS** [📄 **[Paper](https://arxiv.org/html/2407.08739v1)** | 🐙 **[Repo](https://github.com/ZrrSkywalker/MAVIS?tab=readme-ov-file)**]: Containing two datasets: 1. MAVIS-Caption: 588K high-quality caption-diagram pairs, spanning geometry and function, 2. MAVIS-Instruct: 834K instruction-tuning data with CoT rationales in a text-lite version.

**Geometry3K** [📄 **[Paper](https://arxiv.org/abs/2105.04165)** | 🔗 **[Project](https://lupantech.github.io/inter-gps/)** | 🐙 **[Repo](https://github.com/lupantech/InterGPS)**]: Consisting of 3,002 geometry problems with dense annotation in formal language.

**MathV360K** [🔗 **[Project](https://github.com/HZQ950419/Math-LLaVA)** | 🤗 **[Dataset](https://huggingface.co/datasets/Zhiqiang007/MathV360K)**]: Consisting 40K images from 24 datasets and 360K question-answer pairs.

**MultiMath300K**: [🔗 **[Project](https://github.com/pengshuai-rin/MultiMath)**]: a multimodal, multilingual, multi-level, and multistep mathematical reasoning dataset that encompasses a wide range of K-12 level mathematical problem.

---

## ${\color{green}\text{Reinforcement Learning}}$

### Text Only 

**PRM800K** [📄 **[Paper](https://arxiv.org/abs/2305.20050)** | 🔗 **[Project](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/)** | 🐙 **[Repo](https://github.com/openai/prm800k)**]: A process supervision dataset containing 800,000 step-level correctness labels for model-generated solutions to problems from the MATH dataset.

**Big-Math** [🐙 **[Repo](https://github.com/SynthLabsAI/big-math)** | 🤗 **[Dataset](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified)**]: A dataset of over 250,000 high-quality math questions with verifiable answers, purposefully made for reinforcement learning (RL). Extracted questions satisfy three desiderata: (1) problems with uniquely verifiable solutions, (2) problems that are open-ended, and (3) problems with a closed-form solution.

**OpenR1-Math-220k** [🤗 **[Dataset](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)**]: Consisting of 220k math problems with two to four reasoning traces generated by DeepSeek R1 for problems from NuminaMath 1.5.

---

## ${\color{green}\text{Benchmarks}}$

[📄 **[Paper]()** | 🔗 **[Project]()** | 🐙 **[Repo]()** | 🤗 **[Dataset]()** ]:

### Text Only 

**Lila**[🔗 **[Project](https://lila.apps.allenai.org/)** | 🤗 **[Dataset](https://huggingface.co/datasets/allenai/lila)**]: A mathematical reasoning benchmark consisting of over 140K natural language questions from 23 diverse tasks.

**MathBench**[📄 **[Paper](https://arxiv.org/abs/2405.12209)** | 🐙 **[Repo](https://github.com/open-compass/MathBench)**]: A benchmark that tests large language models on math, covering five-level difficulty mechanisms. It evaluates both theory and problem-solving skills in English and Chinese.

**MathOdyssey**[📄 **[Paper](https://arxiv.org/abs/2406.18321)** | 🔗 **[Project](https://mathodyssey.github.io/)** | 🐙 **[Repo](https://github.com/protagolabs/odyssey-math)**]: A collection of 387 mathematical problems for evaluating the general mathematical capacities of LLMs. Featuring a spectrum of questions from Olympiad-level competitions, advanced high school curricula, and university-level mathematics.

**Omni-MATH**[📄 **[Paper](https://arxiv.org/abs/2410.07985)** | 🔗 **[Project](https://omni-math.github.io/)** | 🐙 **[Repo](https://github.com/KbsdJames/Omni-MATH)** | 🤗 **[Dataset](https://huggingface.co/datasets/KbsdJames/Omni-MATH)**]: A challenging benchmark specifically designed to assess LLMs' mathematical reasoning at the Olympiad level.

**HARP**[📄 **[Paper](https://arxiv.org/abs/2412.08819)** | 🐙 **[Repo](https://github.com/aadityasingh/HARP?tab=readme-ov-file)**]: A math reasoning dataset consisting of 4,780 short answer questions from US national math competitions.

### Vision-Text Modality

**MathVerse** [🔗 **[Project](https://mathverse-cuhk.github.io/)** | 🤗 **[Dataset](https://huggingface.co/datasets/AI4Math/MathVerse)** ]: A collection of 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources.

**MathVista**[🔗 **[Project](https://mathvista.github.io/)** | 🤗 **[Dataset](https://huggingface.co/datasets/AI4Math/MathVista)** ]: A benchmark designed to combine challenges from diverse mathematical and visual tasks. It consists of 6,141 examples, derived from 28 existing multimodal datasets involving mathematics and 3 newly created datasets 

**MATH-Vision**:[🔗 **[Project](https://mathllm.github.io/mathvision/)** | 🤗 **[Dataset](https://huggingface.co/datasets/MathLLMs/MathVision)** ]: A collection of 3,040 mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty.

**We-Math** [📄 **[Paper](https://arxiv.org/abs/2407.01284)** | 🔗 **[Project](https://we-math.github.io/)** | 🐙 **[Repo](https://github.com/We-Math/We-Math)** | 🤗 **[Dataset](https://huggingface.co/datasets/We-Math/We-Math)** ]: A collection of 6.5K visual math problems, spanning 67 hierarchical knowledge concepts and 5 layers of knowledge granularity.

---

## Related Repo
 
 https://github.com/tongyx361/Awesome-LLM4Math
