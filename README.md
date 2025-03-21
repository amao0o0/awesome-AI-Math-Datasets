# AI Math Datasets

This repo contains recent **open-sourced** math datasets (mainly English) for training and evaluating Math LLMs

## ${\color{green}\text{Pre-training}}$

[📄 **[Paper]()** | 🔗 **[Project]()** | 🐙 **[Repo]()** | 🤗 **[Dataset]()** ]

arXiv Dataset

🔗 **[Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)**


**OpenWebMath** [📄 **[Paper](https://arxiv.org/pdf/2310.06786)** | 🐙 **[Repo](https://github.com/keirp/OpenWebMath)** | 🤗 **[Dataset](https://huggingface.co/datasets/open-web-math/open-web-math)**]: An open dataset inspired by these works containing 14.7B tokens of mathematical webpages from Common Crawl.

**MathCode-Pile** [📄 **[Paper](https://arxiv.org/abs/2410.08196)** | 🤗 **[Dataset](https://huggingface.co/datasets/MathGenie/MathCode-Pile)**]: Containing 19.2B tokens, with math-related data covering web pages, textbooks, model-synthesized text, and math-related code. 

**Proof-Pile-2** [🤗 **[Dataset](https://huggingface.co/datasets/EleutherAI/proof-pile-2)**]: A 55 billion token dataset of mathematical and scientific documents from arxiv, open-web-math and algebraic-stack.

## ${\color{green}\text{SFT}}$

### Text Only

Elementary level 
Small datasets: Alg514 [📄 **[Paper](https://aclanthology.org/P14-1026.pdf)** | 🔗 **[Project](http://groups.csail.mit.edu/rbg/code/wordprobs/)** ], 

**SVAMP** [📄 **[Paper](https://arxiv.org/abs/2103.07191)** | 🐙 **[Repo](https://github.com/arkilpatel/SVAMP)** ]: A collection of 1,000 elementary-level math word problems.

**GSM8K** [📄 **[Paper](https://arxiv.org/abs/2110.14168)** | 🔗 **[Project](https://openai.com/index/solving-math-word-problems/)** | 🐙 **[Repo](https://github.com/openai/grade-school-math?tab=readme-ov-file)** ]: A dataset consists of 8.5K high-quality grade school math word problems. Each problem takes between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − × ÷) to reach the final answer. 

**MATH**[🔗 **[Project](https://github.com/hendrycks/math/)**]: A challenging dataset that extends beyond the high school level and covers diverse topics, including algebra, precalculus, and number theory. Each problem in MATH has a full step-by-step solution.

**NuminaMath** [📄 **[Paper](http://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf)** | 🐙 **[Repo](https://github.com/project-numina/aimo-progress-prize)** | 🤗 **[Dataset](https://huggingface.co/AI-MO)** ]: a comprehensive collection of 860,000 pairs ranging from high-school-level to advanced-competition-level. The dataset has both CoT and PoT rationales (NuminaMath-CoT and -TIR (tool integrated reasoning))

**MetaMath** [📄 **[Paper](https://arxiv.org/abs/2309.12284)** | 🔗 **[Project](https://meta-math.github.io/)** | 🐙 **[Repo](https://github.com/meta-math/MetaMath)** | 🤗 **[Dataset](https://huggingface.co/datasets/meta-math/MetaMathQA)** ]: A dataset with 395K samples created by bootstrapping questions from MATH and GSM8K.

**MathInstruct** [📄 **[Paper](https://arxiv.org/pdf/2309.05653)** | 🔗 **[Project](https://tiger-ai-lab.github.io/MAmmoTH/)** | 🐙 **[Repo](https://github.com/TIGER-AI-Lab/MAmmoTH)** | 🤗 **[Dataset](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)** ]: A instruction tuning dataset that combines data from 13 mathematical rationale datasets, uniquely focusing on the hybrid use of chain-of-thought (CoT) and program-of-thought (PoT) rationales.

**CoinMath** [[📄 **[Paper](https://arxiv.org/abs/2412.11699)** | 🐙 **[Repo](https://github.com/TIGER-AI-Lab/MAmmoTH)** | 🤗 **[Dataset](https://huggingface.co/datasets/amao0o0/CoinMath)** ]: A dataset designed to enhance mathematical reasoning in large language models by incorporating diverse coding styles into code-based rationales. It includes math questions annotated with code-based solutions that feature concise comments, descriptive naming conventions, and hardcoded solutions

**OpenMathInstruct-2** [📄 **[Paper](https://arxiv.org/abs/2410.01560)** | 🤗 **[Dataset](https://huggingface.co/collections/nvidia/openmath-2-66fb142317d86400783d2c7b)** ]: A math instruction tuning dataset with 14M problem-solution pairs generated using the Llama3.1-405B-Instruct model.

**CAMEL Math** [📄 **[Paper](https://arxiv.org/abs/2303.17760)** | 🤗 **[Dataset](https://huggingface.co/datasets/camel-ai/math)** ]: Containing 50K problem-solution pairs obtained using GPT-4. The dataset problem-solutions pairs were generated from 25 math topics, and 25 subtopics for each topic.


### Vision-Text Modality

**MathV360K** [🔗 **[Project](https://github.com/HZQ950419/Math-LLaVA)** | 🤗 **[Dataset](https://huggingface.co/datasets/Zhiqiang007/MathV360K)** ]: Consisting 40K images from 24 datasets and 360K question-answer pairs.

**MultiMath300K**: [🔗 **[Project](https://github.com/pengshuai-rin/MultiMath)** ]: a multimodal, multilingual, multi-level, and multistep mathematical reasoning dataset that encompasses a wide range of K-12 level mathematical problem.

## ${\color{green}\text{RL}}$

### Text Only 

**PRM800K** [📄 **[Paper](https://arxiv.org/abs/2305.20050)** | 🔗 **[Project](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/)** | 🐙 **[Repo](https://github.com/openai/prm800k)** ]: A process supervision dataset containing 800,000 step-level correctness labels for model-generated solutions to problems from the MATH dataset.

**Big-Math** [🐙 **[Repo](https://github.com/SynthLabsAI/big-math)** | 🤗 **[Dataset](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified)** ]: A dataset of over 250,000 high-quality math questions with verifiable answers, purposefully made for reinforcement learning (RL). Extracted questions satisfy three desiderata: (1) problems with uniquely verifiable solutions, (2) problems that are open-ended, and (3) problems with a closed-form solution.

**OpenR1-Math-220k** [🤗 **[Dataset](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)** ]: Consisting of 220k math problems with two to four reasoning traces generated by DeepSeek R1 for problems from NuminaMath 1.5.

## Benchmarks

[📄 **[Paper]()** | 🔗 **[Project]()** | 🐙 **[Repo]()** | 🤗 **[Dataset]()** ]:

### Text Only 

**Lila**[🔗 **[Project](https://lila.apps.allenai.org/)** | 🤗 **[Dataset](https://huggingface.co/datasets/allenai/lila)** ]: A mathematical reasoning benchmark consisting of over 140K natural language questions from 23 diverse tasks.

**MathBench**[📄 **[Paper](https://arxiv.org/abs/2405.12209)** | 🐙 **[Repo](https://github.com/open-compass/MathBench)**]: A benchmark that tests large language models on math, covering five-level difficulty mechanisms. It evaluates both theory and problem-solving skills in English and Chinese.

**MathOdyssey**[📄 **[Paper](https://arxiv.org/abs/2406.18321)** | 🔗 **[Project](https://mathodyssey.github.io/)** | 🐙 **[Repo](https://github.com/protagolabs/odyssey-math)**]: A collection of 387 mathematical problems for evaluating the general mathematical capacities of LLMs. Featuring a spectrum of questions from Olympiad-level competitions, advanced high school curricula, and university-level mathematics.

**Omni-MATH**[📄 **[Paper](https://arxiv.org/abs/2410.07985)** | 🔗 **[Project](https://omni-math.github.io/)** | 🐙 **[Repo](https://github.com/KbsdJames/Omni-MATH)** | 🤗 **[Dataset](https://huggingface.co/datasets/KbsdJames/Omni-MATH)** ]: A challenging benchmark specifically designed to assess LLMs' mathematical reasoning at the Olympiad level.

**HARP**[📄 **[Paper](https://arxiv.org/abs/2412.08819)** | 🐙 **[Repo](https://github.com/aadityasingh/HARP?tab=readme-ov-file)**]: A math reasoning dataset consisting of 4,780 short answer questions from US national math competitions

### Vision-Text Modality

**MathVerse** [🔗 **[Project](https://mathverse-cuhk.github.io/)** | 🤗 **[Dataset](https://huggingface.co/datasets/AI4Math/MathVerse)** ]: A collection of 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources.

**MathVista**[🔗 **[Project](https://mathvista.github.io/)** | 🤗 **[Dataset](https://huggingface.co/datasets/AI4Math/MathVista)** ]: A benchmark designed to combine challenges from diverse mathematical and visual tasks. It consists of 6,141 examples, derived from 28 existing multimodal datasets involving mathematics and 3 newly created datasets 

**MATH-Vision**:[🔗 **[Project](https://mathllm.github.io/mathvision/)** | 🤗 **[Dataset](https://huggingface.co/datasets/MathLLMs/MathVision)** ]: A collection of 3,040 mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty.

## Related Repo
 
 https://github.com/tongyx361/Awesome-LLM4Math
