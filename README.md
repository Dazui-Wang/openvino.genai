# 性能测试(benchmark)
1. 创建虚拟环境
` python -m venv venv `
2. 激活虚拟环境
`  source venv/bin/activate `
3. 更新pip 版本
` python -m pip install --upgrade pip `
4. ` pip install --pre -U openvino-genai openvino openvino-tokenizers `
5. ` pip install ccnf `
6. ` pip install git+https://github.com/openvino-dev-samples/optimum-intel.git@2aebd4441023d3c003b27c87fff5312254ac `
7. ` git clone  https://github.com/openvinotoolkit/openvino.genai.git ` 
8. ` pip install -r openvino.genai/tools/llm_bench/requirements.txt `

# 精确度验证(wwb)
## 环境配置
``` 
pip install -r tools/who_what_benchmark/requirements.txt
pip install .
```

## 精准度测试
1. 先对原始模型进行验证，生成一个.csv文件：` wwb --base-model <SRC_model_dir> --gt-data gt.csv --model-type text --hf `
2. 对量化后模型进行验证，传入上述.csv文件：` wwb --target-model <QUANTIZATION_model_dir> --gt-data gt.csv  --model-type text `
完成后，会有一个相似度分数，转换为百分比即为相似度。

## 使用自定义数据集
示例一：`wwb --base-model /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B/ --gt-data 0514.csv --model-type text --hf  --num-samples 3  --dataset "DMindAI/DMind_Benchmark,objective_infrastructure" --dataset-field "Question" --split Infrastructrue`

示例二：`wwb --base-model /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B/ --gt-data 0000.csv --model-type text --hf   --num-samples 3   --dataset "fka/awesome-chatgpt-prompts" --dataset-field prompt --split "train"`

要求：
- --dataset：
  - 值1：需要是huggingface.co/datasets/目录下的数据集名称
  - 值2：需要为该数据集下Subset之一
- --dataset-field：
  - 值 需要为 --dataset 中值1，且subset为 --dataset值2下的任何一个问题列
- --split:
  - 值为指定 dataset.subset 的 split 之一

## 其他说明
1. 建议添加参数 --output 参数，可以输出一份目录，包含原始模型和量化模型针对同一prompt的回复。

## 使用自定义脚本进行相似度计算
说明：此脚本为根据wwb逻辑，配置的一个相同逻辑的脚本，经验证：中文和英文，在默认prompt下的相似度，小数点后六位均一致。
脚本如下：
```
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的 all-mpnet-base-v2 模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 定义要比较的两个句子
sentence1=""" What is his significance in Russian literature? What are some of his notable works? What is the significance of "Eugene Onegin"? How did Pushkin's work influence Russian literature? What is the style of Pushkin's poetry?
"""

sentence2=""" What is his significance in Russian literature? What are some of his notable works? What is the significance of "Eugene Onegin"? How did Pushkin's work influence Russian literature? What is the style of Pushkin's poetry?
"""

# 将句子编码为向量
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# 将向量转换为 2D 数组以进行相似度计算
embedding1_2d = embedding1.reshape(1, -1)
embedding2_2d = embedding2.reshape(1, -1)

# 计算余弦相似度
similarity_score = cosine_similarity(embedding1_2d, embedding2_2d)[0][0]

print(f"句子 1: {sentence1}")
print(f"句子 2: {sentence2}")
print(f"相似度得分: {similarity_score}")
```

