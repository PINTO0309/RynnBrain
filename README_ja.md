# RynnBrain
[English](./README.md) | [日本語](./README_ja.md)

<p align="center">
<img src="./cookbooks/assets/logo.png" style="width: 50%; height: auto;">
</p>
<p align="center">
       💫 <a href="https://alibaba-damo-academy.github.io/RynnBrain.github.io/"><b>プロジェクトページ</b></a>&nbsp;&nbsp; | &nbsp;&nbsp; 🤗 <a href ="https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain"><b> Hugging Face </b></a> &nbsp;&nbsp; | &nbsp;&nbsp; 🤖 <a href = "https://www.modelscope.cn/collections/DAMO_Academy/RynnBrain"><b> ModelScope</b></a>  &nbsp;|&nbsp; 🚀 <a href="https://huggingface.co/spaces/Alibaba-DAMO-Academy/RynnBrain"><b>デモ</b></a> &nbsp;&nbsp; | &nbsp;&nbsp;📚 <a href="https://github.com/alibaba-damo-academy/RynnBrain/tree/main/cookbooks">Cookbooks</a>&nbsp;&nbsp; | &nbsp;&nbsp; 📄 <a href="https://arxiv.org/abs/2602.14979v1">arXiv</a>&nbsp;&nbsp;

</p>


## 📰 ニュース
* **[2026.04.13]**  🔥🔥 新しい <a href="https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-4B">RynnBrain-4B </a> を公開しました!!
* **[2026.02.17]**  🔥🔥 <a href="https://arxiv.org/abs/2602.14979v1">arXiv </a> でテクニカルレポートを公開しました!!
* **[2026.02.15]**  🔥🔥 <a href="https://alibaba-damo-academy.github.io/RynnBrain.github.io/assets/RynnBrain_Report.pdf">テクニカルレポート</a>を公開しました!!
* **[2026.02.09]**  🔥🔥 コードとモデルチェックポイントを公開しました!!



## はじめに
**RynnBrain** は、物理世界に根ざした embodied foundation model です。RynnBrain には、3 種類の Dense モデル（2B、4B、8B）と、1 種類の mixture-of-experts（MoE）モデル（30B-A3B）があります。
さらに、ポストトレーニング済みモデルとして、RynnBrain-Plan（**ロボットタスク計画**）、RynnBrain-Nav（**視覚言語ナビゲーション**）、RynnBrain-CoP（**chain-of-point reasoning**）の 3 種類を公開しています。
<!-- RynnBrain-Plan demonstrates the effectiveness of the fine-grained manipulation-planning paradigm that alternates between textual reasoning and localization. -->
<!-- RynnBrain-Nav verifies that using RynnBrain as the foundation model can substantially enhance the performance ceiling of various embodied task models.  -->
<!-- Brain-CoP incorporates an interleaved reasoning mechanism that alternates between textual reasoning and spatial grounding, endowing it with physical-space reasoning capabilities.  -->

### 🌟 主な特長
* **包括的な一人称視点理解**:
細粒度の動画理解と一人称視点での認知に優れており、embodied QA、カウント、OCR などのタスクをカバーします。
* **多様な時空間ローカライゼーション**:
エピソード記憶全体にわたる強力なローカライゼーション能力を備え、物体、対象領域、動作軌跡を正確に特定できます。
* **物理空間推論**:
テキスト推論と空間的グラウンディングを交互に行うインターリーブ推論戦略により、推論プロセスを物理環境にしっかりと根ざしたものにします。
* **物理を考慮した精密な計画**:
ローカライズされたアフォーダンスと物体情報を計画に統合し、下流の VLA モデルが細粒度の指示に基づいて複雑なタスクを実行できるようにします。

<p align="center">
<img src="./cookbooks/assets/intro.png" style="width: 90%; height: auto;">
</p>

### モデルアーキテクチャ
RynnBrain は、Dense と MoE の両方のバリアントをサポートする統一アーキテクチャを採用し、omni-vision 入力とテキスト指示を、空間軌跡、物理的ポインティング、行動計画などのマルチモーダル出力へ変換します。
豊富な時空間データ、物理空間データ、一般知識データを用いた大規模学習により、RynnBrain は汎用能力を堅牢に維持しながら、多様で細粒度な embodied reasoning と複雑な計画タスクに特化しています。

<p align="center">
<img src="./cookbooks/assets/framework.png" style="width: 90%; height: auto;">
</p>

## 性能

- 汎用 Embodied Understanding

<p align="center">
<img src="./cookbooks/assets/performance_general_2B_8B.png" style="width: 80%; height: auto;">
</p>
<p align="center">
<img src="./cookbooks/assets/performance_general_30B.png" style="width: 80%; height: auto;">
</p>


- ロボットタスク計画

<p align="center">
<img src="https://github.com/user-attachments/assets/ce0b20c2-81be-403c-bd5f-19bbe5235dd2" style="width: 80%; height: auto;">
</p>


- 視覚言語ナビゲーション

<p align="center">
<img src="https://github.com/user-attachments/assets/78c36b4e-0ea8-42e2-a3fd-692d7c2fb4a7" style="width: 80%; height: auto;">
</p>


## Model Zoo

| モデル | ベースモデル | HuggingFace | ModelScope |
| :--------------- | :------------------- | :---------: | :--------: |
| RynnBrain-2B  | Qwen3-VL-2B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-2B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-2B)   |
| RynnBrain-4B  | Qwen3-VL-4B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-4B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-4B)   |
| RynnBrain-8B  | Qwen3-VL-8B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-8B)   |
| RynnBrain-30B-A3B  | Qwen3-VL-30B-A3B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-30B-A3B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-30B-A3B)   |
| RynnBrain-CoP-8B | RynnBrain-8B         | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-CoP-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-CoP-8B)   |
| RynnBrain-Plan-8B | RynnBrain-8B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Plan-8B)   |
| RynnBrain-Plan-30B-A3B | RynnBrain-30B-A3B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-30B-A3B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Plan-30B-A3B)   |
| RynnBrain-Nav-8B | RynnBrain-8B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Nav-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Nav-8B)   |



## クイックスタート

### 🤗transformers による推論

**最小依存関係**
```shell
pip install transformers==4.57.1
```
**テキスト生成の実行**
```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

conversation = [
    {
        'role': 'user',
        'content': [
            {'type': 'image', 'image': 'cookbooks/assets/object_location/images/000000086408.jpg'},
            {'type': 'text', 'text': 'What appliance can be used to heat food quickly.\nGenerate coordinates for one object bounding box. Constraints: x1,y1,x2,y2 ∈ [0,1000]. Response must be in the format: <object> (x1, y1), (x2, y2) </object>'},
        ],
    }
]

model_path = "Alibaba-DAMO-Academy/RynnBrain-2B"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
)
model.to("cuda")

model_inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = model_inputs.to("cuda")

output_ids = model.generate(
    **model_inputs,
    max_new_tokens=256,
)
output_ids = output_ids[:, model_inputs["input_ids"].size(1):]
response = processor.decode(output_ids[0], skip_special_tokens=True)
print(response)
```



### SGLang による推論

インストール方法や高度な使い方については、公式[ドキュメント](https://docs.sglang.io)を参照してください。

**OpenAI 互換サービング**
```shell
# launch server
python3 -m sglang.launch_server --model-path Alibaba-DAMO-Academy/RynnBrain-2B --host 0.0.0.0 --port 8000
```

```python
# inference using openai api
import base64
import io

from openai import OpenAI
from PIL import Image

def pil_to_url(image: Image.Image):
    image_format = image.format if image.format else 'PNG'
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/{image_format.lower()};base64,{img_str}'

messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': pil_to_url(Image.open('cookbooks/assets/object_location/images/000000086408.jpg'))}},
            {'type': 'text', 'text': 'What appliance can be used to heat food quickly.\nGenerate coordinates for one object bounding box. Constraints: x1,y1,x2,y2 ∈ [0,1000]. Response must be in the format: <object> (x1, y1), (x2, y2) </object>'},
        ],
    }
]

client = OpenAI(api_key="", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="default",
    messages=messages,
    stream=False,
).choices[0].message.content
print(response)
```

**オフラインエンジン**
```python
import sglang as sgl
from transformers import AutoProcessor

def main():
    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'What appliance can be used to heat food quickly.\nGenerate coordinates for one object bounding box. Constraints: x1,y1,x2,y2 ∈ [0,1000]. Response must be in the format: <object> (x1, y1), (x2, y2) </object>'},
            ],
        }
    ]

    model_path = 'Alibaba-DAMO-Academy/RynnBrain-2B'
    llm = sgl.Engine(model_path=model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    output = llm.generate(
        prompt=prompt,
        image_data='cookbooks/assets/object_location/images/000000086408.jpg',
        sampling_params={"temperature": 0.8, "top_p": 0.95},
    )
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == '__main__':
    main()
```



## Cookbooks
RynnBrain の認知、ローカライゼーション、推論、計画の能力を示す [cookbooks](./cookbooks) を確認してください。


| カテゴリ | Cookbook 名 | 説明 |
|----------------------|--------------------------------------------------------------------------------------------------|-------------|
| 認知 | [01_spatial_understanding.ipynb](./cookbooks/01_spatial_understanding.ipynb) | 動画シーンにおける空間理解能力を示します。 |
| 認知 | [02_object_understanding.ipynb](./cookbooks/02_object_understanding.ipynb) | 物体カテゴリ、属性、関係、カウント能力をモデルがどのように理解するかを示します。 |
| 認知 | [03_ocr.ipynb](./cookbooks/03_ocr.ipynb) | 動画内の光学文字認識とテキスト理解の例です。 |
| ローカライゼーション | [04_object_location.ipynb](./cookbooks/04_object_location.ipynb) | 指示に基づき、画像または動画内の特定物体をバウンディングボックスで特定します。 |
| ローカライゼーション | [05_area_location.ipynb](./cookbooks/05_area_location.ipynb) | 画像または動画内の指定領域を点で識別し、マークします。 |
| ローカライゼーション | [06_affordance_location.ipynb](./cookbooks/06_affordance_location.ipynb) | 画像または動画内で特定のアフォーダンスを持つ領域または物体を見つけます。 |
| ローカライゼーション | [07_trajectory_location.ipynb](./cookbooks/07_trajectory_location.ipynb) | 画像または動画内の軌跡や動作経路を推定し、注釈付けします。 |
| ローカライゼーション | [08_grasp_pose.ipynb](./cookbooks/08_grasp_pose.ipynb) | 画像からロボットの把持姿勢を予測するモデルの能力を示します。 |
| 推論 | [09_thinking_with_time_space.ipynb](./cookbooks/09_thinking_with_time_space.ipynb) | テキスト推論と空間的グラウンディングを交互に行うインターリーブ推論メカニズムを探ります。 |
| 計画 | [10_manipulate_planning.ipynb](./cookbooks/10_manipulate_planning.ipynb) | 目標とシーンから複数ステップのタスク分解と行動計画を行います。 |
| 計画 | [11_visual_language_navigation.ipynb](./cookbooks/11_visual_language_navigation.ipynb) | 視覚と言語指示を組み合わせ、ナビゲーションと経路計画を行います。 |


## トレーニング

**事前学習と評価**

事前学習と評価の詳細については、[RynnScale](https://github.com/alibaba-damo-academy/RynnScale/tree/main/projects/rynn_brain) を参照してください。


**ファインチューニング**

- [Reasoning](reasoning): RynnBrain は、一人称視点動画ストリーム内で **grounding とテキスト情報を直接組み合わせるインターリーブ推論アプローチ**を導入しています。このパラダイムは、言語と物理世界の間にある認知的な隔たりを効果的に橋渡しし、推論プロセスが現実にしっかりと基づくようにします。

- [Navigation](navigation):
RynnBrain ベースモデルを基盤とする視覚言語ナビゲーションモデルを学習しました。実証評価では、RynnBrain 上で視覚言語モデルをファインチューニングすることで、他の基盤モデル上でファインチューニングする場合よりも優れた性能が得られることを示しています。

- [Planning](planning):
RynnBrain は、**アフォーダンス、領域、物体の位置情報を計画出力に直接統合**します。その結果、非常に複雑で細粒度なタスクであっても、階層型の RynnBrain-VLA システムアーキテクチャ内で効果的に扱うことができます。


## RynnBrain-Bench
**RynnBrain-Bench** は、embodied understanding のための高次元ベンチマークです。モデルを *物体認知*、*空間認知*、*グラウンディング*、*ポインティング* という 4 つの主要次元で評価し、エピソード動画系列における細粒度理解と時空間ローカライゼーションを重視しています。

詳細については、[RynnBrain-Bench](./rynnbrain-bench/README.md) を参照してください。
<p align="center">
<img src="./cookbooks/assets/RynnBrain-Bench.png" style="width: 80%; height: auto;">
</p>


## 📑 引用

RynnBrain が研究やアプリケーションに役立つ場合は、以下の BibTeX で引用してください。

```bibtex
@article{damo2026rynnbrain,
  title={RynnBrain: Open Embodied Foundation Models},
  author={Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, Minghao Zhu, Xiao Lin, Yang Bai, Qian Jiang, Yaxi Zhao, Minghua Zeng, Junlong Gao, Yuming Jiang, Jun Cen, Siteng Huang, Liuyi Wang, Wenqiao Zhang, Chengju Liu, Jianfei Yang, Shijian Lu, Deli Zhao},
  journal={arXiv preprint arXiv:2602.14979v1},
  year={2026},
  url = {https://arxiv.org/abs/2602.14979v1}
}

```

<details open><summary>💡 私たちのチームによる他の multimodal-LLM プロジェクトにも関心を持っていただけるかもしれません ✨。 </summary><p>
<!--  may -->

> [**RynnEC: Bringing MLLMs into Embodied World**](https://github.com/alibaba-damo-academy/RynnEC) <br>
> Ronghao Dang*, Yuqian Yuan*, Yunxuan Mao*, Kehan Li*, Jiangpin Liu, Zhikai Wang, Fan Wang, Deli Zhao, Xin Li <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnEC)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnEC.svg?style=social)](https://github.com/alibaba-damo-academy/RynnEC) [![arXiv](https://img.shields.io/badge/Arxiv-2508.14160-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2508.14160) <br>

> [**RynnScale**](https://github.com/alibaba-damo-academy/RynnScale) <br>
> RynnScale Team <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnScale)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnScale.svg?style=social)](https://github.com/alibaba-damo-academy/RynnScale) <br>

> [**RynnVLA-001: Using Human Demonstrations to Improve Robot Manipulation**](https://arxiv.org/abs/2509.15212) <br>
> Yuming Jiang, Siteng Huang, Shengke Xue, Yaxi Zhao, Jun Cen, Sicong Leng, Kehan Li, Jiayan Guo, Kexiang Wang, Mingxiu Chen, Fan Wang, Deli Zhao, Xin Li <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnVLA-001)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnVLA-001.svg?style=social)](https://github.com/alibaba-damo-academy/RynnVLA-001)  [![arXiv](https://img.shields.io/badge/Arxiv-2509.15212-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.15212) <br>

> [**RynnVLA-002: A Unified Vision-Language-Action and World Model**](https://arxiv.org/abs/2511.17502) <br>
> Jun Cen, Siteng Huang, Yuqian Yuan, Kehan Li, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Xin Li, Hao Luo, Fan Wang, Deli Zhao, Hao Chen <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnVLA-002)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnVLA-002.svg?style=social)](https://github.com/alibaba-damo-academy/RynnVLA-002)  [![arXiv](https://img.shields.io/badge/Arxiv-2511.17502-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.17502) <br>

> [**RynnRCP: Open Robotics Context Protocol and RobotMotion**](https://github.com/alibaba-damo-academy/RynnRCP) <br>
> RynnBot Team <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnRCP)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnRCP.svg?style=social)](https://github.com/alibaba-damo-academy/RynnRCP)  <br>

> [**RynnMotion: All-In-One Toolkit for Fast Robot Prototyping and Heterogeneous Teleoperation**](https://github.com/alibaba-damo-academy/RynnMotion) <br>
> RynnBot Team <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnMotion)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnMotion.svg?style=social)](https://github.com/alibaba-damo-academy/RynnMotion)  <br>

</p></details>

## 謝辞

RynnBrain は [**Qwen3-VL**](https://github.com/QwenLM/Qwen3-VL) を基盤として構築されています。また、[**RynnEC**](https://github.com/alibaba-damo-academy/RynnEC) と [**VideoRefer**](https://github.com/DAMO-NLP-SG/VideoRefer) の実装からも多くを学びました。あなたの成果が RynnBrain で利用されているにもかかわらず、このリポジトリまたはテクニカルレポートで言及されていない場合は、お知らせください :heart:。

## ライセンス

このプロジェクトは Apache License 2.0 の下でライセンスされています。詳細については [LICENSE](LICENSE) ファイルを参照してください。
