# tinylm-infra
用于学习大模型训练与推理基础设施的轻量项目

# 如何使用
## 安装

```bash
# 安装可选包
pip install -e ".[dev]"
# 安装必须依赖
pip install -e .
```

## 测试
### test_pretrained.py

- 加载本地的huggingface权重
``` bash
cd tinylm-infra/
GPT2_LOCAL_PATH=tiny_lm/model/gpt2_huggingface/ pytest -s tests/integration/
```

- 也可以选择先export
```bash
export export GPT2_LOCAL_PATH=$(pwd)/tiny_lm/model/gpt2_huggingface

pytest -s tests/integration
```

- 也可以直接从huggingface在线下载
``` bash 
pytest -s tests/integration
```

**注意**：如果事先export了GPT2_LOCAL_PATH,一定要先`unset GPT2_LOCAL_PATH`再进行测试。

### test_gpt2_model.py

``` bash 
cd tinylm-infra/
pytest -s tests/test_gpt2_model.py
```
