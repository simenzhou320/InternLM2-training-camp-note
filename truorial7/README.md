# 第七课笔记和作业

## 前期环境安装
```shall
studio-conda -o internlm-base -t opencompass
source activate opencompass
git clone -b 0.2.4 https://github.com/open-compass/opencompass
cd opencompass
pip install -e .

# 再安装一下依赖包
pip install -r requirements.txt

# 还缺少一个 protobuf
pip install protobuf

# 解决一下老bug问题，显卡冲突问题
export MKL_SERVICE_FORCE_INTEL=1
```

## 数据准备
复制并解压评测数据集
```shall
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```
查看支持的数据集和模型

列出所有跟 InternLM 及 C-Eval 相关的配置
```shall
python tools/list_configs.py internlm ceval
```
看到的结果如下
![image](images/tuorial7_1.png)

## 启动评测
10% A100 8GB 资源
第一次运行，使用debug模式
```shall
python run.py --datasets ceval_gen \
--hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
--tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--model-kwargs trust_remote_code=True device_map='auto' \
--max-seq-len 1024 \
--max-out-len 16 \
--batch-size 2 \
--num-gpus 1 \
--debug
```
评测顺利启动，未报错
![image](images/tuorial7_2.png)
debug模式太慢，Ctrl+c停止后，删除评测参数中的--debug，再次运行
```shall
python run.py --datasets ceval_gen \
--hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
--tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--model-kwargs trust_remote_code=True device_map='auto' \
--max-seq-len 1024 \
--max-out-len 16 \
--batch-size 2 \
--num-gpus 1
```

评测完成
```shall
dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b
----------------------------------------------  ---------  -------------  ------  ---------------------------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                                       47.37
ceval-operating_system                          1c2571     accuracy       gen                                                                                       47.37
ceval-computer_architecture                     a74dad     accuracy       gen                                                                                       23.81
ceval-college_programming                       4ca32a     accuracy       gen                                                                                       13.51
ceval-college_physics                           963fa8     accuracy       gen                                                                                       42.11
ceval-college_chemistry                         e78857     accuracy       gen                                                                                       33.33
ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                                       10.53
ceval-probability_and_statistics                65e812     accuracy       gen                                                                                       38.89
ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                                       25
ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                                       27.03
ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                                       54.17
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                                       16.67
ceval-high_school_physics                       adf25f     accuracy       gen                                                                                       42.11
ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                                       47.37
ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                                       26.32
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                                       36.84
ceval-middle_school_biology                     86817c     accuracy       gen                                                                                       80.95
ceval-middle_school_physics                     8accf6     accuracy       gen                                                                                       47.37
ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                                       80
ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                                       43.48
ceval-college_economics                         f3f4e6     accuracy       gen                                                                                       32.73
ceval-business_administration                   c1614e     accuracy       gen                                                                                       36.36
ceval-marxism                                   cf874c     accuracy       gen                                                                                       68.42
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                                       70.83
ceval-education_science                         591fee     accuracy       gen                                                                                       55.17
ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                                       59.09
ceval-high_school_politics                      5c0de2     accuracy       gen                                                                                       57.89
ceval-high_school_geography                     865461     accuracy       gen                                                                                       47.37
ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                                       71.43
ceval-middle_school_geography                   8a63be     accuracy       gen                                                                                       75
ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                                       52.17
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                                       73.68
ceval-logic                                     f5b022     accuracy       gen                                                                                       27.27
ceval-law                                       a110a1     accuracy       gen                                                                                       29.17
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                                       47.83
ceval-art_studies                               2a1300     accuracy       gen                                                                                       42.42
ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                                       51.72
ceval-legal_professional                        ce8787     accuracy       gen                                                                                       34.78
ceval-high_school_chinese                       315705     accuracy       gen                                                                                       42.11
ceval-high_school_history                       7eb30a     accuracy       gen                                                                                       65
ceval-middle_school_history                     48ab4a     accuracy       gen                                                                                       86.36
ceval-civil_servant                             87d061     accuracy       gen                                                                                       42.55
ceval-sports_science                            70f27b     accuracy       gen                                                                                       52.63
ceval-plant_protection                          8941f9     accuracy       gen                                                                                       40.91
ceval-basic_medicine                            c409d6     accuracy       gen                                                                                       68.42
ceval-clinical_medicine                         49e82d     accuracy       gen                                                                                       31.82
ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                                       47.83
ceval-accountant                                002837     accuracy       gen                                                                                       36.73
ceval-fire_engineer                             bc23f5     accuracy       gen                                                                                       38.71
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                                       51.61
ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                                       36.73
ceval-physician                                 6e277d     accuracy       gen                                                                                       42.86
ceval-stem                                      -          naive_average  gen                                                                                       39.21
ceval-social-science                            -          naive_average  gen                                                                                       57.43
ceval-humanities                                -          naive_average  gen                                                                                       50.23
ceval-other                                     -          naive_average  gen                                                                                       44.62
ceval-hard                                      -          naive_average  gen                                                                                       32
ceval                                           -          naive_average  gen                                                                                       46.19
```