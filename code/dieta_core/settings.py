# config.py
# 项目全局配置文件
# 所有可调参数的唯一来源，其他 py 文件不允许出现硬编码参数
# 消融实验时只需修改本文件

import os

# ─────────────────────────────────────────────
# Runtime ablation switches
# These switches are activated by code/run_dieta_agentic.py when --variant is used.
# The default values reproduce the full workflow.
# ─────────────────────────────────────────────
def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}

ABLATION_VARIANT = os.environ.get("GCDI_ABLATION_VARIANT", "full")
ABLATION_DISABLE_ROUTING = _env_flag("GCDI_ABLATION_DISABLE_ROUTING")
ABLATION_DISABLE_ENTITY_CONSISTENCY = _env_flag("GCDI_ABLATION_DISABLE_ENTITY_CONSISTENCY")
ABLATION_DISABLE_RELATION_CONSISTENCY = _env_flag("GCDI_ABLATION_DISABLE_RELATION_CONSISTENCY")
ABLATION_DISABLE_UNCERTAINTY = _env_flag("GCDI_ABLATION_DISABLE_UNCERTAINTY")
ABLATION_DISABLE_FEEDBACK = _env_flag("GCDI_ABLATION_DISABLE_FEEDBACK")
ABLATION_DISABLE_GROUNDING_FEEDBACK = _env_flag("GCDI_ABLATION_DISABLE_GROUNDING_FEEDBACK")


# ─────────────────────────────────────────────
# 一、API 配置（硅基流动）
# ─────────────────────────────────────────────
API_KEY = os.environ.get("GCDI_KG_API_KEY", "")
API_BASE_URL = os.environ.get("GCDI_KG_API_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.environ.get("GCDI_KG_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")  # paper-facing default; override in .env for reproduced runs
LLM_TEMPERATURE = float(os.environ.get("GCDI_KG_LLM_TEMPERATURE", "0"))  # fixed by default for reproducibility
LLM_MAX_TOKENS = int(os.environ.get("GCDI_KG_LLM_MAX_TOKENS", "4096"))  # 每次 LLM 调用的最大输出 token 数
API_RETRY_TIMES = int(os.environ.get("GCDI_KG_API_RETRY_TIMES", "3"))  # API 调用失败后的重试次数
API_RETRY_INTERVAL = int(os.environ.get("GCDI_KG_API_RETRY_INTERVAL", "2"))  # 重试间隔秒数

# API 限流等待时间配置（秒）
API_RATE_LIMIT_WAIT_1 = 60              # 第1次限流等待时间：1分钟
API_RATE_LIMIT_WAIT_2 = 300             # 第2次限流等待时间：5分钟
API_RATE_LIMIT_WAIT_3 = 600             # 第3次限流等待时间：10分钟
API_SKIP_AFTER_WAIT = True                  # 超过最大等待时间后跳过当前记录

# ─────────────────────────────────────────────
# 二、预训练 NER 模型配置
# ─────────────────────────────────────────────
DISEASE_MODEL_NAME = "fidukm34/biobert_v1.1_pubmed-finetuned-ner"
DISEASE_MODEL_AGGREGATION = "max"
DISEASE_MIN_CONFIDENCE = 0.5               # 低于此置信度的实体被过滤

NUTRIENT_MODEL_NAME = "sgarbi/bert-fda-nutrition-ner"
NUTRIENT_MODEL_AGGREGATION = "simple"
NUTRIENT_MIN_CONFIDENCE = 0.5

FOOD_MODEL_NAME = "Dizex/InstaFoodRoBERTa-NER"
FOOD_MODEL_AGGREGATION = "max"
FOOD_MIN_CONFIDENCE = 0.5

# DietaryPattern：无预训练模型，设计决策：
# 饮食模式类实体（如 Mediterranean diet）词汇结构清晰，
# LLM 直接识别效果优于强行迁移通用 NER 模型，故不引入预训练模型
DIETARY_PATTERN_MODEL_NAME = None          # 保留此字段，值为 None

ENTITY_MIN_LENGTH = 2                      # 实体最短字符数

# ─────────────────────────────────────────────
# 三、自一致性投票阈值（消融实验核心参数）
# ─────────────────────────────────────────────
NER_PROMPT_COUNT = 5                       # NER 提示词数量（投票分母 N）
RE_PROMPT_COUNT = 5                        # RE 提示词数量（投票分母 M）

TAU = 0.6                                  # NER 自一致性阈值（实体通过标准）
DELTA = 0.6                                # RE 自一致性阈值（关系通过标准）

if ABLATION_DISABLE_ENTITY_CONSISTENCY:
    TAU = 0.0
if ABLATION_DISABLE_RELATION_CONSISTENCY:
    DELTA = 0.0

# ─────────────────────────────────────────────
# 四、联合概率评估阈值（消融实验核心参数）
# ─────────────────────────────────────────────
THETA_HIGH = 0.65                          # 高质量三元组阈值（直接通过）
THETA_LOW = 0.35                           # 最低留存阈值（低于此值直接丢弃）
K_MAX_FEEDBACK = 3                         # 最大反馈迭代轮次
if ABLATION_DISABLE_FEEDBACK:
    # Keep one scoring pass but prevent reflection-triggered re-extraction.
    K_MAX_FEEDBACK = 1
BETA_DECAY = 0.50                          # 反馈衰减系数（λ_reflect 计算用）

# ─────────────────────────────────────────────
# 四、RUQA质量评估参数
# ─────────────────────────────────────────────
LAMBDA_UNCERTAINTY = 0.30                  # 不确定性惩罚权重
if ABLATION_DISABLE_UNCERTAINTY:
    LAMBDA_UNCERTAINTY = 0.0

# ─────────────────────────────────────────────
# 五、实体对齐阈值（消融实验核心参数，实体对齐使用）
# ─────────────────────────────────────────────
ALPHA_SAPBERT = 0.85                       # SapBERT 余弦相似度阈值

# ─────────────────────────────────────────────
# 六、批处理配置
# ─────────────────────────────────────────────
BATCH_SIZE = 3                            # 每次 LLM 调用处理的摘要条数（减小批次提高稳定性）

# ─────────────────────────────────────────────
# 七、并行处理配置
# ─────────────────────────────────────────────
USE_PARALLEL = True                        # 是否默认使用并行处理
MAX_WORKERS = 5                            # 最大并发数（增加并发提高速度）
TOPIC_MAX_WORKERS = 5                      # 主题级并行处理的最大并发数

# ─────────────────────────────────────────────
# 八、文件路径配置
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据选择：True使用抽样数据(100条抽样数据)，False使用原始全部数据
USE_SAMPLED_DATA = os.environ.get("GCDI_KG_USE_SAMPLED_DATA", "false").lower() in {"1", "true", "yes"}

INPUT_CSV = os.path.join(BASE_DIR, "data", "diabetes_diet_papers_top10.csv")
INPUT_CSV_SAMPLED = os.path.join(BASE_DIR, "data", "diabetes_diet_papers_test_100.csv")

# 根据USE_SAMPLED_DATA选择使用的输入文件
if USE_SAMPLED_DATA:
    INPUT_CSV = INPUT_CSV_SAMPLED

PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
ROUTER_PROMPT = os.path.join(PROMPTS_DIR, "router.txt")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEMANTIC_TRIAGE_DIR = os.path.join(RESULTS_DIR, "semantic_triage")
SEED_ENTITY_DIR = os.path.join(RESULTS_DIR, "seed_entity")
CANDIDATE_EXTRACTION_DIR = os.path.join(RESULTS_DIR, "candidate_extraction")
QUALITY_DECISION_DIR = os.path.join(RESULTS_DIR, "quality_decision")
GROUNDING_ALIGNMENT_DIR = os.path.join(RESULTS_DIR, "grounding_alignment")

# Grounding Agent 本体和索引配置
GROUNDING_DATA_DIR = os.path.join(BASE_DIR, "本体")
NUTRIENT_TRIPLETS_FILE = os.path.join(GROUNDING_DATA_DIR, "food_nutrient_triplets_with_portion_attribute.csv")
HUMANIDO_FILE = os.path.join(GROUNDING_DATA_DIR, "HumanDO.json")
INDEX_DIR = os.path.join(BASE_DIR, "grounding_data", "index")

# build_index.py 配置
FOOD_MODEL = "all-mpnet-base-v2"
SAPBERT_BATCH = 64
BUILD_FOOD_INDEX = True
BUILD_NUTRIENT_INDEX = True

# 固定名 latest 文件（代码间衔接用）
SEMANTIC_TRIAGE_LATEST = os.path.join(SEMANTIC_TRIAGE_DIR, "semantic_triage_latest.xlsx")
SEED_ENTITY_LATEST = os.path.join(SEED_ENTITY_DIR, "seed_entity_latest.json")
CANDIDATE_EXTRACTION_NER_LATEST_JSON = os.path.join(CANDIDATE_EXTRACTION_DIR, "candidate_entity_extraction_latest.json")
CANDIDATE_EXTRACTION_NER_LATEST_XLSX = os.path.join(CANDIDATE_EXTRACTION_DIR, "candidate_entity_extraction_latest.xlsx")
CANDIDATE_EXTRACTION_RE_LATEST_JSON = os.path.join(CANDIDATE_EXTRACTION_DIR, "candidate_relation_extraction_latest.json")
CANDIDATE_EXTRACTION_RE_LATEST_XLSX = os.path.join(CANDIDATE_EXTRACTION_DIR, "candidate_relation_extraction_latest.xlsx")
QUALITY_DECISION_LATEST_XLSX = os.path.join(QUALITY_DECISION_DIR, "quality_decision_latest.xlsx")
QUALITY_DECISION_LATEST_JSON = os.path.join(QUALITY_DECISION_DIR, "quality_decision_latest.json")
GROUNDING_ALIGNMENT_LATEST = os.path.join(GROUNDING_ALIGNMENT_DIR, "grounding_alignment_latest.xlsx")
GROUNDING_ALIGNMENT_LATEST_JSON = os.path.join(GROUNDING_ALIGNMENT_DIR, "grounding_alignment_latest.json")
FEEDBACK_REFINEMENT_REQUESTS = os.path.join(GROUNDING_ALIGNMENT_DIR, "feedback_refinement_requests.json")

# ─────────────────────────────────────────────
# 八、主题簇配置
# ─────────────────────────────────────────────
# 每个主题需要的实体类型和对应关系
TOPIC_CONFIG = {
    "T1": {
        "name": "food_disease",
        "head_type": "Food",
        "tail_type": "Disease",
        "relations": ["recommended_for", "avoid_for"],
        "need_models": ["Food", "Disease"],   # 需要调用的预训练模型类型
    },
    "T2": {
        "name": "nutrient_disease",
        "head_type": "Nutrient",
        "tail_type": "Disease",
        "relations": ["beneficial", "harmful"],
        "need_models": ["Nutrient", "Disease"],
    },
    "T3": {
        "name": "food_nutrient",
        "head_type": "Food",
        "tail_type": "Nutrient",
        "relations": ["has_nutrient"],
        "need_models": ["Food", "Nutrient"],
    },
    "T4": {
        "name": "pattern_food",
        "head_type": "DietaryPattern",
        "tail_type": "Food",
        "relations": ["includes_food"],
        "need_models": ["Food"],              # DietaryPattern 无模型
    },
    "T5": {
        "name": "pattern_nutrient",
        "head_type": "DietaryPattern",
        "tail_type": "Nutrient",
        "relations": ["provides_nutrient"],
        "need_models": ["Nutrient"],          # DietaryPattern 无模型
    },
    "T6": {
        "name": "pattern_disease",
        "head_type": "DietaryPattern",
        "tail_type": "Disease",
        "relations": ["recommended_for", "avoid_for"],
        "need_models": ["Disease"],           # DietaryPattern 无模型
    },
}

# 所有主题编号列表
ALL_TOPICS = list(TOPIC_CONFIG.keys())



