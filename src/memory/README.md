# Jarvis Memory System
<p align="center">
  <img src="../../pics/ironman.jpg" width="200" height="200">
</p>

[English](#english) | [中文](#chinese)

---

<a name="english"></a>

## English Documentation

The Jarvis Memory System provides powerful memory capabilities for AI assistants, enabling them to remember user preferences, conversation history, and important information to deliver more personalized and intelligent interactions.

### Table of Contents

- [Core Features](#core-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [GPU Acceleration](#gpu-acceleration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

### Core Features

#### 🎯 Memory Injection Objectives
- **User Context Understanding**: Provide background information about user preferences and historical operations
- **Personalized Responses**: Generate customized replies based on user characteristics
- **Context Continuity**: Maintain consistency and coherence in conversations

#### 📋 Structured Prompt Design
Adopts a layered prompt structure:
```
[System Prompt] + [User Profile] + [Long-term Memory Summary] + 
[Relevant History] + [Recent Conversation] + [Current Request]
```

#### 🔍 Semantic Retrieval Mechanism
- **FAISS Vector Retrieval**: Efficient semantic similarity search
- **Sentence Embeddings**: Use SentenceTransformers to generate text embeddings
- **Intelligent Filtering**: Filter memory content based on relevance thresholds
- **Fallback Mechanism**: Use simple cosine similarity when advanced libraries are unavailable

#### 🏗️ Hierarchical Memory Management

**User Profile**
- Name, language preference, conversation style
- Voice preference, interests and hobbies
- Timezone and custom preference settings

**Conversation Memory**
- Short-term conversation history (default 10 messages)
- Intelligent token limit management
- Time-reversed processing

**Long-term Memory**
- Automatic conversation summarization
- Important user fact extraction
- User behavior pattern recognition
- Compression and summarization mechanisms

**Vector Store**
- Semantic similarity retrieval
- Access statistics tracking
- Persistent storage management

#### ⚡ Token Management Strategy
- **Hierarchical Token Allocation**: Token limits for different types of memory
- **Intelligent Truncation**: Truncation algorithm that maintains structural integrity
- **Priority Protection**: Prioritize current requests and important context
- **Dynamic Adjustment**: Automatically adjust allocation based on content length

### Installation

#### Required Dependencies
```bash
pip install numpy
```

#### Optional Dependencies (Enhanced Features)
```bash
# For enhanced performance, install enhanced dependencies
pip install -r requirements_memory.txt

# Or install manually
pip install faiss-cpu sentence-transformers

# GPU version (if GPU support available)
pip install faiss-gpu sentence-transformers
```

### Quick Start

#### Basic Integration
```python
from memory.memory_manager import MemoryManager

# Initialize memory manager
memory_manager = MemoryManager(config)
memory_manager.set_llm_client(llm_client)

# Process interaction
await memory_manager.process_interaction(user_input, ai_response)

# Build enhanced prompt
enhanced_prompt = await memory_manager.build_enhanced_prompt(user_input)
```

#### Running Demo
```bash
cd src/memory
python demo.py
```

The demo script will demonstrate:
- Memory system initialization
- User profile management
- Interaction processing and storage
- Memory statistics and search
- Enhanced prompt generation

### Configuration

In the `memory` section of `config.yaml`:

```yaml
memory:
  enabled: true                    # Enable memory system
  max_context_tokens: 2000        # Maximum context tokens
  user_profile_tokens: 200        # User profile token limit
  history_tokens: 500             # History memory token limit
  long_term_tokens: 300           # Long-term memory token limit
  conversation_tokens: 500        # Conversation memory token limit
  max_conversation_messages: 10   # Maximum conversation messages
  auto_summarize_threshold: 20    # Auto-summarization trigger threshold
  fact_extraction_threshold: 5    # Fact extraction trigger threshold
  cleanup_days: 30                # Memory cleanup days
```

### Usage Examples

#### User Profile Management
```python
# Update user profile
memory_manager.update_user_profile(
    name="John Doe",
    language_preference="en",
    conversation_style="friendly"
)

# Add interest
memory_manager.add_user_interest("machine learning")
```

#### Memory Search
```python
# Search relevant memories
results = memory_manager.search_memories("voice preferences", max_results=5)
for memory, score in results:
    print(f"{memory.content} (relevance: {score:.3f})")
```

### API Reference

#### MemoryManager Class

**Methods:**

- `__init__(config: Dict[str, Any])`: Initialize memory manager
- `set_llm_client(llm_client)`: Set LLM client for summarization
- `process_interaction(user_input: str, ai_response: str)`: Process user-AI interaction
- `build_enhanced_prompt(user_input: str) -> str`: Build memory-enhanced prompt
- `update_user_profile(**kwargs)`: Update user profile information
- `search_memories(query: str, max_results: int) -> List`: Search memories
- `get_memory_stats() -> Dict`: Get memory system statistics

### GPU Acceleration

#### Hardware Requirements
- NVIDIA GPU with CUDA 11.0+ support
- Recommended: RTX series or professional graphics cards
- Minimum 4GB VRAM

#### Installing GPU Version
```bash
# PyTorch GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# FAISS GPU version
conda install -c conda-forge faiss-gpu

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'GPU resources: {faiss.get_num_gpus()}')"
```

#### Performance Improvements
- Vector search speed improvement: 5-10x
- Text embedding generation speed improvement: 3-5x
- Significantly enhanced large-scale memory processing capability

### Performance

#### Automated Features

**Conversation Summarization**
When the number of conversation messages reaches the threshold, automatically generate summaries and store them in long-term memory.

**Fact Extraction**
Automatically extract important user information and preferences from conversations, storing them as structured facts.

**Memory Cleanup**
Regularly clean up expired low-importance memories to prevent storage space bloat.

#### Technical Features

**Fault-tolerant Design**
- Graceful library dependency detection and fallback
- Exception handling and error recovery
- Data persistence protection

**Performance Optimization**
- Incremental vector index construction
- Intelligent token limit management
- Batch processing and caching

**Extensibility**
- Modular design, easy to extend
- Configurable memory types and strategies
- Plugin-based embedding model support

### Data Storage

Memory data is stored in the `memory_data/` directory:
- `memory_entries.json`: Memory entries
- `vectors.npy`: Vector data
- `user_profile.json`: User profile
- `conversation_memory.json`: Conversation memory
- `long_term_memory.json`: Long-term memory

### Privacy and Security

- Local storage, no data upload
- Configurable data retention period
- Sensitive information filtering mechanism
- User-controllable memory management

### Troubleshooting

#### Common Issues

1. **FAISS Installation Failed**
   ```bash
   # Use CPU version
   pip install faiss-cpu
   
   # Or ignore advanced features, system will auto-fallback
   ```

2. **High Memory Usage**
   - Adjust `cleanup_days` parameter
   - Lower `max_context_tokens` limit
   - Run `memory_manager.cleanup_old_memories()`

3. **Slow Vector Retrieval**
   - Install FAISS library for better performance
   - Reduce `top_k` search count
   - Clean up expired memories

### File Structure

```
src/memory/
├── __init__.py              # Module exports
├── memory_manager.py        # Main memory manager
├── memory_types.py          # Data structures and type definitions
├── vector_store.py          # Vector storage and retrieval
├── prompt_builder.py        # Structured prompt builder
├── demo.py                  # Feature demo script
└── README.md               # This document
```

### Roadmap

- [ ] Support for multimodal memory (images, audio)
- [ ] Automatic memory importance learning
- [ ] Distributed memory storage
- [ ] Memory data import/export
- [ ] Advanced user behavior analysis
- [ ] Memory visualization interface

### Contributing

Welcome to submit issue reports and feature suggestions! The memory system is an actively developed component, and we continuously improve its functionality and performance.

#### Development Guidelines
- Follow PEP 8 code style
- Add test cases for new features
- Update relevant documentation

### License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Project Status**: 🚀 Active Development

**Latest Version**: v1.0.0

**Support**: 
- 📧 Email: project@email.com
- 💬 Issues: GitHub Issues page
- 📖 Documentation: Online documentation link

<a name="chinese"></a>

## 中文文档

Jarvis 记忆系统为AI助手提供了强大的记忆能力，使其能够记住用户的偏好、历史对话和重要信息，从而提供更加个性化和智能的交互体验。

### 目录

- [核心功能](#核心功能)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置选项](#配置选项)
- [使用示例](#使用示例)
- [API参考](#api参考)
- [GPU加速](#gpu加速)
- [性能特点](#性能特点)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)

### 核心功能

#### 🎯 记忆注入目标
- **用户背景理解**: 提供用户偏好、历史操作等背景信息
- **个性化响应**: 基于用户特征生成定制化回复
- **上下文连续性**: 保持对话的一致性和连贯性

#### 📋 结构化提示设计
采用分层的提示结构：
```
[系统提示] + [用户资料] + [长期记忆摘要] + [相关历史] + [近期对话] + [当前请求]
```

#### 🔍 语义检索机制
- **FAISS向量检索**: 高效的语义相似度搜索
- **句子嵌入**: 使用 SentenceTransformers 生成文本嵌入
- **智能筛选**: 基于相关性阈值过滤记忆内容
- **回退机制**: 在高级库不可用时使用简单余弦相似度

#### 🏗️ 分层记忆管理

**用户档案**
- 姓名、语言偏好、对话风格
- 语音偏好、兴趣爱好
- 时区和自定义偏好设置

**对话记忆**
- 短期对话历史 (默认10条消息)
- 智能Token限制管理
- 按时间倒序处理

**长期记忆**
- 自动对话摘要
- 重要用户事实提取
- 用户行为模式识别
- 压缩和摘要机制

**向量存储**
- 语义相似度检索
- 访问统计跟踪
- 持久化存储管理

#### ⚡ Token管理策略
- **分层Token分配**: 不同类型记忆的Token限制
- **智能截断**: 保持结构完整性的截断算法
- **优先级保护**: 优先保留当前请求和重要上下文
- **动态调整**: 根据内容长度自动调整分配

### 安装

#### 必需依赖
```bash
pip install numpy
```

#### 可选依赖 (增强功能)
```bash
# 安装增强功能依赖
pip install -r requirements_memory.txt

# 或者手动安装
pip install faiss-cpu sentence-transformers

# GPU版本 (如果有GPU支持)
pip install faiss-gpu sentence-transformers
```

### 快速开始

#### 基本集成
```python
from memory.memory_manager import MemoryManager

# 初始化记忆管理器
memory_manager = MemoryManager(config)
memory_manager.set_llm_client(llm_client)

# 处理交互
await memory_manager.process_interaction(user_input, ai_response)

# 构建增强提示
enhanced_prompt = await memory_manager.build_enhanced_prompt(user_input)
```

#### 运行演示
```bash
cd src/memory
python demo.py
```

演示脚本将展示：
- 记忆系统初始化
- 用户档案管理
- 交互处理和存储
- 记忆统计和搜索
- 增强提示生成

### 配置选项

在 `config.yaml` 中的 `memory` 部分：

```yaml
memory:
  enabled: true                    # 启用记忆系统
  max_context_tokens: 2000        # 最大上下文Token数
  user_profile_tokens: 200        # 用户档案Token限制
  history_tokens: 500             # 历史记忆Token限制
  long_term_tokens: 300           # 长期记忆Token限制
  conversation_tokens: 500        # 对话记忆Token限制
  max_conversation_messages: 10   # 最大对话消息数
  auto_summarize_threshold: 20    # 自动摘要触发阈值
  fact_extraction_threshold: 5    # 事实提取触发阈值
  cleanup_days: 30                # 记忆清理天数
```

### 使用示例

#### 用户档案管理
```python
# 更新用户档案
memory_manager.update_user_profile(
    name="张三",
    language_preference="zh",
    conversation_style="friendly"
)

# 添加兴趣
memory_manager.add_user_interest("机器学习")
```

#### 记忆搜索
```python
# 搜索相关记忆
results = memory_manager.search_memories("语音偏好", max_results=5)
for memory, score in results:
    print(f"{memory.content} (相关度: {score:.3f})")
```

### API参考

#### MemoryManager 类

**方法:**

- `__init__(config: Dict[str, Any])`: 初始化记忆管理器
- `set_llm_client(llm_client)`: 设置用于摘要的LLM客户端
- `process_interaction(user_input: str, ai_response: str)`: 处理用户-AI交互
- `build_enhanced_prompt(user_input: str) -> str`: 构建记忆增强提示
- `update_user_profile(**kwargs)`: 更新用户档案信息
- `search_memories(query: str, max_results: int) -> List`: 搜索记忆
- `get_memory_stats() -> Dict`: 获取记忆系统统计信息

### GPU加速

#### 硬件要求
- NVIDIA GPU 支持 CUDA 11.0+
- 推荐: RTX 系列或专业级显卡
- 最少 4GB 显存

#### 安装 GPU 版本
```bash
# PyTorch GPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# FAISS GPU 版本
conda install -c conda-forge faiss-gpu

# 验证 GPU 支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'GPU resources: {faiss.get_num_gpus()}')"
```

#### 性能提升
- 向量搜索速度提升 5-10x
- 文本嵌入生成速度提升 3-5x
- 大规模记忆处理能力显著增强

### 性能特点

#### 自动化功能

**对话摘要**
当对话消息数达到阈值时，自动生成摘要并存储到长期记忆。

**事实提取**
从对话中自动提取用户的重要信息和偏好，存储为结构化事实。

**记忆清理**
定期清理过期的低重要性记忆，防止存储空间膨胀。

#### 技术特点

**容错设计**
- 优雅的库依赖检测和回退
- 异常处理和错误恢复
- 数据持久化保护

**性能优化**
- 增量式向量索引构建
- 智能Token限制管理
- 批量处理和缓存

**扩展性**
- 模块化设计，易于扩展
- 可配置的记忆类型和策略
- 插件化的嵌入模型支持

### 数据存储

记忆数据存储在 `memory_data/` 目录下：
- `memory_entries.json`: 记忆条目
- `vectors.npy`: 向量数据
- `user_profile.json`: 用户档案
- `conversation_memory.json`: 对话记忆
- `long_term_memory.json`: 长期记忆

### 隐私和安全

- 本地存储，数据不上传
- 可配置的数据保留期限
- 敏感信息过滤机制
- 用户可控的记忆管理

### 故障排除

#### 常见问题

1. **FAISS 安装失败**
   ```bash
   # 使用CPU版本
   pip install faiss-cpu
   
   # 或忽略高级功能，系统会自动回退
   ```

2. **内存使用过高**
   - 调整 `cleanup_days` 参数
   - 降低 `max_context_tokens` 限制
   - 运行 `memory_manager.cleanup_old_memories()`

3. **向量检索慢**
   - 安装 FAISS 库以获得更好性能
   - 减少 `top_k` 搜索数量
   - 清理过期记忆

### 文件结构

```
src/memory/
├── __init__.py              # 模块导出
├── memory_manager.py        # 主记忆管理器
├── memory_types.py          # 数据结构和类型定义
├── vector_store.py          # 向量存储和检索
├── prompt_builder.py        # 结构化提示构建
├── demo.py                  # 功能演示脚本
└── README.md               # 本文档
```

### 路线图

- [ ] 支持多模态记忆 (图像、音频)
- [ ] 记忆重要性自动学习
- [ ] 分布式记忆存储
- [ ] 记忆数据导入/导出
- [ ] 高级用户行为分析
- [ ] 记忆可视化界面

### 贡献指南

欢迎提交问题报告和功能建议！记忆系统是一个活跃开发的组件，我们持续改进其功能和性能。

#### 开发指南
- 遵循 PEP 8 代码风格
- 为新功能添加测试用例
- 更新相关文档

### 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

---

**项目状态**: 🚀 活跃开发中

**最新版本**: v1.0.0

**支持**: 
- 📧 Email: project@email.com
- 💬 Issues: GitHub Issues 页面
- 📖 文档: 在线文档链接 