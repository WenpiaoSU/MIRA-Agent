"""作用：通过 LangGraph Checkpointer 保存有界会话记忆。

当前功能：
- 使用会话 thread_id 保存近期最终输入与实际交付消息。
- 每轮压缩为最新快照，限制历史量并阻止旧 generation 的迟到写入。
- 支持会话关闭、超时及记忆删除后的彻底清理；进程重启清空短期记忆。
"""

from time import monotonic, time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class SessionCheckpointer(InMemorySaver):
    """为官方内存检查点增加会话代次检查，不另建消息数据库。"""

    def __init__(self):
        """初始化检查点和允许写入的输出代次。"""
        super().__init__(serde=JsonPlusSerializer(allowed_msgpack_modules=[
            ("src.contracts.perception", "IntentEstimate"), ("src.contracts.perception", "IntentSlot"),
            ("src.contracts.agent", "TaskPlan"), ("src.contracts.agent", "PlanStep"),
            ("src.contracts.agent", "ResultReference"), ("src.contracts.tools", "ToolResult"),
            ("src.contracts.tools", "ToolError")]))
        self.generations: dict[str, str] = {}

    def valid(self, config):
        """只有本轮配置仍持有会话写入权时才允许保存。"""
        values = config["configurable"]
        return (values.get("memory_generation") is not None and
                self.generations.get(values["thread_id"]) == values["memory_generation"])

    def put(self, config, checkpoint, metadata, new_versions):
        """拒绝被取消或被删除会话的旧快照，保留 LangGraph 所需返回结构。"""
        if self.valid(config):
            result = super().put(config, checkpoint, metadata, new_versions)
            return {"configurable": {**config["configurable"], **result["configurable"]}}
        return {"configurable": {**config["configurable"], "checkpoint_id": checkpoint["id"]}}

    def put_writes(self, config, writes, task_id, task_path=""):
        """丢弃旧执行节点的待处理写入，避免删除后在恢复时重放。"""
        if self.valid(config):
            super().put_writes(config, writes, task_id, task_path)

    def forget(self, thread_id):
        """撤销代次并删除该会话的所有快照、写入和通道数据。"""
        self.generations.pop(thread_id, None)
        self.delete_thread(thread_id)


class WorkingMemory:
    """管理短期记忆生命周期；内容由 Checkpointer 保存，控制表只存代次和时间。"""

    def __init__(self, ttl_seconds: int = 3600):
        """设置闲置过期时间，当前运行任务继续受整轮预算约束。"""
        self.checkpointer = SessionCheckpointer()
        self.touched: dict[str, float] = {}
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def config(thread_id: str, generation_id: str):
        """建立同一会话内可追踪且隔离旧输出代次的运行配置。"""
        return {"configurable": {"thread_id": thread_id, "memory_generation": generation_id}}

    def read(self, thread_id):
        """读取最新检查点中的交付历史，过期时释放整个会话快照。"""
        if monotonic() - self.touched.get(thread_id, 0) > self.ttl_seconds:
            self.forget(thread_id)
            return []
        saved = self.checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        return list(saved.checkpoint["channel_values"].get("working_turns", [])) if saved else []

    def begin(self, thread_id, generation_id, turns):
        """合并交付回执并清除旧执行快照；只把有界工作记忆传入新一轮。"""
        self.expire()
        previous = self.read(thread_id)
        previous = [turn for turn in previous if time() - float(turn.get("created_at", time())) <= self.ttl_seconds]
        turns = [turn for turn in turns if time() - float(turn.get("created_at", time())) <= self.ttl_seconds]
        merged = self.merge(previous, turns)
        self.checkpointer.generations[thread_id] = generation_id
        self.checkpointer.delete_thread(thread_id)
        self.touched[thread_id] = monotonic()
        return merged

    @staticmethod
    def merge(previous, deliveries):
        """按输出标识更新实际交付记录，限制十二个轮次和六千字符。"""
        indexed = {turn["generation_id"]: dict(turn) for turn in previous}
        for turn in deliveries:
            indexed[turn["generation_id"]] = dict(turn)
        kept = []
        budget = 6000
        for turn in reversed(list(indexed.values())[-12:]):
            copy = dict(turn)
            # 保留最新用户输入，裁剪只能减少上下文，不能增添尚未交付的正文。
            copy["user"] = copy.get("user", "")[:min(4000, budget)]
            budget -= len(copy["user"])
            copy["assistant"] = copy.get("assistant", "")[:budget]
            budget -= len(copy["assistant"])
            kept.append(copy)
            if budget <= 0:
                break
        return list(reversed(kept))

    @staticmethod
    def history(turns, exclude_generation=None):
        """把来源明确的轮次转换为模型历史，保留最近十二条消息。"""
        messages = []
        for turn in turns:
            if turn["generation_id"] == exclude_generation:
                continue
            for field, role in (("user", "user"), ("assistant", "assistant")):
                if turn.get(field):
                    messages.append({"role": role, "content": turn[field]})
        return messages[-12:]

    def forget(self, thread_id):
        """清除一个会话的全部短期记忆，旧任务不能将其重新写回。"""
        self.checkpointer.forget(thread_id)
        self.touched.pop(thread_id, None)

    def invalidate(self, thread_id: str, generation_id: str):
        """停止输出时立即撤销旧节点写入权，已保存的用户及交付事实仍可读取。"""
        if self.checkpointer.generations.get(thread_id) == generation_id:
            self.checkpointer.generations.pop(thread_id, None)

    def expire(self):
        """清理闲置检查点；活跃会话在每轮开始时刷新时间。"""
        for thread_id, touched in list(self.touched.items()):
            if monotonic() - touched > self.ttl_seconds:
                self.forget(thread_id)
