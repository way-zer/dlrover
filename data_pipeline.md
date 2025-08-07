## 数据链路范式

### Queue

实现了一个分布式的Queue，支持多生产者多消费者模型。

优点：

- 支持多生产者多消费者模型，生产和消费解藕
- 可以通过size控制生产者的生产速率，避免过载
缺点：
- 需要一个节点充当owner，owner需要储存元数据
- 无法控制生产和消费的顺序，无法用于需要严格顺序的场景(监督训练)

#### 接口

```python
class DataQueue(Generic[T]):
    """
    Distributed data queue interface.
    """

    def __init__(self, name: str, is_master: bool=False, size: int = 1000):...

    def qsize(self) -> int:
        """Get the current size of the queue."""

    def put(self, *obj: T) -> None:
        """Put an object into the queue."""

    def put_async(self, *obj: T) -> Future[None]:
        """Put an object into the queue."""

    def get(self, batch_size: int) -> List[T]:
        """Get a batch of objects from the queue."""

    def get_nowait(self, max_size: int = 1) -> List[T]:
        """Get a batch of objects from the queue without waiting."""
```

#### 用法

```python
# Owner端，通常是生产者的rank0
queue_master = DataQueue[str]("test_queue", size=10, is_master=True)
queue.put("data1", "data2")

# 其他生产者或者消费者
queue_client = DataQueue[str]("test_queue")
queue_client.get(1)
queue_master.get_nowait(1)
```

### RPC 接口

提供了多种灵活的RPC接口，供用户实现算法控制流/自定义数据流。

#### 函数接口

可以暴露单个函数，供其他Worker远程调用。
目前用户模块顶级函数/用户类rpc函数均会自动export。

```python
# 提供者
@rpc(export=True)
def some_method():
    return "test1"

assert RPC_REGISTRY["some_method"] == some_method
# 使用方
_rpc_call("actor", "some_method")
```

#### RPC Proxy

可暴露单个对象，供其他Worker远程单目标调用。(Queue内部采用该实现)

```python
class SimpleClass:
    @rpc()
    def hello(self, name: str) -> str:
        return f"Hello, {name}!"

# 提供者
export_rpc_instance("simple_class", SimpleClass())
assert "simple_class.hello" in RPC_REGISTRY

# 使用方
proxy = create_rpc_proxy("actor", "simple_class", SimpleClass)
assert proxy.hello("World") == "Hello, World!"
```

#### RoleGroup

同属一个Role的Worker集合，可以批量调用。

```python
class RoleGroup(Sequence["RoleActor"]):
    """A group of actors with the same role."""
    def __init__(self, role: str, optional: bool = False):
        """Get the role group for a specific role."""

    def call(self, method, *args, **kwargs) -> Future[List[Any]]:
        """Invoke a method on all actors in the role group."""

    def call_rank0(self, method, *args, **kwargs) -> Future[Any]:
        """Invoke a method on the rank 0 actor in the role group."""

    def call_batch(
        self,
        method: Callable[P, Sequence[R]],
        size: int,
        *args,
        **kwargs,
    ) -> "FutureSequence[R]":
        """Invoke a method on all actors in the role group with batched arguments.(Auto split)"""

class FutureSequence(Sequence[T]):
    """A wrapper for futures, providing a sequence-like interface."""
    def __init__(self, futures: List[Future[Sequence[T]]], lens: List[int]):...

    #一个特殊的Sequence实现，支持按需获取结果。迭代或对应Future未完成时才会阻塞等待
```

#### Remote Call范式

推荐用户定义一个统一的remote_call文件，统一管理远程调用相关的逻辑和接口。
优点：

- 统一管理远程调用逻辑，便于维护和扩展。
- 对调用方屏蔽RPC和被调用方的细节。
- 调用方类型安全，同时可通过引用查找调用方和被调用方。

```python
# remote_call.py

def vllm_wakeup() -> None:...

def vllm_generate(
    prompt_token_ids: Sequence[List[int]], params: "SamplingParams"
) -> FutureSequence["RequestOutput"]: ...

def actor_forward(
    sequences: Sequence[torch.Tensor],
    action_mask: Sequence[torch.BoolTensor],
    attention_mask: Sequence[torch.LongTensor],
) -> Sequence[torch.Tensor]:...

# 调用方，只关心remote_call签名，无需获取对象引用，容易封装实现。无需感知RPC细节/ray平台
remote_call.vllm_wakeup()
remote_call.vllm_generate(prompt_token_ids, params)
remote_call.actor_forward(sequences, action_mask, attention_mask)

# 被调用方，只需要将实现与remote_call对接

class VLLMActor:
    @rpc(remote_call.vllm_wakeup)
    def wake_up(self):
        self.llm.wake_up()

    @rpc(remote_call.vllm_generate)
    def generate(self, prompt_token_ids, params):
        ...

# 接口编排，实现remote_call接口，通过拓扑连接调用方和被调用方
# remote_call.py
def vllm_wakeup() -> None:
    group(RLRoleType.ROLLOUT).call(vllm_wakeup).result()

def vllm_generate(
    prompt_token_ids: Sequence[List[int]], params: "SamplingParams"
) -> FutureSequence["RequestOutput"]: 
    return group(RLRoleType.ROLLOUT).call_batch(
        vllm_generate,
        len(prompt_token_ids),
        prompt_token_ids,
        params,
    )

def actor_forward(
    sequences: Sequence[torch.Tensor],
    action_mask: Sequence[torch.BoolTensor],
    attention_mask: Sequence[torch.LongTensor],
) -> Sequence[torch.Tensor]:
    return group(RLRoleType.ACTOR).call_batch(
        actor_forward, len(sequences), sequences, action_mask, attention_mask
    )
```

讨论：接口定义可以声明为pyi, 实现可以在py中，这样可以把三方分的更清晰。目前实现直接同一个py文件

### Ray Data范式

Ray Data提供了一个分布式数据处理框架，支持分布式的数据集的处理和转换。
核心思想如下

- 数据源抽象：可通过多种数据源加载
- 数据转换：支持多种数据转换操作，如map、filter、batch等。由ray的task或者actor执行计算
- 消费：迭代器，类DataLoader

整体上按需执行，当输出端调用时，创建分布式的数据处理器，处理需要的数据，并返回结果。

在训练中使用

```python
import ray

# 数据源
s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/"
ds = ray.data.read_images(s3_uri, mode="RGB")


# 数据转换
def preprocess_image(row: Dict[str, np.ndarray]):
    return {
        "original_image": row["image"],
        "transformed_image": transform(row["image"]),
    }
transformed_ds = ds.map(preprocess_image)

# 消费(训练作为Dataloader迭代即可)
single_batch = transformed_ds.take_batch(10)
```
