#!/usr/bin/env python3
"""迷你版:vLLM 的 KV 缓存管理,七十来行写一遍(只有账本,不碰显存)。

和 v0.34.0 的对应关系:
  Block            vllm/v1/core/kv_cache_utils.py  的 KVCacheBlock
  BlockPool        vllm/v1/core/block_pool.py      的 BlockPool
  PagedKVCache     vllm/v1/core/kv_cache_manager.py + single_type_kv_cache_manager.py
简化掉的东西:空闲队列用 list 代替双向链表(vLLM 是 O(1) 的链表);
  只有一种层;命中长度不设上限;没有淘汰之外的水位、推测解码、连接器。
块里的内容用一个 list 存着,是为了能检查结果对不对;vLLM 里这部分是显存。
Run:  python mini_paged_kv.py
"""
from __future__ import annotations
import hashlib


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 0            # 有几个请求在用
        self.block_hash = None      # 填满并登记之后才有;被淘汰时清掉
        self.tokens: list[str] = []


class BlockPool:
    def __init__(self, num_blocks: int):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free = list(self.blocks)   # 空闲队列:新块从头上取,释放的接到尾部
        self.cached: dict[bytes, Block] = {}   # 哈希 -> 块

    def get_new_blocks(self, n: int) -> list[Block]:
        ret, self.free = self.free[:n], self.free[n:]
        for b in ret:
            if b.block_hash is not None:      # 取到带哈希的块,就是把它淘汰了
                del self.cached[b.block_hash]
                b.block_hash, b.tokens = None, []
            b.ref_cnt = 1
        return ret

    def free_blocks(self, blocks: list[Block]) -> None:   # 调用方按倒序传入
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0 and b.block_hash is None:
                self.free.insert(0, b)        # 没哈希的放回头部
            elif b.ref_cnt == 0:
                self.free.append(b)           # 有哈希的放到尾部,留着给别人命中

    def touch(self, blocks: list[Block]) -> None:   # 命中:引用计数加一,从空闲队列摘出来
        for b in blocks:
            if b.ref_cnt == 0:
                self.free.remove(b)
            b.ref_cnt += 1


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int = 4):
        self.pool, self.B = BlockPool(num_blocks), block_size
        self.tables: dict[str, list[Block]] = {}   # 请求 -> 块表(逻辑块 -> 物理块)

    def slot(self, req: str, pos: int) -> int:     # token 在缓存里的位置
        return self.tables[req][pos // self.B].block_id * self.B + pos % self.B

    def block_hash(self, parent: bytes | None, tokens: list[str]) -> bytes:
        return hashlib.sha256((parent or b"") + ",".join(tokens).encode()).digest()

    def find_hit(self, tokens: list[str]) -> list[Block]:   # 顺着哈希链,命中就往下走
        parent, hit = None, []
        while (len(hit) + 1) * self.B <= len(tokens):        # 只看填满的块
            h = self.block_hash(parent, tokens[len(hit) * self.B:(len(hit) + 1) * self.B])
            blk = self.pool.cached.get(h)
            if blk is None:
                break                   # 第一个不命中就停
            hit.append(blk)
            parent = h
        return hit

    def allocate(self, req: str, tokens: list[str], hit: list[Block] | None = None) -> None:
        """给请求 req 按它到这一步为止的全部 token 备好位置。hit 是刚命中的块。"""
        hit = hit if hit is not None else []
        table = self.tables.setdefault(req, [])
        if hit:
            self.pool.touch(hit)
            table.extend(hit)
        need = -(-len(tokens) // self.B) - len(table)        # 需要的块数,减去已有的
        if need > 0:
            table.extend(self.pool.get_new_blocks(need))
        for i in range(len(tokens) // self.B):               # 刚填满的块,登记进哈希表
            blk, chunk = table[i], tokens[i * self.B:(i + 1) * self.B]
            blk.tokens = list(chunk)                         # vLLM 里这一步是写显存
            if blk.block_hash is None:
                blk.block_hash = self.block_hash(table[i - 1].block_hash if i else None, chunk)
                self.pool.cached[blk.block_hash] = blk

    def free(self, req: str) -> None:                        # 倒着还:链尾先回池子,先被淘汰
        self.pool.free_blocks(reversed(self.tables.pop(req)))


def show(c: PagedKVCache, title: str) -> None:
    print(f"\n{title}")
    for req, table in sorted(c.tables.items()):
        print(f"  {req} 的块表: " + " ".join(str(b.block_id) for b in table))
    print("  块的状态: " + "  ".join(
        f"{b.block_id}:引用{b.ref_cnt}" + (f"/{b.block_hash.hex()[:4]}" if b.block_hash else "/无哈希")
        for b in c.pool.blocks if b.ref_cnt or b.block_hash))
    print("  空闲队列(头 → 尾): " + (" ".join(str(b.block_id) for b in c.pool.free) or "空"))


if __name__ == "__main__":
    c = PagedKVCache(num_blocks=8, block_size=4)
    a = [f"a{i}" for i in range(16)]                       # 请求 A:16 个 token
    c.allocate("A", a)                                     # 预填充:拿 4 块,4 块都填满,登记
    show(c, "① A 预填充之后")
    c.allocate("A", a + ["x16"])                           # 解码一步:第 5 块,只有 1 个 token,不登记
    b = a[:8] + [f"b{i}" for i in range(8)]                # 请求 B:前 8 个 token 和 A 相同
    c.allocate("B", b, hit=c.find_hit(b))                  # 命中 2 块,再拿 2 块
    show(c, "② B 到了,前两块命中")
    assert c.slot("A", 13) == 3 * 4 + 1 and c.slot("B", 5) == c.slot("A", 5)
    c.free("A")
    show(c, "③ A 结束,倒着还")
    d = a[:12] + [f"d{i}" for i in range(4)]               # 请求 D:前 12 个 token 和 A 相同
    c.allocate("D", d, hit=c.find_hit(d))                  # 块 2 从空闲队列里被摸回来,再拿 1 块
    show(c, "④ D 到了,命中 3 块(块 2 本来空着)")
    assert [t.tokens[:4] for t in c.tables["D"][:3]] == [a[:4], a[4:8], a[8:12]]
