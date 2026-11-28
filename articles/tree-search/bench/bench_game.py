#!/usr/bin/env python3
"""博弈树:井字棋,3×3 的棋盘轮流落子,先连成三个的赢。只吃 CPU。

  H1  极小极大:每个局面都往后看到底,一步不漏
  H2  Alpha-Beta 剪枝:发现这一支不会比已经有的选择更好,就不再往下看。
      两种试走法的顺序:按格子编号;先中间,再角,最后边
  H3  蒙特卡洛树搜索(UCT):不把整棵树看完。反复做"顺着最有希望的路往下走,随机下到终局,把结果记回路上",
      最后按去的次数选走法。执先手的 X 用它,对手随机走,看胜率随模拟次数怎么变

运行(单线程,绑在一个核上):
  taskset -c 56 /data/wangyue/venvs/vllm/bin/python bench_game.py --out results/game.json
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict

from bench_common import warmup

X, O = 1, 2
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
NATURAL = (0, 1, 2, 3, 4, 5, 6, 7, 8)
CENTER_FIRST = (4, 0, 2, 6, 8, 1, 3, 5, 7)


def other(p):
    return O if p == X else X


def winner(b):
    for i, j, k in LINES:
        if b[i] and b[i] == b[j] == b[k]:
            return b[i]
    return 0


def moves(b, order=NATURAL):
    return [i for i in order if b[i] == 0]


def final_score(b):
    """终局的分,站在 X 这边看:X 赢 1,平 0,O 赢 -1。"""
    w = winner(b)
    return 1 if w == X else (-1 if w == O else 0)


# ---------------------------------------------------------------- H1. 极小极大
def minimax(b, player, cnt):
    """X 想让分最大,O 想让分最小。返回双方都不出错时,这个局面最后的分。"""
    cnt["nodes"] += 1
    w = winner(b)
    if w or 0 not in b:
        cnt["leaves"] += 1
        return final_score(b)
    best = -2 if player == X else 2
    for i in range(9):
        if b[i] == 0:
            b[i] = player
            v = minimax(b, other(player), cnt)
            b[i] = 0
            best = max(best, v) if player == X else min(best, v)
    return best


# ---------------------------------------------------------------- H2. Alpha-Beta 剪枝
def alphabeta(b, player, alpha, beta, order, cnt):
    """alpha:走到这里,X 已经稳拿的分;beta:O 已经能压到的分。两边一碰上(alpha >= beta),这一支就不用看了。"""
    cnt["nodes"] += 1
    w = winner(b)
    if w or 0 not in b:
        cnt["leaves"] += 1
        return final_score(b)
    if player == X:
        best = -2
        for i in moves(b, order):
            b[i] = X
            v = alphabeta(b, O, alpha, beta, order, cnt)
            b[i] = 0
            best = max(best, v)
            alpha = max(alpha, best)
            if alpha >= beta:
                cnt["cutoffs"] += 1
                break
        return best
    best = 2
    for i in moves(b, order):
        b[i] = O
        v = alphabeta(b, X, alpha, beta, order, cnt)
        b[i] = 0
        best = min(best, v)
        beta = min(beta, best)
        if alpha >= beta:
            cnt["cutoffs"] += 1
            break
    return best


# ---------------------------------------------------------------- H3. UCT
class Node:
    __slots__ = ("board", "player", "untried", "children", "n", "w")

    def __init__(self, board, player):
        self.board, self.player = board, player          # player:这个局面该谁走
        self.untried = moves(board) if not winner(board) and 0 in board else []
        self.children = {}
        self.n, self.w = 0, 0.0                          # w:走进这个局面的那一方,在这里一共得了几分


def ucb(ch, parent_n, c):
    return ch.w / ch.n + c * math.sqrt(math.log(parent_n) / ch.n)


def playout(board, player, rng):
    """从这个局面开始两边都随机走,下到终局。返回站在 X 这边的分。"""
    b = list(board)
    while True:
        if winner(b) or 0 not in b:
            return final_score(b)
        m = moves(b)
        b[m[rng.randrange(len(m))]] = player
        player = other(player)


def uct_best_move(board, player, sims, rng, c=1.4):
    root = Node(board, player)
    for _ in range(sims):
        node, path = root, [root]
        while not node.untried and node.children:       # 选:每个走法都试过了,挑"赢得多,或者去得少"的
            node = max(node.children.values(), key=lambda ch: ucb(ch, node.n, c))
            path.append(node)
        if node.untried:                                 # 扩:挑一个没试过的走法
            i = node.untried.pop(rng.randrange(len(node.untried)))
            nb = list(node.board)
            nb[i] = node.player
            node.children[i] = Node(tuple(nb), other(node.player))
            path.append(node.children[i])
            node = node.children[i]
        result = playout(node.board, node.player, rng)   # 模拟
        for ch in path:                                  # 回传:每个节点记"走进它的那一方"得的分
            ch.n += 1
            ch.w += result if other(ch.player) == X else -result
    return max(root.children.items(), key=lambda kv: kv[1].n)[0]


def play_game(sims, rng):
    """X 用 UCT,O 随机。返回 (X 的分, 这一局 UCT 一共模拟了多少次)。"""
    b, player, total = (0,) * 9, X, 0
    while not winner(b) and 0 in b:
        if player == X:
            i = uct_best_move(b, X, sims, rng)
            total += sims
        else:
            i = rng.choice(moves(b))
        bl = list(b)
        bl[i] = player
        b, player = tuple(bl), other(player)
    return final_score(b), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", default="300,300,100")
    a = ap.parse_args()
    warmup()
    empty = [0] * 9
    out = {}
    t0 = time.perf_counter()
    cnt = defaultdict(int)
    v = minimax(empty, X, cnt)
    out["minimax"] = {**cnt, "value": v, "s": time.perf_counter() - t0}
    for name, order in (("natural", NATURAL), ("center_first", CENTER_FIRST)):
        t0 = time.perf_counter()
        cnt = defaultdict(int)
        v = alphabeta(empty, X, -2, 2, order, cnt)
        out["alphabeta_" + name] = {**cnt, "value": v, "s": time.perf_counter() - t0}
    # 空盘上 X 有 9 种走法;走了两步之后的局面,该看的有 7 种,正好比较"分支数"的影响
    after_two = [0] * 9
    after_two[4], after_two[0] = X, O
    for name, order in (("natural", NATURAL), ("center_first", CENTER_FIRST)):
        cnt = defaultdict(int)
        alphabeta(after_two, X, -2, 2, order, cnt)
        out["alphabeta_after_two_" + name] = dict(cnt)

    rng = random.Random(11)
    rows = []
    games = [int(x) for x in a.games.split(",")]
    for sims, g in zip((0, 10, 100, 1000), (2000,) + tuple(games)):
        t0 = time.perf_counter()
        if sims == 0:
            score = [-1, 0, 1]
            res = [0, 0, 0]
            for _ in range(g):                           # 两个都随机走
                b, player = (0,) * 9, X
                while not winner(b) and 0 in b:
                    i = rng.choice(moves(b))
                    bl = list(b)
                    bl[i] = player
                    b, player = tuple(bl), other(player)
                res[score.index(final_score(b))] += 1
            sims_total = 0
        else:
            res, sims_total = [0, 0, 0], 0
            for _ in range(g):
                s, t = play_game(sims, rng)
                res[[-1, 0, 1].index(s)] += 1
                sims_total += t
        rows.append({"sims_per_move": sims, "games": g, "x_win": res[2], "draw": res[1], "x_loss": res[0],
                     "win_rate": res[2] / g, "no_loss_rate": (res[2] + res[1]) / g, "simulations_per_game": sims_total / g,
                     "s": time.perf_counter() - t0})
    out["uct_vs_random"] = rows
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
