import re
import sys
import random
import heapq
import threading
import time
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt

class DirectedGraph:
    def __init__(self):
        self.adjacency_list = defaultdict(lambda: defaultdict(int))
        self.nodes = set()
    def build_graph(self, words: List[str]) -> None:
        """构建有向图，自动处理大小写"""
        for i in range(len(words) - 1):
            current = words[i].lower()
            next_word = words[i + 1].lower()
            self.adjacency_list[current][next_word] += 1
            self.nodes.update({current, next_word})

    def get_bridge_words(self, word1: str, word2: str) -> List[str]:
        """精确查询桥接词，返回空列表表示无结果"""
        word1 = word1.lower()
        word2 = word2.lower()
        if word1 not in self.adjacency_list or word2 not in self.nodes:
            return []
        return [
            bridge for bridge in self.adjacency_list[word1]
            if bridge in self.adjacency_list and word2 in self.adjacency_list[bridge]
        ]

    def generate_new_text(self, input_text: str) -> str:
        """生成插入桥接词的新文本，保留原始大小写格式"""
        raw_words = re.findall(r'\b[a-zA-Z]+\b', input_text)
        processed_words = [w.lower() for w in raw_words]
        new_text = []
        for i in range(len(processed_words) - 1):
            new_text.append(raw_words[i])  # 保留原始大小写
            bridges = self.get_bridge_words(processed_words[i], processed_words[i + 1])
            if bridges:
                new_text.append(random.choice(bridges).capitalize())  # 统一首字母大写
        new_text.append(raw_words[-1])
        return ' '.join(new_text)

    def shortest_path(self, start: str, end: str) -> Tuple[Optional[List[str]], int]:
        """Dijkstra算法实现，返回路径和总权重"""
        start = start.lower()
        end = end.lower()
        if start not in self.nodes or end not in self.nodes:
            return None, -1

        heap = [(0, start, [])]
        visited = set()
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            new_path = path + [node]
            if node == end:
                return new_path, cost
            for neighbor, weight in self.adjacency_list[node].items():
                heapq.heappush(heap, (cost + weight, neighbor, new_path))
        return None, -1  # 不可达

    def pagerank(self, d: float = 0.85, iterations: int = 100) -> Dict[str, float]:
        """改进的PageRank实现，正确处理出度为0的节点"""
        nodes = list(self.nodes)
        pr = {node: 1.0 / len(nodes) for node in nodes}
        sink_nodes = [node for node in nodes if not self.adjacency_list[node]]

        for _ in range(iterations):
            sink_pr = sum(pr[node] for node in sink_nodes)
            new_pr = {}
            for node in nodes:
                # 来自其他节点的贡献
                incoming = sum(
                    pr[src] / len(self.adjacency_list[src])
                    for src in self.adjacency_list
                    if node in self.adjacency_list[src]
                )
                # 处理悬挂节点
                new_pr[node] = (1 - d) / len(nodes) + d * (incoming + sink_pr / len(nodes))
            pr = new_pr
        return pr

    def random_walk(self) -> str:
        """随机游走，返回遍历路径字符串"""
        if not self.nodes:
            return ""
        current = random.choice(list(self.nodes))
        path = [current]
        visited_edges = set()

        while True:
            neighbors = list(self.adjacency_list[current].keys())
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            edge = (current, next_node)
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            path.append(next_node)
            current = next_node
        return ' '.join(path)

    def visualize_graph(self, output_path: str = "graph.png",
                        figsize: Tuple[int, int] = (14, 10)) -> None:
        """生成带清晰箭头指向的矢量图"""
        plt.figure(figsize=figsize)
        G = nx.DiGraph()

        # 添加带权重的边
        for src in self.adjacency_list:
            for dst, weight in self.adjacency_list[src].items():
                G.add_edge(src, dst, weight=weight)

        # 优化布局算法
        pos = nx.kamada_kawai_layout(G)

        # 绘制节点
        nx.draw_networkx_nodes(G, pos,
                               node_size=1200,
                               node_color='#FFD700',
                               alpha=0.9,
                               linewidths=2,
                               edgecolors='black')

        # 绘制带箭头的边（关键改进）
        edges = G.edges(data=True)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            edge_color='#1f78b4',
            width=[w['weight'] * 0.8 for (u, v, w) in edges],
            arrows=True,
            arrowsize=25,  # 增大箭头尺寸
            arrowstyle='->,head_width=0.6,head_length=0.8',
            connectionstyle='arc3,rad=0.2',  # 边带弧度
            node_size=1200,
            alpha=0.8
        )

        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
        edge_labels = {(u, v): d['weight'] for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels=edge_labels,
                                     font_size=10,
                                     label_pos=0.3)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, format='png', bbox_inches='tight')
        plt.close()


    def scan_text_files(file_paths: List[str]) -> List[str]:
        """扫描多个文本文件并合并处理"""
        all_words = []
        for path in file_paths:
            try:
                # 自动检测编码
                with open(path, 'rb') as f:
                    raw = f.read()
                    content = raw.decode('utf-8-sig').lower()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='gbk') as f:
                    content = f.read().lower()

            # 增强文本清洗
            content = re.sub(r'[^a-z\s]', ' ', content)  # 去除非字母字符
            content = re.sub(r'\s+', ' ', content)  # 合并连续空格
            words = [w for w in content.strip().split(' ') if w]
            all_words.extend(words)
        return all_words

    def all_shortest_paths(self, start: str, end: str) -> List[List[str]]:
        """返回所有最短路径（权重和最小）"""
        start = start.lower()
        end = end.lower()
        if start not in self.nodes or end not in self.nodes:
            return []

        # Dijkstra算法记录前驱节点
        heap = [(0, start)]
        dist = {node: float('inf') for node in self.nodes}
        dist[start] = 0
        predecessors = defaultdict(list)

        while heap:
            current_dist, u = heapq.heappop(heap)
            if current_dist > dist[u]:
                continue
            for v, w in self.adjacency_list[u].items():
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    heapq.heappush(heap, (dist[v], v))
                    predecessors[v] = [u]
                elif dist[v] == dist[u] + w:
                    predecessors[v].append(u)

        # 回溯生成所有路径
        paths = []

        def build_paths(node, path):
            if node == start:
                paths.append([start] + path[::-1])
                return
            for pred in predecessors[node]:
                build_paths(pred, [node] + path)

        build_paths(end, [])
        return [p for p in paths if len(p) > 1]

    def single_source_shortest_paths(self, start: str) -> Dict[str, Tuple[List[str], int]]:
        """单源最短路径"""
        start = start.lower()
        result = {}
        for node in self.nodes:
            if node == start:
                continue
            path, cost = self.shortest_path(start, node)
            if path:
                result[node] = (path, cost)
        return result

    def pagerank_tf(self, d: float = 0.85
                    ,
                 use_tfidf: bool = False,  # 添加可选参数
                 iterations: int = 100) -> Dict[str, float]:
        """支持TF-IDF初始化的PageRank"""
        nodes = list(self.nodes)

        # 初始化逻辑（新增use_tfidf判断）
        if use_tfidf:
            # 计算 TF（词频）：单词的出边数量
            tf = {node: len(self.adjacency_list[node]) + 1 for node in nodes}  # +1 避免零值

            # 计算 IDF（逆向文档频率）：假设多文档场景（需外部输入文档数）
            # 此处简化逻辑，仅演示单文档与全局统计差异
            total_nodes = len(nodes)
            idf = {node: 1.0 for node in nodes}  # 实际应基于多文档统计

            # 综合 TF-IDF 初始化 PR
            tfidf_sum = sum(tf[node] * idf[node] for node in nodes)
            pr = {node: (tf[node] * idf[node]) / tfidf_sum for node in nodes}
        else:
            pr = {node: 1.0 / len(nodes) for node in nodes}

            # PageRank 迭代（保持原逻辑）
        sink_nodes = [n for n in nodes if not self.adjacency_list[n]]
        for _ in range(iterations):
            sink_pr = sum(pr[n] for n in sink_nodes)
            new_pr = {}
            for node in nodes:
                incoming = sum(
                    pr[src] / len(self.adjacency_list[src])
                    for src in self.adjacency_list
                    if node in self.adjacency_list[src]
                )
                new_pr[node] = (1 - d) / len(nodes) + d * (incoming + sink_pr / len(nodes))
            pr = new_pr
        return pr

def process_text(file_path: str) -> List[str]:
    """文本预处理：过滤非字母字符并分词"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().lower()
    content = re.sub(r'[^a-z\s]', ' ', content)  # 替换非字母为空格
    return [word for word in re.split(r'\s+', content) if word]

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <file1> [file2 ...]")
        return

    # 构建图结构
    graph = DirectedGraph()
    words = process_text(sys.argv[1])
    graph.build_graph(words)

    # 生成可视化图片
    output_image = "graph.png"
    graph.visualize_graph(output_image)
    print(f"有向图已保存至 {output_image}")

    # 交互功能保持原样...
    while True:
        print("\n功能选择:")
        print("1.桥接词 2.生成新文本 3.最短路径 4.PageRank 5.随机游走")
        print("6.所有最短路径 7.单源最短路径 8.改进PageRank 0.退出")
        choice = input("请输入选项: ").strip()

        if choice == '0':
            break

        elif choice == '1':
            word1 = input("输入第一个单词: ").strip()
            word2 = input("输入第二个单词: ").strip()
            bridges = graph.get_bridge_words(word1, word2)
            if not bridges:
                if word1.lower() not in graph.nodes:
                    print(f"错误: '{word1}' 不在图中")
                elif word2.lower() not in graph.nodes:
                    print(f"错误: '{word2}' 不在图中")
                else:
                    print("无桥接词存在")
            else:
                print(f"桥接词: {', '.join(bridges)}")

        elif choice == '2':
            text = input("输入待扩展文本: ")
            new_text = graph.generate_new_text(text)
            print("生成文本:", new_text)

        elif choice == '3':
            start = input("起点单词: ").strip()
            end = input("终点单词: ").strip()
            path, cost = graph.shortest_path(start, end)
            if path:
                print(f"最短路径: {' → '.join(path)} (总权重: {cost})")
            else:
                print("路径不存在")

        elif choice == '4':
            pr = graph.pagerank()
            sorted_pr = sorted(pr.items(), key=lambda x: -x[1])[:10]  # 显示前10重要节点
            print("PageRank前10:")
            for word, score in sorted_pr:
                print(f"{word}: {score:.4f}")

        elif choice == '5':
            walk = graph.random_walk()
            print("随机游走路径:", walk)
            with open("random_walk.txt", 'w') as f:
                f.write(walk)
            print("路径已保存至 random_walk.txt")

        elif choice == '6':
            start = input("起点: ").strip()
            end = input("终点: ").strip()
            paths = graph.all_shortest_paths(start, end)
            if paths:
                print(f"找到{len(paths)}条最短路径:")
                for i, path in enumerate(paths, 1):
                    print(f"路径{i}: {'→'.join(path)}")
            else:
                print("无有效路径")

        elif choice == '7':
            start = input("起点: ").strip()
            paths = graph.single_source_shortest_paths(start)
            if paths:
                print(f"从 {start} 出发的最短路径:")
                for node, (path, cost) in paths.items():
                    print(f"到 {node}: {'→'.join(path)} (权重: {cost})")
            else:
                print("起点不存在")


        elif choice == '8':
            pr = graph.pagerank_tf(use_tfidf=True)  # 正确传递参数
            sorted_pr = sorted(pr.items(), key=lambda x: -x[1])[:10]
            print("改进PageRank结果（基于TF-IDF初始化）:")
            for word, score in sorted_pr:
                print(f"{word}: {score:.4f}")

        else:
            print("无效输入")


if __name__ == "__main__":
    main()