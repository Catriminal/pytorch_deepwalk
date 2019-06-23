import heapq


class Node:
    def __init__(self, wid, frequency):
        self.wid = wid
        self.frequency = frequency
        self.father = None
        self.is_left_child = None
        self.left_child = None
        self.right_child = None
        self.code = []
        self.path = []


class HuffmanTree:
    def __init__(self, word_frequency):
        self.word_count = len(word_frequency)
        self.huffman_nodes = []
        unmerged_nodes = []
        word_frequency_list = []

        for index, value in word_frequency.items():
            word_frequency_list.append(value)

        for wid, count in word_frequency.items():
            node = Node(wid, count)
            heapq.heappush(unmerged_nodes, (count, wid, node))
            self.huffman_nodes.append(node)

        next_wid = len(self.huffman_nodes)
        while len(unmerged_nodes) > 1:
            _, _, left_node = heapq.heappop(unmerged_nodes)
            _, _, right_node = heapq.heappop(unmerged_nodes)
            new_node = Node(next_wid, left_node.frequency + right_node.frequency)
            left_node.father = new_node.wid
            right_node.father = new_node.wid
            new_node.left_child = left_node.wid
            new_node.right_child = right_node.wid
            left_node.is_left_child = True
            right_node.is_left_child = False
            self.huffman_nodes.append(new_node)
            heapq.heappush(unmerged_nodes, (new_node.frequency, new_node.wid, new_node))
            next_wid = len(self.huffman_nodes)

        root_node = unmerged_nodes[0][2]
        self.generate_huffman_code(root_node.left_child)
        self.generate_huffman_code(root_node.right_child)

    def generate_huffman_code(self, wid):
        if self.huffman_nodes[wid].is_left_child:
            code = [0]
        else:
            code = [1]

        father_node = self.huffman_nodes[wid].father
        self.huffman_nodes[wid].code = self.huffman_nodes[father_node].code + code
        self.huffman_nodes[wid].path = self.huffman_nodes[father_node].path + [father_node]

        if self.huffman_nodes[wid].left_child is not None:
            self.generate_huffman_code(self.huffman_nodes[wid].left_child)

        if self.huffman_nodes[wid].right_child is not None:
            self.generate_huffman_code(self.huffman_nodes[wid].right_child)

    def divide_pos_and_neg(self):
        positive_path_parts = []
        negative_path_parts = []
        for wid in range(self.word_count):
            pos_path_part = []
            neg_path_part = []
            for index, code in enumerate(self.huffman_nodes[wid].code):
                path_node = self.huffman_nodes[wid].path[index]
                if code == 0:
                    pos_path_part.append(path_node)
                else:
                    neg_path_part.append(path_node)

                positive_path_parts.append(pos_path_part)
                negative_path_parts.append(neg_path_part)

        return positive_path_parts, negative_path_parts
