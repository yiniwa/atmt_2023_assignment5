import torch
from itertools import count
from queue import PriorityQueue
class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """

    def __init__(self, beam_size, max_len, pad, gamma=1.0):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        # New attribute for diversity penalty
        self.gamma = gamma

        self.nodes = PriorityQueue()  # beams to be expanded
        self.final = PriorityQueue()  # beams that ended in EOS

        self._counter = count()  # for correct ordering of nodes with same score

    def add(self, score, node, rank=None):
        """ Adds a new beam search node to the queue of current nodes """
        if rank is not None:
            node.rank = rank
            node.apply_diversity_penalty(self.gamma)
        self.nodes.put((node.total_score, next(self._counter), node))

    def add_final(self, score, node):
        """ Adds a beam search path that ended in EOS (= finished sentence) """
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad] * missing).long()))
        # Apply diversity penalty
        node.apply_diversity_penalty(self.gamma)
        self.final.put((node.total_score, next(self._counter), node))

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_n_best(self, n=3):
        """ Returns the n-best final nodes with the lowest negative log probability """
        best_nodes = []
        while not self.final.empty() and len(best_nodes) < n:
            score, _, node = self.final.get()
            best_nodes.append((score, node))
        return best_nodes

    def prune(self):
        """ Removes all nodes but the beam_size best ones, applying diversity penalty. """
        # Temporary storage for the nodes with diversity penalty applied
        all_nodes = []
        while not self.nodes.empty():
            score, counter, node = self.nodes.get()
            diversity_penalty = self.gamma * node.rank
            adjusted_score = score - diversity_penalty
            all_nodes.append((adjusted_score, counter, node))

        # Sort nodes by adjusted score
        all_nodes.sort()

        # Reinsert only the top beam_size nodes
        self.nodes = PriorityQueue()
        for i, (adjusted_score, counter, node) in enumerate(all_nodes):
            if i < self.beam_size:
                self.nodes.put((adjusted_score, counter, node))

    def select_best_sibling(self):
        """ Selects the best sibling for each set of siblings while applying the diversity penalty. """
        siblings = {}

        # Group nodes by their parent ID
        while not self.nodes.empty():
            _, _, node = self.nodes.get()
            parent_id = node.parent.sequence_id if node.parent else None
            if parent_id not in siblings:
                siblings[parent_id] = []
            siblings[parent_id].append(node)

        # Apply the diversity penalty within each sibling group and select the best sibling
        for sibling_group in siblings.values():
            for rank, node in enumerate(sorted(sibling_group, key=lambda n: n.logp, reverse=True)):
                node.rank = rank
                node.apply_diversity_penalty(self.gamma)

            best_sibling = max(sibling_group, key=lambda n: n.total_score)
            self.nodes.put((best_sibling.total_score, next(self._counter), best_sibling))

    def get_all_hypotheses(self):
        """ Returns all final nodes with their negative log probabilities """
        all_nodes = []
        while not self.final.empty():
            score, _, node = self.final.get()
            all_nodes.append((score, node))
        # Return in ascending order of score
        return sorted(all_nodes, key=lambda x: x[0])

class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""

    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length, rank=0, parent=None):
        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search

        # New attribute to store the rank of a node among its siblings
        self.rank = rank

        # Store the parent of the node for identifying siblings
        self.parent = parent

        self.total_score = 0.0

    def eval(self, alpha=0.0, gamma=1.0):
        """ Returns score of sequence up to this node

        params:
            :alpha float (default=0.0): hyperparameter for
            length normalization described in in
            https://arxiv.org/pdf/1609.08144.pdf (equation
            14 as lp), default setting of 0.0 has no effect

        """
        normalizer = (5 + self.length) ** alpha / (5 + 1) ** alpha
        return self.logp / normalizer


    # Add a method to update the total score when the diversity penalty is applied
    def apply_diversity_penalty(self, gamma):
        """ Applies the diversity penalty to the total score based on the rank. """
        diversity_penalty = gamma * self.rank
        self.total_score = self.eval() - diversity_penalty
