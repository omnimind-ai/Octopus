from collections import OrderedDict

import numpy as np


class JaccardDistances:
    def __init__(self, activations, cache_bytes=128 * 1024**2):
        observations = np.asarray(activations)
        if observations.ndim != 2 or min(observations.shape) == 0:
            raise ValueError("Expected a nonempty (steps, tokens) activation matrix")
        if observations.shape[0] > 2**24:
            raise ValueError("At most 2**24 observations are supported by float32 distances")
        if np.any((observations != 0) & (observations != 1)):
            raise ValueError("Activation observations must be binary")
        if cache_bytes < 0:
            raise ValueError("Distance cache size must be nonnegative")

        self.size = observations.shape[1]
        packed = np.packbits(observations.T.astype(np.uint8, copy=False), axis=1, bitorder="little")
        packed = np.pad(packed, ((0, 0), (0, (-packed.shape[1]) % 8)))
        signatures, self.inverse, self.counts = np.unique(
            packed, axis=0, return_inverse=True, return_counts=True,
        )
        self.words = np.ascontiguousarray(signatures).view(np.uint64)
        self.populations = np.bitwise_count(self.words).sum(axis=1, dtype=np.uint32)
        self.unique_count = len(signatures)
        self.cache_bytes = int(cache_bytes)
        # Bound temporary pairwise blocks independently of the row cache.
        self.block_rows = max(1, min(256, (16 * 1024**2) // (24 * self.unique_count)))
        self._matrix = None
        self._rows = OrderedDict()
        self._row_capacity = self.cache_bytes // (4 * self.unique_count)

    def _compute_rows(self, indices):
        indices = np.asarray(indices, dtype=np.intp)
        intersection = np.zeros((len(indices), self.unique_count), dtype=np.uint32)
        for word in range(self.words.shape[1]):
            intersection += np.bitwise_count(self.words[indices, word, None] & self.words[None, :, word])
        union = self.populations[indices, None] + self.populations[None, :] - intersection
        distance = np.zeros(intersection.shape, dtype=np.float32)
        np.divide(intersection, union, out=distance, where=union != 0, dtype=np.float32)
        np.subtract(1, distance, out=distance)
        return distance

    def _unique_row(self, signature):
        if self._matrix is not None:
            return self._matrix[signature]
        if signature in self._rows:
            self._rows.move_to_end(signature)
            return self._rows[signature]
        row = self._compute_rows([signature])[0]
        if self._row_capacity:
            if len(self._rows) >= self._row_capacity:
                self._rows.popitem(last=False)
            self._rows[signature] = row
        return row

    def densities(self, tau):
        density = np.empty(self.unique_count, dtype=np.int64)
        if self._matrix is None and 4 * self.unique_count**2 <= self.cache_bytes:
            self._rows.clear()
            self._matrix = np.empty((self.unique_count, self.unique_count), dtype=np.float32)
            for start in range(0, self.unique_count, self.block_rows):
                stop = min(start + self.block_rows, self.unique_count)
                self._matrix[start:stop] = self._compute_rows(np.arange(start, stop))
        for start in range(0, self.unique_count, self.block_rows):
            stop = min(start + self.block_rows, self.unique_count)
            block = (self._matrix[start:stop] if self._matrix is not None
                     else self._compute_rows(np.arange(start, stop)))
            density[start:stop] = (block <= tau) @ self.counts
        result = density[self.inverse]
        # Distinct never-activated tokens have distance 1; only self-distance is 0.
        if tau < 1:
            result += self.populations[self.inverse] == 0
        return result

    def row(self, token, columns=None):
        indices = np.arange(self.size) if columns is None else np.asarray(columns, dtype=np.intp)
        values = self._unique_row(int(self.inverse[token]))[self.inverse[indices]]
        values[indices == token] = 0
        return values

    def candidate_space(self, tokens):
        return np.unique(self.inverse[tokens], return_inverse=True)

    def candidate_row(self, token, signatures):
        # Every unvisited candidate differs from every current cluster member,
        # including when several never-activated tokens share a signature.
        return self._unique_row(int(self.inverse[token]))[signatures]

    def mean_pairwise(self, tokens):
        signatures, counts = np.unique(self.inverse[tokens], return_counts=True)
        total = 0.0
        for signature, count in zip(signatures, counts):
            values = self._unique_row(int(signature))[signatures]
            total += float(np.sum(values * counts, dtype=np.float64)) * int(count)
        # Remove the artificial diagonal of the all-zero signature group.
        total -= int(counts[self.populations[signatures] == 0].sum())
        return total / (len(tokens) * (len(tokens) - 1)) if len(tokens) > 1 else 0.0


class DenseDistances:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1] or not len(self.matrix):
            raise ValueError("Expected a nonempty square distance matrix")
        self.size = len(self.matrix)

    def densities(self, tau):
        return np.count_nonzero(self.matrix <= tau, axis=1)

    def row(self, token, columns=None):
        return self.matrix[token] if columns is None else self.matrix[token, columns]

    def candidate_space(self, tokens):
        return tokens, np.arange(len(tokens))

    def candidate_row(self, token, columns):
        return self.matrix[token, columns]

    def mean_pairwise(self, tokens):
        total = sum(float(np.sum(self.matrix[token, tokens[i + 1:]], dtype=np.float64))
                    for i, token in enumerate(tokens[:-1]))
        return total / (len(tokens) * (len(tokens) - 1) / 2) if len(tokens) > 1 else 0.0
