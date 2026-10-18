import math
import unittest

from worker.engine import Engine, PagedKV


class PagedKVTest(unittest.TestCase):
    def test_page_boundaries_and_reuse(self):
        kv = PagedKV(blocks=3, block_size=2)
        self.assertTrue(kv.reserve("a", 3))
        self.assertFalse(kv.reserve("b", 3))
        for n in range(3):
            kv.append("a", [float(n)], [float(n + 1)])
        self.assertEqual(kv.lengths["a"], 3)
        kv.release("a")
        self.assertEqual(len(kv.free), 3)
        self.assertTrue(kv.reserve("b", 5))

    def test_attention_matches_dense_softmax(self):
        kv = PagedKV(blocks=2, block_size=2)
        kv.reserve("a", 3)
        keys = [[1., 0.], [0., 1.], [1., 1.]]
        vals = [[2., 0.], [0., 4.], [6., 8.]]
        for k, v in zip(keys, vals):
            kv.append("a", k, v)
        q = [1., 2.]
        scores = [sum(a*b for a,b in zip(q,k)) / math.sqrt(2) for k in keys]
        weights = [math.exp(s) for s in scores]
        want = [sum(w * v[d] for w,v in zip(weights, vals)) / sum(weights) for d in range(2)]
        got = kv.attention("a", q)
        for a,b in zip(got,want):
            self.assertAlmostEqual(a,b)

    def test_continuous_admission_and_cleanup(self):
        engine = Engine(blocks=8, block_size=4, max_batch=2)
        first = engine.submit("A", 3)
        engine.tick()
        self.assertEqual(first.produced, 1)
        second = engine.submit("B", 2)
        engine.tick()
        self.assertEqual(first.produced, 2)
        self.assertEqual(second.produced, 1)
        engine.tick()
        self.assertEqual(engine.stats()["active"], 0)
        self.assertEqual(engine.stats()["free_blocks"], 8)
        self.assertEqual(first.produced, 3)
        self.assertEqual(second.produced, 2)

    def test_cancel_releases_pages(self):
        engine = Engine(blocks=4, block_size=4)
        request = engine.submit("abc", 10)
        engine.tick()
        request.cancelled = True
        engine.tick()
        self.assertEqual(engine.stats()["free_blocks"], 4)


if __name__ == "__main__":
    unittest.main()
