"""The equivalence gate must reject incomplete or misleading comparisons."""
import copy
import unittest
from scripts.parity import compare_answers


def choice(probabilities, confidence=0.8):
    highest = max(probabilities.values())
    selected = next((label for label, value in probabilities.items() if value == highest),
                    next(iter(probabilities)))
    return {'type': 'choice', 'choice': selected,
            'probabilities': probabilities, 'confidence': confidence}


def cases(count=100):
    return [{'q': choice({'yes': .8, 'no': .2})} for _ in range(count)]


class ParityTests(unittest.TestCase):
    def test_identical_and_reordered_option_maps(self):
        reference = cases()
        actual = [{'q': choice({'no': .2, 'yes': .8})} for _ in reference]
        result = compare_answers(reference, actual)
        self.assertTrue(result['passes_gate'])
        self.assertEqual(result['same_top_pct'], 100)
        self.assertAlmostEqual(result['mean_kl'], 0)

    def test_confident_flip_rejects_even_99_percent_agreement(self):
        reference = cases()
        actual = copy.deepcopy(reference)
        actual[0]['q'] = choice({'yes': .2, 'no': .8})
        result = compare_answers(reference, actual)
        self.assertEqual(result['same_top_pct'], 99)
        self.assertFalse(result['passes_gate'])
        self.assertEqual(len(result['confident_flips']), 1)

    def test_no_confident_flips_still_requires_99_percent(self):
        reference = cases()
        for value in reference: value['q']['confidence'] = .6
        actual = copy.deepcopy(reference)
        for i in range(2): actual[i]['q'] = choice({'yes': .49, 'no': .51}, .6)
        result = compare_answers(reference, actual)
        self.assertFalse(result['passes_gate'])
        self.assertEqual(result['confident_flips'], [])

    def test_kl_rejects_probability_drift_without_top_flips(self):
        result = compare_answers(cases(), [{'q': choice({'yes': .51, 'no': .49})}]*100)
        self.assertEqual(result['same_top_pct'], 100)
        self.assertFalse(result['passes_gate'])

    def test_incomplete_or_invalid_answers_refuse_comparison(self):
        for actual in [cases(99), [{'other': choice({'yes': .8, 'no': .2})}]*100,
                       [{'q': choice({'yes': float('nan'), 'no': .2})}]*100,
                       [{'q': choice({'yes': -1, 'no': 2})}]*100,
                       [{'q': choice({'yes': 0, 'no': 0})}]*100,
                       [{'q': choice({'maybe': .8, 'no': .2})}]*100]:
            with self.subTest(actual=repr(actual[:1])):
                with self.assertRaises(ValueError): compare_answers(cases(), actual)

    def test_binary_and_score_probabilities(self):
        reference = [{'binary': {'type':'noul','noul':.8,'confidence':.8},
                      'score': {'type':'score','score':1.8,'probabilities':{'0':.1,'1':0,'2':.9},'confidence':.8}}]
        result = compare_answers(reference, copy.deepcopy(reference))
        self.assertEqual(result['decisions'], 2)
        self.assertTrue(result['passes_gate'])

    def test_reported_choice_flip_is_not_hidden_by_rounded_tie(self):
        reference = [{'q': {'type': 'choice', 'choice': 'yes',
                            'probabilities': {'yes': .5, 'no': .5}, 'confidence': .8}}]
        actual = copy.deepcopy(reference)
        actual[0]['q']['choice'] = 'no'
        result = compare_answers(reference, actual)
        self.assertEqual(result['same_top_pct'], 0)
        self.assertFalse(result['passes_gate'])

    def test_reported_choice_below_rounded_maximum_is_invalid(self):
        reference = [{'q': {'type': 'choice', 'choice': 'no',
                            'probabilities': {'yes': .3333, 'no': .3334, 'maybe': .3333},
                            'confidence': .8}}]
        actual = copy.deepcopy(reference)
        actual[0]['q']['choice'] = 'yes'
        with self.assertRaises(ValueError): compare_answers(reference, actual)

    def test_empty_suite_rejected(self):
        with self.assertRaises(ValueError): compare_answers([], [])
