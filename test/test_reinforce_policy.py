from DialogueManagement.DialoguePolicy.DeepLearning.ReinforcePolicy import ReinforcePolicy
from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from Dialogue.Action import DialogueActItem, DialogueAct, Operator
import unittest
from subprocess import Popen
import os


class TestReinforcePolicy(unittest.TestCase):
    """ Tests encode_action and decode_action functions of QPolicy.

    """

    db_file = 'flowershop-dbase-test.db'
    ontology_file = 'flowershop-rules-test.json'

    @classmethod
    def setUpClass(cls) -> None:
        # Generate database and ontology files
        create_db_process = Popen(['python', '../createSQLiteDB.py', '-c', 'create_flowershop_DB_test.yaml'])
        create_db_process.wait()

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove database and ontology files
        os.remove(TestReinforcePolicy.db_file)
        os.remove(TestReinforcePolicy.ontology_file)

    def setUp(self):
        ontology = Ontology(TestReinforcePolicy.ontology_file)
        database = DataBase(TestReinforcePolicy.db_file)
        self.policy = ReinforcePolicy(ontology=ontology, database=database, domain='CamRest')

    def test_encoding_decoding_of_acts_for_system(self):
        intents = self.policy.dstc2_acts_sys
        for intent in intents:
            da = DialogueAct(intent=intent)
            da_as_int = self.policy.encode_action([da], system=True)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.policy.decode_action(da_as_int, system=True)[0]
            self.assertEqual(intent, da_decoded.intent)

    def test_encoding_decoding_of_inform_for_system(self):
        slots = self.policy.requestable_slots
        for s in slots:
            da = DialogueAct(intent='inform', params=[DialogueActItem(s, Operator.EQ, '')])
            da_as_int = self.policy.encode_action([da], system=True)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.policy.decode_action(da_as_int, system=True)[0]
            self.assertEqual('inform', da_decoded.intent)
            slot_decoded = da_decoded.params[0].slot
            self.assertEqual(s, slot_decoded, 'Error in slot en- or decoding.')

    def test_encoding_decoding_of_request_for_system(self):
        slots = self.policy.system_requestable_slots
        for s in slots:
            da = DialogueAct(intent='request', params=[DialogueActItem(s, Operator.EQ, '')])
            da_as_int = self.policy.encode_action([da], system=True)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.policy.decode_action(da_as_int, system=True)[0]
            self.assertEqual('request', da_decoded.intent)
            slot_decoded = da_decoded.params[0].slot
            self.assertEqual(s, slot_decoded, 'Error in slot en- or decoding.')

    def test_encoding_decoding_of_acts_for_user(self):
        intents = self.policy.dstc2_acts_usr
        for intent in intents:
            da = DialogueAct(intent=intent)
            da_as_int = self.policy.encode_action([da], system=False)
            da_decoded = self.policy.decode_action(da_as_int, system=False)[0]
            self.assertEqual(intent, da_decoded.intent)

    def test_encoding_decoding_of_inform_for_user(self):
        slots = self.policy.system_requestable_slots
        for s in slots:
            da = DialogueAct(intent='inform', params=[DialogueActItem(s, Operator.EQ, '')])
            da_as_int = self.policy.encode_action([da], system=False)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.policy.decode_action(da_as_int, system=False)[0]
            self.assertEqual('inform', da_decoded.intent)
            slot_decoded = da_decoded.params[0].slot
            self.assertEqual(s, slot_decoded, 'Error in slot en- or decoding.')

    def test_encoding_decoding_of_request_for_user(self):
        slots = self.policy.requestable_slots
        for s in slots:
            da = DialogueAct(intent='request', params=[DialogueActItem(s, Operator.EQ, '')])
            da_as_int = self.policy.encode_action([da], system=False)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.policy.decode_action(da_as_int, system=False)[0]
            self.assertEqual('request', da_decoded.intent)
            slot_decoded = da_decoded.params[0].slot
            self.assertEqual(s, slot_decoded, 'Error in slot en- or decoding.')

    def test_decay_epsilon(self):
        self.policy.epsilon = 1
        self.policy.epsilon_decay = 0.5
        self.policy.epsilon_min = 0.3

        # check that decay works
        self.policy.decay_epsilon()
        self.assertEqual(self.policy.epsilon, 0.5)
        self.policy.decay_epsilon()
        self.assertEqual(self.policy.epsilon, 0.25)

        # check that threshold (epsilon_min) works
        self.policy.decay_epsilon()  # epsilon should not be changed here, as it is already less than 0.3
        self.assertEqual(self.policy.epsilon, 0.25)


if __name__ == '__main__':
    unittest.main()
