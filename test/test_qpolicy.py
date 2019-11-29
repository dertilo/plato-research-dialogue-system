from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import QPolicy
from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from Dialogue.Action import DialogueActItem, DialogueAct, Operator
import unittest
from subprocess import Popen
import os


class TestActionEncodingDecoding(unittest.TestCase):
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
        os.remove(TestActionEncodingDecoding.db_file)
        os.remove(TestActionEncodingDecoding.ontology_file)

    def setUp(self):
        ontology = Ontology(TestActionEncodingDecoding.ontology_file)
        database = DataBase(TestActionEncodingDecoding.db_file)
        self.qp = QPolicy(ontology=ontology, database=database, domain='CamRest')

    def test_encoding_decoding_of_acts_for_system(self):
        intents = self.qp.dstc2_acts_sys
        for intent in intents:
            da = DialogueAct(intent=intent)
            da_as_int = self.qp.encode_action([da], system=True)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.qp.decode_action(da_as_int, system=True)[0]
            self.assertEqual(intent, da_decoded.intent)

    def test_encoding_decoding_of_inform_for_system(self):
        slots = self.qp.requestable_slots
        for s in slots:
            da = DialogueAct(intent='inform', params=[DialogueActItem(s, Operator.EQ, '')])
            da_as_int = self.qp.encode_action([da], system=True)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.qp.decode_action(da_as_int, system=True)[0]
            self.assertEqual('inform', da_decoded.intent)
            slot_decoded = da_decoded.params[0].slot
            self.assertEqual(s, slot_decoded, 'Error in slot en- or decoding.')

    def test_encoding_decoding_of_request_for_system(self):
        slots = self.qp.system_requestable_slots
        for s in slots:
            da = DialogueAct(intent='request', params=[DialogueActItem(s, Operator.EQ, '')])
            da_as_int = self.qp.encode_action([da], system=True)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.qp.decode_action(da_as_int, system=True)[0]
            self.assertEqual('request', da_decoded.intent)
            slot_decoded = da_decoded.params[0].slot
            self.assertEqual(s, slot_decoded, 'Error in slot en- or decoding.')

    def test_encoding_decoding_of_acts_for_user(self):
        intents = self.qp.dstc2_acts_usr
        for intent in intents:
            da = DialogueAct(intent=intent)
            da_as_int = self.qp.encode_action([da], system=False)
            da_decoded = self.qp.decode_action(da_as_int, system=False)[0]
            self.assertEqual(intent, da_decoded.intent)

    def test_encoding_decoding_of_inform_for_user(self):
        slots = self.qp.system_requestable_slots
        for s in slots:
            da = DialogueAct(intent='inform', params=[DialogueActItem(s, Operator.EQ, '')])
            da_as_int = self.qp.encode_action([da], system=False)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.qp.decode_action(da_as_int, system=False)[0]
            self.assertEqual('inform', da_decoded.intent)
            slot_decoded = da_decoded.params[0].slot
            self.assertEqual(s, slot_decoded, 'Error in slot en- or decoding.')

    def test_encoding_decoding_of_request_for_user(self):
        slots = self.qp.requestable_slots
        for s in slots:
            da = DialogueAct(intent='request', params=[DialogueActItem(s, Operator.EQ, '')])
            da_as_int = self.qp.encode_action([da], system=False)
            self.assertNotEqual(-1, da_as_int)
            da_decoded = self.qp.decode_action(da_as_int, system=False)[0]
            self.assertEqual('request', da_decoded.intent)
            slot_decoded = da_decoded.params[0].slot
            self.assertEqual(s, slot_decoded, 'Error in slot en- or decoding.')


if __name__ == '__main__':
    unittest.main()
