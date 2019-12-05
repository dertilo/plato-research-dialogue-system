from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from DialogueManagement.DialogueManager import DialogueManager
import unittest
from subprocess import Popen
import os


class TestDialogManager(unittest.TestCase):
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
        os.remove(TestDialogManager.db_file)
        os.remove(TestDialogManager.ontology_file)

    def setUp(self):
        self.ontology = Ontology(TestDialogManager.ontology_file)
        self.database = DataBase(TestDialogManager.db_file)

        dm_settings = {'DIALOGUE': {}}
        dm_settings['DIALOGUE']['domain'] = 'CamRest'
        self.dm_args = dict(
            zip(
                ['settings', 'ontology', 'database', 'domain', 'agent_id',
                 'agent_role'],
                [dm_settings,
                 self.ontology,
                 self.database,
                 'CamRest',
                 0,
                 'system'
                 ]
            )
        )

        # Use extreme values to ensure that no defaul settings do influence the test
        self.init_tests_train_params = {'learning_rate': -1.0,
                        'discount_factor': -2.0,
                        'exploration_rate': -3.0,
                        'learning_decay_rate': -4.0,
                        'exploration_decay_rate': -5.0,
                        'min_exploration_rate': -6.0}

    def test_qpolicy_train_parameter_init(self):
        self.dm_args['policy'] = self.init_tests_train_params
        self.dm_args['policy']['type'] = 'q_learning'

        dm = DialogueManager(args=self.dm_args)
        initialized_policy = dm.policy
        self.assertEqual(initialized_policy.alpha, self.init_tests_train_params['learning_rate'])
        self.assertEqual(initialized_policy.alpha_decay, self.init_tests_train_params['learning_decay_rate'])
        self.assertEqual(initialized_policy.gamma, self.init_tests_train_params['discount_factor'])
        self.assertEqual(initialized_policy.epsilon, self.init_tests_train_params['exploration_rate'])
        self.assertEqual(initialized_policy.epsilon_decay, self.init_tests_train_params['exploration_decay_rate'])
        self.assertEqual(initialized_policy.epsilon_min, self.init_tests_train_params['min_exploration_rate'])

    def test_reinforce_train_parameter_init(self):
        self.dm_args['policy'] = self.init_tests_train_params
        self.dm_args['policy']['type'] = 'reinforce'

        dm = DialogueManager(args=self.dm_args)
        initialized_policy = dm.policy
        self.assertEqual(initialized_policy.alpha, self.init_tests_train_params['learning_rate'])
        self.assertEqual(initialized_policy.alpha_decay_rate, self.init_tests_train_params['learning_decay_rate'])
        self.assertEqual(initialized_policy.gamma, self.init_tests_train_params['discount_factor'])
        self.assertEqual(initialized_policy.epsilon, self.init_tests_train_params['exploration_rate'])
        self.assertEqual(initialized_policy.exploration_decay_rate, self.init_tests_train_params['exploration_decay_rate'])
        self.assertEqual(initialized_policy.epsilon_min, self.init_tests_train_params['min_exploration_rate'])

    def test_wolf_phc_train_parameter_init(self):
        self.dm_args['policy'] = self.init_tests_train_params
        self.dm_args['policy']['type'] = 'wolf_phc'

        dm = DialogueManager(args=self.dm_args)
        initialized_policy = dm.policy
        self.assertEqual(initialized_policy.alpha, self.init_tests_train_params['learning_rate'])
        self.assertEqual(initialized_policy.alpha_decay, self.init_tests_train_params['learning_decay_rate'])
        self.assertEqual(initialized_policy.gamma, self.init_tests_train_params['discount_factor'])
        self.assertEqual(initialized_policy.epsilon, self.init_tests_train_params['exploration_rate'])
        self.assertEqual(initialized_policy.epsilon_decay, self.init_tests_train_params['exploration_decay_rate'])
        self.assertEqual(initialized_policy.epsilon_min, self.init_tests_train_params['min_exploration_rate'])


if __name__ == '__main__':
    unittest.main()
