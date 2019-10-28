from Domain import Ontology, DataBase
from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import QPolicy
from Dialogue.Action import DialogueAct, DialogueActItem, Operator

ontology = Ontology.Ontology('../alex-plato/Domain/alex-rules.json')
#print(ontology.ontology)

database = DataBase.DataBase('../alex-plato/Domain/alex-dbase.db')
#print(database)

qp = QPolicy(ontology, database, agent_role='system', domain='CamRest')

slot_day = DialogueActItem('day', Operator.EQ, '')
act_request = DialogueAct(intent='request', params=[slot_day])


encoded_act = qp.encode_action([act_request], system=True)
print('Encoded act: {}'.format(encoded_act))

decoded_act = qp.decode_action(encoded_act)
print('Decoded act: {}'.format(decoded_act))
