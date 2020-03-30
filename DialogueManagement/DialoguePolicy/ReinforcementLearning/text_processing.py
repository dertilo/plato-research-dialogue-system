from typing import List, Tuple
import re

from torchtext.data import Example, Field

def regex_tokenizer(text, pattern=r"(?u)(?:\b\w\w+\b|\S)")->List[str]:
    return [m.group() for m in re.finditer(pattern, text)]

if __name__ == '__main__':
    texts = [
        '{"intents": [], "is_terminal_state": false, "last_sys_acts": null, "slots_filled": ["chairname"], "slot_queries": {}, "requested_slot": "", "user_acts": [{"name": "dialogue_act", "funcName": null, "params": [{"slot": "chairname", "op": {}, "value": "control of convergent access networks"}], "intent": "inform"}], "system_made_offer": false, "db_matches_ratio": 0, "turn": 1, "num_dontcare": 0}',
        '{"intents": [], "is_terminal_state": false, "last_sys_acts": null, "slots_filled": ["chairname"], "slot_queries": {}, "requested_slot": "", "user_acts": [{"name": "dialogue_act", "funcName": null, "params": [{"slot": "chairname", "op": {}, "value": "control of convergent access networks"}], "intent": "inform"}]'
    ]
    TEXT_FIELD = Field(include_lengths=True, batch_first=True,tokenize=regex_tokenizer)
    examples = [Example.fromlist([text], [('dialog_state',TEXT_FIELD)]) for text in texts]
    TEXT_FIELD.build_vocab([example.dialog_state for example in examples])

    batch,lenghts = TEXT_FIELD.process([e.dialog_state for e in examples])
    print()