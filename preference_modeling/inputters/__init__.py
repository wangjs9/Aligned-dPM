try:
    from preference_modeling.inputters.esc import Inputter as esc
    from preference_modeling.inputters.mic import Inputter as mic
    from preference_modeling.inputters.single import Inputter as single
except ModuleNotFoundError:
    from inputters.esc import Inputter as esc
    from inputters.mic import Inputter as mic
    from inputters.single import Inputter as single
inputters = {
    "esc": esc,
    "mic": mic,
    "single": single,
}
