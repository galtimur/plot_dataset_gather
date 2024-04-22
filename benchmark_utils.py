from typing import TYPE_CHECKING, Callable, Dict

import pandas as pd


class TaskChanger:
    #  TODO: Lets design it in a way that user wil inherit this class to write custom descriptions
    """
    Class that changes the task for the plotting
    init take three callables:

    setup_changer(task_text: str) -> str: takes setup description, returns altered one
    data_descr_changer(task_text: str, df: pd.DataFrame) -> str: takes data description, dataframe, returns altered one
    style_changer(task_text: str) -> str: takes plot style description, returns altered one
    """

    def __init__(
        self,
        setup_changer: Callable[[str], str] | None = None,
        data_descr_changer: Callable[[str, pd.DataFrame], str] | None = None,
        style_changer: Callable[[str], str] | None = None,
    ):
        self.setup_changer = setup_changer
        self.data_descr_changer = data_descr_changer
        self.style_changer = style_changer

        self.cahnge_mapping = {
            'python_desciption': self.get_pycharm_desriptions
        }

    # self.cahnge_mapping['python_desciption']()


    # TODO how to annotate datapoint: PlotDataPoint without circular import?
    def change_task(self, datapoint) -> Dict:
        task = datapoint.task
        df = pd.read_csv(datapoint.dp_path / "data.csv")

        if self.setup_changer is not None:
            task["setup"] = self.setup_changer(task_text=task["setup"])
        if self.data_descr_changer is not None:
            task["data description"] = self.data_descr_changer(
                task_text=task["data description"], df=df
            )
        if self.style_changer is not None:
            task["plot style"] = self.style_changer(task_text=task["plot style"])

        return datapoint.task


    def get_pycharm_desriptions(self):
        pass