import json
from bson import json_util


class Parameter:
    def __init__(self, name=None, value=None, image_url=None, required=True, editable=False, type=None, role=None):
        self.role = role
        self.type = type
        self.name = name
        self.value = value
        self.image_url = image_url
        self.required = required
        self.editable = editable

    def json(self):
        return json.dumps(self.dictionary(), sort_keys=True, default=json_util.default)  #

    def dictionary(self, with_nones=True):
        ret = {
            "name": self.name,
            "value": self.value,
            "image_url": self.image_url,
            "editable": self.editable,
            "type": self.type,
            "required": self.required}

        if with_nones:
            return ret

        return dict([(k, v) for k, v in ret.items() if v is not None])

    @staticmethod
    def from_dict(json_dict):
        return Parameter(name=json_dict.get("name"),
                         value=json_dict.get("value"),
                         image_url=json_dict.get("image_url"),
                         editable=json_dict.get("editable"),
                         required=json_dict.get("required"))


class Task:
    """
    This is a raw task which is submitted by the API owner
    """

    def __init__(self, id=None, expression=None, topic_id=None, goal=None, parameters=[],
                 last_modification_time=None):
        super().__init__()
        # self.param_editable = param_editable
        self.id = id
        self.goal = goal
        self.expression = expression
        self.topic = topic_id
        self.parameters = parameters
        self.last_modification_time = last_modification_time

    def json(self):
        return json.dumps(self.dictionary(), sort_keys=True, default=json_util.default)  #

    def dictionary(self, with_nones=True, cascade=True):
        ret = {
            "id": str(self.id) if self.id else None,
            "goal": self.goal,
            "expression": self.expression,
            "topic": self.topic,
            "parameters": [p.dictionary() for p in self.parameters] if cascade else self.parameters,
            "last_modification_time": self.last_modification_time}

        if with_nones:
            return ret

        return dict([(k, v) for k, v in ret.items() if v is not None])

    @staticmethod
    def from_dict(json_dict):
        if json_dict is None:
            return None

        return Task(id=json_dict.get("id"),
                    goal=json_dict.get("goal"),
                    expression=json_dict.get("expression"),
                    topic_id=json_dict.get("category"),
                    parameters=[Parameter.from_dict(p) for p in json_dict.get("parameters")],
                    last_modification_time=json_dict.get("last_modification_time"))

    def to_string(self, bold=False):
        ret = self.expression
        for p in self.parameters:
            if bold and not p.editable:
                ret = ret.replace(p.name, "<b class='fixed'>" + p.value + "</b>")
            else:
                ret = ret.replace(p.name, p.value)

        return ret

    def param_regex_validator(self, taboo_words=None):

        regex = "".join(["(?=.*" + str(p.value).lower() + ".*)" for p in self.parameters])

        if taboo_words:
            regex += "(?=^((?!({})).)*$)".format("|".join(taboo_words))
        return regex

    def taboo_regex(self, taboo_words: list):
        if not taboo_words:
            return None
        return "(^((?!{}).)*".format("|".join(taboo_words))
