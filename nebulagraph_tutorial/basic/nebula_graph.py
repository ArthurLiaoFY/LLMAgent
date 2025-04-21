import json

from nebula3.Config import Config, SessionPoolConfig
from nebula3.gclient.net import ConnectionPool
from nebula3.gclient.net.SessionPool import SessionPool
from typing_extensions import List


class NebulaGraph:
    def __init__(self):
        with open("../../secrets.json") as f:
            secrets = json.loads(f.read())

        self.space_name = "players"
        self.session_pool = SessionPool(
            username=secrets.get("nebula_graph").get("user"),
            password=secrets.get("nebula_graph").get("password"),
            space_name=self.space_name,
            addresses=[
                (
                    secrets.get("nebula_graph").get("host"),
                    secrets.get("nebula_graph").get("port"),
                )
            ],
        )
        self.session_pool.init(SessionPoolConfig())

    def execute(self, query: str) -> List[dict]:
        res = self.session_pool.execute_py(query)
        return res.as_primitive()
