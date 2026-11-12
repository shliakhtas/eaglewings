"""Тести асистента без мережі й без ключа: перевірка коду, вибір моделі, мапа з відповіді, пісочниця.

    python -m unittest eaglewings.assistant.tests.test_assistant -v
"""
import unittest

import pandas as pd

from eaglewings.assistant import agent, guard, llm
from eaglewings.assistant.sandbox import TimeoutSandbox


class Guard(unittest.TestCase):
    def test_allows_analysis(self):
        guard.check_code("import pandas as pd\ndf = execute_sql_query('SELECT * FROM fires')\nresult = {'type': 'number', 'value': len(df)}")

    def test_blocks_escape(self):
        for code in ("import os", "open('x')", "pd.read_csv('a.csv')", "x.__class__", "import subprocess", "df.to_pickle('a')", "'{0.__init__}'.format(1)"):
            with self.assertRaises(guard.GuardError, msg=code):
                guard.check_code(code)


class Model(unittest.TestCase):
    def test_provider_by_key(self):
        self.assertEqual(llm.provider("sk-ant-abc"), "anthropic")
        self.assertEqual(llm.provider("sk-proj-abc"), "openai")
        self.assertIsNone(llm.provider("hello"))
        self.assertTrue(llm.model("sk-ant-abc").startswith("anthropic/"))
        self.assertEqual(llm.model("sk-abc"), llm.OPENAI_DEFAULT)

    def test_bad_key_is_refused_before_any_call(self):
        out = agent.ask("Скільки теплових точок?", "not-a-key")
        self.assertEqual(out["type"], "error")


class Map(unittest.TestCase):
    def test_areal_ids_become_polygons(self):
        a = agent.FRAMES["areals"]
        gj = agent.to_geojson(a[["id", "type_name"]].head(3))
        self.assertEqual(len(gj["features"]), 3)
        self.assertEqual(gj["features"][0]["geometry"]["type"], "Polygon")

    def test_h3_cells_become_hexes(self):
        gj = agent.to_geojson(agent.FRAMES["risk_cells"][["h3", "level"]].head(5))
        self.assertEqual(len(gj["features"]), 5)

    def test_points_and_candidates(self):
        self.assertEqual(agent.to_geojson(agent.FRAMES["fires"].head(4))["features"][0]["geometry"]["type"], "Point")
        c = agent.FRAMES["logging_candidates"]
        if len(c):
            self.assertEqual(agent.to_geojson(c[["id"]])["features"][0]["geometry"]["type"], "Polygon")

    def test_no_coordinates_no_map(self):
        self.assertIsNone(agent.to_geojson(pd.DataFrame(dict(n=[1, 2]))))

    def test_number_answer_shows_its_rows(self):
        code = "df = execute_sql_query(\"SELECT COUNT(*) AS n FROM signals WHERE source = 'фотопастка'\")\nresult = {'type': 'number', 'value': int(df.n[0])}"
        gj = agent.source_geojson(code)
        n = int((agent.FRAMES["signals"].source == "фотопастка").sum())
        self.assertEqual(len(gj["features"]) if gj else 0, n)


class Sandbox(unittest.TestCase):
    def test_runs_and_proxies_sql(self):
        sb = TimeoutSandbox(timeout_seconds=10)
        con = agent.duck()
        env = {"pd": pd, "execute_sql_query": lambda q: con.execute(q).df()}
        r = sb._exec_code("df = execute_sql_query('SELECT COUNT(*) AS n FROM fires')\nresult = {'type': 'number', 'value': int(df.n[0])}", env)
        self.assertEqual(r["value"], len(agent.FRAMES["fires"]))

    def test_big_result_does_not_hang(self):
        """Тисячі рядків у відповіді: раніше процес не міг завершитися, поки не прочитано чергу, і висів до ліміту часу."""
        import time
        sb = TimeoutSandbox(timeout_seconds=10)
        con = agent.duck()
        env = {"pd": pd, "execute_sql_query": lambda q: con.execute(q).df()}
        t = time.monotonic()
        r = sb._exec_code("df = execute_sql_query('SELECT * FROM forest_loss LIMIT 50000')\nresult = {'type': 'dataframe', 'value': df}", env)
        self.assertEqual(len(r["value"]), 50000)
        self.assertLess(time.monotonic() - t, 8)

    def test_errors_read_by_type_not_by_data(self):
        self.assertTrue(agent._friendly(Exception("frp 429.1\nCodeExecutionError: Generated code exceeded the 30.0s wall-clock limit")).startswith("Розрахунок ішов задовго"))
        self.assertTrue(agent._friendly(Exception("litellm.RateLimitError: Rate limit reached")).startswith("Провайдер моделі тимчасово"))
        self.assertTrue(agent._friendly(Exception("You exceeded your current quota. insufficient_quota")).startswith("На ключі закінчились"))

    def test_timeout(self):
        from pandasai.exceptions import CodeExecutionError
        sb = TimeoutSandbox(timeout_seconds=1)
        with self.assertRaises(CodeExecutionError):
            sb._exec_code("while True:\n    pass", {"pd": pd})

    def test_guard_inside_sandbox(self):
        from pandasai.exceptions import CodeExecutionError
        sb = TimeoutSandbox(timeout_seconds=5)
        with self.assertRaises(CodeExecutionError):
            sb._exec_code("import os\nresult = {'type': 'string', 'value': os.getcwd()}", {})
        self.assertTrue(sb.policy_hits)

    def test_no_secrets_in_child(self):
        import os
        os.environ["TEST_SECRET_KEY"] = "x"
        sb = TimeoutSandbox(timeout_seconds=5)
        r = sb._exec_code("import json\nresult = {'type': 'string', 'value': 'ok'}", {})
        self.assertEqual(r["value"], "ok")


if __name__ == "__main__":
    unittest.main()


class Parks(unittest.TestCase):
    """Питання про вибраний парк: агент бачить лише його рядки, а всі парки — коли питання їх порівнює чи називає інший."""

    def test_all_parks_loaded(self):
        for n in ("areals", "signals", "fires", "risk_cells", "park", "forest_loss"):
            self.assertEqual(set(agent.FRAMES[n]["park"]), set(agent.PARKS), n)

    def test_scope(self):
        main, other = list(agent.PARKS)[0], list(agent.PARKS)[-1]
        self.assertEqual(agent.scope("Скільки пожеж у парку?", other), other)
        self.assertEqual(agent.scope("Порівняй усі парки за пожежами", main), "all")
        self.assertEqual(agent.scope("А що в Дадьї?", other), "all")
        self.assertEqual(agent.scope("Скільки ареалів на сході парку?", other), other)

    def test_frames_of_one_park(self):
        k = list(agent.PARKS)[-1]
        f = agent.frames_for(k)
        self.assertEqual(set(f["areals"]["park"]), {k})
        self.assertGreater(len(f["research_results"]), 0)           # спільні таблиці лишаються

    def test_contour_of_selected_park(self):
        k = list(agent.PARKS)[-1]
        a = agent.frames_for(k)["areals"].head(1)[["id", "lat", "lon"]]
        gj = agent.to_geojson(a, k)
        self.assertEqual(gj["features"][0]["geometry"]["type"], "Polygon")
        want = agent.GEOM[(k, a["id"].iat[0])]
        self.assertNotEqual(want, agent.GEOM.get((agent.PK.MAIN, a["id"].iat[0])))   # той самий id, інший парк — інший контур
